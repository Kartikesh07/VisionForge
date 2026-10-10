"""Train the custom CNN or two-phase ResNet18 using training and validation only."""

from __future__ import annotations

import os
# Required by CUDA's deterministic matrix operations; set before CUDA initializes.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from dataclasses import asdict, dataclass, replace
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score

from .dataset import IMAGENET_MEAN, IMAGENET_STD, create_dataloaders, seed_everything
from .models import ARCHITECTURE_ID, CustomCNN, ResNet18Classifier, load_checkpoint
from .split_data import CLASSES, ROOT


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    batch_size: int = 32
    learning_rate: float = 0.0003
    weight_decay: float = 0.01
    max_epochs: int = 20
    patience: int = 5
    dropout: float = 0.3
    num_workers: int = 0
    cpu_threads: int = 1
    mixed_precision: bool = True
    architecture: str = "custom_cnn"
    head_epochs: int = 5
    fine_tune_learning_rate: float = 1e-5

    def validate(self):
        if not 1 <= self.max_epochs <= 20 or self.patience < 1:
            raise ValueError("max_epochs must be 1..20 and patience must be positive")
        if self.batch_size < 1 or self.num_workers < 0 or self.cpu_threads < 1:
            raise ValueError("Invalid batch size, worker count, or CPU thread count")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Learning rate must be positive and weight decay nonnegative")
        if self.architecture not in ("custom_cnn", "resnet18"):
            raise ValueError("architecture must be custom_cnn or resnet18")
        if self.architecture == "resnet18" and not 1 <= self.head_epochs < self.max_epochs:
            raise ValueError("ResNet18 requires at least one epoch for each phase")
        if self.fine_tune_learning_rate <= 0:
            raise ValueError("fine-tune learning rate must be positive")


def configure_reproducibility(seed: int):
    seed_everything(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def run_epoch(model, loader, criterion, device, *, optimizer=None, scaler=None) -> dict:
    """Sample-weighted loss and six-class metrics; gradients only during training."""
    training = optimizer is not None
    model.train(training)
    total_loss, total_samples = 0.0, 0
    targets, predictions = [], []
    amp = scaler is not None and scaler.is_enabled()
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=loader.pin_memory)
            labels = labels.to(device, non_blocking=loader.pin_memory)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(images)
                loss = criterion(logits, labels)
            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite loss encountered")
            if training:
                if amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            size = labels.shape[0]
            total_loss += float(loss.item()) * size
            total_samples += size
            targets.extend(labels.detach().cpu().tolist())
            predictions.extend(logits.detach().argmax(dim=1).cpu().tolist())
    if not total_samples:
        raise ValueError("Cannot compute metrics from an empty DataLoader")
    return {"loss": total_loss / total_samples,
            "accuracy": float(accuracy_score(targets, predictions)),
            "macro_f1": float(f1_score(targets, predictions, labels=list(range(6)),
                                       average="macro", zero_division=0)),
            "samples": total_samples}


def save_checkpoint(path: Path, model: nn.Module, config: TrainingConfig, epoch: int,
                    validation: dict, *, phase: str = "custom_cnn", phase_epoch: int | None = None) -> None:
    checkpoint = {
        "schema_version": 1, "architecture": model.architecture_id,
        "pretrained": model.pretrained, "model_config": model.model_config,
        "pretrained_weights": getattr(model, "pretrained_weights", None),
        "phase": phase, "phase_epoch": phase_epoch,
        "model_state_dict": {name: tensor.detach().cpu().clone()
                             for name, tensor in model.state_dict().items()},
        "class_names": list(CLASSES),
        "class_to_idx": {name: index for index, name in enumerate(CLASSES)},
        "training_config": asdict(config), "epoch": epoch,
        "validation_metrics": validation,
        "preprocessing": {"image_size": [224, 224], "mean": list(IMAGENET_MEAN),
                          "std": list(IMAGENET_STD)},
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def train_attempt(config: TrainingConfig, processed_dir: Path, checkpoint_path: Path,
                  device: torch.device) -> dict:
    configure_reproducibility(config.seed)
    torch.set_num_threads(config.cpu_threads)
    loaders = create_dataloaders(processed_dir, batch_size=config.batch_size,
                                seed=config.seed, num_workers=config.num_workers,
                                splits=("train", "val"))
    transfer = config.architecture == "resnet18"
    model = (ResNet18Classifier() if transfer else CustomCNN(dropout=config.dropout)).to(device)
    phases = ([("head", config.head_epochs, config.learning_rate),
               ("fine_tune", config.max_epochs - config.head_epochs, config.fine_tune_learning_rate)]
              if transfer else [("custom_cnn", config.max_epochs, config.learning_rate)])
    criterion = nn.CrossEntropyLoss()
    history, best_score, best_epoch, phase_results = [], -1.0, 0, []
    started = time.perf_counter()
    for phase, budget, learning_rate in phases:
        if transfer:
            if phase == "fine_tune":
                best_head = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                model.load_state_dict(best_head["model_state_dict"])
                del best_head
            model.set_phase(phase)
        optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad),
                                      lr=learning_rate, weight_decay=config.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=config.mixed_precision and device.type == "cuda")
        phase_best, stale_epochs = -1.0, 0
        phase_started = time.perf_counter()
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        print(f"Phase {phase}: LR={learning_rate:g}, trainable parameters={trainable:,}", flush=True)
        for phase_epoch in range(1, budget + 1):
            epoch = len(history) + 1
            epoch_started = time.perf_counter()
            training = run_epoch(model, loaders["train"], criterion, device, optimizer=optimizer, scaler=scaler)
            validation = run_epoch(model, loaders["val"], criterion, device, scaler=scaler)
            improved = validation["macro_f1"] > best_score
            if improved:
                best_score, best_epoch = validation["macro_f1"], epoch
                save_checkpoint(checkpoint_path, model, config, epoch, validation,
                                phase=phase, phase_epoch=phase_epoch)
            if validation["macro_f1"] > phase_best:
                phase_best, stale_epochs = validation["macro_f1"], 0
            else:
                stale_epochs += 1
            row = {"epoch": epoch, "phase": phase, "phase_epoch": phase_epoch,
                   "learning_rate": learning_rate, "trainable_parameters": trainable,
                   "train": training, "val": validation,
                   "duration_seconds": time.perf_counter() - epoch_started, "checkpoint_improved": improved}
            history.append(row)
            print(f"Epoch {epoch:02} ({phase} {phase_epoch}/{budget}) | "
                  f"train loss={training['loss']:.4f} acc={training['accuracy']:.4f} F1={training['macro_f1']:.4f} | "
                  f"val loss={validation['loss']:.4f} acc={validation['accuracy']:.4f} F1={validation['macro_f1']:.4f} | "
                  f"{row['duration_seconds']:.1f}s{' *best' if improved else ''}", flush=True)
            if stale_epochs >= config.patience:
                break
        phase_results.append({"phase": phase, "epochs_completed": phase_epoch,
                              "early_stopped": stale_epochs >= config.patience,
                              "best_macro_f1": phase_best, "learning_rate": learning_rate,
                              "trainable_parameters": trainable,
                              "duration_seconds": time.perf_counter() - phase_started})
    selected = history[best_epoch - 1]
    return {"history": history, "best_epoch": best_epoch,
            "best_phase": selected["phase"], "best_phase_epoch": selected["phase_epoch"],
            "phases": phase_results, "best_validation": selected["val"],
            "highest_validation_accuracy": max(row["val"]["accuracy"] for row in history),
            "epochs_completed": len(history), "early_stopped": phase_results[-1]["early_stopped"],
            "duration_seconds": time.perf_counter() - started,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "dataset_counts": {split: len(loader.dataset) for split, loader in loaders.items()}}


def plot_learning_curves(history: list[dict], best_epoch: int, path: Path):
    figure, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    epochs = [row["epoch"] for row in history]
    for axis, metric, title in zip(axes, ("loss", "accuracy", "macro_f1"), ("Cross-entropy loss", "Accuracy", "Macro-F1")):
        for split, label in (("train", "Training"), ("val", "Validation")):
            axis.plot(epochs, [row[split][metric] for row in history], marker="o", markersize=3, label=label)
        axis.axvline(best_epoch, color="gray", linestyle="--", alpha=0.7, label="Selected epoch")
        axis.set(xlabel="Epoch", title=title)
        if metric != "loss":
            axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.25)
        axis.legend()
    transfer = any(row.get("phase") in ("head", "fine_tune") for row in history)
    if transfer:
        fine_tune_start = next((row["epoch"] for row in history if row["phase"] == "fine_tune"), None)
        if fine_tune_start is not None:
            for axis in axes:
                axis.axvline(fine_tune_start - 0.5, color="purple", linestyle=":", label="Phase 2 begins")
                axis.legend()
    figure.suptitle(f"VisionForge {'ResNet18' if transfer else 'custom CNN'} — training and validation")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def train(config: TrainingConfig = TrainingConfig(), *, processed_dir: Path = ROOT / "data/processed",
          checkpoint_path: Path | None = None,
          report_dir: Path = ROOT / "artifacts/reports", require_cuda: bool = False) -> dict:
    config.validate()
    transfer = config.architecture == "resnet18"
    prefix = "resnet18" if transfer else "baseline_cnn"
    checkpoint_path = checkpoint_path or ROOT / "artifacts/models" / ("resnet18_best.pth" if transfer else "baseline_cnn.pth")
    if transfer:
        torch.hub.set_dir(str(ROOT / ".cache/torch"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required for this run but unavailable")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    attempt_path = checkpoint_path.with_name("." + checkpoint_path.name + ".attempt")
    original_config, retries = config, []
    started = time.perf_counter()
    print(f"Device: {device}; GPU: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'none'}", flush=True)
    while True:
        try:
            result = train_attempt(config, processed_dir, attempt_path, device)
            break
        except torch.cuda.OutOfMemoryError:
            if device.type != "cuda" or config.batch_size == 1:
                raise
            reduced = max(1, config.batch_size // 2)
            retries.append({"failed_batch_size": config.batch_size, "retry_batch_size": reduced})
            config = replace(config, batch_size=reduced)
            print(f"CUDA out of memory: restarting from seed {config.seed} with batch size {reduced}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    attempt_path.replace(checkpoint_path)
    # Reconstruct and re-evaluate the saved checkpoint on validation only.
    best_model, saved = load_checkpoint(checkpoint_path, device)
    validation_loader = create_dataloaders(processed_dir, batch_size=config.batch_size,
                                          seed=config.seed, num_workers=config.num_workers,
                                          splits=("val",))["val"]
    verification_scaler = torch.amp.GradScaler("cuda", enabled=config.mixed_precision and device.type == "cuda")
    verified = run_epoch(best_model, validation_loader, nn.CrossEntropyLoss(), device, scaler=verification_scaler)
    if any(abs(verified[key] - saved["validation_metrics"][key]) > 1e-6
           for key in ("loss", "accuracy", "macro_f1")):
        raise RuntimeError("Reloaded checkpoint validation metrics differ from saved metrics")
    result.update({"status": "completed", "architecture": saved.get("architecture", ARCHITECTURE_ID),
                   "pretrained": transfer, "class_names": list(CLASSES),
                   "pretrained_weights": saved.get("pretrained_weights"),
                   "training_config": asdict(config), "requested_config": asdict(original_config),
                   "device": str(device), "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                   "torch_version": str(torch.__version__), "cuda_runtime": torch.version.cuda,
                   "deterministic_algorithms": True, "oom_retries": retries,
                   "total_duration_seconds": time.perf_counter() - started,
                   "checkpoint": str(checkpoint_path), "checkpoint_reload_verified": True,
                   "reloaded_validation": verified, "test_split_used": False,
                   "split_manifest_sha256": hashlib.sha256((processed_dir / "split_manifest.json").read_bytes()).hexdigest()})
    plot_learning_curves(result["history"], result["best_epoch"], report_dir / f"{prefix}_learning_curves.png")
    (report_dir / f"{prefix}_training.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("best_epoch", "best_validation", "epochs_completed",
                                                  "total_duration_seconds", "oom_retries")}, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--architecture", choices=("custom_cnn", "resnet18"), default="custom_cnn")
    args = parser.parse_args()
    if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        parser.error("Use the project's .venv Python interpreter")
    train(TrainingConfig(batch_size=args.batch_size, num_workers=args.num_workers,
                         architecture=args.architecture), require_cuda=args.require_cuda)


if __name__ == "__main__":
    main()
