"""Audit split leakage before evaluating the two frozen checkpoints on test data."""

from __future__ import annotations

from .train import configure_reproducibility  # Sets deterministic CUDA environment first.

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
from torch.utils.data import SequentialSampler

from .dataset import IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE, create_dataloaders
from .models import load_checkpoint
from .split_data import CLASSES, ROOT, DatasetError, inspect_raw, make_manifest, verify_processed


# Fixed before inspecting the test split; these are audit criteria, not model tuning.
HASH_DISTANCE = 6
SSIM_THRESHOLD = 0.95
ORIENTATIONS = ("original", "horizontal_flip", "vertical_flip", "rotate_180")
CHECKPOINTS = {"custom_cnn": "baseline_cnn.pth", "resnet18": "resnet18_best.pth"}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def orient(pixels: np.ndarray, orientation: str) -> np.ndarray:
    if orientation == "original":
        return pixels
    if orientation == "horizontal_flip":
        return pixels[:, ::-1]
    if orientation == "vertical_flip":
        return pixels[::-1, :]
    if orientation == "rotate_180":
        return pixels[::-1, ::-1]
    raise ValueError(f"Unknown orientation: {orientation}")


def perceptual_hash(pixels: np.ndarray) -> int:
    """63 low-frequency DCT bits, excluding DC (brightness)."""
    small = cv2.resize(pixels.astype(np.float32), (32, 32), interpolation=cv2.INTER_AREA)
    coefficients = cv2.dct(small)[:8, :8].ravel()[1:]
    result = 0
    for bit in coefficients > np.median(coefficients):
        result = (result << 1) | int(bit)
    return result


def structural_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Local SSIM on original pixels, Gaussian 11x11, sigma 1.5, range 255."""
    if left.shape != right.shape or min(left.shape) < 11:
        raise ValueError("SSIM requires equal image shapes at least 11x11")
    left, right = left.astype(np.float64), right.astype(np.float64)
    blur = lambda values: cv2.GaussianBlur(values, (11, 11), 1.5)
    mu_left, mu_right = blur(left), blur(right)
    variance_left = np.maximum(blur(left * left) - mu_left * mu_left, 0)
    variance_right = np.maximum(blur(right * right) - mu_right * mu_right, 0)
    covariance = blur(left * right) - mu_left * mu_right
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    score = ((2 * mu_left * mu_right + c1) * (2 * covariance + c2)
             / ((mu_left ** 2 + mu_right ** 2 + c1) * (variance_left + variance_right + c2)))
    return float(score[5:-5, 5:-5].mean())


def find_cross_split_duplicates(records: list[dict], pixels: list[np.ndarray]) -> dict:
    """Compare all cross-split pairs without filtering by class label.

    DCT hashes screen candidates; full-resolution SSIM confirms suspicious pairs.
    Flips and 180-degree rotations are considered without writing transformed files.
    Exact transformed pixel matches bypass the perceptual screen.
    """
    if len(records) != len(pixels):
        raise ValueError("Records and decoded image arrays must have equal lengths")
    hashes = [[perceptual_hash(orient(array, variant)) for variant in ORIENTATIONS]
              for array in pixels]
    transformed_sha = [[hashlib.sha256(orient(array, variant).tobytes()).hexdigest()
                        for variant in ORIENTATIONS] for array in pixels]
    exact, near = [], []
    pairs_checked, candidates = 0, 0
    for i, left in enumerate(records):
        for j in range(i + 1, len(records)):
            right = records[j]
            if left["split"] == right["split"]:
                continue
            pairs_checked += 1
            pair = {"left": {key: left[key] for key in ("source", "split", "class")},
                    "right": {key: right[key] for key in ("source", "split", "class")}}
            if left["sha256"] == right["sha256"] or left["pixel_sha256"] == right["pixel_sha256"]:
                exact.append({**pair, "reason": "identical original file or decoded pixels"})
                continue
            best = None
            for k, variant in enumerate(ORIENTATIONS):
                distance = (hashes[i][0] ^ hashes[j][k]).bit_count()
                transformed_exact = transformed_sha[i][0] == transformed_sha[j][k]
                if distance > HASH_DISTANCE and not transformed_exact:
                    continue
                candidates += 1
                similarity = structural_similarity(pixels[i], orient(pixels[j], variant))
                if transformed_exact or similarity >= SSIM_THRESHOLD:
                    match = {**pair, "orientation_applied_to_right": variant,
                             "hash_distance": distance, "ssim": similarity,
                             "transformed_pixels_identical": transformed_exact}
                    if best is None or similarity > best["ssim"]:
                        best = match
            if best is not None:
                near.append(best)
    return {"exact_cross_split_pairs": exact, "suspicious_near_duplicate_pairs": near,
            "cross_split_pairs_checked": pairs_checked, "candidate_orientations_checked": candidates}


def audit_dataset(raw: Path, processed: Path, *, expected_per_class: int = 300) -> dict:
    started = time.perf_counter()
    protocol = {"perceptual_hash": "63-bit DCT, 32x32 resize, 8x8 low frequencies excluding DC",
                "maximum_hash_distance": HASH_DISTANCE, "minimum_ssim": SSIM_THRESHOLD,
                "ssim": "original-resolution local Gaussian 11x11, sigma 1.5, range 255",
                "orientations": list(ORIENTATIONS), "all_cross_split_class_pairs_checked": True,
                "limitation": "Screens aligned and flipped near copies; cannot rule out all crops, translations or derived images."}
    try:
        records, duplicates = inspect_raw(raw, expected_per_class)
        manifest = make_manifest(records, seed=42)
        counts = verify_processed(processed, manifest)
        pixels = []
        for record in manifest["records"]:
            with Image.open(raw / record["source"]) as image:
                pixels.append(np.array(image))
        findings = find_cross_split_duplicates(manifest["records"], pixels)
        blocked = bool(findings["exact_cross_split_pairs"] or findings["suspicious_near_duplicate_pairs"])
        return {"status": "potential_leakage" if blocked else "passed", "evaluation_allowed": not blocked,
                "all_images_readable": True, "copies_byte_identical_to_raw": True,
                "split_counts": counts, "totals": {split: sum(values.values()) for split, values in counts.items()},
                "within_split_duplicate_pixel_groups": duplicates["pixel_sha256"],
                "unique_original_pixel_contents": len({record["pixel_sha256"] for record in records}),
                "protocol": protocol, **findings, "duration_seconds": time.perf_counter() - started}
    except (DatasetError, OSError, ValueError) as exc:
        return {"status": "integrity_failed", "evaluation_allowed": False, "error": str(exc),
                "protocol": protocol, "duration_seconds": time.perf_counter() - started}


def leakage_visualizations(audit: dict, raw: Path, output: Path) -> list[str]:
    pairs = audit.get("exact_cross_split_pairs", []) + audit.get("suspicious_near_duplicate_pairs", [])
    names = []
    for start in range(0, len(pairs), 8):
        page = pairs[start:start + 8]
        fig, axes = plt.subplots(len(page), 2, figsize=(9, 3 * len(page)), squeeze=False)
        for row, pair in enumerate(page):
            for column, side in enumerate(("left", "right")):
                record = pair[side]
                with Image.open(raw / record["source"]) as image:
                    axes[row, column].imshow(image, cmap="gray", vmin=0, vmax=255)
                text = f"{record['split']}: {record['source']}"
                if side == "right" and "ssim" in pair:
                    text += f"\n{pair['orientation_applied_to_right']}; SSIM {pair['ssim']:.5f}"
                axes[row, column].set_title(text, fontsize=9)
                axes[row, column].axis("off")
        fig.suptitle("Potential cross-split leakage (original images shown)")
        fig.tight_layout()
        name = f"evaluation_leakage_evidence_{start // 8 + 1:02d}.png"
        fig.savefig(output / name, dpi=130)
        plt.close(fig)
        names.append(name)
    return names


def calculate_metrics(targets: list[int], predictions: list[int]) -> dict:
    if not targets or len(targets) != len(predictions):
        raise ValueError("Metrics require equal nonempty targets and predictions")
    if any(label not in range(len(CLASSES)) for label in targets + predictions):
        raise ValueError("Labels must be in the six-class index range")
    report = classification_report(targets, predictions, labels=list(range(len(CLASSES))),
                                   target_names=list(CLASSES), output_dict=True, zero_division=0)
    return {"samples": len(targets), "accuracy": float(accuracy_score(targets, predictions)),
            "macro_precision": report["macro avg"]["precision"],
            "macro_recall": report["macro avg"]["recall"], "macro_f1": report["macro avg"]["f1-score"],
            "per_class": {name: report[name] for name in CLASSES},
            "confusion_matrix": confusion_matrix(targets, predictions, labels=list(range(len(CLASSES)))).tolist(),
            "confusion_matrix_axes": "rows actual, columns predicted", "class_order": list(CLASSES),
            "misclassification_count": sum(left != right for left, right in zip(targets, predictions))}


def predict_test(model, loader, device: torch.device, *, mixed_precision: bool = True) -> list[dict]:
    if not isinstance(loader.sampler, SequentialSampler) or loader.drop_last:
        raise ValueError("Test inference requires sequential ordering and every sample")
    model.eval()
    predictions = []
    samples = loader.dataset.samples
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=loader.pin_memory)
            with torch.autocast(device_type=device.type, enabled=mixed_precision and device.type == "cuda"):
                logits = model(images)
            if logits.shape != (len(targets), len(CLASSES)) or not torch.isfinite(logits).all():
                raise ValueError("Expected finite six-class logits")
            confidence, labels = logits.float().softmax(dim=1).max(dim=1)
            for actual, predicted, score in zip(targets.tolist(), labels.cpu().tolist(), confidence.cpu().tolist()):
                path, expected = samples[len(predictions)]
                if actual != expected:
                    raise ValueError("Test sample order differs from dataset paths")
                predictions.append({"image": str(Path(path).resolve()), "actual_index": actual,
                                    "predicted_index": predicted, "actual_class": CLASSES[actual],
                                    "predicted_class": CLASSES[predicted], "confidence": score})
    if len(predictions) != len(samples):
        raise ValueError("Test inference did not cover every image")
    return predictions


def save_model_visualizations(name: str, metrics: dict, predictions: list[dict], output: Path) -> list[str]:
    matrix = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(9, 8))
    color = ax.imshow(matrix, cmap="Blues", vmin=0)
    fig.colorbar(color, ax=ax)
    ax.set(xticks=range(6), yticks=range(6), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="Predicted class", ylabel="Actual class", title=f"{name}: test confusion matrix (counts)")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    for row in range(6):
        for column in range(6):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center",
                    color="white" if matrix[row, column] > matrix.max() / 2 else "black")
    fig.tight_layout()
    names = [f"{name}_test_confusion_matrix.png"]
    fig.savefig(output / names[0], dpi=150)
    plt.close(fig)
    mistakes = [record for record in predictions if record["actual_index"] != record["predicted_index"]]
    for start in range(0, len(mistakes), 12):
        page = mistakes[start:start + 12]
        fig, axes = plt.subplots((len(page) + 2) // 3, 3, figsize=(12, 3.5 * ((len(page) + 2) // 3)), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, record in zip(axes.flat, page):
            with Image.open(record["image"]) as image:
                ax.imshow(image, cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{Path(record['image']).name}\nActual: {record['actual_class']}\n"
                         f"Predicted: {record['predicted_class']} ({record['confidence']:.2%})", fontsize=9)
        fig.suptitle(f"{name}: misclassified test images; confidence = uncalibrated softmax")
        fig.tight_layout()
        filename = f"{name}_test_misclassified_{start // 12 + 1:02d}.png"
        fig.savefig(output / filename, dpi=130)
        plt.close(fig)
        names.append(filename)
    return names


def preservation_snapshot() -> dict:
    """Protect dataset bytes/mtimes and all artifacts produced in previous steps."""
    paths = []
    for directory in (ROOT / "data/raw", ROOT / "data/processed", ROOT / "artifacts/models"):
        paths.extend(path for path in directory.rglob("*") if path.is_file())
    # Previous reports are never changed by evaluation.
    for pattern in ("baseline_cnn*", "resnet18_training*", "resnet18_learning*", "resnet18_tests*", "model_comparison.json"):
        paths.extend((ROOT / "artifacts/reports").glob(pattern))
    return {str(path): [hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns]
            for path in paths}


def run_evaluation(*, raw: Path = ROOT / "data/raw", processed: Path = ROOT / "data/processed",
                   output: Path = ROOT / "artifacts/reports", audit_only: bool = False,
                   batch_size: int = 32) -> dict:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    output.mkdir(parents=True, exist_ok=True)
    before = preservation_snapshot()
    audit = audit_dataset(raw, processed)
    audit["evidence_visualizations"] = leakage_visualizations(audit, raw, output)
    write_json(output / "evaluation_dataset_audit.json", audit)
    result = {"status": "blocked" if not audit["evaluation_allowed"] else "audit_passed",
              "dataset_audit": "evaluation_dataset_audit.json", "test_inference_executed": False,
              "models": {}, "seed": 42}
    if not audit["evaluation_allowed"]:
        result["reason"] = "Dataset integrity or potential cross-split leakage requires review; no model test predictions made."
    elif not audit_only:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for final evaluation; no CPU fallback")
        device = torch.device("cuda:0")
        gpu = torch.cuda.get_device_name(device)
        if "3050" not in gpu:
            raise RuntimeError(f"Requested RTX 3050; detected {gpu}")
        torch.set_num_threads(1)
        configure_reproducibility(42)
        result["device"] = gpu
        result["precision"] = "CUDA float16 autocast, as used for validation"
        result["confidence_definition"] = "Uncalibrated maximum softmax score, not a calibrated probability"
        for name, filename in CHECKPOINTS.items():
            checkpoint_path = ROOT / "artifacts/models" / filename
            model, metadata = load_checkpoint(checkpoint_path, device)
            expected_preprocessing = {"image_size": list(IMAGE_SIZE), "mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)}
            if metadata.get("preprocessing") != expected_preprocessing:
                raise ValueError("Checkpoint preprocessing differs from existing evaluation pipeline")
            retries = []
            effective_batch_size = batch_size
            started = time.perf_counter()
            while True:
                loader = create_dataloaders(processed, batch_size=effective_batch_size, seed=42, splits=("test",))["test"]
                try:
                    predictions = predict_test(model, loader, device)
                    torch.cuda.synchronize()
                    break
                except torch.cuda.OutOfMemoryError:
                    if effective_batch_size == 1:
                        raise
                    retries.append(effective_batch_size)
                    effective_batch_size = max(1, effective_batch_size // 2)
                    torch.cuda.empty_cache()
            metrics = calculate_metrics([r["actual_index"] for r in predictions], [r["predicted_index"] for r in predictions])
            validation = metadata["validation_metrics"]
            metrics.update({"validation_accuracy": validation["accuracy"], "validation_macro_f1": validation["macro_f1"],
                            "accuracy_change_from_validation": metrics["accuracy"] - validation["accuracy"],
                            "macro_f1_change_from_validation": metrics["macro_f1"] - validation["macro_f1"],
                            "checkpoint": str(checkpoint_path), "selected_epoch": metadata["epoch"],
                            "selected_phase": metadata.get("phase", "custom_cnn"),
                            "parameter_count": sum(p.numel() for p in model.parameters()),
                            "inference_duration_seconds": time.perf_counter() - started,
                            "effective_batch_size": effective_batch_size, "oom_failed_batch_sizes": retries})
            metrics["visualizations"] = save_model_visualizations(name, metrics, predictions, output)
            with (output / f"{name}_test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
                writer.writeheader()
                writer.writerows(predictions)
            mistakes = [r for r in predictions if r["actual_index"] != r["predicted_index"]]
            write_json(output / f"{name}_test_misclassifications.json", {"images": mistakes})
            write_json(output / f"{name}_test_metrics.json", metrics)
            result["models"][name] = metrics
            del model
            torch.cuda.empty_cache()
        result.update(status="completed", test_inference_executed=True)
    result["previous_artifacts_and_datasets_unchanged"] = before == preservation_snapshot()
    if not result["previous_artifacts_and_datasets_unchanged"]:
        raise RuntimeError("Dataset/checkpoint/previous report preservation check failed")
    if audit_only and audit["evaluation_allowed"]:
        # An audit rerun must not replace a previous measured evaluation report.
        return result
    write_json(output / "final_evaluation.json", result)
    write_comparison_report(result, audit, output)
    return result


def write_comparison_report(result: dict, audit: dict, output: Path) -> None:
    """Render existing measured results without performing further inference."""
    lines = ["# Final model evaluation", "", f"Status: **{result['status']}**", ""]
    if result["models"]:
        lines += ["| Model | Test accuracy | Macro precision | Macro recall | Macro-F1 | Errors | Validation accuracy | Validation macro-F1 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for name, m in result["models"].items():
            lines.append(f"| {name} | {m['accuracy']:.4%} | {m['macro_precision']:.6f} | {m['macro_recall']:.6f} | {m['macro_f1']:.6f} | {m['misclassification_count']} | {m['validation_accuracy']:.4%} | {m['validation_macro_f1']:.6f} |")
        lines += ["", "Per-class metrics, confusion matrices, validation deltas, predictions and uncalibrated confidence scores are in the adjacent JSON/CSV/PNG artifacts.",
                  "These test results characterize this fixed split only; no model selection or tuning follows from them."]
        for name, m in result["models"].items():
            lines += ["", f"## {name}", "",
                      f"Test versus validation: accuracy {m['accuracy_change_from_validation'] * 100:+.4f} percentage points; macro-F1 {m['macro_f1_change_from_validation']:+.6f}.",
                      "", f"![Confusion matrix]({name}_test_confusion_matrix.png)", "",
                      "| Class | Precision | Recall | F1 | Test images |", "| --- | ---: | ---: | ---: | ---: |"]
            for class_name, values in m["per_class"].items():
                lines.append(f"| {class_name} | {values['precision']:.6f} | {values['recall']:.6f} | {values['f1-score']:.6f} | {int(values['support'])} |")
    else:
        lines += [result.get("reason", "Audit completed; test inference was not requested."), "",
                  f"Exact cross-split pairs: {len(audit.get('exact_cross_split_pairs', []))}.",
                  f"Suspicious near-duplicate cross-split pairs: {len(audit.get('suspicious_near_duplicate_pairs', []))}.",
                  "Test metrics, confusion matrices and misclassification counts are unavailable because inference was not executed."]
    (output / "final_evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        parser.error("Use the project's existing .venv Python interpreter")
    try:
        result = run_evaluation(audit_only=args.audit_only, batch_size=args.batch_size)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 2 if result["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
