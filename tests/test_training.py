"""Baseline CNN, training metrics, selection, checkpoint, and OOM-retry tests."""

from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import create_dataloaders
from src.models import ARCHITECTURE_ID, CustomCNN, load_checkpoint
from src.split_data import CLASSES
from src.train import TrainingConfig, run_epoch, save_checkpoint, train_attempt
import src.train as training_module


@pytest.fixture(scope="module", autouse=True)
def modest_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("size", [64, 224])
def test_output_dimensions_and_raw_logits(size):
    model = CustomCNN().eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 3, size, size))
    assert logits.shape == (2, 6) and torch.isfinite(logits).all()
    assert not torch.allclose(logits.sum(dim=1), torch.ones(2))
    assert sum(isinstance(layer, nn.Conv2d) for layer in model.modules()) == 4
    assert sum(isinstance(layer, nn.BatchNorm2d) for layer in model.modules()) == 4
    assert sum(isinstance(layer, nn.MaxPool2d) for layer in model.modules()) == 4
    assert isinstance(model.pool, nn.AdaptiveAvgPool2d)


def test_forward_backward_and_optimizer_update():
    model = CustomCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
    before = model.classifier.weight.detach().clone()
    loss = nn.CrossEntropyLoss()(model(torch.randn(3, 3, 64, 64)), torch.tensor([0, 2, 5]))
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    optimizer.step()
    assert not torch.equal(before, model.classifier.weight)


def test_checkpoint_roundtrip_preserves_outputs_and_metadata(tmp_path):
    model = CustomCNN().eval()
    images = torch.randn(2, 3, 64, 64)
    path = tmp_path / "model.pth"
    config = TrainingConfig()
    validation = {"loss": 0.5, "accuracy": 0.8, "macro_f1": 0.78, "samples": 270}
    save_checkpoint(path, model, config, 3, validation)
    restored, metadata = load_checkpoint(path)
    with torch.no_grad():
        torch.testing.assert_close(model(images), restored(images), rtol=0, atol=0)
    assert not restored.training
    assert metadata["architecture"] == ARCHITECTURE_ID
    assert metadata["class_names"] == list(CLASSES)
    assert metadata["epoch"] == 3 and metadata["validation_metrics"] == validation
    assert metadata["training_config"]["learning_rate"] == 0.0003
    assert metadata["training_config"]["seed"] == 42 and metadata["pretrained"] is False


@pytest.mark.parametrize("field,value", [("architecture", "unknown"), ("class_names", ["wrong"] * 6),
                                        ("class_to_idx", {"wrong": 0})])
def test_incompatible_checkpoint_is_rejected(tmp_path, field, value):
    path = tmp_path / "model.pth"
    save_checkpoint(path, CustomCNN(), TrainingConfig(), 1, {})
    checkpoint = torch.load(path, weights_only=True)
    checkpoint[field] = value
    torch.save(checkpoint, path)
    with pytest.raises(ValueError):
        load_checkpoint(path)


def test_epoch_metrics_use_all_samples_and_six_class_macro_f1():
    predictions = torch.tensor([0, 1, 2, 3, 4, 0, 5])
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 0])
    logits = torch.nn.functional.one_hot(predictions, num_classes=6).float() * 8

    class FixedLogits(nn.Module):
        def forward(self, indices):
            return logits[indices]

    loader = DataLoader(TensorDataset(torch.arange(7), labels), batch_size=4)
    report = run_epoch(FixedLogits(), loader, nn.CrossEntropyLoss(), torch.device("cpu"))
    assert report["samples"] == 7
    assert report["accuracy"] == pytest.approx(5 / 7)
    assert report["macro_f1"] == pytest.approx(0.75)
    assert report["loss"] == pytest.approx(nn.CrossEntropyLoss()(logits, labels).item())


def test_validation_does_not_change_parameters_or_batchnorm():
    model = CustomCNN()
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    loader = DataLoader(TensorDataset(torch.randn(4, 3, 64, 64), torch.tensor([0, 1, 2, 3])), batch_size=2)
    run_epoch(model, loader, nn.CrossEntropyLoss(), torch.device("cpu"))
    assert all(torch.equal(before[name], tensor) for name, tensor in model.state_dict().items())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_early_stopping_and_best_macro_f1_checkpoint(tmp_path, monkeypatch):
    class TinyDataset:
        def __len__(self):
            return 6

    class TinyLoader:
        dataset = TinyDataset()

    monkeypatch.setattr(training_module, "create_dataloaders", lambda *args, **kwargs: {"train": TinyLoader(), "val": TinyLoader()})
    val_scores = iter([0.3, 0.5, 0.4, 0.5, 0.45, 0.4, 0.49])

    def fake_epoch(*args, optimizer=None, **kwargs):
        score = 0.2 if optimizer is not None else next(val_scores)
        return {"loss": 1.0, "accuracy": 0.9 if score < 0.5 else 0.8, "macro_f1": score, "samples": 6}

    monkeypatch.setattr(training_module, "run_epoch", fake_epoch)
    path = tmp_path / "best.pth"
    report = train_attempt(TrainingConfig(mixed_precision=False), tmp_path, path, torch.device("cpu"))
    assert report["best_epoch"] == 2 and report["epochs_completed"] == 7
    assert report["early_stopped"] and report["best_validation"]["macro_f1"] == 0.5
    assert report["highest_validation_accuracy"] == 0.9
    _, metadata = load_checkpoint(path)
    assert metadata["epoch"] == 2 and metadata["validation_metrics"]["accuracy"] == 0.8


def test_loader_can_use_train_val_without_a_test_directory(tmp_path):
    from PIL import Image
    for split in ("train", "val"):
        for name in CLASSES:
            folder = tmp_path / split / name
            folder.mkdir(parents=True)
            Image.new("L", (200, 200), 128).save(folder / "image.bmp")
    loaders = create_dataloaders(tmp_path, splits=("train", "val"), pin_memory=False)
    assert set(loaders) == {"train", "val"} and not (tmp_path / "test").exists()


def test_cuda_oom_restarts_with_reduced_batch_and_same_seed(tmp_path, monkeypatch):
    calls, cache_clears = [], []
    metrics = {"loss": 0.5, "accuracy": 0.8, "macro_f1": 0.79, "samples": 6}

    def fake_attempt(config, processed_dir, path, device):
        calls.append((config.batch_size, config.seed))
        if len(calls) == 1:
            raise torch.cuda.OutOfMemoryError("simulated OOM")
        path.write_bytes(b"new successful checkpoint")
        return {"history": [{"epoch": 1, "train": metrics, "val": metrics}],
                "best_epoch": 1, "best_validation": metrics, "epochs_completed": 1}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "mock GPU")
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: cache_clears.append(True))
    monkeypatch.setattr(training_module, "train_attempt", fake_attempt)
    monkeypatch.setattr(training_module, "load_checkpoint", lambda *args: (nn.Identity(), {"validation_metrics": metrics}))
    monkeypatch.setattr(training_module, "create_dataloaders", lambda *args, **kwargs: {"val": object()})
    monkeypatch.setattr(training_module, "run_epoch", lambda *args, **kwargs: metrics)
    monkeypatch.setattr(training_module, "plot_learning_curves", lambda *args: None)
    (tmp_path / "split_manifest.json").write_text("{}")
    result = training_module.train(TrainingConfig(mixed_precision=False), processed_dir=tmp_path,
                                   checkpoint_path=tmp_path / "baseline.pth", report_dir=tmp_path / "reports",
                                   require_cuda=True)
    assert calls == [(32, 42), (16, 42)] and len(cache_clears) == 1
    assert result["training_config"]["batch_size"] == 16
    assert result["requested_config"]["batch_size"] == 32
    assert result["oom_retries"] == [{"failed_batch_size": 32, "retry_batch_size": 16}]
    assert (tmp_path / "baseline.pth").read_bytes() == b"new successful checkpoint"
    assert not (tmp_path / ".baseline.pth.attempt").exists()


@pytest.mark.parametrize("changes", [{"max_epochs": 21}, {"patience": 0}, {"batch_size": 0}, {"learning_rate": 0}])
def test_invalid_training_configuration(changes):
    with pytest.raises(ValueError):
        replace(TrainingConfig(), **changes).validate()
