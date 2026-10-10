"""Final-evaluation metrics, leakage gate, and inference/report regression tests."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import evaluate
from src.split_data import CLASSES, prepare_splits


def record(array, split, name, class_name="crazing", encoding="bmp"):
    digest = hashlib.sha256(array.tobytes()).hexdigest()
    return {"source": f"{class_name}/{name}", "class": class_name, "split": split,
            "pixel_sha256": digest, "sha256": digest + encoding}


def test_metrics_known_errors_fixed_class_order():
    actual = [0, 0, 1, 1, 2, 3, 4, 5]
    predicted = [0, 1, 1, 1, 2, 3, 4, 5]
    metrics = evaluate.calculate_metrics(actual, predicted)
    assert metrics["accuracy"] == 7 / 8
    assert metrics["misclassification_count"] == 1
    assert metrics["macro_precision"] == pytest.approx((1 + 2 / 3 + 4) / 6)
    assert metrics["macro_recall"] == pytest.approx((0.5 + 1 + 4) / 6)
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 0.8 + 4) / 6)
    assert metrics["confusion_matrix"][0][:2] == [1, 1]
    assert metrics["class_order"] == list(CLASSES)
    assert metrics["per_class"]["inclusion"]["support"] == 2


def test_metrics_missing_classes_zero_division():
    metrics = evaluate.calculate_metrics([0, 0], [0, 0])
    assert metrics["accuracy"] == 1
    assert metrics["macro_f1"] == pytest.approx(1 / 6)
    assert np.array(metrics["confusion_matrix"]).shape == (6, 6)


@pytest.mark.parametrize("targets,predictions", [([], []), ([0], []), ([6], [0]), ([0], [-1])])
def test_metrics_reject_invalid_inputs(targets, predictions):
    with pytest.raises(ValueError):
        evaluate.calculate_metrics(targets, predictions)


def test_exact_pixels_different_encoding_or_label_block_cross_split():
    pixels = np.random.default_rng(1).integers(0, 256, (200, 200), dtype=np.uint8)
    records = [record(pixels, "train", "first.bmp"),
               record(pixels, "test", "second.png", "patches", encoding="png")]
    result = evaluate.find_cross_split_duplicates(records, [pixels, pixels.copy()])
    assert len(result["exact_cross_split_pairs"]) == 1
    assert not result["suspicious_near_duplicate_pairs"]


def test_within_split_duplicate_does_not_block():
    pixels = np.zeros((200, 200), dtype=np.uint8)
    records = [record(pixels, "train", "first.bmp"), record(pixels, "train", "second.bmp")]
    result = evaluate.find_cross_split_duplicates(records, [pixels, pixels])
    assert not result["exact_cross_split_pairs"]
    assert result["cross_split_pairs_checked"] == 0


@pytest.mark.parametrize("orientation", evaluate.ORIENTATIONS)
def test_detects_slightly_altered_near_copies_and_flips(orientation):
    original = np.random.default_rng(42).integers(10, 240, (200, 200), dtype=np.uint8)
    # A slight brightness change preserves structure but changes pixel SHA256.
    modified = evaluate.orient(original + 1, orientation).copy()
    records = [record(original, "val", "first.bmp"), record(modified, "test", "second.bmp")]
    result = evaluate.find_cross_split_duplicates(records, [original, modified])
    assert not result["exact_cross_split_pairs"]
    assert len(result["suspicious_near_duplicate_pairs"]) == 1
    pair = result["suspicious_near_duplicate_pairs"][0]
    assert pair["orientation_applied_to_right"] == orientation
    assert pair["ssim"] > 0.99


def test_unrelated_images_not_flagged():
    rng = np.random.default_rng(12)
    arrays = [rng.integers(0, 256, (200, 200), dtype=np.uint8) for _ in range(3)]
    records = [record(array, split, f"{split}.bmp") for array, split in zip(arrays, ("train", "val", "test"))]
    result = evaluate.find_cross_split_duplicates(records, arrays)
    assert result["cross_split_pairs_checked"] == 3
    assert not result["exact_cross_split_pairs"] and not result["suspicious_near_duplicate_pairs"]


def test_integrity_corruption_blocks_audit(tmp_path):
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    rng = np.random.default_rng(42)
    for name in CLASSES:
        (raw / name).mkdir(parents=True)
        for i in range(20):
            Image.fromarray(rng.integers(0, 256, (200, 200), dtype=np.uint8)).save(raw / name / f"{i}.bmp")
    prepare_splits(raw, processed, expected_per_class=20)
    manifest_before = (processed / "split_manifest.json").read_bytes()
    assert evaluate.audit_dataset(raw, processed, expected_per_class=20)["evaluation_allowed"]
    next((processed / "test").rglob("*.bmp")).write_bytes(b"corrupt")
    audit = evaluate.audit_dataset(raw, processed, expected_per_class=20)
    assert audit["status"] == "integrity_failed" and not audit["evaluation_allowed"]
    assert (processed / "split_manifest.json").read_bytes() == manifest_before


def test_leakage_gate_prevents_checkpoint_load_and_test_loader(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "audit_dataset", lambda *args: {
        "status": "potential_leakage", "evaluation_allowed": False,
        "exact_cross_split_pairs": [], "suspicious_near_duplicate_pairs": []})
    monkeypatch.setattr(evaluate, "preservation_snapshot", lambda: {})
    def forbidden(*args, **kwargs):
        pytest.fail("Leakage gate must stop before loading checkpoints or test DataLoaders")
    monkeypatch.setattr(evaluate, "load_checkpoint", forbidden)
    monkeypatch.setattr(evaluate, "create_dataloaders", forbidden)
    result = evaluate.run_evaluation(output=tmp_path)
    assert result["status"] == "blocked" and not result["test_inference_executed"]
    assert result["models"] == {}
    assert json.loads((tmp_path / "final_evaluation.json").read_text())["status"] == "blocked"
    assert not list(tmp_path.glob("*test_predictions*"))


def test_audit_only_does_not_initialize_model_or_loader(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "audit_dataset", lambda *args: {"status": "passed", "evaluation_allowed": True})
    monkeypatch.setattr(evaluate, "preservation_snapshot", lambda: {})
    def forbidden(*args, **kwargs):
        pytest.fail("Audit only must not initialize test inference")
    monkeypatch.setattr(evaluate, "load_checkpoint", forbidden)
    monkeypatch.setattr(evaluate, "create_dataloaders", forbidden)
    saved_report = tmp_path / "final_evaluation.json"
    saved_report.write_text('{"previous_measured_result": true}')
    previous_bytes = saved_report.read_bytes()
    result = evaluate.run_evaluation(output=tmp_path, audit_only=True)
    assert result["status"] == "audit_passed" and not result["test_inference_executed"]
    assert saved_report.read_bytes() == previous_bytes


def test_cuda_evaluation_uses_only_test_split_and_recovers_oom(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "audit_dataset", lambda *args: {"status": "passed", "evaluation_allowed": True})
    monkeypatch.setattr(evaluate, "preservation_snapshot", lambda: {})
    monkeypatch.setattr(evaluate, "configure_reproducibility", lambda seed: None)
    monkeypatch.setattr(torch, "set_num_threads", lambda threads: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "NVIDIA RTX 3050")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    model_loads, batches = [], []
    def load(path, device):
        assert device.type == "cuda"
        model_loads.append(path.name)
        return nn.Linear(6, 6), {
            "preprocessing": {"image_size": list(evaluate.IMAGE_SIZE),
                              "mean": list(evaluate.IMAGENET_MEAN), "std": list(evaluate.IMAGENET_STD)},
            "validation_metrics": {"accuracy": 1., "macro_f1": 1.}, "epoch": 1}
    def loaders(path, **kwargs):
        assert kwargs["splits"] == ("test",) and kwargs["seed"] == 42
        batches.append(kwargs["batch_size"])
        return {"test": kwargs["batch_size"]}
    def predict(model, batch_size, device):
        if batch_size == 32:
            raise torch.cuda.OutOfMemoryError("simulated")
        return [{"image": str(tmp_path / f"{i}.bmp"), "actual_index": i, "predicted_index": i,
                 "actual_class": name, "predicted_class": name, "confidence": .9}
                for i, name in enumerate(CLASSES)]
    monkeypatch.setattr(evaluate, "load_checkpoint", load)
    monkeypatch.setattr(evaluate, "create_dataloaders", loaders)
    monkeypatch.setattr(evaluate, "predict_test", predict)
    result = evaluate.run_evaluation(output=tmp_path)
    assert result["status"] == "completed" and result["test_inference_executed"]
    assert model_loads == ["baseline_cnn.pth", "resnet18_best.pth"]
    assert batches == [32, 16, 32, 16]
    for name, metrics in result["models"].items():
        assert metrics["macro_f1"] == 1 and metrics["misclassification_count"] == 0
        assert metrics["oom_failed_batch_sizes"] == [32] and metrics["effective_batch_size"] == 16
        assert (tmp_path / f"{name}_test_predictions.csv").is_file()
        assert json.loads((tmp_path / f"{name}_test_misclassifications.json").read_text()) == {"images": []}


def test_no_cpu_fallback_when_cuda_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "audit_dataset", lambda *args: {"status": "passed", "evaluation_allowed": True})
    monkeypatch.setattr(evaluate, "preservation_snapshot", lambda: {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is required"):
        evaluate.run_evaluation(output=tmp_path)


class FixedLogits(nn.Module):
    def forward(self, values):
        return values


def prediction_loader(tmp_path, *, shuffle=False, drop_last=False):
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 0])
    logits = torch.full((7, 6), -2.0)
    logits[torch.arange(7), labels] = 2
    dataset = TensorDataset(logits, labels)
    dataset.samples = [(str(tmp_path / f"{i}.bmp"), int(label)) for i, label in enumerate(labels)]
    return DataLoader(dataset, batch_size=3, shuffle=shuffle, drop_last=drop_last)


def test_prediction_paths_labels_confidence_and_partial_batch(tmp_path):
    model = FixedLogits().train()
    result = evaluate.predict_test(model, prediction_loader(tmp_path), torch.device("cpu"))
    assert len(result) == 7 and not model.training
    assert all(record["actual_index"] == record["predicted_index"] for record in result)
    assert all(0 < record["confidence"] < 1 for record in result)
    assert [Path(record["image"]).name for record in result] == [f"{i}.bmp" for i in range(7)]
    assert result[0]["confidence"] == pytest.approx(1 / (1 + 5 * np.exp(-4)))


@pytest.mark.parametrize("options", [{"shuffle": True}, {"drop_last": True}])
def test_inference_rejects_shuffling_or_dropped_samples(tmp_path, options):
    with pytest.raises(ValueError):
        evaluate.predict_test(FixedLogits(), prediction_loader(tmp_path, **options), torch.device("cpu"))


def test_nonfinite_logits_rejected(tmp_path):
    loader = prediction_loader(tmp_path)
    loader.dataset.tensors[0][0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        evaluate.predict_test(FixedLogits(), loader, torch.device("cpu"))


def test_confusion_and_misclassification_visualizations(tmp_path):
    path = tmp_path / "sample.bmp"
    Image.new("L", (200, 200), 120).save(path)
    metrics = evaluate.calculate_metrics([0], [1])
    predictions = [{"image": str(path), "actual_index": 0, "predicted_index": 1,
                    "actual_class": CLASSES[0], "predicted_class": CLASSES[1], "confidence": 0.75}]
    outputs = evaluate.save_model_visualizations("synthetic", metrics, predictions, tmp_path)
    assert len(outputs) == 2
    for name in outputs:
        with Image.open(tmp_path / name) as image:
            image.verify()
    assert len(evaluate.save_model_visualizations("perfect", evaluate.calculate_metrics([0], [0]), [], tmp_path)) == 1
