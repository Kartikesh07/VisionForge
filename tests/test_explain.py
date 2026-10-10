"""Grad-CAM dimensions, normalization, predictions, state preservation and outputs."""

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
from PIL import Image
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import explain
from src.dataset import build_transforms
from src.models import CustomCNN, ResNet18Classifier
from src.split_data import CLASSES
from src.train import TrainingConfig, save_checkpoint


@pytest.fixture(scope="module", autouse=True)
def modest_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def model():
    torch.manual_seed(42)
    result = ResNet18Classifier(pretrained=False).eval()
    with torch.no_grad():
        result.backbone.fc.weight.zero_()
        result.backbone.fc.bias.zero_()
        result.backbone.fc.weight[0].fill_(1.)
    # Explainability must work even when every model parameter is frozen.
    for parameter in result.parameters():
        parameter.requires_grad_(False)
    return result


def sample_image(size=(200, 200)):
    return Image.fromarray(np.random.default_rng(42).integers(0, 256, (size[1], size[0]), dtype=np.uint8))


@pytest.mark.parametrize("size", [(200, 200), (180, 120)])
def test_heatmap_dimensions_normalization_and_prediction_consistency(model, size):
    image = sample_image(size)
    tensor = build_transforms()["val"](image).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)[0]
        confidence, predicted = logits.softmax(dim=0).max(dim=0)
    result = explain.explain_image(model, image)
    assert result.original.shape == result.heatmap.shape == (size[1], size[0])
    assert result.input_heatmap.shape == (224, 224)
    assert result.overlay_rgb.shape == result.heatmap_rgb.shape == (size[1], size[0], 3)
    assert result.input_heatmap.dtype == result.heatmap.dtype == np.float32
    assert np.isfinite(result.heatmap).all()
    assert 0 <= result.heatmap.min() <= result.heatmap.max() <= 1
    assert result.input_heatmap.min() == pytest.approx(0, abs=1e-6)
    assert result.input_heatmap.max() == pytest.approx(1, abs=1e-6)
    assert result.predicted_index == predicted.item()
    assert result.predicted_class == CLASSES[predicted.item()]
    assert result.confidence == pytest.approx(confidence.item(), abs=1e-7)
    np.testing.assert_allclose(result.logits, logits.numpy(), atol=1e-6)
    np.testing.assert_array_equal(result.original, np.array(image))


def test_gradients_hooks_and_model_state_do_not_persist(model):
    before = {name: value.clone() for name, value in model.state_dict().items()}
    flags = [parameter.requires_grad for parameter in model.parameters()]
    target = model.backbone.layer4[-1]
    hooks = len(target._forward_hooks)
    result = explain.explain_image(model, sample_image())
    assert not model.training and not target.training
    assert np.any(result.heatmap)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert [parameter.requires_grad for parameter in model.parameters()] == flags
    assert len(target._forward_hooks) == hooks
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    repeated = explain.explain_image(model, sample_image())
    np.testing.assert_array_equal(repeated.heatmap, result.heatmap)


def test_zero_attribution_produces_finite_zero_map(model):
    with torch.no_grad():
        model.backbone.fc.weight.zero_()
    result = explain.explain_image(model, sample_image())
    assert np.isfinite(result.heatmap).all() and not np.any(result.heatmap)
    assert result.confidence == pytest.approx(1 / 6)


def test_inference_mode_rejected_and_wrong_architecture_rejected(model):
    with torch.inference_mode(), pytest.raises(RuntimeError, match="requires gradients"):
        explain.explain_image(model, sample_image())
    with pytest.raises(ValueError, match="ResNet18Classifier"):
        explain.explain_image(CustomCNN(), sample_image())


def test_lossless_outputs_metadata_and_rgb_color_order(model, tmp_path):
    result = explain.explain_image(model, sample_image())
    metadata = explain.save_explanation(result, tmp_path, source=tmp_path / "source.bmp", actual_class="crazing")
    assert metadata["target_class"] == result.predicted_class
    assert metadata["predicted_class"] == result.predicted_class
    assert metadata["confidence"] == result.confidence
    assert "not precise defect segmentation" in metadata["limitation"]
    for name in ("original.png", "heatmap.png", "overlay.png", "heatmap_gray.png", "explanation.png"):
        with Image.open(tmp_path / name) as image:
            image.verify()
    with Image.open(tmp_path / "original.png") as image:
        np.testing.assert_array_equal(np.array(image), result.original)
    with Image.open(tmp_path / "overlay.png") as image:
        np.testing.assert_array_equal(np.array(image), result.overlay_rgb)
    with Image.open(tmp_path / "heatmap.png") as image:
        np.testing.assert_array_equal(np.array(image), result.heatmap_rgb)
    np.testing.assert_array_equal(np.load(tmp_path / "heatmap.npy", allow_pickle=False), result.heatmap)
    np.testing.assert_array_equal(np.load(tmp_path / "input_heatmap.npy", allow_pickle=False), result.input_heatmap)


@pytest.mark.parametrize("heatmap", [np.full((20, 20), np.nan), np.full((20, 20), -0.1), np.full((20, 20), 1.1), np.zeros((5, 5))])
def test_invalid_heatmap_rejected(heatmap):
    with pytest.raises(ValueError):
        explain.colorize_heatmap(np.zeros((20, 20), dtype=np.uint8), heatmap)


def test_offline_checkpoint_loading_validates_metadata(model, tmp_path):
    path = tmp_path / "resnet.pth"
    save_checkpoint(path, model, TrainingConfig(architecture="resnet18"), 1,
                    {"accuracy": 1., "macro_f1": 1., "loss": 0., "samples": 270}, phase="head", phase_epoch=1)
    restored = explain.load_explanation_model(path, device="cpu")
    assert isinstance(restored, ResNet18Classifier) and not restored.training
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
    saved = torch.load(path, weights_only=True)
    saved["preprocessing"]["image_size"] = [100, 100]
    torch.save(saved, path)
    with pytest.raises(ValueError, match="preprocessing"):
        explain.load_explanation_model(path, device="cpu")


def test_demonstrations_construct_validation_only_and_reject_test_image(tmp_path, monkeypatch):
    samples = [(str(tmp_path / "val" / name / f"{index}.bmp"), label)
               for label, name in enumerate(CLASSES) for index in range(2)]
    def validation_only(**kwargs):
        assert kwargs == {"splits": ("val",)}
        return {"val": SimpleNamespace(samples=samples)}
    monkeypatch.setattr(explain, "create_datasets", validation_only)
    selected = explain.select_validation_images(samples_per_class=1)
    assert [actual for _, actual in selected] == list(CLASSES)
    assert all(path.name == "0.bmp" for path, _ in selected)
    assert len(explain.select_validation_images(Path(samples[1][0]))) == 1
    with pytest.raises(ValueError, match="existing image"):
        explain.select_validation_images(tmp_path / "test/crazing/0.bmp")
    with pytest.raises(ValueError, match="between 1 and 45"):
        explain.select_validation_images(samples_per_class=0)
