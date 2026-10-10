"""Reusable inference input, prediction, device, and Grad-CAM regressions."""

from io import BytesIO
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import inference
from src.dataset import CLASS_TO_IDX, build_transforms
from src.models import RESNET18_ARCHITECTURE_ID, CustomCNN, ResNet18Classifier
from src.split_data import CLASSES
from src.train import TrainingConfig, save_checkpoint


@pytest.fixture(scope="module", autouse=True)
def modest_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    torch.manual_seed(42)
    path = tmp_path_factory.mktemp("inference-model") / "resnet.pth"
    model = ResNet18Classifier(pretrained=False).eval()
    save_checkpoint(path, model, TrainingConfig(architecture="resnet18"), 1,
                    {"accuracy": 1., "macro_f1": 1., "loss": 0., "samples": 270}, phase="head", phase_epoch=1)
    return path


@pytest.fixture(scope="module")
def pipeline(checkpoint):
    return inference.InferencePipeline(checkpoint, device="cpu")


def image():
    return Image.fromarray(np.random.default_rng(42).integers(0, 256, (200, 200, 3), dtype=np.uint8))


def encode(value):
    stream = BytesIO()
    value.save(stream, format="PNG")
    return stream.getvalue()


@pytest.mark.parametrize("input_format", ["path", "string", "pil", "bytes", "bytearray", "memoryview"])
def test_formats_produce_identical_predictions(pipeline, tmp_path, input_format):
    pil = image()
    data = encode(pil)
    path = tmp_path / "image.png"
    path.write_bytes(data)
    sources = {"path": path, "string": str(path), "pil": pil, "bytes": data,
               "bytearray": bytearray(data), "memoryview": memoryview(data)}
    expected = pipeline.predict(pil).to_dict()
    assert pipeline.predict(sources[input_format]).to_dict() == expected
    assert pil.mode == "RGB" and pil.size == (200, 200)


def test_predictions_match_direct_evaluation_and_preserve_class_mapping(pipeline):
    value = image()
    tensor = build_transforms()["test"](value.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        probabilities = pipeline.model(tensor).float().softmax(dim=1)[0]
    result = pipeline.predict(value)
    assert result.predicted_index == probabilities.argmax().item()
    assert result.predicted_class == CLASSES[result.predicted_index]
    assert result.confidence == pytest.approx(probabilities.max().item(), abs=1e-7)
    assert list(result.probabilities) == list(CLASSES)
    assert sum(result.probabilities.values()) == pytest.approx(1, abs=1e-6)
    assert all(0 <= score <= 1 for score in result.probabilities.values())
    np.testing.assert_allclose(list(result.probabilities.values()), probabilities.numpy(), atol=1e-7)
    assert dict(pipeline.class_to_idx) == CLASS_TO_IDX
    assert result.model_info["architecture"] == RESNET18_ARCHITECTURE_ID
    assert result.model_info["class_to_idx"] == CLASS_TO_IDX
    assert len(result.model_info["checkpoint_sha256"]) == 64
    assert "Uncalibrated" in result.model_info["confidence_definition"]
    json.dumps(result.to_dict(), allow_nan=False)
    result.model_info["class_to_idx"]["crazing"] = 100
    assert pipeline.model_info["class_to_idx"] == CLASS_TO_IDX


def test_eval_and_inference_mode_no_weight_or_buffer_mutation(pipeline, monkeypatch):
    before = {name: value.clone() for name, value in pipeline.model.state_dict().items()}
    pipeline.model.train()
    forward = pipeline.model.forward
    def check(value):
        assert not pipeline.model.training
        assert torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
        return forward(value)
    monkeypatch.setattr(pipeline.model, "forward", check)
    pipeline.predict(image())
    assert not torch.is_inference_mode_enabled()
    for name, value in pipeline.model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_decoder_matches_imagefolder_rgb_conversion(mode):
    source = image().convert(mode)
    original = np.array(source)
    decoded = inference.decode_image(source)
    assert decoded.mode == "RGB"
    np.testing.assert_array_equal(np.array(decoded), np.array(source.convert("RGB")))
    np.testing.assert_array_equal(np.array(source), original)
    source.close()
    assert decoded.getpixel((0, 0)) is not None


@pytest.mark.parametrize("source", [None, 42, b"", b"not an image", bytearray(b"invalid")])
def test_invalid_input_errors(pipeline, source):
    with pytest.raises(inference.InvalidImageError):
        pipeline.predict(source)


def test_missing_directory_corrupt_and_truncated_inputs(pipeline, tmp_path):
    for path in (tmp_path / "missing.png", tmp_path):
        with pytest.raises(inference.InvalidImageError):
            pipeline.predict(path)
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"corrupt image data")
    with pytest.raises(inference.InvalidImageError):
        pipeline.predict(corrupt)
    with pytest.raises(inference.InvalidImageError):
        pipeline.predict(encode(image())[:100])
    closed = image()
    closed.close()
    with pytest.raises(inference.InvalidImageError):
        pipeline.predict(closed)


def test_input_resource_limits(monkeypatch):
    monkeypatch.setattr(inference, "MAX_IMAGE_BYTES", 10)
    with pytest.raises(inference.InvalidImageError, match="20 MiB"):
        inference.decode_image(b"x" * 11)
    monkeypatch.setattr(inference, "MAX_IMAGE_PIXELS", 10)
    with pytest.raises(inference.InvalidImageError, match="pixels"):
        inference.decode_image(image())


def test_nonfinite_or_wrong_shape_logits_rejected(pipeline, monkeypatch):
    for output in (torch.full((1, 6), float("nan")), torch.zeros(1, 5)):
        monkeypatch.setattr(pipeline.model, "forward", lambda tensor: output)
        with pytest.raises(RuntimeError, match="logits"):
            pipeline.predict(image())


def test_cpu_fallback_and_explicit_missing_cuda(checkpoint, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    automatic = inference.InferencePipeline(checkpoint)
    result = automatic.predict(image())
    assert result.model_info["device"] == "cpu"
    assert result.model_info["fallback_reason"] == "CUDA is unavailable"
    assert result.model_info["precision"] == "float32"
    with pytest.raises(ValueError, match="explicitly requested"):
        inference.InferencePipeline(checkpoint, device="cuda")


def test_auto_cuda_placement_failure_falls_back(checkpoint, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    move = ResNet18Classifier.to
    def unavailable(self, device, *args, **kwargs):
        if torch.device(device).type == "cuda":
            raise RuntimeError("simulated unavailable CUDA driver")
        return move(self, device, *args, **kwargs)
    monkeypatch.setattr(ResNet18Classifier, "to", unavailable)
    automatic = inference.InferencePipeline(checkpoint)
    assert automatic.predict(image()).model_info["device"] == "cpu"
    assert "placement failed" in automatic.fallback_reason


def test_auto_cuda_oom_retries_cpu(checkpoint, monkeypatch):
    automatic = inference.InferencePipeline(checkpoint, device="cpu")
    automatic._automatic_device = True
    automatic.device = torch.device("cuda:0")
    forward = automatic._forward_probabilities
    devices = []
    def simulate(tensor):
        devices.append(automatic.device.type)
        if automatic.device.type == "cuda":
            raise torch.cuda.OutOfMemoryError("simulated OOM")
        return forward(tensor)
    monkeypatch.setattr(automatic, "_forward_probabilities", simulate)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    prediction = automatic.predict(image())
    assert devices == ["cuda", "cpu"]
    assert prediction.model_info["device"] == "cpu" and "out of memory" in automatic.fallback_reason


def test_explain_separate_gradients_prediction_and_cleanup(pipeline):
    value = image()
    prediction = pipeline.predict(value)
    explanation = pipeline.explain(encode(value))
    assert explanation.predicted_class == prediction.predicted_class
    assert explanation.confidence == pytest.approx(prediction.confidence, abs=1e-7)
    assert explanation.heatmap.shape == (200, 200)
    assert np.isfinite(explanation.heatmap).all()
    assert all(parameter.grad is None for parameter in pipeline.model.parameters())
    assert pipeline.predict(value).to_dict() == prediction.to_dict()
    with torch.inference_mode(), pytest.raises(RuntimeError, match="outside"):
        pipeline.explain(value)


@pytest.mark.parametrize("device", ["invalid", "mps"])
def test_invalid_device(checkpoint, device):
    with pytest.raises(ValueError):
        inference.InferencePipeline(checkpoint, device=device)


def test_invalid_checkpoints_and_metadata(checkpoint, tmp_path):
    with pytest.raises(inference.ModelLoadError):
        inference.InferencePipeline(tmp_path / "missing.pth", device="cpu")
    corrupt = tmp_path / "corrupt.pth"
    corrupt.write_bytes(b"invalid checkpoint")
    with pytest.raises(inference.ModelLoadError):
        inference.InferencePipeline(corrupt, device="cpu")
    metadata = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata["class_to_idx"]["crazing"] = 100
    invalid = tmp_path / "mapping.pth"
    torch.save(metadata, invalid)
    with pytest.raises(inference.ModelLoadError, match="mapping"):
        inference.InferencePipeline(invalid, device="cpu")
    metadata = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata["preprocessing"]["image_size"] = [100, 100]
    torch.save(metadata, invalid)
    with pytest.raises(inference.ModelLoadError, match="preprocessing"):
        inference.InferencePipeline(invalid, device="cpu")


def test_custom_cnn_checkpoint_rejected(tmp_path):
    path = tmp_path / "custom.pth"
    save_checkpoint(path, CustomCNN(), TrainingConfig(), 1,
                    {"accuracy": 1., "macro_f1": 1., "loss": 0., "samples": 270})
    with pytest.raises(inference.ModelLoadError, match="ResNet18"):
        inference.InferencePipeline(path, device="cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_actual_gpu_cpu_agreement_and_gradcam(checkpoint):
    cpu = inference.InferencePipeline(checkpoint, device="cpu")
    cuda = inference.InferencePipeline(checkpoint, device="cuda:0", mixed_precision=False)
    value = image()
    cpu_prediction, gpu_prediction = cpu.predict(value), cuda.predict(value)
    assert gpu_prediction.predicted_class == cpu_prediction.predicted_class
    assert gpu_prediction.model_info["device"] == "cuda:0"
    # CUDA's default FP32 kernels may use TF32; compare scores within 1e-4,
    # while requiring the exact same top class. Do not change global backends.
    np.testing.assert_allclose(list(cpu_prediction.probabilities.values()),
                               list(gpu_prediction.probabilities.values()), atol=1e-4, rtol=1e-4)
    explanation = cuda.explain(value)
    assert explanation.predicted_class == gpu_prediction.predicted_class
    assert explanation.confidence == pytest.approx(gpu_prediction.confidence, abs=1e-6)
    assert explanation.heatmap.shape == (200, 200)
