"""Transfer phases, pretrained construction, checkpoint loading, and selection tests."""

from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.models as model_module
import src.train as training_module
from src.models import ResNet18Classifier, RESNET18_ARCHITECTURE_ID, load_checkpoint
from src.train import TrainingConfig, save_checkpoint, train_attempt
from src.compare_models import benchmark_latency


@pytest.fixture(scope="module", autouse=True)
def modest_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_official_weight_enum_is_used(monkeypatch):
    calls = []

    def offline_builder(*, weights):
        calls.append(weights)
        return resnet18(weights=None)

    monkeypatch.setattr(model_module, "resnet18", offline_builder)
    model = ResNet18Classifier()
    assert calls == [ResNet18_Weights.IMAGENET1K_V1]
    assert model.backbone.fc.out_features == 6 and model.backbone.fc.in_features == 512


@pytest.mark.parametrize("phase", ["head", "fine_tune"])
def test_phase_gradient_and_batchnorm_freezing(phase):
    model = ResNet18Classifier(pretrained=False)
    model.set_phase(phase)
    model.train()
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    expected = {name for name, _ in model.named_parameters()
                if name.startswith("backbone.fc.") or (phase == "fine_tune" and name.startswith("backbone.layer4."))}
    assert trainable == expected
    before = {name: buffer.clone() for name, buffer in model.named_buffers()}
    frozen_parameter = model.backbone.conv1.weight.detach().clone()
    layer4_parameter = model.backbone.layer4[0].conv1.weight.detach().clone()
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad),
                                  lr=0.0003 if phase == "head" else 1e-5)
    logits = model(torch.randn(2, 3, 64, 64))
    assert logits.shape == (2, 6)
    nn.CrossEntropyLoss()(logits, torch.tensor([1, 5])).backward()
    for name, parameter in model.named_parameters():
        assert (parameter.grad is not None) == (name in expected)
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
    optimizer.step()
    assert torch.equal(frozen_parameter, model.backbone.conv1.weight)
    assert torch.equal(layer4_parameter, model.backbone.layer4[0].conv1.weight) == (phase == "head")
    for name, buffer in model.named_buffers():
        if phase == "head" or not name.startswith("backbone.layer4."):
            assert torch.equal(before[name], buffer)
    assert not model.backbone.bn1.training
    assert model.backbone.layer4[0].bn1.training == (phase == "fine_tune")
    model.eval()
    assert not any(layer.training for layer in model.modules())


def test_resnet_checkpoint_load_does_not_download_and_preserves_logits(tmp_path, monkeypatch):
    model = ResNet18Classifier(pretrained=False)
    model.set_phase("fine_tune")
    model.eval()
    images = torch.randn(2, 3, 64, 64)
    path = tmp_path / "resnet.pth"
    config = TrainingConfig(architecture="resnet18")
    save_checkpoint(path, model, config, 8, {"accuracy": 0.9, "macro_f1": 0.89},
                    phase="fine_tune", phase_epoch=3)
    original = model_module.resnet18

    def must_not_download(*, weights):
        assert weights is None
        return original(weights=None)

    monkeypatch.setattr(model_module, "resnet18", must_not_download)
    restored, metadata = load_checkpoint(path)
    with torch.no_grad():
        torch.testing.assert_close(model(images), restored(images), rtol=0, atol=0)
    assert metadata["architecture"] == RESNET18_ARCHITECTURE_ID
    assert metadata["phase"] == "fine_tune" and metadata["phase_epoch"] == 3
    assert metadata["pretrained_weights"] == "ResNet18_Weights.IMAGENET1K_V1"
    assert metadata["training_config"]["fine_tune_learning_rate"] == 1e-5


def test_phases_restore_best_head_then_select_global_best(tmp_path, monkeypatch):
    class TinyModel(nn.Module):
        architecture_id = RESNET18_ARCHITECTURE_ID
        pretrained = True
        model_config = {"num_classes": 6}

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(0.0))

        def set_phase(self, phase):
            self.phase = phase

    model = TinyModel()
    monkeypatch.setattr(training_module, "ResNet18Classifier", lambda: model)

    class TinyLoader:
        dataset = list(range(6))

    monkeypatch.setattr(training_module, "create_dataloaders", lambda *args, **kwargs: {"train": TinyLoader(), "val": TinyLoader()})
    scores = iter([0.8, 0.7, 0.6, 0.85])
    phase2_start, learning_rates = [], []

    def fake_epoch(current, loader, criterion, device, *, optimizer=None, **kwargs):
        if optimizer is not None:
            learning_rates.append(optimizer.param_groups[0]["lr"])
            if current.phase == "fine_tune" and not phase2_start:
                phase2_start.append(current.weight.item())
            with torch.no_grad():
                current.weight.add_(1)
            score = 0.1
        else:
            score = next(scores)
        return {"loss": 1.0, "accuracy": score, "macro_f1": score, "samples": 6}

    monkeypatch.setattr(training_module, "run_epoch", fake_epoch)
    config = TrainingConfig(architecture="resnet18", head_epochs=2, max_epochs=4, mixed_precision=False)
    path = tmp_path / "best.pth"
    report = train_attempt(config, tmp_path, path, torch.device("cpu"))
    assert learning_rates == [0.0003, 0.0003, 1e-5, 1e-5]
    assert phase2_start == [1.0], "Phase 2 should start from epoch 1, the best head checkpoint"
    assert report["best_epoch"] == 4 and report["best_phase"] == "fine_tune"
    saved = torch.load(path, weights_only=True)
    assert saved["phase_epoch"] == 2 and saved["model_state_dict"]["weight"].item() == 3.0


@pytest.mark.parametrize("changes", [{"head_epochs": 0}, {"head_epochs": 20}, {"fine_tune_learning_rate": 0}])
def test_invalid_phase_configuration(changes):
    with pytest.raises(ValueError):
        TrainingConfig(architecture="resnet18", **changes).validate()


def test_benchmark_rejects_non_cuda_inputs():
    with pytest.raises(ValueError, match="CUDA-resident"):
        benchmark_latency(nn.Identity(), torch.zeros(1, 3, 224, 224))
