"""Custom CNN and ImageNet ResNet18 classifiers with checkpoint reconstruction."""

from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from .split_data import CLASSES


ARCHITECTURE_ID = "visionforge_custom_cnn_v1"
RESNET18_ARCHITECTURE_ID = "visionforge_resnet18_imagenet_v1"


class CustomCNN(nn.Module):
    """Four Conv/BatchNorm/ReLU/MaxPool blocks and a pooled six-class head."""

    architecture_id = ARCHITECTURE_ID
    pretrained = False

    def __init__(self, num_classes: int = 6, dropout: float = 0.3):
        super().__init__()
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError("num_classes must be a positive integer")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        blocks = []
        incoming = 3
        for outgoing in (32, 64, 128, 256):
            blocks.append(nn.Sequential(
                nn.Conv2d(incoming, outgoing, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(outgoing),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
            ))
            incoming = outgoing
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(256, num_classes)
        self.model_config = {"num_classes": num_classes, "dropout": dropout}

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.pool(self.features(images)).flatten(1)
        return self.classifier(self.dropout(features))


class ResNet18Classifier(nn.Module):
    """Official ImageNet ResNet18 with a six-class head and explicit phase freezing."""

    architecture_id = RESNET18_ARCHITECTURE_ID
    pretrained = True
    pretrained_weights = "ResNet18_Weights.IMAGENET1K_V1"

    def __init__(self, num_classes: int = 6, *, pretrained: bool = True):
        super().__init__()
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError("num_classes must be a positive integer")
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        self.model_config = {"num_classes": num_classes}
        self.set_phase("head")

    def set_phase(self, phase: str) -> None:
        if phase not in ("head", "fine_tune"):
            raise ValueError("ResNet18 phase must be head or fine_tune")
        self.phase = phase
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        for parameter in self.backbone.fc.parameters():
            parameter.requires_grad_(True)
        if phase == "fine_tune":
            for parameter in self.backbone.layer4.parameters():
                parameter.requires_grad_(True)
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            # Frozen BatchNorm buffers must not adapt to training batches.
            self.backbone.eval()
            self.backbone.fc.train()
            if self.phase == "fine_tune":
                self.backbone.layer4.train()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[nn.Module, dict]:
    """Reconstruct either classifier from saved tensor weights and validated metadata."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") not in (ARCHITECTURE_ID, RESNET18_ARCHITECTURE_ID):
        raise ValueError(f"Unsupported architecture: {checkpoint.get('architecture')}")
    if checkpoint.get("class_names") != list(CLASSES):
        raise ValueError("Checkpoint class names differ from the six NEU-CLS classes")
    expected_mapping = {name: index for index, name in enumerate(CLASSES)}
    if checkpoint.get("class_to_idx") != expected_mapping:
        raise ValueError("Checkpoint class-to-index mapping is inconsistent")
    if checkpoint["architecture"] == RESNET18_ARCHITECTURE_ID:
        # Reconstruction uses saved weights; never downloads ImageNet weights.
        model = ResNet18Classifier(pretrained=False, **checkpoint["model_config"])
        model.set_phase(checkpoint.get("phase", "head"))
    else:
        model = CustomCNN(**checkpoint["model_config"])
    if model.model_config["num_classes"] != len(CLASSES):
        raise ValueError("Checkpoint classifier dimension differs from class names")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint
