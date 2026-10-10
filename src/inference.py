"""Reusable, validated ResNet18 classification with separate optional Grad-CAM."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
from io import BytesIO
import json
from pathlib import Path
import pickle
import sys
from threading import RLock
from types import MappingProxyType
from typing import TYPE_CHECKING
import warnings

from PIL import Image, UnidentifiedImageError
import torch

from .dataset import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, build_transforms
from .models import RESNET18_ARCHITECTURE_ID, ResNet18Classifier, load_checkpoint
from .split_data import ROOT

if TYPE_CHECKING:
    from .explain import Explanation

ImageInput = str | Path | Image.Image | bytes | bytearray | memoryview
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
CONFIDENCE_DEFINITION = "Uncalibrated maximum softmax score, not a calibrated probability"


class InvalidImageError(ValueError):
    """Unsupported, missing, corrupt, or excessively large image input."""


class ModelLoadError(RuntimeError):
    """Checkpoint cannot be reconstructed with the required model metadata."""


def _check_dimensions(image: Image.Image) -> None:
    if min(image.size) < 1 or image.width * image.height > MAX_IMAGE_PIXELS:
        raise InvalidImageError(f"Image must have 1..{MAX_IMAGE_PIXELS:,} pixels")


def decode_image(source: ImageInput) -> Image.Image:
    """Fully decode into an owned RGB image, matching ImageFolder's PIL loader.

    File handles are closed; caller-owned PIL images are never closed or changed.
    No EXIF orientation correction, alpha compositing, or other new preprocessing
    is introduced. Encoded data is limited to 20 MiB and decoded data to 20M pixels.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            if isinstance(source, Image.Image):
                _check_dimensions(source)
                source.load()
                return source.convert("RGB").copy()
            if isinstance(source, (str, Path)):
                path = Path(source)
                if not path.is_file():
                    raise InvalidImageError(f"Image file does not exist or is not a regular file: {path}")
                if path.stat().st_size > MAX_IMAGE_BYTES:
                    raise InvalidImageError("Encoded image exceeds 20 MiB")
                data = path.read_bytes()
            elif isinstance(source, (bytes, bytearray, memoryview)):
                if len(source) > MAX_IMAGE_BYTES:
                    raise InvalidImageError("Encoded image exceeds 20 MiB")
                data = bytes(source)
            else:
                raise InvalidImageError("Expected an image path, PIL image, or encoded image bytes")
            if not data or len(data) > MAX_IMAGE_BYTES:
                raise InvalidImageError("Encoded image is empty or exceeds 20 MiB")
            with Image.open(BytesIO(data)) as image:
                _check_dimensions(image)
                image.verify()
            with Image.open(BytesIO(data)) as image:
                image.load()
                return image.convert("RGB").copy()
    except InvalidImageError:
        raise
    except (OSError, ValueError, SyntaxError, UnidentifiedImageError,
            Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise InvalidImageError(f"Cannot decode image: {exc}") from exc


@dataclass(frozen=True)
class Prediction:
    predicted_class: str
    predicted_index: int
    confidence: float
    probabilities: dict[str, float]
    model_info: dict

    def to_dict(self) -> dict:
        """Return an independent JSON-serializable representation."""
        return asdict(self)


class InferencePipeline:
    """Load once, reuse for images, and serialize model operations per instance.

    Auto selects CUDA or CPU. Auto mode also falls back to CPU on CUDA placement
    failure or inference OOM, recording why; explicit CUDA requests fail clearly.
    CUDA defaults to FP16 autocast, matching evaluation; CPU always uses FP32.
    Grad-CAM is a separate FP32 operation with its own gradients-enabled scope.
    """

    def __init__(self, checkpoint: str | Path = ROOT / "artifacts/models/resnet18_best.pth",
                 *, device: str | torch.device = "auto", mixed_precision: bool = True):
        if not isinstance(mixed_precision, bool):
            raise ValueError("mixed_precision must be a bool")
        self._automatic_device = device == "auto"
        self.fallback_reason = None
        if self._automatic_device:
            selected = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if selected.type == "cpu":
                self.fallback_reason = "CUDA is unavailable"
        else:
            try:
                selected = torch.device(device)
            except (TypeError, RuntimeError) as exc:
                raise ValueError("device must be auto, cpu, cuda, or cuda:N") from exc
            if selected.type not in ("cpu", "cuda"):
                raise ValueError("Only CPU and CUDA devices are supported")
            if selected.type == "cuda":
                if not torch.cuda.is_available():
                    raise ValueError("CUDA was explicitly requested but is unavailable")
                index = selected.index if selected.index is not None else torch.cuda.current_device()
                if index >= torch.cuda.device_count():
                    raise ValueError(f"CUDA device {index} does not exist")
                selected = torch.device(f"cuda:{index}")
        self.checkpoint = Path(checkpoint).resolve()
        try:
            # Validate and reconstruct on CPU first; never downloads weights.
            model, metadata = load_checkpoint(self.checkpoint, "cpu")
            if not isinstance(model, ResNet18Classifier) or metadata["architecture"] != RESNET18_ARCHITECTURE_ID:
                raise ValueError("Inference requires the saved ResNet18 architecture")
            expected = {"image_size": list(IMAGE_SIZE), "mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)}
            if metadata.get("preprocessing") != expected:
                raise ValueError("Checkpoint preprocessing differs from the evaluation pipeline")
            self.checkpoint_sha256 = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        except (OSError, ValueError, KeyError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
            raise ModelLoadError(f"Cannot load ResNet18 checkpoint: {exc}") from exc
        self.model = model.eval()
        try:
            self.model.to(selected)
        except RuntimeError as exc:
            if not self._automatic_device or selected.type != "cuda":
                raise
            self.model.to("cpu")
            selected = torch.device("cpu")
            self.fallback_reason = f"CUDA placement failed: {exc}"
        self.device = selected
        self.mixed_precision = mixed_precision
        self.class_names = tuple(metadata["class_names"])
        self.class_to_idx = MappingProxyType(dict(metadata["class_to_idx"]))
        self._architecture = metadata["architecture"]
        self._selected_epoch = metadata.get("epoch")
        self._selected_phase = metadata.get("phase")
        self.transform = build_transforms()["test"]
        self._lock = RLock()

    @property
    def model_info(self) -> dict:
        return {"architecture": self._architecture, "checkpoint": str(self.checkpoint),
                "checkpoint_sha256": self.checkpoint_sha256, "class_names": list(self.class_names),
                "class_to_idx": dict(self.class_to_idx), "num_classes": len(self.class_names),
                "device": str(self.device), "precision": "float16 autocast" if self.device.type == "cuda" and self.mixed_precision else "float32",
                "selected_epoch": self._selected_epoch, "selected_phase": self._selected_phase,
                "fallback_reason": self.fallback_reason, "confidence_definition": CONFIDENCE_DEFINITION}

    def _forward_probabilities(self, tensor: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.inference_mode(), torch.autocast(device_type=self.device.type,
                                                   enabled=self.device.type == "cuda" and self.mixed_precision):
            logits = self.model(tensor.to(self.device))
            if logits.shape != (1, len(self.class_names)) or not torch.isfinite(logits).all():
                raise RuntimeError("Model returned invalid or non-finite six-class logits")
            return logits.float().softmax(dim=1)[0].cpu()

    def predict(self, source: ImageInput) -> Prediction:
        """Return plain scalar predictions; inference tensors never escape the scope."""
        image = decode_image(source)
        tensor = self.transform(image).unsqueeze(0)
        with self._lock:
            try:
                scores = self._forward_probabilities(tensor)
            except torch.cuda.OutOfMemoryError:
                if not self._automatic_device or self.device.type != "cuda":
                    raise
                self.model.to("cpu")
                self.device = torch.device("cpu")
                self.fallback_reason = "CUDA inference ran out of memory; retried on CPU"
                torch.cuda.empty_cache()
                scores = self._forward_probabilities(tensor)
            values = scores.tolist()
            index = max(range(len(values)), key=values.__getitem__)
            return Prediction(self.class_names[index], index, values[index],
                              dict(zip(self.class_names, values)), self.model_info)

    def explain(self, source: ImageInput) -> Explanation:
        """Run existing FP32 Grad-CAM separately; never inside inference_mode().

        Returns the Step 7 Explanation, including its own prediction/confidence;
        FP32 explanations can differ slightly from a FP16 classification result.
        """
        if torch.is_inference_mode_enabled():
            raise RuntimeError("Call explain() outside torch.inference_mode(); Grad-CAM requires gradients")
        from .explain import explain_image
        image = decode_image(source)
        with self._lock, torch.enable_grad():
            return explain_image(self.model, image)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--fp32", action="store_true", help="Disable CUDA FP16 autocast")
    args = parser.parse_args()
    if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        parser.error("Use the project's existing .venv Python interpreter")
    try:
        result = InferencePipeline(device=args.device, mixed_precision=not args.fp32).predict(args.image)
        print(json.dumps(result.to_dict(), indent=2, allow_nan=False))
    except (InvalidImageError, ModelLoadError, ValueError, RuntimeError, OSError) as exc:
        print(f"Inference failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
