"""Grad-CAM explanations of the saved ResNet18 using validation images only."""

from __future__ import annotations

from .train import configure_reproducibility  # Configure CUDA before initialization.

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
import torch

from .dataset import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, build_transforms, create_datasets
from .models import RESNET18_ARCHITECTURE_ID, ResNet18Classifier, load_checkpoint
from .split_data import CLASSES, ROOT


OUTPUT = ROOT / "artifacts/reports/gradcam"
LIMITATION = "Grad-CAM shows regions influencing the predicted class; it is model interpretability, not precise defect segmentation."


@dataclass
class Explanation:
    original: np.ndarray
    input_heatmap: np.ndarray
    heatmap: np.ndarray
    heatmap_rgb: np.ndarray
    overlay_rgb: np.ndarray
    predicted_index: int
    predicted_class: str
    confidence: float
    logits: np.ndarray


def load_explanation_model(checkpoint: Path = ROOT / "artifacts/models/resnet18_best.pth",
                           device: str | torch.device = "cuda:0") -> ResNet18Classifier:
    model, metadata = load_checkpoint(checkpoint, device)
    if metadata["architecture"] != RESNET18_ARCHITECTURE_ID:
        raise ValueError("Grad-CAM requires the saved ResNet18 checkpoint")
    expected = {"image_size": list(IMAGE_SIZE), "mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)}
    if metadata.get("preprocessing") != expected:
        raise ValueError("Checkpoint preprocessing differs from the existing validation pipeline")
    return model.eval()


def colorize_heatmap(original: np.ndarray, heatmap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV JET heatmap and 50/50 overlay on original grayscale pixels, in RGB."""
    if original.ndim != 2 or original.dtype != np.uint8 or heatmap.shape != original.shape:
        raise ValueError("Expected uint8 grayscale original and matching two-dimensional heatmap")
    if not np.isfinite(heatmap).all() or heatmap.min() < 0 or heatmap.max() > 1:
        raise ValueError("Heatmap must contain finite values in [0, 1]")
    colored_bgr = cv2.applyColorMap(np.rint(heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    grayscale_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    overlay_bgr = cv2.addWeighted(grayscale_bgr, 0.5, colored_bgr, 0.5, 0)
    return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB), cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)


def explain_image(model: ResNet18Classifier, image: Image.Image) -> Explanation:
    """Compute predicted-class Grad-CAM with one FP32 forward and backward pass.

    The wrapper's backbone.layer4[-1] is model.layer4[-1] on bare ResNet18.
    GradCAM targets=None selects the top logit from this same forward, so reported
    predictions and CAM targets cannot disagree. No augmentation/smoothing is used.
    An input requiring gradients also supports a completely frozen backbone.
    """
    if not isinstance(model, ResNet18Classifier):
        raise ValueError("Expected VisionForge ResNet18Classifier")
    if torch.is_inference_mode_enabled():
        raise RuntimeError("Grad-CAM requires gradients; call outside torch.inference_mode()")
    model.eval()
    original = np.array(image.convert("L"), dtype=np.uint8)
    tensor = build_transforms()["val"](image).unsqueeze(0)
    device = next(model.parameters()).device
    try:
        with torch.enable_grad(), torch.autocast(device_type=device.type, enabled=False):
            tensor = tensor.to(device=device, dtype=torch.float32).requires_grad_(True)
            with GradCAM(model=model, target_layers=[model.backbone.layer4[-1]]) as cam:
                batch_maps = cam(input_tensor=tensor, targets=None, aug_smooth=False, eigen_smooth=False)
                logits = cam.outputs.detach().float()
                if logits.shape != (1, len(CLASSES)) or not torch.isfinite(logits).all():
                    raise ValueError("Expected finite six-class logits")
                confidence, predicted = logits.softmax(dim=1).max(dim=1)
                input_heatmap = np.asarray(batch_maps[0], dtype=np.float32).copy()
                logits_array = logits.cpu().numpy()[0].copy()
                index, score = int(predicted.item()), float(confidence.item())
        if input_heatmap.shape != IMAGE_SIZE or not np.isfinite(input_heatmap).all():
            raise ValueError("Grad-CAM returned invalid heatmap dimensions or values")
        input_heatmap = np.clip(input_heatmap, 0, 1)
        heatmap = cv2.resize(input_heatmap, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_LINEAR)
        heatmap = np.clip(heatmap, 0, 1).astype(np.float32)
        colored, overlay = colorize_heatmap(original, heatmap)
        return Explanation(original, input_heatmap, heatmap, colored, overlay,
                           index, CLASSES[index], score, logits_array)
    finally:
        # The library backpropagates but never updates parameters. Do not retain gradients.
        model.zero_grad(set_to_none=True)


def save_explanation(result: Explanation, directory: Path, *, source: Path,
                     actual_class: str | None = None) -> dict:
    """Save lossless images, numeric CAMs, prediction metadata, and a triptych."""
    directory.mkdir(parents=True, exist_ok=True)
    images = {"original.png": result.original,
              "heatmap.png": cv2.cvtColor(result.heatmap_rgb, cv2.COLOR_RGB2BGR),
              "overlay.png": cv2.cvtColor(result.overlay_rgb, cv2.COLOR_RGB2BGR),
              "heatmap_gray.png": np.rint(result.heatmap * 255).astype(np.uint8)}
    for filename, pixels in images.items():
        if not cv2.imwrite(str(directory / filename), pixels):
            raise OSError(f"Failed to save {directory / filename}")
    np.save(directory / "heatmap.npy", result.heatmap, allow_pickle=False)
    np.save(directory / "input_heatmap.npy", result.input_heatmap, allow_pickle=False)
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    for ax, pixels, title in zip(axes, (result.original, result.heatmap_rgb, result.overlay_rgb),
                                 ("Original grayscale image", "Grad-CAM (low blue, high red)", "50/50 overlay")):
        ax.imshow(pixels, cmap="gray" if pixels.ndim == 2 else None, vmin=0, vmax=255)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    actual = f"Actual: {actual_class} | " if actual_class else ""
    fig.suptitle(f"{actual}Predicted: {result.predicted_class}\nConfidence: {result.confidence:.2%} (uncalibrated softmax)", fontsize=11)
    fig.text(0.5, 0.025, "Model interpretability; not precise defect segmentation.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.86))
    fig.savefig(directory / "explanation.png", dpi=150)
    plt.close(fig)
    metadata = {"source_image": str(source.resolve()), "actual_class": actual_class,
                "predicted_index": result.predicted_index, "predicted_class": result.predicted_class,
                "confidence": result.confidence, "confidence_definition": "uncalibrated maximum softmax score",
                "target_class": result.predicted_class, "target_layer": "backbone.layer4[-1] (bare ResNet18: model.layer4[-1])",
                "method": "pytorch-grad-cam GradCAM", "precision": "float32 with gradients enabled",
                "original_shape": list(result.original.shape), "input_heatmap_shape": list(result.input_heatmap.shape),
                "heatmap_shape": list(result.heatmap.shape), "heatmap_min": float(result.heatmap.min()),
                "heatmap_max": float(result.heatmap.max()), "all_zero_heatmap": bool(not np.any(result.heatmap)),
                "normalization": "Per-image Grad-CAM [0,1], then bilinear resize to original dimensions",
                "limitation": LIMITATION, "files": list(images) + ["heatmap.npy", "input_heatmap.npy", "explanation.png"]}
    (directory / "prediction.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return metadata


def protected_snapshot() -> dict:
    paths = [path for root in (ROOT / "data/raw", ROOT / "data/processed", ROOT / "artifacts/models")
             for path in root.rglob("*") if path.is_file()]
    reports = ROOT / "artifacts/reports"
    paths += [path for path in reports.rglob("*") if path.is_file()
              and path.relative_to(reports).parts[0] != "gradcam"]
    return {str(path): [hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns] for path in paths}


def select_validation_images(image: Path | None = None, *, samples_per_class: int = 1) -> list[tuple[Path, str]]:
    """Select deterministic demonstrations; do not construct or read test images."""
    dataset = create_datasets(splits=("val",))["val"]
    if image is not None:
        requested = image.resolve()
        matches = [(Path(path), CLASSES[label]) for path, label in dataset.samples if Path(path).resolve() == requested]
        if len(matches) != 1:
            raise ValueError("For Step 7, --image must select an existing image under data/processed/val/<class>/")
        return matches
    if isinstance(samples_per_class, bool) or not 1 <= samples_per_class <= 45:
        raise ValueError("samples_per_class must be between 1 and 45")
    return [(Path(path), name) for index, name in enumerate(CLASSES)
            for path, label in [item for item in dataset.samples if item[1] == index][:samples_per_class]]


def run_explanations(image: Path | None = None, *, samples_per_class: int = 1) -> dict:
    before = protected_snapshot()
    selected = select_validation_images(image, samples_per_class=samples_per_class)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Grad-CAM demonstrations; no CPU fallback")
    device = torch.device("cuda:0")
    gpu = torch.cuda.get_device_name(device)
    if "3050" not in gpu:
        raise RuntimeError(f"Requested RTX 3050; detected {gpu}")
    torch.set_num_threads(1)
    configure_reproducibility(42)
    model = load_explanation_model(device=device)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    examples = []
    for path, actual in selected:
        with Image.open(path) as source_image:
            result = explain_image(model, source_image)
        directory = OUTPUT / f"{actual}_{path.stem}"
        metadata = save_explanation(result, directory, source=path, actual_class=actual)
        metadata["output_directory"] = str(directory)
        examples.append(metadata)
    unchanged = before == protected_snapshot()
    if not unchanged:
        raise RuntimeError("Dataset, checkpoints or previous reports changed during Grad-CAM generation")
    report = {"device": gpu, "seed": 42, "precision": "float32", "sample_count": len(examples),
              "source_split": "val", "test_images_explained": 0,
              "datasets_checkpoints_and_previous_reports_unchanged": unchanged,
              "limitation": LIMITATION, "examples": examples}
    (OUTPUT / "gradcam_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Validation Grad-CAM examples", "", LIMITATION, "",
             "Confidence is an uncalibrated softmax score. Heatmaps are normalized separately per image.", ""]
    for example in examples:
        directory = Path(example["output_directory"]).name
        lines += [f"## {example['actual_class']}: {Path(example['source_image']).name}", "",
                  f"Predicted: {example['predicted_class']}; confidence: {example['confidence']:.2%}.", "",
                  f"![Original, heatmap and overlay]({directory}/explanation.png)", ""]
    (OUTPUT / "index.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="Select one existing validation image")
    parser.add_argument("--samples-per-class", type=int, default=1, help="First N validation images per class (default 1)")
    args = parser.parse_args()
    if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        parser.error("Use the project's existing .venv Python interpreter")
    if args.image is not None and args.samples_per_class != 1:
        parser.error("--image and a non-default --samples-per-class are mutually exclusive")
    try:
        print(json.dumps(run_explanations(args.image, samples_per_class=args.samples_per_class), indent=2))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Grad-CAM failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
