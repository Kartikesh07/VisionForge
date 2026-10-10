"""Validation checkpoint comparison and GPU forward-latency benchmark; no test data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .train import configure_reproducibility
import numpy as np
import torch

from .dataset import create_dataloaders
from .models import load_checkpoint
from .split_data import ROOT


def benchmark_latency(model, images, *, warmup: int = 30, repetitions: int = 200) -> dict:
    """CUDA-event timings for a resident tensor; omit decoding and host/device transfer."""
    if images.device.type != "cuda":
        raise ValueError("The comparison benchmark requires CUDA-resident images")
    if warmup < 1 or repetitions < 1:
        raise ValueError("warmup and repetitions must be positive")
    model.eval()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for _ in range(warmup):
            model(images)
        torch.cuda.synchronize()
        for start, end in zip(starts, ends):
            start.record()
            model(images)
            end.record()
        torch.cuda.synchronize()
    timings = np.array([start.elapsed_time(end) for start, end in zip(starts, ends)])
    return {"mean_ms": float(timings.mean()), "median_ms": float(np.median(timings)),
            "p95_ms": float(np.percentile(timings, 95)), "std_ms": float(timings.std()),
            "repetitions": repetitions, "warmup": warmup}


def compare_checkpoints(baseline_path: Path, resnet_path: Path, output_path: Path) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU latency comparison requires CUDA")
    configure_reproducibility(42)
    torch.set_num_threads(1)
    loader = create_dataloaders(batch_size=1, seed=42, splits=("val",))["val"]
    images, _ = next(iter(loader))
    images = images.to("cuda")
    rows = {}
    for name, path in (("custom_cnn", baseline_path), ("resnet18", resnet_path)):
        model, checkpoint = load_checkpoint(path, "cuda")
        rows[name] = {
            "checkpoint": str(path), "architecture": checkpoint["architecture"],
            "validation_accuracy": checkpoint["validation_metrics"]["accuracy"],
            "validation_macro_f1": checkpoint["validation_metrics"]["macro_f1"],
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "selected_epoch": checkpoint["epoch"], "selected_phase": checkpoint.get("phase", "custom_cnn"),
            "latency": benchmark_latency(model, images),
        }
        del model
        torch.cuda.empty_cache()
    result = {"models": rows, "protocol": {
        "device": "cuda:0", "gpu_name": torch.cuda.get_device_name(0),
        "batch_size": 1, "input_shape": [1, 3, 224, 224], "precision": "autocast float16",
        "input": "first validation image, transformed using the existing evaluation pipeline",
        "timing": "CUDA events after 30 warmup forwards; 200 measured forwards",
        "includes": "model forward only", "excludes": "image decoding, preprocessing, host-to-device transfer",
        "deterministic_algorithms": True, "test_split_used": False,
        "metrics_source": "saved best-checkpoint validation metrics (both verified after reloading)"}}
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        parser.error("Use the project's .venv Python interpreter")
    compare_checkpoints(ROOT / "artifacts/models/baseline_cnn.pth",
                        ROOT / "artifacts/models/resnet18_best.pth",
                        ROOT / "artifacts/reports/model_comparison.json")


if __name__ == "__main__":
    main()
