"""Load one real NEU-CLS batch, transfer it to CUDA, and verify dataset immutability."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.dataset import CLASS_TO_IDX, create_dataloaders


def dataset_snapshot() -> dict:
    snapshot = {}
    for folder in (ROOT / "data/raw", ROOT / "data/processed"):
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                with path.open("rb") as source:
                    digest = hashlib.file_digest(source, "sha256").hexdigest()
                snapshot[path.relative_to(ROOT).as_posix()] = (digest, path.stat().st_mtime_ns)
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        parser.error("Use the project's .venv Python interpreter")
    if not torch.cuda.is_available():
        raise RuntimeError("The requested NVIDIA GPU smoke test requires CUDA availability")
    before = dataset_snapshot()
    loaders = create_dataloaders(seed=args.seed, num_workers=args.num_workers)
    images, labels = next(iter(loaders["train"]))
    assert images.shape == (32, 3, 224, 224)
    assert images.dtype == torch.float32 and labels.dtype == torch.int64
    assert labels.shape == (32,) and 0 <= labels.min().item() <= labels.max().item() < 6
    assert torch.isfinite(images).all().item()
    gpu_images = images.to("cuda", non_blocking=loaders["train"].pin_memory)
    gpu_labels = labels.to("cuda", non_blocking=loaders["train"].pin_memory)
    torch.cuda.synchronize()
    assert torch.equal(images, gpu_images.cpu()) and torch.equal(labels, gpu_labels.cpu())
    assert dataset_snapshot() == before, "Raw images or processed splits were modified"
    report = {
        "status": "passed", "seed": args.seed, "num_workers": args.num_workers,
        "batch_shape": list(images.shape), "label_shape": list(labels.shape),
        "image_dtype": str(images.dtype), "label_dtype": str(labels.dtype),
        "class_to_idx": CLASS_TO_IDX,
        "dataset_counts": {split: len(loader.dataset) for split, loader in loaders.items()},
        "cuda_available": True, "gpu_name": torch.cuda.get_device_name(0),
        "gpu_image_device": str(gpu_images.device), "gpu_label_device": str(gpu_labels.device),
        "gpu_transfer_verified": True, "dataset_files_unchanged": True,
    }
    output = ROOT / "artifacts/reports/dataloader-smoke.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
