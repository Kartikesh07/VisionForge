"""In-memory NEU-CLS transforms and reproducible PyTorch DataLoaders."""

from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from .split_data import CLASSES, ROOT, SPLITS


IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLASS_TO_IDX = {name: index for index, name in enumerate(CLASSES)}


def build_transforms() -> dict[str, transforms.Compose]:
    """Return fresh pipelines; only the training pipeline augments images."""
    pipelines = {}
    for split in SPLITS:
        operations = [
            transforms.Resize(IMAGE_SIZE, interpolation=InterpolationMode.BILINEAR, antialias=True),
            transforms.Grayscale(num_output_channels=3),
        ]
        if split == "train":
            operations.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=10, interpolation=InterpolationMode.BILINEAR),
            ])
        operations.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        pipelines[split] = transforms.Compose(operations)
    return pipelines


def seed_everything(seed: int = 42) -> None:
    """Seed process RNGs, including transforms when num_workers is zero."""
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def seed_worker(worker_id: int) -> None:
    """Picklable worker initializer; PyTorch already seeds each worker's torch RNG."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def create_datasets(processed_dir: str | Path = ROOT / "data/processed", *,
                    splits: tuple[str, ...] = SPLITS) -> dict[str, datasets.ImageFolder]:
    """Load existing splits without writing images or changing split membership."""
    processed_dir = Path(processed_dir)
    pipelines = build_transforms()
    if not splits or len(set(splits)) != len(splits) or any(split not in SPLITS for split in splits):
        raise ValueError("splits must contain unique names from train, val, test")
    result = {}
    for split in splits:
        directory = processed_dir / split
        if not directory.is_dir():
            raise ValueError(f"Missing {split} split directory: {directory}. Complete Step 2 first.")
        class_directories = sorted(path.name for path in directory.iterdir() if path.is_dir())
        if class_directories != list(CLASSES):
            raise ValueError(f"{split} class directories must be {list(CLASSES)}; found {class_directories}")
        dataset = datasets.ImageFolder(root=str(directory), transform=pipelines[split])
        if dataset.class_to_idx != CLASS_TO_IDX:
            raise ValueError(f"Inconsistent class-to-index mapping in {split}: {dataset.class_to_idx}")
        result[split] = dataset
    return result


def create_dataloaders(
    processed_dir: str | Path = ROOT / "data/processed",
    *,
    batch_size: int = 32,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    splits: tuple[str, ...] = SPLITS,
) -> dict[str, DataLoader]:
    """Create loaders with shuffled training and sequential evaluation.

    Recreate and iterate the loaders in the same order to reproduce a run with
    the same seed and worker count. With zero workers, augmentations use the
    process's torch RNG, which this factory seeds; other random operations in
    that process can change subsequent training augmentations. Successive
    iterations intentionally advance RNG state rather than repeat an epoch.

    For num_workers > 0, call from an entry point protected by
    ``if __name__ == '__main__':``. Spawn is used explicitly and worker callbacks
    are module-level functions. Batches remain on CPU until the caller moves them.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    if pin_memory is not None and not isinstance(pin_memory, bool):
        raise ValueError("pin_memory must be a bool or None")
    seed_everything(seed)
    loaded = create_datasets(processed_dir, splits=splits)
    pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory
    result = {}
    for split in splits:
        offset = SPLITS.index(split)
        options = {}
        if num_workers > 0:
            options["multiprocessing_context"] = "spawn"
            options["prefetch_factor"] = 2
        result[split] = DataLoader(
            loaded[split],
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=False,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(seed + offset),
            **options,
        )
    return result
