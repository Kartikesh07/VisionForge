"""Tests for transforms, labels, loader ordering, seeds, and Windows spawning."""

from pathlib import Path
import random
import shutil
import sys

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import (
    CLASS_TO_IDX, build_transforms, create_dataloaders, create_datasets, seed_everything,
)
from src.split_data import CLASSES


@pytest.fixture(scope="module", autouse=True)
def modest_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def processed_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("loader-data")
    for split in ("train", "val", "test"):
        for class_index, class_name in enumerate(CLASSES):
            directory = root / split / class_name
            directory.mkdir(parents=True)
            for index in range(7):
                Image.new("L", (200, 200), 20 + class_index * 30 + index).save(directory / f"image_{index}.bmp")
    return root


def asymmetric_image():
    array = ((np.arange(200)[:, None] * 3 + np.arange(200)[None, :] * 7) % 256).astype(np.uint8)
    array[10:40, 20:60] = 255
    return Image.fromarray(array)


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_shapes_dtype_and_grayscale_channels(split):
    seed_everything(42)
    tensor = build_transforms()[split](asymmetric_image())
    assert tensor.shape == (3, 224, 224) and tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    restored = tensor * std + mean
    torch.testing.assert_close(restored[0], restored[1], atol=1e-6, rtol=0)
    torch.testing.assert_close(restored[0], restored[2], atol=1e-6, rtol=0)
    assert restored.min() >= -1e-6 and restored.max() <= 1 + 1e-6


@pytest.mark.parametrize("intensity", [0, 128, 255])
def test_imagenet_normalization_numeric_values(intensity):
    image = Image.new("L", (200, 200), intensity)
    tensor = build_transforms()["val"](image)
    expected = (torch.tensor(intensity / 255.0) - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])
    torch.testing.assert_close(tensor, expected[:, None, None].expand(3, 224, 224))


def test_training_augmentation_and_seed_repeatability():
    pipelines = build_transforms()
    operations = pipelines["train"].transforms
    horizontal = next(item for item in operations if isinstance(item, transforms.RandomHorizontalFlip))
    vertical = next(item for item in operations if isinstance(item, transforms.RandomVerticalFlip))
    rotation = next(item for item in operations if isinstance(item, transforms.RandomRotation))
    assert horizontal.p == vertical.p == 0.5
    assert rotation.degrees == [-10.0, 10.0]
    image = asymmetric_image()
    seed_everything(42)
    first = pipelines["train"](image)
    seed_everything(42)
    assert torch.equal(first, pipelines["train"](image))
    seed_everything(43)
    assert not torch.equal(first, pipelines["train"](image))
    assert not torch.equal(first, pipelines["val"](image))


def test_evaluation_has_no_augmentation_and_ignores_random_seed():
    pipelines = build_transforms()
    image = asymmetric_image()
    for split in ("val", "test"):
        assert [type(item) for item in pipelines[split].transforms] == [
            transforms.Resize, transforms.Grayscale, transforms.ToTensor, transforms.Normalize,
        ]
        seed_everything(1)
        first = pipelines[split](image)
        seed_everything(999)
        assert torch.equal(first, pipelines[split](image))
    assert torch.equal(pipelines["val"](image), pipelines["test"](image))


def test_imagefolder_mapping_and_all_label_ranges(processed_dataset):
    loaded = create_datasets(processed_dataset)
    expected = {name: index for index, name in enumerate(CLASSES)}
    assert expected == CLASS_TO_IDX
    for dataset in loaded.values():
        assert isinstance(dataset, datasets.ImageFolder)
        assert dataset.class_to_idx == expected and dataset.classes == list(CLASSES)
        assert len(dataset) == 42 and set(dataset.targets) == set(range(6))
        image, label = dataset[0]
        assert image.shape == (3, 224, 224) and 0 <= label < 6


def test_dataloader_defaults_shuffle_order_and_last_batch(processed_dataset):
    loaders = create_dataloaders(processed_dataset, pin_memory=False)
    assert isinstance(loaders["train"].sampler, RandomSampler)
    for split, loader in loaders.items():
        assert loader.batch_size == 32 and loader.num_workers == 0
        assert not loader.drop_last and not loader.persistent_workers
        assert loader.multiprocessing_context is None
        batches = list(loader)
        assert [len(labels) for _, labels in batches] == [32, 10]
        for images, labels in batches:
            assert images.shape[1:] == (3, 224, 224)
            assert images.device.type == labels.device.type == "cpu"
            assert labels.dtype == torch.int64 and 0 <= labels.min() <= labels.max() < 6
        if split != "train":
            assert isinstance(loader.sampler, SequentialSampler)
            assert torch.cat([labels for _, labels in batches]).tolist() == loader.dataset.targets


def test_loader_run_is_reproducible_and_epochs_advance(processed_dataset):
    def first_batch(seed):
        loader = create_dataloaders(processed_dataset, seed=seed, pin_memory=False)["train"]
        return next(iter(loader))

    first = first_batch(42)
    repeated = first_batch(42)
    assert all(torch.equal(left, right) for left, right in zip(first, repeated))
    different = first_batch(43)
    assert not torch.equal(first[0], different[0])
    loader = create_dataloaders(processed_dataset, pin_memory=False)["train"]
    assert list(loader.sampler) != list(loader.sampler)


def test_validation_and_test_loader_order_is_repeatable(processed_dataset):
    loaders = create_dataloaders(processed_dataset, pin_memory=False)
    for split in ("val", "test"):
        first = list(loaders[split])
        second = list(loaders[split])
        assert all(torch.equal(a, b) and torch.equal(x, y)
                   for (a, x), (b, y) in zip(first, second))


def test_optional_worker_uses_spawn_and_loads_a_batch(processed_dataset):
    loader = create_dataloaders(processed_dataset, num_workers=1, pin_memory=False)["val"]
    assert loader.multiprocessing_context.get_start_method() == "spawn"
    images, labels = next(iter(loader))
    assert images.shape == (32, 3, 224, 224)
    assert labels.tolist() == loader.dataset.targets[:32]


def test_seed_covers_python_numpy_and_torch():
    seed_everything(42)
    first = (random.random(), np.random.random(), torch.rand(3))
    seed_everything(42)
    second = (random.random(), np.random.random(), torch.rand(3))
    assert first[:2] == second[:2] and torch.equal(first[2], second[2])


@pytest.mark.parametrize("options", [
    {"batch_size": 0}, {"batch_size": 1.5}, {"num_workers": -1},
    {"num_workers": True}, {"seed": -1}, {"seed": 2**32}, {"pin_memory": "yes"},
])
def test_invalid_loader_options_are_rejected(processed_dataset, options):
    with pytest.raises(ValueError):
        create_dataloaders(processed_dataset, **options)


def test_missing_split_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Complete Step 2"):
        create_datasets(tmp_path)


@pytest.mark.parametrize("change", ["missing_class", "extra_class"])
def test_mismatched_split_classes_are_rejected(processed_dataset, tmp_path, change):
    root = tmp_path / "mismatch"
    shutil.copytree(processed_dataset, root)
    if change == "missing_class":
        (root / "val" / "scratches").rename(root / "removed_class")
    else:
        (root / "test" / "unexpected").mkdir()
    with pytest.raises(ValueError, match="class directories"):
        create_datasets(root)
