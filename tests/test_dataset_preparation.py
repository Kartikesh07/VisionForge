"""Dataset integrity and safe-rerun regression checks using synthetic images."""

import json
from pathlib import Path
import shutil
import sys

from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import download_data
import split_data


@pytest.fixture
def dataset(tmp_path):
    raw = tmp_path / "raw"
    for class_index, class_name in enumerate(split_data.CLASSES):
        directory = raw / class_name
        directory.mkdir(parents=True)
        for index in range(20):
            Image.new("L", (200, 200), class_index * 20 + index).save(directory / f"image_{index:02}.bmp")
    return raw, tmp_path / "processed"


def prepare(raw, processed, **options):
    return split_data.prepare_splits(raw, processed, expected_per_class=20, **options)


def test_stratification_reproducibility_and_no_mutation_on_rerun(dataset):
    raw, processed = dataset
    result = prepare(raw, processed)
    assert result["totals"] == {"train": 84, "val": 18, "test": 18}
    assert all(set(result["split_counts"][split].values()) == {count}
               for split, count in (("train", 14), ("val", 3), ("test", 3)))
    original = (processed / split_data.MANIFEST).read_bytes()
    mtimes = {str(path): path.stat().st_mtime_ns for path in processed.rglob("*") if path.is_file()}
    assert prepare(raw, processed)["action"] == "verified_existing"
    assert mtimes == {str(path): path.stat().st_mtime_ns for path in processed.rglob("*") if path.is_file()}
    assert (processed / split_data.MANIFEST).read_bytes() == original
    second = processed.with_name("processed_again")
    prepare(raw, second)
    assert (second / split_data.MANIFEST).read_bytes() == original


def test_duplicate_images_are_reported_and_kept_in_one_split(dataset):
    raw, processed = dataset
    first = raw / "crazing/image_00.bmp"
    second = raw / "crazing/image_01.bmp"
    shutil.copy2(first, second)
    report = prepare(raw, processed)
    assert report["duplicate_files"] == [["crazing/image_00.bmp", "crazing/image_01.bmp"]]
    manifest = json.loads((processed / split_data.MANIFEST).read_text())
    duplicates = [entry for entry in manifest["records"] if entry["source"] in report["duplicate_files"][0]]
    assert len({entry["split"] for entry in duplicates}) == 1
    assert report["cross_split_overlap"] == 0


def test_identical_pixels_with_different_encoding_are_detected(dataset):
    raw, processed = dataset
    second = raw / "crazing/image_01.bmp"
    second.unlink()
    with Image.open(raw / "crazing/image_00.bmp") as image:
        image.save(second.with_suffix(".png"))
    report = prepare(raw, processed)
    assert not report["duplicate_files"]
    assert len(report["duplicate_pixel_content"]) == 1
    assert report["cross_split_overlap"] == 0


@pytest.mark.parametrize("problem", ["unreadable", "missing", "conflicting_label", "wrong_size", "loose_image"])
def test_invalid_raw_data_does_not_publish_splits(dataset, problem):
    raw, processed = dataset
    path = raw / "crazing/image_00.bmp"
    if problem == "unreadable":
        path.write_bytes(b"not an image")
    elif problem == "missing":
        path.unlink()
    elif problem == "conflicting_label":
        shutil.copy2(path, raw / "inclusion/image_00.bmp")
    elif problem == "loose_image":
        shutil.copy2(path, raw / "unorganized.bmp")
    else:
        Image.new("L", (64, 64)).save(path)
    with pytest.raises(split_data.DatasetError):
        prepare(raw, processed)
    assert not processed.exists()


def test_corrupted_split_requires_explicit_rebuild(dataset):
    raw, processed = dataset
    prepare(raw, processed)
    copied = next((processed / "train/crazing").glob("*.bmp"))
    copied.write_bytes(b"corrupt")
    with pytest.raises(split_data.DatasetError, match="Unreadable"):
        prepare(raw, processed)
    assert copied.read_bytes() == b"corrupt"
    assert prepare(raw, processed, rebuild=True)["all_images_readable"]


def test_unmanaged_directory_is_preserved(dataset):
    raw, processed = dataset
    processed.mkdir()
    sentinel = processed / "keep.txt"
    sentinel.write_text("keep me")
    with pytest.raises(split_data.DatasetError, match="unmanaged"):
        prepare(raw, processed, rebuild=True)
    assert sentinel.read_text() == "keep me"


def test_copy_failure_preserves_existing_split(dataset, monkeypatch):
    raw, processed = dataset
    prepare(raw, processed)
    before = (processed / split_data.MANIFEST).read_bytes()

    def failed_copy(*args, **kwargs):
        raise OSError("simulated copy failure")

    monkeypatch.setattr(split_data.shutil, "copy2", failed_copy)
    with pytest.raises(OSError, match="simulated"):
        prepare(raw, processed, rebuild=True)
    assert (processed / split_data.MANIFEST).read_bytes() == before
    assert not list(processed.parent.glob(".split-stage-*"))


def test_raw_and_processed_cannot_overlap(dataset):
    raw, _ = dataset
    with pytest.raises(split_data.DatasetError, match="separate"):
        prepare(raw, raw / "processed")


def test_archive_paths_reject_traversal_and_detection_data():
    for member in ("../bad.bmp", "NEU-CLS/../bad.bmp", "C:/bad.bmp", "NEU-DET/images/crazing_1.jpg"):
        with pytest.raises(split_data.DatasetError):
            download_data.validate_archive_members([member])
