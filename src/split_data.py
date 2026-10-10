"""Validate NEU-CLS and copy deterministic, stratified, leakage-free splits."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import tempfile

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CLASSES = (
    "crazing", "inclusion", "patches", "pitted_surface",
    "rolled_in_scale", "scratches",
)
SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
MANIFEST = "split_manifest.json"


class DatasetError(ValueError):
    """Dataset validation failed; no partial split should be published."""


def image_fingerprints(path: Path) -> tuple[str, str]:
    if path.is_symlink() or not path.is_file():
        raise DatasetError(f"Expected a regular image file: {path}")
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            if image.size != (200, 200) or image.mode != "L":
                raise DatasetError(f"Expected original 200x200 grayscale image: {path}")
            pixels = hashlib.sha256(image.tobytes()).hexdigest()
    except (OSError, SyntaxError) as exc:
        raise DatasetError(f"Unreadable image: {path}: {exc}") from exc
    return hashlib.sha256(path.read_bytes()).hexdigest(), pixels


def inspect_raw(raw: Path, expected_per_class: int = 300) -> tuple[list[dict], dict]:
    if not raw.is_dir() or raw.is_symlink():
        raise DatasetError(f"Missing regular raw dataset directory: {raw}")
    unexpected = [path.name for path in raw.iterdir() if path.is_dir() and path.name not in CLASSES]
    if unexpected:
        raise DatasetError(f"Unexpected raw directories (use NEU-CLS, not NEU-DET): {unexpected}")
    loose_images = [path.name for path in raw.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES]
    if loose_images:
        raise DatasetError(f"Raw images must be organized in the six class directories: {loose_images}")
    records = []
    for class_name in CLASSES:
        directory = raw / class_name
        if not directory.is_dir() or directory.is_symlink():
            raise DatasetError(f"Missing regular class directory: {directory}")
        files = sorted(directory.iterdir(), key=lambda path: path.name)
        if len(files) != expected_per_class:
            raise DatasetError(f"{class_name}: expected {expected_per_class} images, found {len(files)}")
        for path in files:
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                raise DatasetError(f"Unexpected file in class directory: {path}")
            file_hash, pixel_hash = image_fingerprints(path)
            records.append({"source": path.relative_to(raw).as_posix(), "class": class_name,
                            "sha256": file_hash, "pixel_sha256": pixel_hash})
    duplicates = {}
    for field in ("sha256", "pixel_sha256"):
        grouped = defaultdict(list)
        for record in records:
            grouped[record[field]].append(record)
        duplicates[field] = [[item["source"] for item in group]
                             for group in grouped.values() if len(group) > 1]
        for group in grouped.values():
            if len({item["class"] for item in group}) > 1:
                raise DatasetError(f"Duplicate content has conflicting class labels: {group}")
    return records, duplicates


def make_manifest(records: list[dict], seed: int = 42) -> dict:
    rng = random.Random(seed)
    assigned = []
    for class_name in CLASSES:
        class_records = [record for record in records if record["class"] == class_name]
        count = len(class_records)
        if count % 20:
            raise DatasetError("Exact 70/15/15 splits require a per-class count divisible by 20")
        remaining = {"train": count * 14 // 20, "val": count * 3 // 20, "test": count * 3 // 20}
        grouped = defaultdict(list)
        for record in class_records:
            grouped[record["pixel_sha256"]].append(record)
        groups = list(grouped.values())
        rng.shuffle(groups)
        # Place any duplicate groups first, keeping their content in one split.
        groups.sort(key=len, reverse=True)
        for group in groups:
            choices = [split for split in SPLITS if remaining[split] >= len(group)]
            if not choices:
                raise DatasetError("Cannot fit duplicate groups into the exact requested split counts")
            split = rng.choices(choices, weights=[remaining[item] for item in choices], k=1)[0]
            remaining[split] -= len(group)
            for record in group:
                assigned.append({**record, "split": split,
                                 "destination": f"{split}/{record['source']}"})
        if any(remaining.values()):
            raise DatasetError(f"Unfilled split counts: {remaining}")
    return {"schema_version": 1, "dataset": "NEU-CLS", "seed": seed,
            "ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
            "records": sorted(assigned, key=lambda record: record["source"])}


def verify_processed(processed: Path, manifest: dict) -> dict:
    if processed.is_symlink():
        raise DatasetError(f"Processed directory cannot be a symlink: {processed}")
    expected = {record["destination"] for record in manifest["records"]} | {MANIFEST}
    actual = set()
    expected_directories = set(SPLITS) | {f"{split}/{name}" for split in SPLITS for name in CLASSES}
    for path in processed.rglob("*"):
        relative = path.relative_to(processed).as_posix()
        if path.is_symlink():
            raise DatasetError(f"Symlink in processed dataset: {path}")
        if path.is_dir() and relative not in expected_directories:
            raise DatasetError(f"Unexpected processed directory: {path}")
        if path.is_file() and relative != ".gitkeep":
            actual.add(relative)
    if actual != expected:
        raise DatasetError(f"Processed files differ: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
    stored_manifest = json.loads((processed / MANIFEST).read_text(encoding="utf-8"))
    if stored_manifest != manifest:
        raise DatasetError("Stored manifest differs from the current raw data or split plan")
    memberships = {field: {} for field in ("source", "sha256", "pixel_sha256")}
    counts = {split: Counter() for split in SPLITS}
    for record in manifest["records"]:
        file_hash, pixel_hash = image_fingerprints(processed / record["destination"])
        if (file_hash, pixel_hash) != (record["sha256"], record["pixel_sha256"]):
            raise DatasetError(f"Split copy differs from its original image: {record['destination']}")
        for field, membership in memberships.items():
            previous = membership.setdefault(record[field], record["split"])
            if previous != record["split"]:
                raise DatasetError(f"Image leakage across splits: {record['source']} ({field})")
        counts[record["split"]][record["class"]] += 1
    return {split: dict(counts[split]) for split in SPLITS}


def prepare_splits(raw: Path, processed: Path, *, seed: int = 42,
                   rebuild: bool = False, verify_only: bool = False,
                   expected_per_class: int = 300) -> dict:
    if raw.is_symlink() or processed.is_symlink():
        raise DatasetError("Dataset root directories cannot be symlinks")
    raw, processed = raw.resolve(), processed.resolve()
    if raw == processed or raw.is_relative_to(processed) or processed.is_relative_to(raw):
        raise DatasetError("Raw and processed paths must be separate directories")
    records, duplicates = inspect_raw(raw, expected_per_class)
    manifest = make_manifest(records, seed)
    if verify_only or ((processed / MANIFEST).exists() and not rebuild):
        counts = verify_processed(processed, manifest)
        action = "verified_existing"
    else:
        if processed.exists():
            existing = {path.name for path in processed.iterdir()} - {".gitkeep"}
            if existing and not (processed / MANIFEST).is_file():
                raise DatasetError("Refusing to replace unmanaged files in processed directory")
        processed.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".split-stage-", dir=processed.parent))
        backup = stage.with_name(stage.name + "-backup")
        try:
            for split in SPLITS:
                for class_name in CLASSES:
                    (stage / split / class_name).mkdir(parents=True)
            if (processed / ".gitkeep").is_file() and not (processed / ".gitkeep").is_symlink():
                shutil.copy2(processed / ".gitkeep", stage / ".gitkeep")
            for record in manifest["records"]:
                shutil.copy2(raw / record["source"], stage / record["destination"])
            (stage / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            counts = verify_processed(stage, manifest)
            if processed.exists():
                processed.rename(backup)
            try:
                stage.rename(processed)
            except BaseException:
                if backup.exists():
                    backup.rename(processed)
                raise
            if backup.exists():
                shutil.rmtree(backup)
            action = "created"
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    return {"dataset": "NEU-CLS", "seed": seed, "action": action,
            "raw_counts": dict(Counter(record["class"] for record in records)),
            "split_counts": counts,
            "totals": {split: sum(counts[split].values()) for split in SPLITS},
            "duplicate_files": duplicates["sha256"],
            "duplicate_pixel_content": duplicates["pixel_sha256"],
            "unique_pixel_content": len({record["pixel_sha256"] for record in records}),
            "unique_pixel_content_per_split": {
                split: len({record["pixel_sha256"] for record in manifest["records"]
                            if record["split"] == split}) for split in SPLITS
            },
            "cross_split_overlap": 0, "all_images_readable": True,
            "copies_byte_identical_to_raw": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rebuild", action="store_true", help="Replace only a previously managed split")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.rebuild and args.verify_only:
        parser.error("--rebuild and --verify-only are mutually exclusive")
    if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        parser.error("Use the project's .venv Python interpreter")
    try:
        report = prepare_splits(args.raw_dir, args.processed_dir, seed=args.seed,
                                rebuild=args.rebuild, verify_only=args.verify_only)
    except (DatasetError, OSError, json.JSONDecodeError) as exc:
        print(f"Dataset preparation failed: {exc}", file=sys.stderr)
        return 1
    output = ROOT / "artifacts/reports/dataset-integrity.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
