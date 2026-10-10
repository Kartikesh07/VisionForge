"""Download and organize the official image-only NEU-CLS archive."""

from __future__ import annotations

import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile

import requests

from split_data import CLASSES, DatasetError, ROOT, inspect_raw


OFFICIAL_PAGE = "https://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/"
DRIVE_ID = "1NGlXT9sIaQpyxUoT6MLKm1Pr6x8oxOvc"
DRIVE_URL = f"https://drive.google.com/file/d/{DRIVE_ID}/view"
PREFIXES = {"Cr": "crazing", "In": "inclusion", "Pa": "patches",
            "PS": "pitted_surface", "RS": "rolled_in_scale", "Sc": "scratches"}
MANUAL_HELP = f"""Manual download:
1. Open {OFFICIAL_PAGE}
2. Select Google-Drive or Baidu-Pan on the NEU-CLS row (not NEU-DET or NEU-CLS-64).
3. Save NEU-CLS.rar to data/raw/NEU-CLS.rar. Direct official link: {DRIVE_URL}
4. Run .\\.venv\\Scripts\\python.exe src/download_data.py --archive data/raw/NEU-CLS.rar
5. If tar cannot extract RAR, extract it using 7-Zip and run the same script with
   --extracted-dir PATH_TO_THE_EXTRACTED_NEU_CLS_FOLDER instead.
6. Run .\\.venv\\Scripts\\python.exe src/split_data.py
"""


class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action = None
        self.params = {}
        self.active = False

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "form" and attributes.get("id") == "download-form":
            self.action = attributes.get("action")
            self.active = True
        if tag == "input" and self.active and attributes.get("name"):
            self.params[attributes["name"]] = attributes.get("value", "")

    def handle_endtag(self, tag):
        if tag == "form":
            self.active = False


def download_archive(destination: Path) -> None:
    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise DatasetError(f"Archive must be a regular file: {destination}")
        print(f"Using existing archive: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix + ".part")
    if part.exists() or part.is_symlink():
        raise DatasetError(f"Remove or inspect the previous partial download before retrying: {part}")
    try:
        with requests.Session() as session:
            response = session.get("https://drive.google.com/uc",
                                   params={"export": "download", "id": DRIVE_ID},
                                   stream=True, timeout=(15, 60))
            with response:
                response.raise_for_status()
                if "text/html" in response.headers.get("Content-Type", ""):
                    form = DownloadForm()
                    form.feed(response.text)
                    if (form.action != "https://drive.usercontent.google.com/download"
                            or form.params.get("id") != DRIVE_ID):
                        raise DatasetError("Google Drive did not provide a public download confirmation")
                    response = session.get(form.action, params=form.params,
                                           stream=True, timeout=(15, 60))
                with response:
                    response.raise_for_status()
                    if "text/html" in response.headers.get("Content-Type", ""):
                        raise DatasetError("Google Drive returned an HTML error or login page")
                    with part.open("xb") as output:
                        for chunk in response.iter_content(1024 * 1024):
                            output.write(chunk)
        if not part.read_bytes().startswith(b"Rar!\x1a\x07"):
            raise DatasetError("Download is not a RAR archive")
        part.rename(destination)
        print(f"Downloaded {destination.stat().st_size:,} bytes from the official NEU-CLS link")
    finally:
        if part.exists():
            part.unlink()


def validate_archive_members(members: list[str]) -> None:
    images = set()
    for member in members:
        path = PurePosixPath(member.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts or ":" in member:
            raise DatasetError(f"Unsafe archive path: {member}")
        if path.as_posix().rstrip("/") == "NEU-CLS":
            continue
        if path.as_posix() == "NEU-CLS/Thumbs.db":
            continue
        if len(path.parts) != 2 or path.parts[0] != "NEU-CLS":
            raise DatasetError(f"Unexpected archive layout; expected official NEU-CLS: {member}")
        match = re.fullmatch(r"(Cr|In|Pa|PS|RS|Sc)_([1-9][0-9]*)\.bmp", path.name)
        if not match or not 1 <= int(match.group(2)) <= 300 or path.name in images:
            raise DatasetError(f"Unexpected or duplicate archive image: {member}")
        images.add(path.name)
    if len(images) != 1800:
        raise DatasetError(f"Expected 1800 original BMP images in the archive, found {len(images)}")


def organize_images(source: Path, raw: Path) -> dict:
    if source.is_symlink() or raw.is_symlink():
        raise DatasetError("Dataset roots cannot be symlinks")
    with tempfile.TemporaryDirectory(prefix=".neu-organize-", dir=raw.parent) as temporary:
        stage = Path(temporary)
        for class_name in CLASSES:
            (stage / class_name).mkdir()
        for path in sorted(source.iterdir()):
            if path.name == "Thumbs.db" and path.is_file() and not path.is_symlink():
                continue
            match = re.fullmatch(r"(Cr|In|Pa|PS|RS|Sc)_([1-9][0-9]*)\.bmp", path.name)
            if not match or not 1 <= int(match.group(2)) <= 300 or path.is_symlink() or not path.is_file():
                raise DatasetError(f"Unexpected original NEU-CLS file: {path}")
            shutil.copy2(path, stage / PREFIXES[match.group(1)] / path.name)
        records, duplicates = inspect_raw(stage)
        existing = [class_name for class_name in CLASSES if (raw / class_name).exists()]
        if existing:
            current, _ = inspect_raw(raw)
            if current != records:
                raise DatasetError("Existing raw images differ; refusing to overwrite originals")
            action = "verified_existing"
        else:
            moved = []
            try:
                for class_name in CLASSES:
                    (stage / class_name).rename(raw / class_name)
                    moved.append(class_name)
            except BaseException:
                for class_name in moved:
                    (raw / class_name).rename(stage / class_name)
                raise
            action = "organized"
        return {"dataset": "NEU-CLS", "official_page": OFFICIAL_PAGE,
                "official_download": DRIVE_URL, "action": action, "image_count": len(records),
                "classes": list(CLASSES), "byte_duplicate_groups": duplicates["sha256"],
                "pixel_duplicate_groups": duplicates["pixel_sha256"]}


def prepare_raw(raw: Path, archive: Path | None = None, extracted: Path | None = None) -> dict:
    if raw.is_symlink():
        raise DatasetError("Raw root cannot be a symlink")
    raw.mkdir(parents=True, exist_ok=True)
    if extracted is not None:
        report = organize_images(extracted, raw)
        report["source_method"] = "manual_extraction"
    else:
        archive = archive or raw / "NEU-CLS.rar"
        download_archive(archive)
        if not archive.read_bytes().startswith(b"Rar!\x1a\x07"):
            raise DatasetError("Expected the official NEU-CLS RAR archive")
        tar = shutil.which("tar")
        if tar is None:
            raise DatasetError("RAR extraction requires tar or manual extraction with 7-Zip")
        listed = subprocess.run([tar, "-tf", str(archive.resolve())], capture_output=True, text=True, check=True)
        validate_archive_members(listed.stdout.splitlines())
        details = subprocess.run([tar, "-tvf", str(archive.resolve())], capture_output=True, text=True, check=True)
        if any(line and line[0] not in "-d" for line in details.stdout.splitlines()):
            raise DatasetError("Archive contains unsupported special files or links")
        with tempfile.TemporaryDirectory(prefix=".neu-extract-", dir=raw.parent) as temporary:
            subprocess.run([tar, "-xf", str(archive.resolve()), "-C", temporary],
                           capture_output=True, text=True, check=True)
            report = organize_images(Path(temporary) / "NEU-CLS", raw)
        report["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
        report["archive_bytes"] = archive.stat().st_size
        report["source_method"] = "official_archive"
    (raw / "dataset_source.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, epilog=MANUAL_HELP,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--archive", type=Path, help="Use a manually downloaded official NEU-CLS.rar")
    source.add_argument("--extracted-dir", type=Path, help="Folder containing the original Cr_1.bmp, etc.")
    args = parser.parse_args()
    if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        parser.error("Use the project's .venv Python interpreter")
    try:
        report = prepare_raw(ROOT / "data/raw", args.archive, args.extracted_dir)
    except (DatasetError, OSError, requests.RequestException, subprocess.CalledProcessError) as exc:
        print(f"NEU-CLS download/organization failed: {exc}\n\n{MANUAL_HELP}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
