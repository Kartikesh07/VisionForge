"""Verify or download a checksum-pinned checkpoint, then start Streamlit."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parents[1]


def verify_checkpoint(path: Path, manifest: dict) -> bool:
    return (path.is_file() and path.stat().st_size == manifest["bytes"]
            and hashlib.sha256(path.read_bytes()).hexdigest() == manifest["sha256"])


def ensure_checkpoint(path: Path, manifest: dict, url: str | None = None) -> Path:
    """Never replace an existing checkpoint; validate before atomic publication."""
    if path.exists():
        if not verify_checkpoint(path, manifest):
            raise RuntimeError("Existing checkpoint does not match the manifest; refusing to overwrite it")
        return path
    if not url or urlparse(url).scheme != "https":
        raise RuntimeError("Provide the trained checkpoint locally or set VISIONFORGE_CHECKPOINT_URL to an HTTPS download URL")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".checkpoint-", suffix=".partial", dir=path.parent)
    staged = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream, requests.get(url, stream=True, timeout=(10, 60)) as response:
            response.raise_for_status()
            if urlparse(response.url).scheme != "https":
                raise RuntimeError("Checkpoint redirect must remain HTTPS")
            received = 0
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                received += len(chunk)
                if received > manifest["bytes"]:
                    raise RuntimeError("Downloaded checkpoint exceeds expected size")
                stream.write(chunk)
        if not verify_checkpoint(staged, manifest):
            raise RuntimeError("Downloaded checkpoint failed SHA256 or size verification")
        # Hard-link publication is atomic and refuses to replace a concurrent file.
        try:
            os.link(staged, path)
        except FileExistsError:
            if not verify_checkpoint(path, manifest):
                raise RuntimeError("Another checkpoint appeared; refusing to replace it")
        return path
    finally:
        staged.unlink(missing_ok=True)


def main() -> int:
    manifest = json.loads((ROOT / "deployment/model_manifest.json").read_text())
    try:
        ensure_checkpoint(ROOT / "artifacts/models" / manifest["filename"], manifest,
                          os.environ.get("VISIONFORGE_CHECKPOINT_URL"))
    except (OSError, RuntimeError, requests.RequestException) as exc:
        print(f"Startup failed: {exc}", file=sys.stderr)
        return 1
    command = [sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
               "--server.address", "0.0.0.0", "--server.port", "7860", "--server.headless", "true"]
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
