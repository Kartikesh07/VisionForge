"""Deployment artifacts and checksum-pinned checkpoint delivery."""

import hashlib
import json
from pathlib import Path
import sys

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deployment.start import ROOT, ensure_checkpoint, verify_checkpoint


@pytest.fixture
def manifest():
    return {"bytes": 5, "sha256": hashlib.sha256(b"model").hexdigest()}


class Response:
    url = "https://example.com/model.pth"

    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def raise_for_status(self):
        pass

    def iter_content(self, **kwargs):
        yield self.data


def test_real_manifest_and_report_match_preserved_artifacts():
    manifest = json.loads((ROOT / "deployment/model_manifest.json").read_text())
    assert verify_checkpoint(ROOT / "artifacts/models" / manifest["filename"], manifest)
    original = json.loads((ROOT / "artifacts/reports/final_evaluation.json").read_text())
    for name, model in original["models"].items():
        model["checkpoint"] = "artifacts/models/" + ("baseline_cnn.pth" if name == "custom_cnn" else "resnet18_best.pth")
    assert json.loads((ROOT / "deployment/evaluation.json").read_text()) == original


def test_local_checkpoint_never_requires_network(tmp_path, manifest, monkeypatch):
    path = tmp_path / "model.pth"
    path.write_bytes(b"model")
    monkeypatch.setattr(requests, "get", lambda *a, **kw: pytest.fail("Unexpected network"))
    assert ensure_checkpoint(path, manifest) == path


def test_existing_mismatch_preserved(tmp_path, manifest):
    path = tmp_path / "model.pth"
    path.write_bytes(b"other")
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        ensure_checkpoint(path, manifest, Response.url)
    assert path.read_bytes() == b"other"


@pytest.mark.parametrize("url", [None, "http://example.com/model.pth", "file:///tmp/model.pth"])
def test_missing_checkpoint_requires_https(tmp_path, manifest, url):
    with pytest.raises(RuntimeError, match="HTTPS"):
        ensure_checkpoint(tmp_path / "model.pth", manifest, url)


def test_download_verified_and_temporary_removed(tmp_path, manifest, monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **kw: Response(b"model"))
    path = tmp_path / "model.pth"
    assert ensure_checkpoint(path, manifest, Response.url) == path
    assert path.read_bytes() == b"model"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("data", [b"other", b"oversized"])
def test_invalid_download_never_published(tmp_path, manifest, monkeypatch, data):
    monkeypatch.setattr(requests, "get", lambda *a, **kw: Response(data))
    with pytest.raises(RuntimeError):
        ensure_checkpoint(tmp_path / "model.pth", manifest, Response.url)
    assert list(tmp_path.iterdir()) == []


def test_connection_failure_removes_temporary(tmp_path, manifest, monkeypatch):
    def fail(*args, **kwargs):
        raise requests.ConnectionError("Unavailable")
    monkeypatch.setattr(requests, "get", fail)
    with pytest.raises(requests.ConnectionError):
        ensure_checkpoint(tmp_path / "model.pth", manifest, Response.url)
    assert list(tmp_path.iterdir()) == []


def test_insecure_redirect_rejected(tmp_path, manifest, monkeypatch):
    response = Response(b"model")
    response.url = "http://example.com/model.pth"
    monkeypatch.setattr(requests, "get", lambda *a, **kw: response)
    with pytest.raises(RuntimeError, match="remain HTTPS"):
        ensure_checkpoint(tmp_path / "model.pth", manifest, Response.url)
    assert list(tmp_path.iterdir()) == []
