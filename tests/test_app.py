"""Streamlit components and real inference/Grad-CAM integration."""

from io import BytesIO
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app
from src.split_data import CLASSES, ROOT


@pytest.fixture(autouse=True)
def preserve_main_module_for_windows_spawn():
    # Streamlit AppTest replaces __main__ with its temporary runner module.
    # Restore pytest's module so later Windows DataLoader workers can spawn.
    original = sys.modules["__main__"]
    yield
    sys.modules["__main__"] = original


@pytest.fixture(scope="module", autouse=True)
def modest_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def inspection_component():
    from app import render_inspection
    render_inspection()


def test_cached_resource_reused():
    first = app.get_pipeline()
    assert app.get_pipeline() is first
    assert first.model_info["architecture"] == "visionforge_resnet18_imagenet_v1"


def test_actual_evaluation_metrics_and_probability_chart():
    report = app.load_evaluation()
    source = json.loads(app.REPORT.read_text())
    assert report == source
    predicted = app.get_pipeline().predict(ROOT / "data/processed/val/inclusion/In_1.bmp").to_dict()
    frame = app.probability_frame(predicted)
    assert len(frame) == 6 and frame["Predicted"].sum() == 1
    assert list(frame["Probability"]) == [predicted["probabilities"][name] for name in CLASSES]
    spec = app.probability_chart(frame).to_dict()
    assert spec["encoding"]["x"]["field"] == "Probability"
    assert spec["encoding"]["y"]["field"] == "Defect"


@pytest.mark.parametrize("change", ["blocked", "mapping", "metric"])
def test_corrupted_evaluation_report_rejected(tmp_path, change):
    report = app.load_evaluation()
    if change == "blocked":
        report["status"] = "blocked"
    elif change == "mapping":
        report["models"]["resnet18"]["class_order"] = []
    else:
        report["models"]["resnet18"]["accuracy"] = float("nan")
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        app.load_evaluation(path)


def test_main_navigation_actual_performance_and_notices():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=45).run()
    assert not at.exception
    assert at.button[1].disabled
    assert any("defect-free" in text.value and "uncalibrated" in text.value for text in at.caption)
    at.sidebar.radio[0].set_value("Model Performance").run()
    assert not at.exception
    measured = app.load_evaluation()["models"]["resnet18"]
    assert next(m.value for m in at.metric if m.label == "Test accuracy") == f"{measured['accuracy']:.2%}"
    assert next(m.value for m in at.metric if m.label == "Misclassified") == str(measured["misclassification_count"])
    assert len(at.dataframe) == 2
    at.selectbox[0].set_value("custom_cnn").run()
    assert not at.exception


def test_real_validation_inspection_and_gradcam():
    at = AppTest.from_function(inspection_component, default_timeout=45).run()
    assert not at.exception
    at.selectbox[0].set_value("inclusion").run()
    at.button[0].click().run()
    assert not at.button[1].disabled
    at.button[1].click().run()
    assert not at.exception
    prediction = at.session_state["prediction"]
    direct = app.get_pipeline().predict(at.session_state["sample_data"]).to_dict()
    assert prediction == direct
    assert len(prediction["probabilities"]) == 6
    at.button[2].click().run()
    assert not at.exception
    explanation = at.session_state["explanation"]
    assert np.isfinite(explanation.heatmap).all()
    assert explanation.heatmap.shape == (200, 200)
    assert not torch.is_inference_mode_enabled()
    assert any("not precise defect segmentation" in element.value for element in at.info)
    at.selectbox[0].set_value("scratches").run()
    at.button[0].click().run()
    assert "prediction" not in at.session_state and "explanation" not in at.session_state


class UploadedImage(BytesIO):
    name = "uploaded.bmp"


def test_upload_bytes_integration(monkeypatch):
    data = (ROOT / "data/processed/val/inclusion/In_1.bmp").read_bytes()
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: UploadedImage(data))
    at = AppTest.from_function(inspection_component, default_timeout=45).run()
    assert not at.exception and not at.button[1].disabled
    at.button[1].click().run()
    assert at.session_state["prediction"] == app.get_pipeline().predict(data).to_dict()
    assert not at.exception


def test_corrupt_upload_disables_analysis(monkeypatch):
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: UploadedImage(b"invalid image"))
    at = AppTest.from_function(inspection_component, default_timeout=45).run()
    assert not at.exception and at.button[1].disabled
    assert any("Invalid image" in error.value for error in at.error)
    assert "prediction" not in at.session_state


def test_pipeline_load_failure_is_actionable(monkeypatch):
    def unavailable():
        raise app.ModelLoadError("Checkpoint unavailable")
    monkeypatch.setattr(app, "get_pipeline", unavailable)
    at = AppTest.from_function(inspection_component, default_timeout=45).run()
    at.button[0].click().run()
    at.button[1].click().run()
    assert not at.exception
    assert any("Checkpoint unavailable" in error.value for error in at.error)
    assert "prediction" not in at.session_state


def test_gradcam_failure_preserves_classification(monkeypatch):
    pipeline = app.get_pipeline()
    def failure(data):
        raise RuntimeError("Grad-CAM unavailable")
    monkeypatch.setattr(pipeline, "explain", failure)
    at = AppTest.from_function(inspection_component, default_timeout=45).run()
    at.button[0].click().run()
    at.button[1].click().run()
    at.button[2].click().run()
    assert not at.exception
    assert "prediction" in at.session_state and "explanation" not in at.session_state
    assert any("Grad-CAM unavailable" in error.value for error in at.error)
