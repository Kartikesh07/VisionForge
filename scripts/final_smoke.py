"""Exercise real dashboard inference and explanations with CUDA hidden."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["VISIONFORGE_DEVICE"] = "cpu"

from io import BytesIO
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import streamlit as st
from streamlit.testing.v1 import AppTest
import torch
import app


class Upload(BytesIO):
    name = "inspection.png"


def inspection():
    from app import render_inspection
    render_inspection()


def main():
    assert not torch.cuda.is_available(), "CPU smoke must run without CUDA"
    torch.set_num_threads(2)
    started = time.perf_counter()
    original_main = sys.modules["__main__"]
    results = {}
    image_path = ROOT / "data/processed/val/inclusion/In_1.bmp"
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    try:
        for extension, format_name in [("png", "PNG"), ("jpg", "JPEG"), ("jpeg", "JPEG"), ("bmp", "BMP")]:
            encoded = BytesIO()
            image.save(encoded, format=format_name)
            data = encoded.getvalue()
            uploaded = Upload(data)
            uploaded.name = f"inspection.{extension}"
            with patch.object(st, "file_uploader", return_value=uploaded):
                at = AppTest.from_function(inspection, default_timeout=60).run()
                assert not at.exception and not at.button[1].disabled
                at.button[1].click().run()
                assert not at.exception
                prediction = at.session_state["prediction"]
                assert prediction == app.get_pipeline().predict(data).to_dict()
                assert len(app.probability_frame(prediction)) == 6
                charts = [element for element in at if "vega" in element.type]
                assert len(charts) == 1, [(element.type) for element in at]
                assert prediction["model_info"]["device"] == "cpu"
                at.button[2].click().run()
                assert not at.exception
                explanation = at.session_state["explanation"]
                assert explanation.heatmap.shape == (200, 200)
                assert np.isfinite(explanation.heatmap).all()
                assert 0 <= explanation.heatmap.min() <= explanation.heatmap.max() <= 1
                assert explanation.overlay_rgb.shape == (200, 200, 3)
                assert explanation.predicted_class == prediction["predicted_class"]
                results[extension] = {"prediction": prediction, "gradcam_shape": list(explanation.heatmap.shape), "chart_rendered": True}
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        assert not at.exception
        at.sidebar.radio[0].set_value("Model Performance").run()
        assert not at.exception and len(at.dataframe) == 2
        metrics = {metric.label: metric.value for metric in at.metric}
        output = {"cuda_available": torch.cuda.is_available(), "formats": results,
                  "performance_metrics": metrics, "duration_seconds": time.perf_counter() - started,
                  "method": "Streamlit AppTest with uploaded bytes; browser upload/visual inspection remains manual"}
        destination = ROOT / "artifacts/reports/final/cpu_dashboard_smoke.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(json.dumps({"status": "passed", "formats": list(results), "cuda_available": False,
                          "metrics": metrics, "duration_seconds": output["duration_seconds"]}, indent=2))
    finally:
        sys.modules["__main__"] = original_main


if __name__ == "__main__":
    main()
