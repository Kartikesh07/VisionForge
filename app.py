"""VisionForge Streamlit dashboard; models, datasets and reports are read-only."""

from pathlib import Path
import hashlib
import json
import logging
import os

import altair as alt
import pandas as pd
import streamlit as st

from src.inference import InferencePipeline, InvalidImageError, ModelLoadError, decode_image
from src.split_data import CLASSES, ROOT

CHECKPOINT = ROOT / "artifacts/models/resnet18_best.pth"
REPORT = ROOT / "artifacts/reports/final_evaluation.json"
if not REPORT.is_file():
    REPORT = ROOT / "deployment/evaluation.json"
LABELS = {name: name.replace("_", " ").title() for name in CLASSES}
NOTICE = "This system classifies six known defect types. It does not identify defect-free surfaces. Confidence scores are uncalibrated."
LOGGER = logging.getLogger(__name__)


@st.cache_resource(show_spinner="Loading the saved ResNet18 model…", max_entries=1)
def get_pipeline() -> InferencePipeline:
    return InferencePipeline(CHECKPOINT, device=os.environ.get("VISIONFORGE_DEVICE", "auto"))


def load_evaluation(path: Path = REPORT) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "completed" or not report.get("test_inference_executed"):
        raise ValueError("A completed final evaluation report is required")
    for name in ("custom_cnn", "resnet18"):
        model = report["models"][name]
        for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
            if not 0 <= model[key] <= 1:
                raise ValueError("Evaluation report contains invalid metrics")
        if model["class_order"] != list(CLASSES):
            raise ValueError("Evaluation class order differs from the model")
    return report


def probability_frame(prediction: dict) -> pd.DataFrame:
    scores = prediction["probabilities"]
    if set(scores) != set(CLASSES):
        raise ValueError("Prediction must contain all six class probabilities")
    return pd.DataFrame([{"Defect": LABELS[name], "Probability": scores[name],
                          "Predicted": name == prediction["predicted_class"]} for name in CLASSES])


def probability_chart(frame: pd.DataFrame):
    return alt.Chart(frame).mark_bar(cornerRadiusEnd=5, height=22).encode(
        x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", title=None)),
        y=alt.Y("Defect:N", sort="-x", axis=alt.Axis(title=None, labelLimit=180)),
        color=alt.condition(alt.datum.Predicted, alt.value("#59ddc4"), alt.value("#36506c")),
        tooltip=["Defect:N", alt.Tooltip("Probability:Q", format=".2%")],
    ).properties(height=270).configure_view(strokeWidth=0).configure_axis(
        labelColor="#b6c4d6", gridColor="#243146", domain=False, tickColor="#243146", labelFontSize=12)


def clear_input():
    for key in ("sample_data", "sample_name", "input_digest", "prediction", "explanation", "inspection_error"):
        st.session_state.pop(key, None)


def sync_input(data: bytes | None):
    digest = hashlib.sha256(data).hexdigest() if data is not None else None
    if st.session_state.get("input_digest") != digest:
        for key in ("prediction", "explanation", "inspection_error"):
            st.session_state.pop(key, None)
        st.session_state["input_digest"] = digest


def analyze(data: bytes) -> bool:
    st.session_state.pop("explanation", None)
    st.session_state.pop("prediction", None)
    try:
        with st.spinner("Inspecting the surface…"):
            st.session_state["prediction"] = get_pipeline().predict(data).to_dict()
        st.session_state.pop("inspection_error", None)
        return True
    except (InvalidImageError, ModelLoadError, RuntimeError, ValueError, OSError) as exc:
        LOGGER.exception("Surface classification failed")
        st.session_state["inspection_error"] = f"Unable to inspect this image. {exc}"
        return False


def generate_explanation(data: bytes) -> bool:
    try:
        with st.spinner("Computing Grad-CAM with gradients enabled…"):
            st.session_state["explanation"] = get_pipeline().explain(data)
        st.session_state.pop("inspection_error", None)
        return True
    except (InvalidImageError, ModelLoadError, RuntimeError, ValueError, OSError) as exc:
        LOGGER.exception("Grad-CAM failed")
        st.session_state.pop("explanation", None)
        st.session_state["inspection_error"] = f"Unable to generate Grad-CAM. {exc}"
        return False


def render_inspection():
    st.caption("INSPECTION / SINGLE IMAGE")
    st.title("Understand every surface.")
    st.write("Inspect a steel image, compare defect scores, and see what influenced the model.")
    left, right = st.columns([1, 1.15], gap="large")
    with left, st.container(border=True):
        st.subheader("Surface image")
        uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp"],
                                    max_upload_size=20, key="surface_upload", on_change=clear_input,
                                    help="PNG, JPG, JPEG or BMP · up to 20 MiB. Images stay in session memory.")
        with st.expander("Or explore a validation sample"):
            category = st.selectbox("Sample defect category", CLASSES, format_func=LABELS.get)
            sample_paths = sorted((ROOT / "data/processed/val" / category).glob("*.bmp"))
            if not sample_paths:
                st.caption("Validation samples are available in the local dataset workspace. Upload an image to inspect it here.")
            if st.button("Load validation sample", disabled=uploaded is not None or not sample_paths, width="stretch"):
                try:
                    path = next(iter(sample_paths))
                    st.session_state["sample_data"] = path.read_bytes()
                    st.session_state["sample_name"] = f"Validation sample · {path.name}"
                except (OSError, StopIteration):
                    st.error("No validation sample is available for this category.")
        data = uploaded.getvalue() if uploaded is not None else st.session_state.get("sample_data")
        filename = uploaded.name if uploaded is not None else st.session_state.get("sample_name", "")
        sync_input(data)
        valid = False
        if data is not None:
            try:
                original = decode_image(data)
                st.image(original, caption=filename, width="stretch")
                st.caption(f"{original.width} × {original.height} pixels · evaluated at 224 × 224")
                valid = True
            except InvalidImageError as exc:
                st.error(f"Invalid image. {exc}")
        else:
            st.markdown('<div class="empty-image"><span>▧</span><strong>Your surface, in focus.</strong><p>Upload a steel image to begin an inspection.</p></div>', unsafe_allow_html=True)
        if st.button("Analyze surface", type="primary", disabled=not valid, width="stretch"):
            analyze(data)
    with right, st.container(border=True):
        st.subheader("Inspection result")
        prediction = st.session_state.get("prediction")
        if prediction:
            first, second = st.columns([1.25, 1])
            first.metric("Predicted defect", LABELS[prediction["predicted_class"]])
            second.metric("Softmax confidence", f"{prediction['confidence']:.2%}")
            st.caption("Scores are uncalibrated; a high score does not guarantee a correct classification.")
            st.altair_chart(probability_chart(probability_frame(prediction)), width="stretch")
            st.caption(f"{prediction['model_info']['device'].upper()} · {prediction['model_info']['precision']} · ResNet18")
            if prediction["model_info"].get("fallback_reason"):
                st.info(f"Running on CPU: {prediction['model_info']['fallback_reason']}")
            if st.button("Generate Grad-CAM", width="stretch"):
                generate_explanation(data)
            st.download_button("Download prediction JSON", json.dumps(prediction, indent=2),
                               file_name="visionforge_prediction.json", mime="application/json", width="stretch")
        else:
            st.markdown('<div class="result-placeholder"><div class="status-dot">READY FOR INSPECTION</div><h3>From texture to insight.</h3><p>Select an image and analyze its surface. Your predicted class and six defect scores will appear here.</p><hr><p>01 &nbsp; Select a surface image<br>02 &nbsp; Classify the defect<br>03 &nbsp; Explore the explanation</p></div>', unsafe_allow_html=True)
    if st.session_state.get("inspection_error"):
        st.error(st.session_state["inspection_error"])
    explanation = st.session_state.get("explanation")
    if explanation is not None:
        st.divider()
        st.subheader("What influenced the prediction?")
        st.caption(f"Grad-CAM target: {LABELS[explanation.predicted_class]} · FP32 softmax confidence {explanation.confidence:.2%}")
        heatmap, overlay = st.columns(2, gap="large")
        with heatmap, st.container(border=True):
            st.image(explanation.heatmap_rgb, caption="Grad-CAM · blue: low influence / red: high influence", width="stretch")
        with overlay, st.container(border=True):
            st.image(explanation.overlay_rgb, caption="Heatmap overlay on the original grayscale surface", width="stretch")
        st.info("Grad-CAM is model interpretability, not precise defect segmentation. It highlights coarse regions influencing the predicted class, not defect boundaries.")
        st.caption("Grad-CAM uses FP32. Its prediction or confidence can differ slightly from the FP16 inspection scores.")


def render_performance(report: dict | None):
    st.caption("MODEL PERFORMANCE / HELD-OUT EVALUATION")
    st.title("Evidence behind the model.")
    st.write("Measured results from the preserved final evaluation. No evaluation or training runs in this dashboard.")
    if report is None:
        st.warning("Evaluation results are unavailable. Restore the completed final evaluation report to display measured metrics.")
        return
    resnet = report["models"]["resnet18"]
    columns = st.columns(4)
    for column, label, value in zip(columns, ("Test accuracy", "Macro-F1", "Test images", "Misclassified"),
                                     (f"{resnet['accuracy']:.2%}", f"{resnet['macro_f1']:.6f}", str(resnet["samples"]), str(resnet["misclassification_count"]))):
        column.metric(label, value, border=True)
    st.subheader("ResNet18 vs. custom CNN")
    comparison = [{"Model": "ResNet18" if name == "resnet18" else "Custom CNN",
                   "Test accuracy": f"{metrics['accuracy']:.2%}", "Macro precision": round(metrics["macro_precision"], 6),
                   "Macro recall": round(metrics["macro_recall"], 6), "Macro-F1": round(metrics["macro_f1"], 6),
                   "Validation accuracy": f"{metrics['validation_accuracy']:.2%}", "Errors": metrics["misclassification_count"]}
                  for name, metrics in report["models"].items()]
    st.dataframe(pd.DataFrame(comparison), hide_index=True, width="stretch")
    st.caption(f"ResNet18 test accuracy is {abs(resnet['accuracy_change_from_validation']) * 100:.2f} percentage points {'below' if resnet['accuracy_change_from_validation'] < 0 else 'above'} validation. These scores describe this fixed held-out split.")
    model_name = st.selectbox("Detailed results", ["resnet18", "custom_cnn"], format_func=lambda name: "ResNet18" if name == "resnet18" else "Custom CNN")
    metrics = report["models"][model_name]
    left, right = st.columns([1.05, 1], gap="large")
    with left, st.container(border=True):
        st.subheader("Confusion matrix")
        matrix = pd.DataFrame(metrics["confusion_matrix"], index=[LABELS[n] for n in CLASSES], columns=[LABELS[n] for n in CLASSES])
        long = matrix.rename_axis("Actual").reset_index().melt(id_vars="Actual", var_name="Predicted", value_name="Images")
        base = alt.Chart(long).encode(x=alt.X("Predicted:N", sort=[LABELS[n] for n in CLASSES], axis=alt.Axis(labelAngle=-35, labelLimit=150)),
                                     y=alt.Y("Actual:N", sort=[LABELS[n] for n in CLASSES]))
        chart = base.mark_rect(cornerRadius=3).encode(color=alt.Color("Images:Q", scale=alt.Scale(range=["#162337", "#59ddc4"]), legend=None), tooltip=["Actual", "Predicted", "Images"])
        labels = base.mark_text(fontSize=12).encode(text="Images:Q", color=alt.condition(alt.datum.Images > matrix.to_numpy().max() / 2, alt.value("#07131c"), alt.value("#d8e5f2")))
        st.altair_chart((chart + labels).properties(height=300).configure_view(strokeWidth=0), width="stretch")
        st.caption("Rows: actual class · columns: predicted class")
    with right, st.container(border=True):
        st.subheader("Per-class metrics")
        rows = [{"Class": LABELS[name], "Precision": round(m["precision"], 4), "Recall": round(m["recall"], 4),
                 "F1": round(m["f1-score"], 4), "Images": int(m["support"])} for name, m in metrics["per_class"].items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Macro averages give equal weight to each of the six classes.")
    with st.expander("Evaluation provenance"):
        st.write(f"Source: {REPORT.relative_to(ROOT).as_posix()}")
        st.write(f"Evaluation device: {report.get('device', 'Not recorded')}")
        st.write(f"Precision: {report.get('precision', 'Not recorded')}")
        st.write(f"Selected epoch: {resnet['selected_epoch']} · phase: {resnet['selected_phase']}")


def main():
    st.set_page_config(page_title="VisionForge | Steel Surface Intelligence", page_icon="◈", layout="wide")
    st.markdown("""<style>
    .stApp {background:#0b111b;color:#e8eef7;}
    [data-testid="stSidebar"] {background:#101927;border-right:1px solid #243146;}
    .block-container {max-width:1400px;padding-top:2.5rem;padding-bottom:2rem;}
    h1 {letter-spacing:-.045em;font-weight:700!important;} h3 {letter-spacing:-.02em;}
    [data-testid="stVerticalBlockBorderWrapper"] {border-color:#243146!important;background:#111b29;border-radius:16px;}
    [data-testid="stMetricValue"] {font-size:1.65rem;letter-spacing:-.03em;}
    [data-testid="stFileUploaderDropzone"] {border:1px dashed #426079;background:#172438;border-radius:12px;}
    .brand {font-size:1.6rem;font-weight:750;letter-spacing:-.04em;color:#ecf4ff;margin-bottom:0;}
    .brand b {color:#59ddc4;} .brand-note {font-size:.75rem;letter-spacing:.14em;color:#8b9db3;margin-bottom:2rem;}
    .empty-image {min-height:230px;display:flex;flex-direction:column;align-items:center;justify-content:center;background:linear-gradient(135deg,#162438,#111b29);border-radius:12px;margin:1rem 0;color:#9cb0c7;text-align:center;}
    .empty-image span {font-size:3rem;color:#59ddc4;} .empty-image strong{color:#dce8f7;margin:.5rem;}
    .result-placeholder {padding:2rem 1rem;min-height:340px;color:#a4b5ca;line-height:2;}
    .status-dot {color:#59ddc4;font-size:.7rem;letter-spacing:.15em;margin-bottom:1rem;}
    .result-placeholder h3 {color:#e5effb;} .footer {font-size:.75rem;color:#859ab4;margin-top:1rem;}
    @media(max-width:700px) {.block-container{padding:1rem;} [data-testid="stMetricValue"]{font-size:1.3rem;} .result-placeholder{min-height:200px;padding:1rem;}}
    </style>""", unsafe_allow_html=True)
    try:
        report = load_evaluation()
    except (OSError, ValueError, KeyError, TypeError):
        report = None
    with st.sidebar:
        st.markdown('<div class="brand"><b>◈</b> VisionForge</div><div class="brand-note">STEEL SURFACE INTELLIGENCE</div>', unsafe_allow_html=True)
        page = st.radio("Workspace", ["Inspection", "Model Performance"], label_visibility="collapsed")
        st.divider()
        st.caption("MODEL")
        st.write("**ResNet18**")
        st.caption("ImageNet transfer learning · six-class classifier")
        if report:
            st.metric("Held-out test accuracy", f"{report['models']['resnet18']['accuracy']:.2%}")
        if st.session_state.get("prediction"):
            info = st.session_state.prediction["model_info"]
            st.caption(f"Active device: {info['device']} · {info['precision']}")
        st.caption("SUPPORTED DEFECTS")
        for name in CLASSES:
            st.write(f"· {LABELS[name]}")
        st.divider()
        st.caption("Six known defects. No defect-free class. Uncalibrated confidence.")
        st.caption("Local inference · uploads are not written to disk")
    if page == "Inspection":
        render_inspection()
    else:
        render_performance(report)
    st.divider()
    st.caption(NOTICE)
    with st.expander("Model information"):
        try:
            st.json(get_pipeline().model_info)
        except (ModelLoadError, RuntimeError, ValueError, OSError) as exc:
            st.error(f"Model unavailable. {exc}")
    st.markdown('<div class="footer">VISIONFORGE &nbsp; / &nbsp; INDUSTRIAL COMPUTER VISION &nbsp; / &nbsp; LOCAL WORKSPACE</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
