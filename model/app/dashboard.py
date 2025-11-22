# model/app/dashboard.py
"""
Premium MediGuard AI — Clinician Dashboard (FULL VERSION with auto-PDF)
"""

import sys, os, json, hashlib
from datetime import datetime
import tempfile

# ---------- ensure project root & utils folder are importable ----------
CURRENT = os.path.dirname(os.path.abspath(__file__))          # model/app
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT, "..", ".."))
UTILS_PATH = os.path.join(PROJECT_ROOT, "utils")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if UTILS_PATH not in sys.path:
    sys.path.insert(0, UTILS_PATH)

# ---------- external libs ----------
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# backend imports
from utils.scaling_bridge import scale_input
from utils.pdf_report import generate_pdf_report   # <-- NEW PDF SUPPORT

# optional blockchain
try:
    from utils.blockchain import Blockchain
except:
    Blockchain = None

# ---------- files ----------
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "xgb_model.pkl")
LABEL_ENCODER_PATH = os.path.join(PROJECT_ROOT, "model", "label_encoder.pkl")
MEDICAL_RANGES_PATH = os.path.join(PROJECT_ROOT, "utils", "medical_ranges.json")

# ---------- UI CONFIG ----------
st.set_page_config(page_title="MediGuard AI — Clinician Dashboard", layout="wide")

st.markdown("""
<style>

tbody tr td {
    color: #1b2a49 !important;
    font-weight: 500 !important;
}

thead tr th {
    color: #0b3d91 !important;
    font-weight: 700 !important;
}

body, .block-container {
    background-color: #f5faff !important;
    color: #1b2a49 !important;
}

.block-container {
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
}

h1, h2, h3, h4 {
    color: #0b3d91 !important;
    font-weight: 700 !important;
}

.card {
    background: #ffffff;
    border-radius: 16px;
    padding: 18px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.08);
}

.stButton>button {
    background-color: #0b5cff !important;
    color: white !important;
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)
st.set_page_config(page_title="MediGuard AI — Clinician Dashboard", layout="wide")

st.markdown("""
<style>
    ... YOUR CSS ...
</style>
""", unsafe_allow_html=True)

# ⭐⭐⭐ INSERT HEADER HERE ⭐⭐⭐
# ---------- HEADER SPACING + STYLING ----------
st.markdown(
    """
    <style>
        .header-space {
            padding-top: 5px;      /* distance from top */
            padding-bottom: 10px;   /* space below header */
            background: #f3f8ff;    /* soft medical blue */
            text-align: left;       /* align like old UI */
            margin-bottom: 10px;
        }
        .big-header {
            font-size: 28px;
            font-weight: 700;
            color: #0b3d91;
            margin-bottom: 4px;
        }
        .muted-text {
            font-size: 15px;
            color: #5a6b7a;
            margin-top: -4px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='header-space'>", unsafe_allow_html=True)
st.markdown("<div class='big-header'>MediGuard AI — Clinician Triage Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='muted-text'>Enter raw lab values (real units) and press <b>Predict</b>. This demo logs predictions to a local blockchain for audit.</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)




# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model(path=MODEL_PATH):
    return joblib.load(path)

@st.cache_resource
def load_label_encoder(path=LABEL_ENCODER_PATH):
    try:
        return joblib.load(path)
    except:
        return None


# ---------- PREDICTOR WRAPPER ----------
def predict_with_model(model, X_df, label_encoder=None):
    """
    Returns:
    {
      "prediction": str,
      "probabilities": {label: prob},
      "issues": [],
      "top_contributing_features": [(feat, importance), ...]
    }
    """
    out = {"prediction": None, "probabilities": {}, "issues": [], "top_contributing_features": []}

    # --- MODEL PRED ---
    try:
        proba = model.predict_proba(X_df)[0].tolist()
        pred_idx = int(model.predict(X_df)[0])
    except:
        proba = []
        pred_idx = int(model.predict(X_df)[0])

    # --- LABELS ---
    if label_encoder:
        try:
            class_names = list(label_encoder.classes_)
            pred_label = label_encoder.inverse_transform([pred_idx])[0]
        except:
            class_names = [str(i) for i in range(len(proba))]
            pred_label = str(pred_idx)
    else:
        class_names = [str(i) for i in range(len(proba))]
        pred_label = str(pred_idx)

    # probabilities dict
    if proba and len(proba) == len(class_names):
        out["probabilities"] = {class_names[i]: float(proba[i]) for i in range(len(proba))}

    out["prediction"] = pred_label

    # feature importance
    if hasattr(model, "feature_importances_"):
        feat = list(X_df.columns)
        imp = model.feature_importances_
        pairs = sorted(zip(feat, imp), key=lambda x: x[1], reverse=True)[:10]
        out["top_contributing_features"] = [(f, float(v)) for f, v in pairs]

    return out
# ---------- PART 2: medical ranges, defaults, presets, input UI ----------

# ---------- load medical ranges to get feature order ----------
try:
    with open(MEDICAL_RANGES_PATH, "r") as f:
        medical_ranges = json.load(f)
    FEATURE_ORDER = list(medical_ranges.keys())
except Exception as e:
    st.error(f"Could not load medical ranges: {e}")
    FEATURE_ORDER = []

# sensible defaults (raw units)
DEFAULTS = {f: 0.0 for f in FEATURE_ORDER}
if "Glucose" in DEFAULTS: DEFAULTS["Glucose"] = 100.0
if "Hemoglobin" in DEFAULTS: DEFAULTS["Hemoglobin"] = 14.5
if "Platelets" in DEFAULTS: DEFAULTS["Platelets"] = 250000.0
if "HbA1c" in DEFAULTS: DEFAULTS["HbA1c"] = 5.3

# session state for inputs
if "inputs" not in st.session_state:
    st.session_state["inputs"] = DEFAULTS.copy()

# quick clinical presets
HEALTHY_SAMPLE = {
    "Glucose": 95, "Cholesterol": 170, "Hemoglobin": 14.2, "Platelets": 260000,
    "White Blood Cells": 6500, "Red Blood Cells": 4.7, "Hematocrit": 44,
    "Mean Corpuscular Volume": 90, "Mean Corpuscular Hemoglobin": 30,
    "Mean Corpuscular Hemoglobin Concentration": 33, "Insulin": 10, "BMI": 23,
    "Systolic Blood Pressure": 118, "Diastolic Blood Pressure": 75,
    "Triglycerides": 100, "HbA1c": 5.3, "LDL Cholesterol": 110, "HDL Cholesterol": 50,
    "ALT": 20, "AST": 22, "Heart Rate": 78, "Creatinine": 0.9, "Troponin": 0.01,
    "C-reactive Protein": 0.6
}

DIABETES_SAMPLE = {
    "Glucose": 180, "HbA1c": 8.5, "Insulin": 30, "BMI": 29, "Triglycerides": 200,
    "LDL Cholesterol": 140, "HDL Cholesterol": 38, "Cholesterol": 220,
    "Hemoglobin": 14, "Heart Rate": 85, "Creatinine": 1.1
}

HEART_SAMPLE = {
    "Troponin": 1.2, "ALT": 45, "AST": 80, "Heart Rate": 120, "Creatinine": 1.6,
    "C-reactive Protein": 12, "Systolic Blood Pressure": 160, "Diastolic Blood Pressure": 100,
    "LDL Cholesterol": 160, "HDL Cholesterol": 35, "Triglycerides": 240, "Cholesterol": 240,
    "Glucose": 100, "Hemoglobin": 14.2, "Platelets": 300000, "White Blood Cells": 12000,
    "Red Blood Cells": 4.9, "Hematocrit": 47, "Mean Corpuscular Volume": 89,
    "Mean Corpuscular Hemoglobin": 30, "Mean Corpuscular Hemoglobin Concentration": 33,
    "Insulin": 15, "BMI": 28, "HbA1c": 4
}

# Sidebar: presets + model check
with st.sidebar:
    st.header("Controls")
    st.markdown("**Model & Data**")
    try:
        _ = load_model()
        st.success("Model loaded ✓")
    except Exception as e:
        st.error(f"Could not load model: {e}")

    st.markdown("---")
    st.markdown("**Quick presets**")
    if st.button("Load Healthy sample"):
        for k, v in HEALTHY_SAMPLE.items():
            if k in st.session_state["inputs"]:
                st.session_state["inputs"][k] = v
    if st.button("Load Diabetes sample"):
        for k, v in DIABETES_SAMPLE.items():
            if k in st.session_state["inputs"]:
                st.session_state["inputs"][k] = v
    if st.button("Load Heart-like sample"):
        for k, v in HEART_SAMPLE.items():
            if k in st.session_state["inputs"]:
                st.session_state["inputs"][k] = v

    st.markdown("---")
    st.caption("Tip: use presets, then tweak numeric fields before Predict")


# ---------- layout: left inputs, right outputs ----------
col_left, col_right = st.columns([1.0, 1.1], gap="large")

with col_left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Patient Information")
    patient_id = st.text_input("Patient ID (anonymized)", value=st.session_state.get("patient_id", "patient_001"))
    st.session_state["patient_id"] = patient_id

    st.markdown("**Enter raw lab values (real units)**")
    st.write("")

    # split features into three groups for nicer layout
    n = len(FEATURE_ORDER)
    group_a = FEATURE_ORDER[: n//3]
    group_b = FEATURE_ORDER[n//3 : 2*(n//3)]
    group_c = FEATURE_ORDER[2*(n//3) :]

    def render_group(features):
        for feat in features:
            default = st.session_state["inputs"].get(feat, DEFAULTS.get(feat, 0.0))
            # choose step and format heuristics
            if "Platelet" in feat or "Platelets" in feat:
                step = 100.0; fmt = "%.0f"
            elif "HbA1c" in feat or "Troponin" in feat:
                step = 0.01; fmt = "%.3f"
            elif "Rate" in feat or "Heart" in feat:
                step = 1.0; fmt = "%.0f"
            else:
                step = 0.1; fmt = "%.2f"

            val = st.number_input(label=feat, value=float(default), step=float(step), format=fmt)
            st.session_state["inputs"][feat] = float(val)

    st.markdown("**Basic labs / CBC**")
    render_group(group_a)
    st.markdown("**Red-cell indices / hormones**")
    render_group(group_b)
    st.markdown("**Lipids / Enzymes / Vitals**")
    render_group(group_c)
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    predict_btn = st.button("🔮 Predict", type="primary")
# ---------- PART 3: Prediction handling, output UI, PDF report, feature charts ----------

# Load cached model + encoder
try:
    model = load_model()
except Exception as e:
    st.error(f"Model load error: {e}")
    st.stop()

le = load_label_encoder()

# Prepare right column containers
with col_right:
    st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

    mcol1, mcol2, mcol3 = st.columns([1, 1, 1])
    pred_card = mcol1.empty()
    conf_card = mcol2.empty()
    issues_card = mcol3.empty()

    result_banner = st.container()
    probs_box = st.container()
    warns_box = st.container()
    contrib_box = st.container()
    pdf_box = st.container()   # NEW — PDF report box


# ----------------- RUN PREDICTION --------------------
if predict_btn:

    # raw inputs in correct order
    raw_inputs = {f: float(st.session_state["inputs"].get(f, 0.0)) for f in FEATURE_ORDER}

    # scale inputs using v2 with alerts
    try:
        scaled, warnings, alerts = scale_input(raw_inputs)
    except Exception as e:
        st.error(f"Scaling failed: {e}")
        st.stop()

    # model-ready dataframe
    X = pd.DataFrame([scaled], columns=FEATURE_ORDER).fillna(0.5)

    # model prediction
    try:
        out = predict_with_model(model, X, label_encoder=le)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # combine warnings + alerts into issues list
    issues_list = []
    if warnings:
        for w in warnings:
            issues_list.append({"issue": str(w)})
    if alerts:
        for a in alerts:
            issues_list.append({"issue": str(a)})

    pred_label = out.get("prediction", "N/A")
    probs = out.get("probabilities", {})
    top_features = out.get("top_contributing_features", [])

    confidence = max(probs.values()) if probs else 0.0


    # ------------ Metric Cards ------------
    def display_metric(container, label, value, color="#0b5cff"):
        container.markdown(f"""
            <div class='metric-card' style="
                background:#ffffff;
                padding:14px;
                border-radius:12px;
                box-shadow:0px 3px 12px rgba(0,0,0,0.08);
                margin-bottom:12px;
                border-left:5px solid {color};
            ">
                <div style="font-size:13px; color:#1b2a49; opacity:0.75;">{label}</div>
                <div style="font-size:24px; font-weight:700; color:{color}; margin-top:4px;">{value}</div>
            </div>
        """, unsafe_allow_html=True)

    display_metric(pred_card, "Prediction", f"{pred_label}", "#0b5cff")
    display_metric(conf_card, "Confidence", f"{confidence*100:.1f}%", "#198754")
    display_metric(issues_card, "Input issues", str(len(issues_list)), "#dc3545" if issues_list else "#6c757d")

    # ------------ Banner ------------
    color_map = {
        "Healthy": "#198754",
        "Diabetes": "#d63384",
        "Anemia": "#fd7e14",
        "Thalasse": "#0d6efd",
        "Thromboc": "#6f42c1",
    }
    banner_color = color_map.get(pred_label, "#0b5cff")

    with result_banner:
        st.markdown(
            f"""
            <div class='card pred-banner'>
                <h3 style='color:{banner_color}; margin:0px 0px 6px 0px;'>
                    Prediction: <b>{pred_label}</b>
                </h3>
                <div class='small'>Confidence: <b>{confidence*100:.1f}%</b></div>
                <div class='small' style='margin-top:8px;'>
                    Predicted at: {datetime.utcnow().isoformat(timespec='seconds')} UTC
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ------------ Probability table ------------
    with probs_box:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Class probabilities")
        probs_df = (
            pd.DataFrame(list(probs.items()), columns=["Class", "Probability"])
            .sort_values("Probability", ascending=False)
        )
        st.table(probs_df.style.format({"Probability": "{:.3f}"}))
        st.markdown("</div>", unsafe_allow_html=True)

    # ------------ Issues table ------------
    with warns_box:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if issues_list:
            st.subheader("Input warnings / alerts")
            st.table(pd.DataFrame(issues_list))
        else:
            st.success("No data-quality issues detected.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ------------ Feature Importance Chart ------------
    with contrib_box:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Top contributing features")

        if top_features:
            top_df = pd.DataFrame(top_features, columns=["feature", "global_importance"])
            import plotly.express as px
            
            fig = px.pie(
                top_df,
                names="feature",
                values="global_importance",
                hole=0.55,
                color_discrete_sequence=px.colors.sequential.Blues_r
            )
            st.plotly_chart(fig, use_container_width=True)
            st.table(top_df.style.format({"global_importance": "{:.4f}"}))
        else:
            st.write("Model does not expose feature importance.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ------------ PDF Report (AUTO-GENERATED) ------------
    with pdf_box:
        from utils.pdf_report import generate_pdf_report
        import tempfile

        pdf_filename = f"{patient_id}_report.pdf"

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            pdf_path = tmp.name
            generate_pdf_report(
                pdf_path,
                patient_id,
                raw_inputs,
                scaled,
                pred_label,
                probs
            )

            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()

        st.download_button(
            label="📄 Download Medical PDF Report",
            data=pdf_bytes,
            file_name=pdf_filename,
            mime="application/pdf"
        )
# ---------- PART 4: Footer, spacing & end-of-page ----------

st.markdown("---")
st.caption(
    """
    **MediGuard AI — Clinical Decision Support Prototype**

    • Not a medical device •  
    • For educational and research use only •  
    • Add authentication, HTTPS, encryption & audit logging for production deployments •
    """
)

st.markdown(
    """
    <div style='text-align:center; font-size:12px; color:#6c757d; margin-top:10px;'>
        Built with ❤️ using Streamlit, XGBoost, ReportLab & Plotly • 2025
    </div>
    """,
    unsafe_allow_html=True,
)
