# model/app/dashboard.py
"""
Premium MediGuard UI (Clinician Dashboard)
Drop-in replacement for existing dashboard.py — matches style and layout of app/app.py example.
Assumptions:
 - project root contains `utils/` (scaling_bridge.py, medical_ranges.json, optional blockchain.py)
 - model files at model/xgb_model.pkl and model/label_encoder.pkl (joblib)
 - Python env already active and streamlit installed
Run:
  conda activate envname   # or activate your venv
  streamlit run model/app/dashboard.py
"""
import sys, os, json, hashlib
from datetime import datetime

# ---------- ensure project root & utils folder are importable ----------
CURRENT = os.path.dirname(os.path.abspath(__file__))          # model/app
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT, "..", ".."))
UTILS_PATH = os.path.join(PROJECT_ROOT, "utils")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if UTILS_PATH not in sys.path:     # CRUCIAL
    sys.path.insert(0, UTILS_PATH)

# ---------- libs ----------
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------- backend imports ----------
from utils.scaling_bridge import scale_input

# optional blockchain module under utils (if exists)
try:
    from utils.blockchain import Blockchain  # optional; adapt if your blockchain API differs
except Exception:
    Blockchain = None

# ---------- constants / paths ----------
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "xgb_model.pkl")
LABEL_ENCODER_PATH = os.path.join(PROJECT_ROOT, "model", "label_encoder.pkl")
MEDICAL_RANGES_PATH = os.path.join(PROJECT_ROOT, "utils", "medical_ranges.json")

# ---------- Page styling (hospital white premium) ----------
st.set_page_config(page_title="MediGuard AI — Clinician Dashboard", layout="wide")

st.markdown("""
<style>
/* Fix table cell text visibility */
tbody tr td {
    color: #1b2a49 !important;
    font-weight: 500 !important;
}

/* Fix table header */
thead tr th {
    color: #0b3d91 !important;
    font-weight: 700 !important;
}

/* -------- BASE BACKGROUND (Hospital white-blue) -------- */
body, .block-container {
    background-color: #f5faff !important;        /* very soft blue-white */
    color: #1b2a49 !important;                    /* deep navy readable */
}

/* Remove unnecessary padding */
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
}

/* -------- HEADERS -------- */
h1, h2, h3, h4 {
    color: #0b3d91 !important;                   /* medical navy blue */
    font-weight: 700 !important;
}

/* -------- CARDS -------- */
.card {
    background: #ffffff;
    border-radius: 16px;
    padding: 18px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.08);
    transition: transform 0.1s ease, box-shadow 0.1s ease;
}

.card:hover {
    transform: translateY(-3px);
    box-shadow: 0 4px 16px rgba(0,0,0,0.12);
}

/* -------- PREDICTION BANNER -------- */
.pred-banner {
    background: linear-gradient(135deg, #e0f0ff, #ffffff);
    border-left: 6px solid #0b5cff;
}

/* -------- Sidebar -------- */
.css-1lcbmhc {
    background-color: #e3effa !important;
}

.css-1lcbmhc * {
    color: #1b2a49 !important;
}

/* -------- INPUT LABELS -------- */
.stNumberInput label, .stTextInput label {
    color: #1b2a49 !important;
    font-weight: 600 !important;
}

/* -------- INPUT FIELDS -------- */
.stNumberInput input, .stTextInput input {
    background-color: #ffffff !important;
    color: #1b2a49 !important;
}

/* -------- BUTTONS -------- */
.stButton>button {
    background-color: #0b5cff !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
    border: none !important;
    transition: 0.2s ease-in-out;
}

.stButton>button:hover {
    background-color: #0846a8 !important;
}

/* -------- TABLE TEXT FIX -------- */
table, thead tr th, tbody tr td {
    color: #1b2a49 !important;
}

/* Fix for Streamlit dark defaults */
.css-ffhzg2, .css-81oif8 {
    color: #1b2a49 !important;
}
            /* Better spacing between cards */
.card {
    margin-bottom: 18px !important;
}

/* Gap under metric cards */
.metric-card {
    margin-right: 10px;
}


</style>
""", unsafe_allow_html=True)



st.markdown(
    """
    <style>
      .big-header { font-size:28px; font-weight:700; margin-bottom:6px; }
      .muted { color:#6c757d; font-size:14px; }
      .card { background: #ffffff; border-radius:10px; padding:16px; box-shadow: 0 2px 6px rgba(0,0,0,0.06); }
      .metric { font-size:22px; font-weight:700; }
      .small { font-size:13px; color:#666; }
      .pred-banner { padding:12px; border-radius:8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- load model & label encoder (cached) ----------
@st.cache_resource
def load_model(path=MODEL_PATH):
    return joblib.load(path)

@st.cache_resource
def load_label_encoder(path=LABEL_ENCODER_PATH):
    try:
        return joblib.load(path)
    except Exception:
        return None

# ---------- helper wrappers to standardize predictor output ----------
def predict_with_model(model, X_df, label_encoder=None):
    """
    Standardized output dict:
      {
        "prediction": <label_str_or_idx>,
        "probabilities": {class_label: prob, ...},
        "issues": [],                    # forwarded from scaling if present
        "top_contributing_features": []  # list of (feature, importance) tuples (optional)
      }
    """
    out = {"prediction": None, "probabilities": {}, "issues": [], "top_contributing_features": []}
    try:
        proba = model.predict_proba(X_df)[0].tolist()
        pred_idx = int(model.predict(X_df)[0])
    except Exception:
        # if model doesn't support predict_proba
        proba = []
        pred_idx = int(model.predict(X_df)[0])

    # class labels
    if label_encoder is not None:
        try:
            class_names = list(label_encoder.classes_)
            pred_label = label_encoder.inverse_transform([pred_idx])[0]
        except Exception:
            class_names = [str(i) for i in range(len(proba))]
            pred_label = str(pred_idx)
    else:
        class_names = [str(i) for i in range(len(proba))]
        pred_label = str(pred_idx)

    # build probabilities dict
    if proba and len(proba) == len(class_names):
        out["probabilities"] = {class_names[i]: float(proba[i]) for i in range(len(proba))}
    else:
        # fallback: try to get model.classes_ if present
        try:
            classes = getattr(model, "classes_", None)
            if classes is not None:
                out["probabilities"] = {str(classes[i]): float(proba[i]) for i in range(len(proba))}
        except Exception:
            out["probabilities"] = {}

    out["prediction"] = pred_label

    # try feature importance extraction (XGBoost / sklearn)
    try:
        if hasattr(model, "feature_importances_"):
            importances = getattr(model, "feature_importances_")
            features = list(X_df.columns)
            pairs = list(zip(features, importances))
            pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)[:10]
            out["top_contributing_features"] = [(f, float(v)) for f, v in pairs_sorted]
    except Exception:
        out["top_contributing_features"] = []

    return out

# ---------- UI: Title + sidebar presets ----------
st.markdown("<div class='big-header'>MediGuard AI — Clinician Triage Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='muted'>Enter raw lab values (real units) and press <b>Predict</b>. This demo logs predictions to a local blockchain for audit.</div>", unsafe_allow_html=True)
st.write("")

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
        st.session_state['preset'] = "healthy"
    if st.button("Load Diabetes sample"):
        st.session_state['preset'] = "diabetes"
    if st.button("Load Heart-like sample"):
        st.session_state['preset'] = "heart"
    st.markdown("---")
    st.caption("Theme: Hospital White • Compact clinical layout")

# ---------- load medical ranges to get feature order ----------
try:
    with open(MEDICAL_RANGES_PATH, "r") as f:
        medical_ranges = json.load(f)
    FEATURE_ORDER = list(medical_ranges.keys())
except Exception:
    # fallback: try to infer from scaling_bridge if exported
    try:
        from utils.scaling_bridge import FEATURE_ORDER as FEATURE_ORDER  # type: ignore
    except Exception:
        FEATURE_ORDER = []  # will error later if empty

# set default values (simple realistic defaults)
DEFAULTS = {f: 0.0 for f in FEATURE_ORDER}
# you can customize a few sensible defaults if you like
if "Glucose" in DEFAULTS:
    DEFAULTS["Glucose"] = 100.0
if "Hemoglobin" in DEFAULTS:
    DEFAULTS["Hemoglobin"] = 14.5
if "Platelets" in DEFAULTS:
    DEFAULTS["Platelets"] = 250000.0

# session inputs
if "inputs" not in st.session_state:
    st.session_state["inputs"] = DEFAULTS.copy()

# presets
preset = st.session_state.get("preset", None)

HEALTHY_SAMPLE = {
    "Glucose": 95,
    "Cholesterol": 170,
    "Hemoglobin": 14.2,
    "Platelets": 260000,
    "White Blood Cells": 6500,
    "Red Blood Cells": 4.7,
    "Hematocrit": 44,
    "Mean Corpuscular Volume": 90,
    "Mean Corpuscular Hemoglobin": 30,
    "Mean Corpuscular Hemoglobin Concentration": 33,
    "Insulin": 10,
    "BMI": 23,
    "Systolic Blood Pressure": 118,
    "Diastolic Blood Pressure": 75,
    "Triglycerides": 100,
    "HbA1c": 5.3,
    "LDL Cholesterol": 110,
    "HDL Cholesterol": 50,
    "ALT": 20,
    "AST": 22,
    "Heart Rate": 78,
    "Creatinine": 0.9,
    "Troponin": 0.01,
    "C-reactive Protein": 0.6
}

DIABETES_SAMPLE = {
    "Glucose": 180,
    "HbA1c": 8.5,
    "Insulin": 30,
    "BMI": 29,
    "Triglycerides": 200,
    "LDL Cholesterol": 140,
    "HDL Cholesterol": 38,
    "Cholesterol": 220,
    "Hemoglobin": 14,
    "Heart Rate": 85,
    "Creatinine": 1.1
}

HEART_SAMPLE = {
    "Troponin": 1.2,                    # VERY HIGH (heart damage marker)
    "ALT": 45,                          # mildly elevated due to systemic stress
    "AST": 80,                          # AST often increases in myocardial injury
    "Heart Rate": 120,                  # tachycardia
    "Creatinine": 1.6,                  # possible kidney stress
    "C-reactive Protein": 12,           # high inflammation
    "Systolic Blood Pressure": 160,     # hypertension (common in MI)
    "Diastolic Blood Pressure": 100,    
    "LDL Cholesterol": 160,             # high → major risk factor
    "HDL Cholesterol": 35,              # low HDL → higher risk
    "Triglycerides": 240,               # high TG → heart disease risk
    "Cholesterol": 240,                 # total cholesterol high
    "Glucose": 100,                     # stress hyperglycemia
    "Hemoglobin": 14.2,
    "Platelets": 300000,
    "White Blood Cells": 12000,         # elevated WBC (inflammation)
    "Red Blood Cells": 4.9,
    "Hematocrit": 47,
    "Mean Corpuscular Volume": 89,
    "Mean Corpuscular Hemoglobin": 30,
    "Mean Corpuscular Hemoglobin Concentration": 33,
    "Insulin": 15,
    "BMI": 28,                          # overweight → cardiac risk
    "HbA1c": 4                        # borderline diabetic
}

if preset == "healthy":
    for k,v in HEALTHY_SAMPLE.items():
        if k in st.session_state["inputs"]:
            st.session_state["inputs"][k] = v

elif preset == "diabetes":
    for k,v in DIABETES_SAMPLE.items():
        if k in st.session_state["inputs"]:
            st.session_state["inputs"][k] = v

elif preset == "heart":
    for k,v in HEART_SAMPLE.items():
        if k in st.session_state["inputs"]:
            st.session_state["inputs"][k] = v

st.session_state["preset"] = None


# ---------- layout: left inputs, right outputs ----------
col_left, col_right = st.columns([1.0, 1.1], gap="large")

with col_left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Patient Information")
    patient_id = st.text_input("Patient ID (use anonymized ID)", value=st.session_state.get("patient_id", "patient_001"))
    st.session_state["patient_id"] = patient_id

    st.markdown("**Enter raw lab values (units shown in placeholder)**")
    st.write("")

    # group features into three roughly equal groups (or use explicit grouping)
    n = len(FEATURE_ORDER)
    group_a = FEATURE_ORDER[: n//3]
    group_b = FEATURE_ORDER[n//3 : 2*(n//3)]
    group_c = FEATURE_ORDER[2*(n//3) :]

    def render_group(features):
        for feat in features:
            default = st.session_state["inputs"].get(feat, DEFAULTS.get(feat, 0.0))
            # basic heuristics for step/format
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

with col_right:

    # ADD THIS LINE RIGHT HERE ↓↓↓
    st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

    # metric placeholders
    mcol1, mcol2, mcol3 = st.columns([1,1,1])
    pred_card = mcol1.empty()
    conf_card = mcol2.empty()
    issues_card = mcol3.empty()

    result_banner = st.container()
    probs_box = st.container()
    warns_box = st.container()
    contrib_box = st.container()

   

# ---------- load cached model & encoder ----------
try:
    model = load_model()
except Exception as e:
    st.error(f"Model load error: {e}")
    st.stop()

le = load_label_encoder()

# ---------- Predict action ----------
if predict_btn:
    # gather raw inputs in expected FEATURE_ORDER
    raw_inputs = {f: float(st.session_state["inputs"].get(f, 0.0)) for f in FEATURE_ORDER}

    # scale inputs (scale_input returns scaled dict and optional warnings)
    try:
            scaled, warnings, alerts = scale_input(raw_inputs)
    except Exception as e:
        st.error(f"Scaling failed: {e}")
        st.stop()
    
    # create dataframe X with correct column order
    X = pd.DataFrame([scaled], columns=FEATURE_ORDER).fillna(0.5)

    # run prediction and convert to standardized dict
    try:
        out = predict_with_model(model, X, label_encoder=le)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # attach scaling warnings into out["issues"] if present
    if warnings:
        # ensure structured list of dicts for display (key, message)
        for w in warnings:
            out.setdefault("issues", []).append({"issue": str(w)})
    # attach critical alerts also
    if alerts:
        for a in alerts:
            out.setdefault("issues", []).append({"issue": str(a)})


    pred_label = out.get("prediction", "N/A")
    probs = out.get("probabilities", {})
    issues_list = out.get("issues", [])
    top_features = out.get("top_contributing_features", [])

    # top metric cards
    conf_value = max(probs.values()) if probs else 0.0
    # display metrics
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

    display_metric(pred_card, "Prediction", f"{pred_label}", color="#0b5cff")
    display_metric(conf_card, "Confidence", f"{conf_value*100:.1f}%", color="#198754")
    display_metric(issues_card, "Input issues", str(len(issues_list)), color="#dc3545" if issues_list else "#6c757d")

    # result banner with color mapping
    color_map = {
        "Healthy": "#198754",
        "Diabetes": "#d63384",
        "Anemia": "#fd7e14",
        "Thalasse": "#0d6efd",
        "Thromboc": "#6f42c1",
    }
    banner_color = color_map.get(pred_label, "#0b5cff")
    with result_banner:
        st.markdown(f"<div class='card pred-banner'><h3 style='color:{banner_color}; margin:0px 0px 6px 0px;'>Prediction: <b>{pred_label}</b></h3>"
                    f"<div class='small'>Confidence: <b>{conf_value*100:.1f}%</b></div>"
                    f"<div class='small' style='margin-top:8px;'>Predicted at: {datetime.utcnow().isoformat(timespec='seconds')} UTC</div></div>", unsafe_allow_html=True)

    # probabilities box
    with probs_box:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Class probabilities")
        if probs:
            probs_df = pd.DataFrame(list(probs.items()), columns=["Class", "Probability"])
            probs_df["Probability"] = probs_df["Probability"].astype(float)
            probs_df = probs_df.sort_values("Probability", ascending=False).reset_index(drop=True)
            st.table(probs_df.style.format({"Probability":"{:.3f}"}))
        else:
            st.write("No probabilities returned.")
        st.markdown("</div>", unsafe_allow_html=True)

    # warnings / issues
    with warns_box:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if issues_list:
            st.subheader("Input warnings / data-quality issues")
            # turn simple strings into table rows if needed
            if isinstance(issues_list, list) and all(isinstance(x, str) for x in issues_list):
                df_issues = pd.DataFrame([{"issue": s} for s in issues_list])
                st.table(df_issues)
            else:
                df_issues = pd.DataFrame(issues_list)
                if not df_issues.empty:
                    st.table(df_issues)
                else:
                    st.write("No issues detected.")
        else:
            st.success("No data-quality issues detected.")
        st.markdown("</div>", unsafe_allow_html=True)


        # ---- contrib chart (only after predict) ----
    with contrib_box:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Top contributing features (global importance surrogate)")

        if top_features and len(top_features) > 0:

            top_df = pd.DataFrame(top_features, columns=["feature", "global_importance"])
            top_df["global_importance"] = top_df["global_importance"].astype(float)

            import plotly.express as px

            fig = px.pie(
                top_df,
                names="feature",
                values="global_importance",
                hole=0.55,
                color_discrete_sequence=px.colors.sequential.Blues_r
            )

            fig.update_layout(
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                showlegend=True,
                margin=dict(l=20, r=20, t=20, b=20),
                height=350
            )

            fig.update_traces(
                textfont_color="#1b2a49",
                textposition="inside"
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

            st.table(top_df.style.format({"global_importance": "{:.4f}"}))

        else:
            st.write("No feature importance available for this model.")

        st.markdown("</div>", unsafe_allow_html=True)


    


# small footer
st.markdown("---")
st.caption("MediGuard AI — Demo dashboard. For production use, add authentication, HTTPS, logging, and clinical validation.")
