# model/app/dashboard.py

import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # model/app -> model -> root
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import streamlit as st
import pandas as pd
import joblib

# import our scaling bridge
from utils.scaling_bridge import scale_input

MODEL_PATH = "model/xgb_model.pkl"
LABEL_ENCODER_PATH = "model/label_encoder.pkl"

st.set_page_config(page_title="MediGuard AI (XGBoost)", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_label_encoder():
    try:
        return joblib.load(LABEL_ENCODER_PATH)
    except Exception:
        return None

def main():
    st.title("🩺 MediGuard AI — Intelligent Triage Assistant (XGBoost)")

    model = load_model()
    le = load_label_encoder()

    # load features order from medical ranges file to guarantee same column order as training
    import json
    with open("utils/medical_ranges.json", "r") as f:
        medical_ranges = json.load(f)
    features = list(medical_ranges.keys())

    st.header("Enter Raw Clinical Values (real units)")
    raw_inputs = {}
    # two-column layout for nicer UI
    cols = st.columns(2)
    for i, feat in enumerate(features):
        col = cols[i % 2]
        raw_inputs[feat] = col.text_input(feat, "")

    patient_id = st.text_input("Patient ID", "")

    if st.button("Predict"):
        # scale raw inputs to 0-1 using the exact medical ranges
        scaled, warnings = scale_input(raw_inputs)

        st.subheader("Scaled Input (0–1)")
        st.write(pd.DataFrame([scaled]))

        if warnings:
            st.warning("⚠ Data Quality Warnings:")
            for w in warnings:
                st.write("- ", w)

        # prepare dataframe for model with correct column order
        X = pd.DataFrame([scaled], columns=features).fillna(0.5)

        # predict probabilities and label
        proba = model.predict_proba(X)[0].tolist()
        pred_idx = int(model.predict(X)[0])

        if le:
            class_names = list(le.classes_)
            pred_label = le.inverse_transform([pred_idx])[0]
        else:
            class_names = [str(i) for i in range(len(proba))]
            pred_label = str(pred_idx)

        st.success(f"🩸 Prediction: **{pred_label}**")
        st.write("📊 Probability distribution:")
        st.table(pd.DataFrame({"Disease": class_names, "Probability": proba}))

        # optional blockchain logging (if utils/blockchain.py exists)
        try:
            from utils.blockchain import append_block, load_chain
            block = append_block(patient_id, pred_label, proba, raw_inputs)
            st.subheader("🧾 Blockchain Log (Latest Entry)")
            st.json(block)
            st.subheader("Last 5 Blocks")
            st.table(pd.DataFrame(load_chain()[-5:]))
        except Exception:
            st.info("Blockchain module not available — skipping immutable log.")

if __name__ == "__main__":
    main()
