# model/app/dashboard.py

import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import streamlit as st
import pandas as pd
import joblib
import json

from utils.scaling_bridge import scale_input  # NEW: includes alerts

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
    except:
        return None

def main():
    st.title("🩺 MediGuard AI — Intelligent Triage Assistant (XGBoost)")

    model = load_model()
    le = load_label_encoder()

    # load medical ranges to enforce column order
    with open("utils/medical_ranges.json", "r") as f:
        medical_ranges = json.load(f)

    features = list(medical_ranges.keys())

    st.header("Enter Raw Clinical Values (real units)")
    raw_inputs = {}

    cols = st.columns(2)
    for i, feat in enumerate(features):
        raw_inputs[feat] = cols[i % 2].text_input(feat, "")

    patient_id = st.text_input("Patient ID", "")

    if st.button("Predict"):
        # scale raw input with alerts
        scaled, warnings, alerts = scale_input(raw_inputs, medical_ranges)

        st.subheader("Scaled Input (0–1)")
        st.write(pd.DataFrame([scaled]))

        # 🚨 CRITICAL alerts
        if alerts:
            st.error("🚨 CRITICAL MEDICAL ALERTS DETECTED:")
            for a in alerts:
                st.write("- ", a)

        # ⚠ Normal warnings
        if warnings:
            st.warning("⚠ Data Quality Warnings:")
            for w in warnings:
                st.write("- ", w)

        # prepare row
        X = pd.DataFrame([scaled], columns=features).fillna(0.5)

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

        # optional blockchain
        try:
            from utils.blockchain import append_block, load_chain
            block = append_block(patient_id, pred_label, proba, raw_inputs)
            st.subheader("🧾 Blockchain Log (Latest Entry)")
            st.json(block)
            st.subheader("Last 5 Blocks")
            st.table(pd.DataFrame(load_chain()[-5:]))
        except Exception:
            st.info("Blockchain module not available in deployment")

if __name__ == "__main__":
    main()
