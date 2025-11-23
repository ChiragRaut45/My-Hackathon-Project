# model/app/blockchain_viewer.py

import sys, os

# ---------- FIX PYTHON PATH ----------
CURRENT = os.path.dirname(os.path.abspath(__file__))      # model/app
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT, "..", ".."))
UTILS_PATH = os.path.join(PROJECT_ROOT, "utils")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if UTILS_PATH not in sys.path:
    sys.path.insert(0, UTILS_PATH)

# ---------- imports ----------
import streamlit as st
import pandas as pd
from utils.blockchain import load_chain

# ---------- UI ----------
st.set_page_config(page_title="Blockchain Ledger Viewer", layout="wide")

st.title("🔗 MediGuard Blockchain Ledger")
st.caption("This page shows all prediction logs stored on the blockchain.")

try:
    chain = load_chain()
    st.success(f"Loaded {len(chain)} blocks")

    df = pd.DataFrame(chain)
    st.dataframe(df)

    st.subheader("Raw JSON View")
    st.json(chain)

except Exception as e:
    st.error(f"Could not load blockchain: {e}")
