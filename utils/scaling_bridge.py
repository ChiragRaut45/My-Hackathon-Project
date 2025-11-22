# utils/scaling_bridge.py
"""
Scaling bridge: Convert raw clinical values -> scaled (0–1).
Adds advanced data-quality detection (critical alerts).
"""

import json
from typing import Dict, Tuple

MEDICAL_RANGES_PATH = "utils/medical_ranges.json"

def load_medical_ranges(path: str = MEDICAL_RANGES_PATH):
    with open(path, "r") as f:
        return json.load(f)

def scale_value(raw, mn, mx):
    if raw < mn:
        raw = raw  # still show true value for alert
    if raw > mx:
        raw = raw
    # scaled after clipping:
    raw_clipped = max(mn, min(raw, mx))
    return (raw_clipped - mn) / (mx - mn)

def scale_input(raw_inputs: Dict[str, str], ranges=None):
    if ranges is None:
        ranges = load_medical_ranges()

    scaled = {}
    warnings = []
    alerts = []   # 🚨 critical out-of-range alerts

    for feat, val in raw_inputs.items():

        if feat not in ranges:
            warnings.append(f"{feat}: No medical range defined")
            scaled[feat] = 0.5
            continue

        mn, mx = ranges[feat]

        if val is None or val == "":
            warnings.append(f"{feat}: Missing → using neutral 0.5")
            scaled[feat] = 0.5
            continue

        try:
            rv = float(val)
        except:
            warnings.append(f"{feat}: Invalid number → using 0.5")
            scaled[feat] = 0.5
            continue

        # --- critical alert detection ---
        if rv < mn:
            alerts.append(f"⚠️ {feat} critically LOW ({rv}) — expected {mn} to {mx}")
        if rv > mx:
            alerts.append(f"⚠️ {feat} dangerously HIGH ({rv}) — expected {mn} to {mx}")

        # scale to 0–1
        scaled_val = scale_value(rv, mn, mx)
        scaled[feat] = float(scaled_val)

    return scaled, warnings, alerts
