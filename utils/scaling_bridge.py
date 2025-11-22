# utils/scaling_bridge.py
"""
Scaling bridge: Convert raw clinical values -> scaled (0–1)
Includes:
 - Min–max scaling
 - Out-of-range CRITICAL alerts
 - Data quality warnings
"""

import json

RANGE_PATH = "utils/medical_ranges.json"


def load_ranges(path=RANGE_PATH):
    with open(path, "r") as f:
        return json.load(f)


def scale_value(raw, mn, mx):
    """
    Min–max scaling with clipping.
    """
    if raw < mn:
        raw = mn   # clip but allow alert
    if raw > mx:
        raw = mx
    return (raw - mn) / (mx - mn)


def scale_input(raw_inputs, ranges=None):
    """
    Convert raw patient inputs → scaled 0–1 + warnings + critical alerts.
    RETURNS → (scaled_dict, warnings_list, alerts_list)
    """
    if ranges is None:
        ranges = load_ranges()

    scaled = {}
    warnings = []
    alerts = []

    for feat, val in raw_inputs.items():

        if feat not in ranges:
            warnings.append(f"No medical range found for {feat}")
            scaled[feat] = 0.5
            continue

        mn, mx = ranges[feat]

        # Missing
        if val == "" or val is None:
            scaled[feat] = 0.5
            warnings.append(f"{feat}: Missing → Default = 0.5")
            continue

        # Invalid number
        try:
            rv = float(val)
        except:
            scaled[feat] = 0.5
            warnings.append(f"{feat}: Invalid entry → Using 0.5")
            continue

        # -----------------------
        # 🚨 CRITICAL ALERT LOGIC
        # -----------------------
        if rv < mn:
            alerts.append(
                f"⚠️ {feat} critically LOW ({rv}) — below physiological limits ({mn}-{mx})"
            )
        if rv > mx:
            alerts.append(
                f"⚠️ {feat} dangerously HIGH ({rv}) — above physiological limits ({mn}-{mx})"
            )

        # scale to 0–1
        scaled_val = scale_value(rv, mn, mx)
        scaled[feat] = float(scaled_val)

    return scaled, warnings, alerts
