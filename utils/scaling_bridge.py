# utils/scaling_bridge.py
"""
Scaling bridge: convert raw clinical values -> scaled (0-1) values that match training dataset.
Relies on utils/medical_ranges.json which must contain the same ranges used when generating the scaled dataset.
"""

import json
from typing import Dict, Tuple

MEDICAL_RANGES_PATH = "utils/medical_ranges.json"

def load_medical_ranges(path: str = MEDICAL_RANGES_PATH) -> Dict[str, Tuple[float, float]]:
    with open(path, "r") as f:
        return json.load(f)

def _clip(x: float, mn: float, mx: float) -> float:
    if x < mn:
        return mn
    if x > mx:
        return mx
    return x

def scale_raw_to_0_1(raw_inputs: Dict[str, str], medical_ranges: Dict[str, Tuple[float, float]]):
    """
    Convert raw_inputs (strings or numbers) to scaled values in 0-1 using medical_ranges.
    Returns (scaled_dict, warnings_list).
    """
    scaled = {}
    warnings = []

    for feat, raw_val in raw_inputs.items():
        if feat not in medical_ranges:
            warnings.append(f"{feat}: no medical range defined")
            scaled[feat] = None
            continue

        mn, mx = medical_ranges[feat]

        if raw_val is None or raw_val == "":
            warnings.append(f"{feat}: missing -> using neutral 0.5")
            scaled[feat] = 0.5
            continue

        try:
            rv = float(raw_val)
        except Exception:
            warnings.append(f"{feat}: invalid number '{raw_val}' -> using neutral 0.5")
            scaled[feat] = 0.5
            continue

        # clip to medical range to avoid extreme out-of-range mapping
        rv_clipped = _clip(rv, mn, mx)

        if mx == mn:
            med_scaled = 0.5
        else:
            med_scaled = (rv_clipped - mn) / (mx - mn)

        # ensure numeric in [0,1]
        med_scaled = max(0.0, min(1.0, med_scaled))
        scaled[feat] = float(med_scaled)

    return scaled, warnings

# convenience wrapper for your dashboard
def scale_input(raw_inputs: Dict[str, str]):
    med = load_medical_ranges()
    return scale_raw_to_0_1(raw_inputs, med)
