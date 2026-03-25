from pathlib import Path

import joblib
import pandas as pd

from config_loader import load_device_config
from adapter import adapt_raw_payload


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "edge_model.pkl"

FEATURE_ORDER = [
    "Air temperature [K]",
    "temp_diff",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "power_kw",
    "Tool wear [min]",
]


def safe_minmax(value: float, min_val: float, max_val: float) -> float:
    denom = max_val - min_val
    if abs(denom) < 1e-12:
        return 0.0
    result = (value - min_val) / denom
    return max(0.0, min(1.0, result))


def safe_ratio(value: float, max_val: float) -> float:
    if abs(max_val) < 1e-12:
        return 0.0
    result = value / max_val
    return max(0.0, min(1.0, result))


def normalize_runtime(features: dict, norm_cfg: dict) -> dict:
    normalized = {}

    for feature_name, value in features.items():
        cfg = norm_cfg[feature_name]
        mode = cfg["mode"]

        if mode == "minmax":
            normalized[feature_name] = safe_minmax(value, cfg["min"], cfg["max"])
        elif mode == "ratio":
            normalized[feature_name] = safe_ratio(value, cfg["max"])
        else:
            raise ValueError(f"Unsupported normalization mode: {mode}")

    return normalized


class EdgePredictor:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Edge model not found: {MODEL_PATH}")
        self.model = joblib.load(MODEL_PATH)

    def predict(self, raw_payload: dict) -> dict:
        machine_id = raw_payload.get("machine_id")
        config = load_device_config(machine_id)

        features = adapt_raw_payload(raw_payload)
        normalized = normalize_runtime(features, config)

        x = pd.DataFrame(
            [[normalized[col] for col in FEATURE_ORDER]],
            columns=FEATURE_ORDER,
        )

        prediction = int(self.model.predict(x)[0])
        risk_score = float(self.model.predict_proba(x)[0][1])

        return {
            "machine_id": machine_id,
            "prediction": prediction,
            "risk_score": risk_score,
            "features": normalized,
        }