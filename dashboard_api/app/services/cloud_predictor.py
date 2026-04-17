import json
import re
from typing import Any, Dict

import numpy as np
import xgboost as xgb


def normalize_value(value: float, rule: Dict[str, Any]) -> float:
    mode = rule.get("mode")

    if mode == "minmax":
        min_v = rule["min"]
        max_v = rule["max"]
        if max_v == min_v:
            return 0.0
        return (value - min_v) / (max_v - min_v)

    if mode == "ratio":
        max_v = rule["max"]
        if max_v == 0:
            return 0.0
        return value / max_v

    raise ValueError(f"Unsupported normalization mode: {mode}")


def sanitize_feature_name(name: str) -> str:
    name = name.replace("[", "").replace("]", "")
    name = name.replace("(", "").replace(")", "")
    name = name.replace("/", "_")
    name = name.replace("-", "_")
    name = name.replace(" ", "_")
    name = re.sub(r"[^0-9a-zA-Z_]", "", name)
    return name


class CloudPredictor:
    def __init__(self):
        self.model = xgb.Booster()
        self.model.load_model("app/models/ml/cloud_model.json")

        with open("app/models/ml/cloud_feature_list.json") as f:
            self.features = json.load(f)

        with open("app/models/ml/cloud_normalization_config.json") as f:
            self.norm = json.load(f)

        self.safe_features = [sanitize_feature_name(f) for f in self.features]

    def normalize(self, data: dict) -> dict:
        result = {}

        for f in self.features:
            val = data.get(f, 0)
            cfg = self.norm.get(f)

            if cfg is not None:
                try:
                    val = normalize_value(float(val), cfg)
                except Exception as e:
                    print(f"[WARN] Failed to normalize feature '{f}': value={val}, cfg={cfg}, error={e}")
                    val = 0.0

            result[f] = val

        return result

    def predict(self, data: dict):
        normalized = self.normalize(data)

        row = {sanitize_feature_name(f): normalized[f] for f in self.features}
        X = np.array([[row[f] for f in self.safe_features]], dtype=float)

        dmatrix = xgb.DMatrix(X, feature_names=self.safe_features)

        prob = float(self.model.predict(dmatrix)[0])

        return {
            "risk_score": prob,
            "prediction": int(prob > 0.5),
            "risk_level": self._level(prob),
        }

    def _level(self, score: float):
        if score > 0.8:
            return "CRITICAL"
        if score > 0.6:
            return "HIGH"
        if score > 0.4:
            return "WARNING"
        return "NORMAL"