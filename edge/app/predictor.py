from typing import Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd


class EdgePredictor:
    def __init__(self, model_path: str, feature_order: List[str]):
        self.model = joblib.load(model_path)
        self.feature_order = feature_order

    def prepare_vector(self, normalized_features):
        return pd.DataFrame([normalized_features])[self.feature_order]

    def predict(self, normalized_features: Dict[str, float]) -> Tuple[int, float]:
        x = self.prepare_vector(normalized_features)

        pred = int(self.model.predict(x)[0])

        if hasattr(self.model, "predict_proba"):
            proba = float(self.model.predict_proba(x)[0][1])
        else:
            proba = float(pred)

        return pred, proba


def resolve_risk_level(risk_score: float, warning_threshold: float, critical_threshold: float) -> str:
    if risk_score >= critical_threshold:
        return "CRITICAL"
    if risk_score >= warning_threshold:
        return "WARNING"
    return "NORMAL"


def resolve_prediction_label(prediction: int) -> str:
    return "FAILURE_RISK" if prediction == 1 else "NORMAL"