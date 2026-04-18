import numpy as np
import xgboost as xgb
from pathlib import Path


class CloudPredictor:
    def __init__(self, model_path: Path):
        self.model = xgb.Booster()
        self.model.load_model(str(model_path))

    def predict(self, features: list[float]) -> tuple[int, float]:
        x = np.array(features, dtype=float).reshape(1, -1)
        dmatrix = xgb.DMatrix(x)

        risk_score = float(self.model.predict(dmatrix)[0])
        prediction = 1 if risk_score >= 0.5 else 0

        return prediction, risk_score