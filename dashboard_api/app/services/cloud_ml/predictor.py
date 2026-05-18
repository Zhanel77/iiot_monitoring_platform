from pathlib import Path

import numpy as np
import xgboost as xgb

from app.services.cloud_ml.feature_pipeline import sanitize_feature_name


class CloudPredictor:
    def __init__(self, model_path: Path, feature_names: list[str]):
        self.feature_names = [sanitize_feature_name(str(f)) for f in feature_names]

        print("PREDICTOR FEATURE NAMES:", self.feature_names)

        self.model = xgb.Booster()
        self.model.load_model(str(model_path))

    def predict(self, features: list[float]) -> tuple[int, float]:
        x = np.array(features, dtype=float).reshape(1, -1)

        dmatrix = xgb.DMatrix(
            x,
            feature_names=self.feature_names,
        )

        risk_score = float(self.model.predict(dmatrix)[0])
        prediction = 1 if risk_score >= 0.5 else 0

        return prediction, risk_score