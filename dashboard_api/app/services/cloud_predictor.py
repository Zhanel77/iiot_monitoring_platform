import xgboost as xgb
import json
import numpy as np


class CloudPredictor:
    def __init__(self):
        self.model = xgb.Booster()
        self.model.load_model("app/models/ml/cloud_model.pkl")

        with open("app/models/ml/feature_list.json") as f:
            self.features = json.load(f)

        with open("app/models/ml/normalization_config.json") as f:
            self.norm = json.load(f)

    def normalize(self, data: dict):
        result = {}

        for f in self.features:
            val = data.get(f, 0)
            cfg = self.norm.get(f)

            if cfg:
                min_v = cfg["min"]
                max_v = cfg["max"]
                if max_v > min_v:
                    val = (val - min_v) / (max_v - min_v)

            result[f] = val

        return result

    def predict(self, data: dict):
        data = self.normalize(data)

        X = np.array([[data[f] for f in self.features]])
        dmatrix = xgb.DMatrix(X)

        prob = self.model.predict(dmatrix)[0]

        return {
            "risk_score": float(prob),
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