from pathlib import Path

import numpy as np
import shap
import xgboost as xgb


class CloudShapExplainer:
    def __init__(self, model_path: Path, feature_names: list[str]):
        self.model = xgb.Booster()
        self.model.load_model(str(model_path))
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(self.model)

    def explain(self, features: list[float], feature_values: dict[str, float], top_k: int = 5) -> list[dict]:
        x = np.array(features, dtype=float).reshape(1, -1)
        shap_values = self.explainer.shap_values(x)

        values = shap_values[0] if hasattr(shap_values, "ndim") and shap_values.ndim > 1 else shap_values

        rows = []
        for idx, feature_name in enumerate(self.feature_names):
            shap_value = float(values[idx])
            raw_value = float(feature_values.get(feature_name, 0.0))
            rows.append(
                {
                    "feature": feature_name,
                    "feature_value": raw_value,
                    "shap_value": shap_value,
                    "effect": "increase" if shap_value >= 0 else "decrease",
                    "abs_shap": abs(shap_value),
                }
            )

        rows.sort(key=lambda item: item["abs_shap"], reverse=True)
        rows = rows[:top_k]

        for row in rows:
            row.pop("abs_shap", None)

        return rows 