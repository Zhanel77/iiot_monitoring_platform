import json
import re
from typing import Any, Dict, List

import numpy as np
import shap
import xgboost as xgb


def sanitize_feature_name(name: str) -> str:
    name = name.replace("[", "").replace("]", "")
    name = name.replace("(", "").replace(")", "")
    name = name.replace("/", "_")
    name = name.replace("-", "_")
    name = name.replace(" ", "_")
    name = re.sub(r"[^0-9a-zA-Z_]", "", name)
    return name


class ShapExplainer:
    def __init__(self):
        self.model = xgb.Booster()
        self.model.load_model("app/models/ml/cloud_model.json")

        with open("app/models/ml/cloud_feature_list.json") as f:
            self.features = json.load(f)

        self.safe_features = [sanitize_feature_name(f) for f in self.features]
        self.explainer = shap.TreeExplainer(self.model)

    def explain(
        self,
        normalized_features: dict,
        raw_features: dict,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        row = {sanitize_feature_name(f): normalized_features[f] for f in self.features}
        x = np.array([[row[f] for f in self.safe_features]], dtype=float)

        shap_values = self.explainer.shap_values(x)

        if hasattr(shap_values, "ndim") and shap_values.ndim > 1:
            values = shap_values[0]
        else:
            values = shap_values

        factors = []
        for idx, feature_name in enumerate(self.features):
            shap_value = float(values[idx])
            feature_value = float(raw_features.get(feature_name, 0.0))

            factors.append(
                {
                    "feature": feature_name,
                    "feature_value": feature_value,
                    "shap_value": shap_value,
                    "effect": "increase" if shap_value >= 0 else "decrease",
                    "abs_shap": abs(shap_value),
                }
            )

        factors.sort(key=lambda x: x["abs_shap"], reverse=True)
        factors = factors[:top_k]

        for item in factors:
            item.pop("abs_shap", None)

        return factors