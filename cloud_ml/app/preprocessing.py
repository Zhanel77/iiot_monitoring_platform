import json
import math
from pathlib import Path
from app.feature_pipeline import sanitize_feature_name


class CloudPreprocessor:
    def __init__(self, feature_list_path: Path, normalization_config_path: Path):
        with open(feature_list_path, "r", encoding="utf-8") as f:
            self.feature_list = [
                    sanitize_feature_name(f) 
                    for f in json.load(f)
                ]

        with open(normalization_config_path, "r", encoding="utf-8") as f:
            self.normalization_config = json.load(f)

    @staticmethod
    def _safe_float(value: float) -> float:
        return float(value)

    @staticmethod
    def _sanitize_feature_names(row: dict) -> dict:
        return {
            sanitize_feature_name(str(k).strip()): v
            for k, v in row.items()
        }

    @staticmethod
    def _engineer_features(raw: dict) -> dict:
        air_temp = float(raw["Air temperature [K]"])
        process_temp = float(raw["Process temperature [K]"])
        rotational_speed = float(raw["Rotational speed [rpm]"])
        torque = float(raw["Torque [Nm]"])
        tool_wear = float(raw["Tool wear [min]"])

        temp_diff = process_temp - air_temp

        # Если в training у тебя была другая формула/масштаб — замени тут точно на ту же
        power_kw = 2 * math.pi * torque * rotational_speed / 60000.0

        wear_rate = tool_wear / max(rotational_speed, 1.0)
        thermal_stress = temp_diff * torque

        return {
            "Air temperature [K]": air_temp,
            "Process temperature [K]": process_temp,
            "temp_diff": temp_diff,
            "Rotational speed [rpm]": rotational_speed,
            "Torque [Nm]": torque,
            "power_kw": power_kw,
            "Tool wear [min]": tool_wear,
            "wear_rate": wear_rate,
            "thermal_stress": thermal_stress,
        }

    def _normalize_value(self, feature_name: str, value: float) -> float:
        cfg = self.normalization_config.get(feature_name)

        if not cfg:
            return float(value)

        method = cfg.get("method", "none")

        if method == "robust":
            median = cfg.get("median", 0.0)
            iqr = cfg.get("iqr", 1.0)
            if iqr == 0:
                return float(value)
            return (value - median) / iqr

        if method == "standard":
            mean = cfg.get("mean", 0.0)
            std = cfg.get("std", 1.0)
            if std == 0:
                return float(value)
            return (value - mean) / std

        if method == "minmax":
            min_v = cfg.get("min", 0.0)
            max_v = cfg.get("max", 1.0)
            if max_v == min_v:
                return float(value)
            return (value - min_v) / (max_v - min_v)

        return float(value)

    def prepare_features(self, payload: dict) -> tuple[list[float], dict]:
        raw = {
            "Air temperature [K]": self._safe_float(payload["air_temperature_k"]),
            "Process temperature [K]": self._safe_float(payload["process_temperature_k"]),
            "Rotational speed [rpm]": self._safe_float(payload["rotational_speed_rpm"]),
            "Torque [Nm]": self._safe_float(payload["torque_nm"]),
            "Tool wear [min]": self._safe_float(payload["tool_wear_min"]),
        }

        engineered = self._engineer_features(raw)
        engineered = self._sanitize_feature_names(engineered)

        ordered_features = []
        features_used = {}

        for feature_name in self.feature_list:
            value = engineered.get(feature_name)
            if value is None:
                raise ValueError(f"Missing feature after preprocessing: {feature_name}")

            normalized_value = self._normalize_value(feature_name, float(value))
            ordered_features.append(normalized_value)
            features_used[feature_name] = float(value)

        return ordered_features, features_used