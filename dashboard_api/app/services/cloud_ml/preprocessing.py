import json
import math
from pathlib import Path
from app.services.cloud_ml.feature_pipeline import sanitize_feature_name

class CloudPreprocessor:
    def __init__(self, feature_list_path: Path, normalization_config_path: Path):
        with open(feature_list_path, "r", encoding="utf-8") as f:
            self.feature_list = [
                sanitize_feature_name(str(feature).strip())
                for feature in json.load(f)
            ]

        with open(normalization_config_path, "r", encoding="utf-8") as f:
            raw_config = json.load(f)

        self.normalization_config = {
            sanitize_feature_name(str(k).strip()): v
            for k, v in raw_config.items()
        }

        # print("FEATURE LIST:", self.feature_list)
        # print("NORMALIZATION CONFIG KEYS:", list(self.normalization_config.keys()))

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

        mode = cfg.get("mode") or cfg.get("method") or "none"

        if mode == "ratio":
            max_v = cfg.get("max", 1.0)
            if max_v == 0:
                return float(value)
            return float(value) / max_v

        if mode == "minmax":
            min_v = cfg.get("min", 0.0)
            max_v = cfg.get("max", 1.0)
            if max_v == min_v:
                return float(value)
            return (float(value) - min_v) / (max_v - min_v)

        if mode == "robust":
            median = cfg.get("median", 0.0)
            iqr = cfg.get("iqr", 1.0)
            if iqr == 0:
                return float(value)
            return (float(value) - median) / iqr

        if mode == "standard":
            mean = cfg.get("mean", 0.0)
            std = cfg.get("std", 1.0)
            if std == 0:
                return float(value)
            return (float(value) - mean) / std

        return float(value)

    def prepare_features(self, payload: dict) -> tuple[list[float], dict]:
        features_raw = payload.get("features_raw") or {}

        raw = {
            "Air temperature [K]": self._safe_float(
                features_raw.get("Air temperature [K]", payload.get("air_temperature_k", 0))
            ),
            "Process temperature [K]": self._safe_float(
                features_raw.get("Process temperature [K]", payload.get("process_temperature_k", 0))
            ),
            "Rotational speed [rpm]": self._safe_float(
                features_raw.get("Rotational speed [rpm]", payload.get("rotational_speed_rpm", 0))
            ),
            "Torque [Nm]": self._safe_float(
                features_raw.get("Torque [Nm]", payload.get("torque_nm", 0))
            ),
            "Tool wear [min]": self._safe_float(
                features_raw.get("Tool wear [min]", payload.get("tool_wear_min", 0))
            ),
        }

        # если process_temperature не пришел, но temp_diff есть
        if raw["Process temperature [K]"] == 0 and "temp_diff" in features_raw:
            raw["Process temperature [K]"] = (
                raw["Air temperature [K]"] + self._safe_float(features_raw["temp_diff"])
            )

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

        # print("CLOUD RAW:", raw)
        # print("CLOUD FEATURES USED:", features_used)
        # print("CLOUD ORDERED FEATURES:", ordered_features)

        return ordered_features, features_used