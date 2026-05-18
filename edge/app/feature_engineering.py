from typing import Dict, Any


def compute_temp_diff(features: Dict[str, float]) -> float:
    return features["Process temperature [K]"] - features["Air temperature [K]"]


def compute_power_kw(features: Dict[str, float]) -> float:
    rpm = features["Rotational speed [rpm]"]
    torque = features["Torque [Nm]"]

    # Механическая мощность:
    # P(W) = torque * 2*pi*rpm / 60
    # P(kW) = P(W) / 1000
    power_kw = (torque * 2 * 3.141592653589793 * rpm) / 60000.0
    return power_kw


def enrich_features(features: Dict[str, float]) -> Dict[str, float]:
    enriched = dict(features)
    enriched["temp_diff"] = compute_temp_diff(enriched)
    enriched["power_kw"] = compute_power_kw(enriched)
    return enriched


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


def normalize_features(features: Dict[str, float], normalization_config: Dict[str, Any]) -> Dict[str, float]:
    normalized = {}

    for feature_name, rule in normalization_config.items():
        if feature_name not in features:
            raise KeyError(f"Missing feature for normalization: {feature_name}")

        normalized[feature_name] = normalize_value(features[feature_name], rule)

    return normalized