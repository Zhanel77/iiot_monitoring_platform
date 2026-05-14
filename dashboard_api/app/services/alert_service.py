def build_explanation(top_factors: list[dict]) -> str:
    if not top_factors:
        return "Abnormal machine behavior detected."

    explanations = []

    for factor in top_factors[:3]:
        feature = factor.get("feature")
        value = factor.get("feature_value")
        effect = factor.get("effect")

        if effect != "increase":
            continue

        if feature == "Tool_wear_min":
            explanations.append(
                f"tool wear reached abnormal levels ({value} min)"
            )

        elif feature == "Torque_Nm":
            explanations.append(
                f"torque load increased to {value} Nm"
            )

        elif feature == "temp_diff":
            explanations.append(
                f"thermal imbalance detected ({value} K difference)"
            )

        elif feature == "Rotational_speed_rpm":
            explanations.append(
                f"rotational speed became unstable ({value} rpm)"
            )

        elif feature == "Air_temperature_K":
            explanations.append(
                f"air temperature exceeded normal operating range"
            )

        elif feature == "power_kw":
            explanations.append(
                f"power consumption increased significantly"
            )

    if not explanations:
        return "Machine behavior deviates from normal operating conditions."

    return (
        "Failure risk increased because "
        + ", ".join(explanations)
        + "."
    )