def build_explanation(top_factors: list[dict]) -> str:
    reasons = []

    for f in top_factors[:3]:
        name = f["feature"]
        value = f["feature_value"]

        if f["effect"] == "increase":
            reasons.append(f"{name} ({value}) increased risk")
        else:
            reasons.append(f"{name} ({value}) decreased risk")

    return "; ".join(reasons)