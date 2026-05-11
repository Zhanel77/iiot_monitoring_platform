def sanitize_feature_name(name: str) -> str:
    return (
        name.replace(" ", "_")
            .replace("[", "")
            .replace("]", "")
            .replace("<", "")
            .replace(">", "")
    )

