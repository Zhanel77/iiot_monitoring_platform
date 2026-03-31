import json
from pathlib import Path
from typing import Dict, Any


def load_normalization_config(path: str) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Normalization config not found: {path}")

    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)