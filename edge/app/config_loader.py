import json
from pathlib import Path


CONFIG_DIR = Path(__file__).resolve().parent / "config"


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_device_config(machine_id: int | None = None) -> dict:
    if machine_id is not None:
        machine_path = CONFIG_DIR / f"machine_{machine_id}.json"
        if machine_path.exists():
            return load_json(machine_path)

    fallback_path = CONFIG_DIR / "training_fallback.json"
    if fallback_path.exists():
        return load_json(fallback_path)

    raise FileNotFoundError("No config found for device and no training fallback config found.")