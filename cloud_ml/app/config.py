from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "cloud_model.json"
FEATURE_LIST_PATH = MODELS_DIR / "cloud_feature_list.json"
NORMALIZATION_CONFIG_PATH = MODELS_DIR / "cloud_normalization_config.json"

TOP_K_FEATURES = 5
ENABLE_SHAP_FOR_ALL = True
SHAP_ONLY_FOR_HIGH_RISK = False
HIGH_RISK_THRESHOLD = 0.7