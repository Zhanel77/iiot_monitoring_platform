from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


APP_DIR = Path(__file__).resolve().parent.parent
ML_MODELS_DIR = APP_DIR / "models" / "ml"


class Settings(BaseSettings):
    DATABASE_URL: str

    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    OPENWEATHER_API_KEY: str | None = None

    # Cloud ML artifacts
    MODEL_PATH: Path = ML_MODELS_DIR / "cloud_model.json"
    FEATURE_LIST_PATH: Path = ML_MODELS_DIR / "cloud_feature_list.json"
    NORMALIZATION_CONFIG_PATH: Path = ML_MODELS_DIR / "cloud_normalization_config.json"
    SHAP_BACKGROUND_SAMPLE_PATH: Path = ML_MODELS_DIR / "shap_background_sample.pkl"

    # Cloud ML settings
    TOP_K_FEATURES: int = 5
    ENABLE_SHAP_FOR_ALL: bool = True
    SHAP_ONLY_FOR_HIGH_RISK: bool = False
    HIGH_RISK_THRESHOLD: float = 0.7

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()