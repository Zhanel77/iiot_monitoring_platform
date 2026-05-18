import os
from pathlib import Path
from typing import Dict

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Better to define the root relative to this file
APP_DIR = Path(__file__).resolve().parent.parent
ML_MODELS_DIR = APP_DIR / "models" / "ml"

class Settings(BaseSettings):
    # --- Database & Security ---
    DATABASE_URL: str
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # --- External APIs ---
    OPENWEATHER_API_KEY: str = ""
    
    # Use Field for better documentation/metadata
    WEATHER_SENSITIVITY_FACTORS: Dict[str, float] = Field(
        default_factory=lambda: {
            "low": 0.5,
            "medium": 1.0,
            "high": 2.0
        }
    )

    # --- Cloud ML Artifacts ---
    # These will automatically resolve to absolute paths
    MODEL_PATH: Path = ML_MODELS_DIR / "cloud_model.json"
    FEATURE_LIST_PATH: Path = ML_MODELS_DIR / "cloud_feature_list.json"
    NORMALIZATION_CONFIG_PATH: Path = ML_MODELS_DIR / "cloud_normalization_config.json"
    SHAP_BACKGROUND_SAMPLE_PATH: Path = ML_MODELS_DIR / "shap_background_sample.pkl"

    # --- Cloud ML Logic ---
    TOP_K_FEATURES: int = 5
    ENABLE_SHAP_FOR_ALL: bool = True
    SHAP_ONLY_FOR_HIGH_RISK: bool = False
    HIGH_RISK_THRESHOLD: float = Field(0.7, ge=0.0, le=1.0)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # 'ignore' is good for production to avoid crashing on extra env vars
        extra="ignore",
    )

settings = Settings()