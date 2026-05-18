# app/services/cloud_ml/schemas.py
from typing import Optional, Any
from pydantic import BaseModel, Field


class CloudPredictRequest(BaseModel):
    device_id: str = Field(..., description="ID устройства для определения weather-настроек")
    air_temperature_k: float = Field(..., alias="air_temperature_k")
    process_temperature_k: float = Field(..., alias="process_temperature_k")
    rotational_speed_rpm: float = Field(..., alias="rotational_speed_rpm")
    torque_nm: float = Field(..., alias="torque_nm")
    tool_wear_min: float = Field(..., alias="tool_wear_min")
    
    # Опциональные поля для кэширования погоды (если уже известна)
    outside_temp_c: Optional[float] = None
    outside_humidity: Optional[float] = None
    
    class Config:
        populate_by_name = True


class ShapFactor(BaseModel):
    feature: str
    feature_value: float
    shap_value: Optional[float] = None  # Для совместимости с weather фактором
    effect: str  # "increase" или "decrease"
    reason: Optional[str] = None  # Для weather объяснения


class CloudPredictResponse(BaseModel):
    prediction: int
    prediction_label: str
    risk_score: float
    ml_risk_score: Optional[float] = None
    weather_factor: Optional[float] = None
    top_factors: list[ShapFactor]
    features_used: dict[str, Any]