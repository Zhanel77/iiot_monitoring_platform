from typing import Any

from pydantic import BaseModel, Field


class CloudPredictRequest(BaseModel):
    air_temperature_k: float = Field(..., alias="air_temperature_k")
    process_temperature_k: float = Field(..., alias="process_temperature_k")
    rotational_speed_rpm: float = Field(..., alias="rotational_speed_rpm")
    torque_nm: float = Field(..., alias="torque_nm")
    tool_wear_min: float = Field(..., alias="tool_wear_min")

    class Config:
        populate_by_name = True


class ShapFactor(BaseModel):
    feature: str
    feature_value: float
    shap_value: float
    effect: str


class CloudPredictResponse(BaseModel):
    prediction: int
    prediction_label: str
    risk_score: float
    top_factors: list[ShapFactor]
    features_used: dict[str, Any]