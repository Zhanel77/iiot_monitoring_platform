from pydantic import BaseModel
from datetime import datetime
from typing import Any


class ShapFactorRead(BaseModel):
    feature: str
    feature_value: float
    shap_value: float
    effect: str


class PredictionCreate(BaseModel):
    device_id: str
    machine_id: int
    event_time: datetime
    prediction: int
    risk_score: float | None = None
    risk_level: str | None = None
    source: str | None = None
    scenario: str | None = None
    model_type: str | None = "cloud"
    features_used: dict[str, Any] | None = None
    top_factors: list[ShapFactorRead] | None = None


class PredictionRead(BaseModel):
    id: int
    device_id: str
    machine_id: int
    event_time: datetime
    prediction: int
    risk_score: float | None
    risk_level: str | None
    source: str | None = None
    scenario: str | None = None
    model_type: str
    features_used: dict[str, Any] | None = None
    top_factors: list[ShapFactorRead] | None = None

    model_config = {"from_attributes": True}