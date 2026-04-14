from pydantic import BaseModel
from datetime import datetime


class PredictionCreate(BaseModel):
    device_id: str
    machine_id: int
    event_time: datetime
    prediction: int
    risk_score: float | None = None
    risk_level: str | None = None
    source: str | None = None
    scenario: str | None = None


class PredictionRead(BaseModel):
    id: int
    device_id: str
    machine_id: int
    event_time: datetime
    prediction: int
    risk_score: float | None
    risk_level: str | None

    model_config = {"from_attributes": True}