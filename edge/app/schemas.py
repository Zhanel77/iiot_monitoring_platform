from typing import Dict, Any, Optional
from pydantic import BaseModel


class FailureType(BaseModel):
    twf: int = 0
    hdf: int = 0
    pwf: int = 0
    osf: int = 0
    rnf: int = 0


class GroundTruth(BaseModel):
    machine_failure: int = 0
    failure_type: FailureType


class RawSensors(BaseModel):
    air_temperature_k: float
    process_temperature_k: float
    rotational_speed_rpm: float
    torque_nm: float
    tool_wear_min: float


class RawMQTTMessage(BaseModel):
    device_id: str
    machine_id: int
    timestamp: str
    source: str
    scenario: str
    sensors: RawSensors
    ground_truth: Optional[GroundTruth] = None


class AdaptedFeatures(BaseModel):
    device_id: str
    machine_id: int
    timestamp: str
    source: str
    scenario: str
    features: Dict[str, float]
    ground_truth: Optional[Dict[str, Any]] = None


class EdgePredictionMessage(BaseModel):
    device_id: str
    machine_id: int
    timestamp: str
    source: str
    scenario: str
    prediction: int
    prediction_label: str
    risk_score: float
    risk_level: str
    model_name: str
    features_raw: Dict[str, float]
    features_normalized: Dict[str, float]
    ground_truth: Optional[Dict[str, Any]] = None