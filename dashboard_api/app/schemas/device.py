# app/schemas/device.py
from pydantic import BaseModel
from typing import Optional


class DeviceCreate(BaseModel):
    machine_id: int
    device_id: str
    name: str | None = None
    # Weather поля (опциональны при создании)
    weather_dependent: bool = False
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    weather_sensitivity: str = "medium"  # low, medium, high


class DeviceUpdate(BaseModel):
    name: str | None = None
    status: str | None = None
    # Weather поля для обновления
    weather_dependent: Optional[bool] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    weather_sensitivity: Optional[str] = None


class DeviceRead(BaseModel):
    id: int
    machine_id: int
    device_id: str
    name: str | None
    status: str
    # Weather поля для чтения
    weather_dependent: bool
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    weather_sensitivity: str

    model_config = {"from_attributes": True}


class DeviceWeatherConfig(BaseModel):
    """Специальная схема только для weather-конфигурации"""
    device_id: str
    machine_id: int
    weather_dependent: bool
    latitude: Optional[float]
    longitude: Optional[float]
    weather_sensitivity: str
    device_name: Optional[str] = None