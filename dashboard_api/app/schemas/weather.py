from pydantic import BaseModel


class DeviceWeatherStatus(BaseModel):
    device_id: str
    machine_id: int | None = None

    latitude: float | None = None
    longitude: float | None = None

    risk_score: float | None = None
    risk_level: str | None = None

    outside_temp_c: float | None = None
    outside_humidity: float | None = None
    outside_pressure: float | None = None
    wind_speed: float | None = None
    weather_main: str | None = None

    weather_impact: str | None = None
    environmental_reasons: list[str] | None = None