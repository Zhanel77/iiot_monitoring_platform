from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.device import Device
from app.models.prediction import Prediction
from app.schemas.weather import DeviceWeatherStatus
from app.services.weather_service import WeatherService
from app.core.config import settings

router = APIRouter(prefix="/weather", tags=["weather"])

weather_service = WeatherService(settings.OPENWEATHER_API_KEY)


def calculate_weather_impact(
    temp: float | None,
    humidity: float | None,
    wind: float | None,
    weather_main: str | None,
) -> tuple[str, list[str]]:

    score = 0
    reasons = []

    if temp is not None:
        if temp > 35:
            score += 2
            reasons.append("Extreme outside temperature")

        elif temp > 28:
            score += 1
            reasons.append("High outside temperature")

    if humidity is not None and humidity > 80:
        score += 1
        reasons.append("High humidity")

    if wind is not None:
        if wind > 12:
            score += 2
            reasons.append("Strong wind conditions")

        elif wind > 8:
            score += 1
            reasons.append("Elevated wind speed")

    if weather_main:
        weather_upper = weather_main.upper()

        if weather_upper in ["DUST", "SAND", "ASH"]:
            score += 2
            reasons.append("Dust storm conditions")

        elif weather_upper in ["THUNDERSTORM", "TORNADO"]:
            score += 3
            reasons.append("Severe weather event")

    if score >= 4:
        return "HIGH", reasons

    if score >= 2:
        return "MEDIUM", reasons

    return "LOW", reasons


@router.get(
    "/devices-status",
    response_model=list[DeviceWeatherStatus],
)
def get_devices_weather_status(
    db: Session = Depends(get_db),
):
    devices = db.query(Device).all()

    result = []

    for device in devices:

        latest_prediction = (
            db.query(Prediction)
            .filter(Prediction.device_id == device.device_id)
            .order_by(Prediction.created_at.desc())
            .first()
        )

        weather = None

        if device.latitude and device.longitude:
            weather = weather_service.get_current_weather(
                lat=device.latitude,
                lon=device.longitude,
            )

        weather_impact = "LOW"
        environmental_reasons = []

        if weather:
            weather_impact, environmental_reasons = calculate_weather_impact(
                weather.get("outside_temp_c"),
                weather.get("outside_humidity"),
                weather.get("wind_speed"),
                weather.get("weather_main"),
            )

        result.append(
            DeviceWeatherStatus(
                device_id=device.device_id,
                machine_id=device.machine_id,

                latitude=device.latitude,
                longitude=device.longitude,

                risk_score=(
                    latest_prediction.risk_score
                    if latest_prediction
                    else None
                ),

                risk_level=(
                    latest_prediction.risk_level
                    if latest_prediction
                    else None
                ),

                outside_temp_c=(
                    weather.get("outside_temp_c")
                    if weather
                    else None
                ),

                outside_humidity=(
                    weather.get("outside_humidity")
                    if weather
                    else None
                ),

                outside_pressure=(
                    weather.get("outside_pressure")
                    if weather
                    else None
                ),

                wind_speed=(
                    weather.get("wind_speed")
                    if weather
                    else None
                ),

                weather_main=(
                    weather.get("weather_main")
                    if weather
                    else None
                ),

                environmental_reasons=environmental_reasons,

                weather_impact=weather_impact,
            )
        )

    return result