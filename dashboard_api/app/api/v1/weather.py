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
) -> str:
    score = 0

    if temp and temp > 32:
        score += 2
    elif temp and temp > 26:
        score += 1

    if humidity and humidity > 80:
        score += 1

    if wind and wind > 10:
        score += 1

    if score >= 3:
        return "HIGH"

    if score >= 1:
        return "MEDIUM"

    return "LOW"


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

        if weather:
            weather_impact = calculate_weather_impact(
                weather.get("outside_temp_c"),
                weather.get("outside_humidity"),
                weather.get("wind_speed"),
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

                weather_impact=weather_impact,
            )
        )

    return result