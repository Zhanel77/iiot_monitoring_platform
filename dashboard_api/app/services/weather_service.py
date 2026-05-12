import httpx
from typing import Optional
from pydantic import BaseModel
from app.core.config import settings  # нужно добавить OPENWEATHER_API_KEY

class WeatherData(BaseModel):
    outside_temp_c: float
    outside_humidity: float
    outside_pressure: float
    wind_speed: float
    weather_main: str

class WeatherService:
    def __init__(self, api_key: str | None):
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"

    async def get_weather(self, lat: float, lon: float) -> Optional[WeatherData]:
        """Получает текущую погоду по координатам"""
        if not self.api_key:
            return None
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    self.base_url,
                    params={
                        "lat": lat,
                        "lon": lon,
                        "appid": self.api_key,
                        "units": "metric"  # для температуры в Цельсиях
                    },
                    timeout=5.0
                )
                response.raise_for_status()
                data = response.json()
                
                return WeatherData(
                    outside_temp_c=data["main"]["temp"],
                    outside_humidity=data["main"]["humidity"],
                    outside_pressure=data["main"]["pressure"],
                    wind_speed=data["wind"]["speed"],
                    weather_main=data["weather"][0]["main"]
                )
            except Exception as e:
                # Логируем ошибку, но не падаем
                print(f"Weather API error: {e}")
                return None

    def get_current_weather(self, lat: float, lon: float) -> dict | None:
        if not self.api_key:
            return None

        url = "https://api.openweathermap.org/data/2.5/weather"

        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "metric",
        }

        try:
            response = httpx.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()

            return {
                "outside_temp_c": float(data["main"]["temp"]),
                "outside_humidity": float(data["main"]["humidity"]),
                "outside_pressure": float(data["main"]["pressure"]),
                "wind_speed": float(data["wind"]["speed"]),
                "weather_main": data["weather"][0]["main"],
            }

        except Exception:
            return None
    