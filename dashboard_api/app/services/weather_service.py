import httpx


class WeatherService:
    def __init__(self, api_key: str | None):
        self.api_key = api_key

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