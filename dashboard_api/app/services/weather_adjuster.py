import math
from typing import Optional
from app.services.weather_service import WeatherData

class WeatherAdjuster:
    @staticmethod
    def calculate_weather_factor(
        weather: Optional[WeatherData],
        weather_sensitivity: str,  # "low", "medium", "high"
    ) -> tuple[float, Optional[dict]]:
        """
        Возвращает (weather_factor, explanation)
        """
        if not weather:
            return 1.0, None
        
        factor = 1.0
        reasons = []
        
        # Температура
        if weather.outside_temp_c >= 35:
            factor += 0.10
            reasons.append("high outside temperature")
        elif weather.outside_temp_c <= 0:
            factor += 0.05
            reasons.append("freezing temperature")
        
        # Влажность
        if weather.outside_humidity >= 80:
            factor += 0.05
            reasons.append("high humidity")
        elif weather.outside_humidity <= 20:
            factor += 0.03
            reasons.append("very dry conditions")
        
        # Ветер
        if weather.wind_speed >= 12:
            factor += 0.05
            reasons.append("strong wind")
        
        # Учет чувствительности устройства
        sensitivity_mapping = {
            "low": 0.3,
            "medium": 1.0,
            "high": 2.0
        }
        sensitivity_multiplier = sensitivity_mapping.get(weather_sensitivity, 1.0)
        
        # Применяем sensitivity
        final_factor = 1.0 + (factor - 1.0) * sensitivity_multiplier
        
        # Логируем причину корректировки
        explanation = None
        if final_factor > 1.0:
            explanation = {
                "feature": "weather_context",
                "feature_value": round(final_factor, 3),
                "effect": "increase",
                "reason": f"Weather increased risk due to {', '.join(reasons)}",
                "sensitivity": weather_sensitivity
            }
        elif final_factor < 1.0:
            explanation = {
                "feature": "weather_context",
                "feature_value": round(final_factor, 3),
                "effect": "decrease",
                "reason": f"Weather decreased risk",
                "sensitivity": weather_sensitivity
            }
        
        return round(final_factor, 3), explanation