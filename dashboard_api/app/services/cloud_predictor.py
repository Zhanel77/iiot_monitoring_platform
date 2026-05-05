import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import xgboost as xgb

from app.core.config import settings
from app.services.weather_service import WeatherService
from app.services.shap_explainer import ShapExplainer


def normalize_value(value: float, rule: Dict[str, Any]) -> float:
    mode = rule.get("mode")

    if mode == "minmax":
        min_v = rule["min"]
        max_v = rule["max"]
        if max_v == min_v:
            return 0.0
        return (value - min_v) / (max_v - min_v)

    if mode == "ratio":
        max_v = rule["max"]
        if max_v == 0:
            return 0.0
        return value / max_v

    raise ValueError(f"Unsupported normalization mode: {mode}")


def sanitize_feature_name(name: str) -> str:
    name = name.replace("[", "").replace("]", "")
    name = name.replace("(", "").replace(")", "")
    name = name.replace("/", "_")
    name = name.replace("-", "_")
    name = name.replace(" ", "_")
    name = re.sub(r"[^0-9a-zA-Z_]", "", name)
    return name


class CloudPredictor:
    def __init__(self) -> None:
        base_dir = Path(__file__).resolve().parent.parent

        model_path = base_dir / "models" / "ml" / "cloud_model.json"
        feature_list_path = base_dir / "models" / "ml" / "cloud_feature_list.json"
        normalization_path = base_dir / "models" / "ml" / "cloud_normalization_config.json"

        # Загрузка feature list
        with open(feature_list_path, "r", encoding="utf-8") as f:
            self.features = json.load(f)

        # Для совместимости с обоими стилями
        self.feature_list = self.features

        # Загрузка конфигурации нормализации
        with open(normalization_path, "r", encoding="utf-8") as f:
            self.norm = json.load(f)
            self.normalization_config = self.norm

        # Загрузка модели (сохраняем оригинальный тип Booster)
        self.model = xgb.Booster()
        self.model.load_model(str(model_path))

        # Подготовка безопасных имен фич (если нужно для DMatrix)
        self.safe_features = [sanitize_feature_name(f) for f in self.features]
        self.shap_explainer = ShapExplainer()

        # Инициализация weather service с защитой от ошибок
        self.weather_service = None
        try:
            if hasattr(settings, "OPENWEATHER_API_KEY") and settings.OPENWEATHER_API_KEY:
                self.weather_service = WeatherService(settings.OPENWEATHER_API_KEY)
        except Exception as e:
            print(f"[WARN] Failed to initialize WeatherService: {e}")

    def normalize(self, data: dict) -> dict:
        """Нормализация фич согласно конфигурации"""
        result = {}

        for f in self.features:
            val = data.get(f, 0)
            cfg = self.norm.get(f)

            if cfg is not None:
                try:
                    val = normalize_value(float(val), cfg)
                except Exception as e:
                    print(f"[WARN] Failed to normalize feature '{f}': value={val}, cfg={cfg}, error={e}")
                    val = 0.0

            result[f] = val

        return result

    def _normalize_features(self, features: dict) -> dict:
        """Альтернативный метод нормализации для совместимости с новым кодом"""
        return self.normalize(features)

    def predict(self, event: dict, device=None) -> dict:
        """
        Основной метод предсказания
        
        Args:
            event: словарь с данными события, должен содержать 'features_raw'
            device: объект устройства с атрибутами weather_dependent, latitude, longitude
            
        Returns:
            dict: результаты предсказания
        """
        # Извлечение сырых фич
        features_raw = event.get("features_raw", {})

        # Формирование словаря фич (поддерживаем оба формата имен)
        features_used = {
            "Air temperature [K]": float(features_raw.get("Air temperature [K]", 0)),
            "temp_diff": float(features_raw.get("temp_diff", 0)),
            "Rotational speed [rpm]": float(features_raw.get("Rotational speed [rpm]", 0)),
            "Torque [Nm]": float(features_raw.get("Torque [Nm]", 0)),
            "power_kw": float(features_raw.get("power_kw", 0)),
            "Tool wear [min]": float(features_raw.get("Tool wear [min]", 0)),
        }

        # Нормализация фич
        normalized_features = self._normalize_features(features_used)

        # Подготовка входного вектора для модели
        # Используем стандартный список фич
        input_vector = np.array(
            [[normalized_features.get(feature, 0.0) for feature in self.features]]
        )

        # Предсказание с использованием DMatrix (как в оригинале)
        dmatrix = xgb.DMatrix(input_vector, feature_names=self.safe_features)
        risk_score = float(self.model.predict(dmatrix)[0])
        prediction = 1 if risk_score >= 0.5 else 0

        # Weather accounting
        weather_data = None
        weather_factor = 1.0
        weather_explanation = None

        if (
            device is not None
            and self.weather_service is not None
            and getattr(device, "weather_dependent", False)
            and getattr(device, "latitude", None) is not None
            and getattr(device, "longitude", None) is not None
        ):
            try:
                weather_data = self.weather_service.get_current_weather(
                    lat=float(device.latitude),
                    lon=float(device.longitude),
                )

                if weather_data:
                    weather_factor, weather_explanation = self._calculate_weather_factor(
                        weather_data=weather_data,
                        sensitivity=getattr(device, "weather_sensitivity", "medium"),
                    )

                    # Применяем погодный фактор к риску
                    risk_score = min(risk_score * weather_factor, 1.0)
                    prediction = 1 if risk_score >= 0.5 else 0
                    
                    # Добавляем погодные данные в features_used для отладки
                    features_used.update(weather_data)
                    
            except Exception as e:
                print(f"[WARN] Weather service error: {e}")
                weather_data = None

        # Определение уровня риска
        risk_level = self._risk_level(risk_score)

        # Формирование топ факторов влияния
        top_factors = self._build_top_factors(
            features_used=features_used,
            weather_data=weather_data,
            weather_factor=weather_factor,
            weather_explanation=weather_explanation,
        )

        return {
            "prediction": prediction,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "features_used": features_used,
            "top_factors": top_factors,
            "weather_used": weather_data is not None,
            "weather_factor": weather_factor,
        }

    def _calculate_weather_factor(
        self, 
        weather_data: dict, 
        sensitivity: str
    ) -> Tuple[float, Optional[str]]:
        """
        Расчет погодного фактора риска
        
        Args:
            weather_data: словарь с погодными данными
            sensitivity: чувствительность устройства ('low', 'medium', 'high')
            
        Returns:
            tuple: (фактор риска, объяснение)
        """
        factor = 1.0
        reasons = []

        temp = weather_data.get("outside_temp_c", 0)
        humidity = weather_data.get("outside_humidity", 0)
        wind_speed = weather_data.get("wind_speed", 0)

        # Множитель в зависимости от чувствительности
        multiplier = {
            "low": 0.5,
            "medium": 1.0,
            "high": 1.5,
        }.get(sensitivity, 1.0)

        # Высокая температура
        if temp >= 35:
            factor += 0.10 * multiplier
            reasons.append("high outside temperature")

        # Высокая влажность
        if humidity >= 80:
            factor += 0.05 * multiplier
            reasons.append("high outside humidity")

        # Сильный ветер
        if wind_speed >= 12:
            factor += 0.05 * multiplier
            reasons.append("strong wind")

        # Экстремальные условия
        if temp <= -10:
            factor += 0.10 * multiplier
            reasons.append("very low outside temperature")

        # Ограничиваем максимальный фактор
        factor = min(factor, 1.5)

        if not reasons:
            return factor, None

        return factor, ", ".join(reasons)

    def _risk_level(self, risk_score: float) -> str:
        """Определение уровня риска на основе скора"""
        if risk_score >= 0.8:
            return "CRITICAL"
        if risk_score >= 0.6:
            return "HIGH"
        if risk_score >= 0.4:
            return "WARNING"
        return "NORMAL"

    def _build_top_factors(
        self,
        features_used: dict,
        weather_data: Optional[dict],
        weather_factor: float,
        weather_explanation: Optional[str],
    ) -> list:
        """
        Формирование списка топ факторов, влияющих на риск
        """
        factors = []

        # Износ инструмента
        tool_wear = features_used.get("Tool wear [min]", 0)
        if tool_wear >= 180:
            factors.append({
                "feature": "Tool wear [min]",
                "feature_value": tool_wear,
                "effect": "increase",
                "reason": "High tool wear may increase failure risk",
            })
        elif tool_wear >= 120:
            factors.append({
                "feature": "Tool wear [min]",
                "feature_value": tool_wear,
                "effect": "moderate",
                "reason": "Moderate tool wear requires attention",
            })

        # Крутящий момент
        torque = features_used.get("Torque [Nm]", 0)
        if torque >= 55:
            factors.append({
                "feature": "Torque [Nm]",
                "feature_value": torque,
                "effect": "increase",
                "reason": "High torque may indicate mechanical load",
            })

        # Температурная разница
        temp_diff = features_used.get("temp_diff", 0)
        if temp_diff >= 12:
            factors.append({
                "feature": "temp_diff",
                "feature_value": temp_diff,
                "effect": "increase",
                "reason": "High temperature difference may indicate thermal stress",
            })
        elif temp_diff >= 8:
            factors.append({
                "feature": "temp_diff",
                "feature_value": temp_diff,
                "effect": "moderate",
                "reason": "Elevated temperature difference may cause thermal stress",
            })

        # Скорость вращения
        rpm = features_used.get("Rotational speed [rpm]", 0)
        if rpm >= 3000:
            factors.append({
                "feature": "Rotational speed [rpm]",
                "feature_value": rpm,
                "effect": "increase",
                "reason": "High rotational speed increases wear and failure risk",
            })

        # Температура воздуха
        air_temp = features_used.get("Air temperature [K]", 0)
        if air_temp > 310:  # > 37°C
            factors.append({
                "feature": "Air temperature [K]",
                "feature_value": air_temp,
                "effect": "increase",
                "reason": "High operating temperature may reduce component life",
            })

        # Погодные факторы
        if weather_data and weather_factor > 1.0 and weather_explanation:
            factors.append({
                "feature": "weather_context",
                "feature_value": round(weather_factor, 2),
                "effect": "increase",
                "reason": f"Weather increased risk due to {weather_explanation}",
            })

        # Сортируем по важности (эффект увеличения риска важнее)
        effect_priority = {"increase": 0, "moderate": 1}
        factors.sort(key=lambda x: effect_priority.get(x.get("effect", "moderate"), 2))

        return factors[:4]  # Возвращаем топ-4 фактора

    # Legacy метод для обратной совместимости со старым API
    def predict_legacy(self, data: dict) -> dict:
        """
        Старый метод predict для обратной совместимости
        """
        normalized = self.normalize(data)

        row = {sanitize_feature_name(f): normalized[f] for f in self.features}
        X = np.array([[row[f] for f in self.safe_features]], dtype=float)

        dmatrix = xgb.DMatrix(X, feature_names=self.safe_features)
        prob = float(self.model.predict(dmatrix)[0])

        prediction = int(prob > 0.5)
        risk_level = self._level(prob)

        top_factors = self.shap_explainer.explain(
            normalized_features=normalized,
            raw_features=data,
            top_k=5,
        )

        return {
            "risk_score": prob,
            "prediction": prediction,
            "risk_level": risk_level,
            "features_used": data,
            "top_factors": top_factors,
        }

    def _level(self, score: float):
        if score > 0.8:
            return "CRITICAL"
        if score > 0.6:
            return "HIGH"
        if score > 0.4:
            return "WARNING"
        return "NORMAL"
