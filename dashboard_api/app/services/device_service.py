# app/services/device_service.py
from typing import Optional, Tuple
from sqlalchemy.orm import Session
from app.repositories.device_repository import DeviceRepository
from app.schemas.device import DeviceWeatherConfig


class DeviceService:
    """Сервис для работы с устройствами - расширен для weather-aware функциональности"""
    
    def __init__(self, db: Session):
        self.repository = DeviceRepository(db)
    
    def get_weather_config(self, device_id: str) -> Optional[DeviceWeatherConfig]:
        """
        Получает weather-конфигурацию устройства.
        Возвращает None если устройство не найдено или не зависит от погоды.
        """
        device = self.repository.get_by_device_id(device_id)
        print(f"SEARCH DEVICE ID: '{device_id}'")
        if not device:
            return None
        
        # Проверяем, зависит ли устройство от погоды
        if not device.weather_dependent:
            return None
        
        # Проверяем наличие координат
        if device.latitude is None or device.longitude is None:
            return None
        
        return DeviceWeatherConfig(
            device_id=device.device_id,
            machine_id=device.machine_id,
            weather_dependent=device.weather_dependent,
            latitude=device.latitude,
            longitude=device.longitude,
            weather_sensitivity=device.weather_sensitivity or "medium",
            device_name=device.name,
        )
    
    def get_device_with_weather(self, device_id: str) -> Optional[dict]:
        """Получает полную информацию об устройстве включая weather-настройки"""
        device = self.repository.get_by_device_id(device_id)
        # print("FOUND DEVICE:", device.device_id if device else None, flush=True)
        # print("WEATHER DEP:", getattr(device, "weather_dependent", None), flush=True)
        # print("LAT:", getattr(device, "latitude", None), flush=True)
        # print("LON:", getattr(device, "longitude", None), flush=True)
        if not device:
            return None
        
        return {
            "id": device.id,
            "machine_id": device.machine_id,
            "device_id": device.device_id,
            "name": device.name,
            "status": device.status,
            "weather_dependent": device.weather_dependent,
            "latitude": device.latitude,
            "longitude": device.longitude,
            "weather_sensitivity": device.weather_sensitivity or "medium",
        }
    
    def is_weather_dependent(self, device_id: str) -> bool:
        """Проверяет, зависит ли устройство от погоды"""
        device = self.repository.get_by_device_id(device_id)
        return device is not None and device.weather_dependent == True
    
    def get_sensitivity(self, device_id: str) -> str:
        """Возвращает чувствительность устройства к погоде"""
        device = self.repository.get_by_device_id(device_id)
        if not device:
            return "medium"  # значение по умолчанию
        
        return device.weather_sensitivity or "medium"
    
    def get_coordinates(self, device_id: str) -> Optional[Tuple[float, float]]:
        """Возвращает координаты устройства (lat, lon)"""
        device = self.repository.get_by_device_id(device_id)
        if device and device.latitude and device.longitude:
            return (device.latitude, device.longitude)
        return None
    
    def update_weather_settings(
        self, 
        device_id: str, 
        weather_dependent: Optional[bool] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        weather_sensitivity: Optional[str] = None
    ) -> Optional[dict]:
        """Обновляет weather-настройки устройства"""
        device = self.repository.get_by_device_id(device_id)
        if not device:
            return None
        
        update_data = {}
        
        if weather_dependent is not None:
            update_data["weather_dependent"] = weather_dependent
        
        if latitude is not None:
            update_data["latitude"] = latitude
        
        if longitude is not None:
            update_data["longitude"] = longitude
        
        if weather_sensitivity is not None:
            if weather_sensitivity not in ["low", "medium", "high"]:
                raise ValueError("weather_sensitivity must be 'low', 'medium', or 'high'")
            update_data["weather_sensitivity"] = weather_sensitivity
        
        if update_data:
            device = self.repository.update(device, update_data)
        
        return self.get_device_with_weather(device_id)
    
    def get_all_weather_dependent_devices(self) -> list[DeviceWeatherConfig]:
        """Получает все устройства, которые зависят от погоды и имеют координаты"""
        all_devices = self.repository.get_all()
        
        weather_devices = []
        for device in all_devices:
            if device.weather_dependent and device.latitude and device.longitude:
                weather_devices.append(DeviceWeatherConfig(
                    device_id=device.device_id,
                    machine_id=device.machine_id,
                    weather_dependent=device.weather_dependent,
                    latitude=device.latitude,
                    longitude=device.longitude,
                    weather_sensitivity=device.weather_sensitivity or "medium",
                    device_name=device.name,
                ))
        
        return weather_devices