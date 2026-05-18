from fastapi import APIRouter
from app.api.v1.auth import router as auth_router
from app.api.v1.users import router as users_router
from app.api.v1.devices import router as devices_router
from app.api.v1.predictions import router as predictions_router
from app.api.v1.ws import router as ws_router
from app.api.v1.ws_predictions import router as ws_predictions_router
from app.api.v1.cloud import router as cloud_router
from app.api.v1.weather import router as weather_router


api_router = APIRouter(prefix="/api/v1")
api_router.include_router(auth_router)
api_router.include_router(users_router)
api_router.include_router(devices_router)
api_router.include_router(predictions_router)
api_router.include_router(ws_router)
api_router.include_router(ws_predictions_router)
api_router.include_router(cloud_router)
api_router.include_router(weather_router)