from fastapi import APIRouter
from app.services.cloud_predictor import CloudPredictor

router = APIRouter(prefix="/cloud", tags=["cloud"])

predictor = CloudPredictor()


@router.post("/predict")
def predict(data: dict):
    return predictor.predict(data)