from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.repositories.device_repository import DeviceRepository
from app.services.cloud_predictor import CloudPredictor

router = APIRouter(prefix="/cloud", tags=["cloud"])

cloud_predictor = CloudPredictor()


@router.post("/predict")
def predict_cloud(payload: dict, db: Session = Depends(get_db)):
    machine_id = payload.get("machine_id")

    device = None
    if machine_id is not None:
        device = DeviceRepository(db).get_by_machine_id(machine_id)

    result = cloud_predictor.predict(payload, device=device)

    return result