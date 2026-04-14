from sqlalchemy.orm import Session
from app.models.prediction import Prediction
from app.models.user_device import UserDevice


class PredictionRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, data: dict):
        obj = Prediction(**data)
        self.db.add(obj)
        self.db.commit()
        self.db.refresh(obj)
        return obj

    def get_for_user(self, user):
        if user.role == "admin":
            return self.db.query(Prediction).all()

        return (
            self.db.query(Prediction)
            .join(UserDevice, Prediction.machine_id == UserDevice.device_id)
            .filter(UserDevice.user_id == user.id)
            .all()
        )