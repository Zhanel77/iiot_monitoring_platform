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

    def get_for_user(self, user, limit: int = 50):
        query = self.db.query(Prediction)

        if user.role != "admin":
            query = (
                query
                .join(UserDevice, Prediction.machine_id == UserDevice.device_id)
                .filter(UserDevice.user_id == user.id)
            )

        return (
            query
            .order_by(Prediction.event_time.desc())
            .limit(limit)
            .all()
        )