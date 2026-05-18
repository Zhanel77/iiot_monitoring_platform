from sqlalchemy.orm import Session
from app.models.user_device import UserDevice


class UserDeviceRepository:
    def __init__(self, db: Session):
        self.db = db

    def assign(self, user_id: int, device_id: int):
        obj = UserDevice(user_id=user_id, device_id=device_id)
        self.db.add(obj)
        self.db.commit()
        return obj

    def remove(self, user_id: int, device_id: int):
        self.db.query(UserDevice).filter(
            UserDevice.user_id == user_id,
            UserDevice.device_id == device_id
        ).delete()
        self.db.commit()

    def get_user_devices(self, user_id: int):
        return self.db.query(UserDevice).filter(
            UserDevice.user_id == user_id
        ).all()