from sqlalchemy.orm import Session
from app.models.device import Device
from app.models.user_device import UserDevice


class DeviceRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_all(self):
        return self.db.query(Device).all()

    def get_for_user(self, user):
        if user.role == "admin":
            return self.get_all()

        return (
            self.db.query(Device)
            .join(UserDevice, Device.id == UserDevice.device_id)
            .filter(UserDevice.user_id == user.id)
            .all()
        )