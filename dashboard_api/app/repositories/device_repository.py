from sqlalchemy.orm import Session
from app.models.device import Device
from app.models.user_device import UserDevice


class DeviceRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_all(self):
        return self.db.query(Device).all()

    def create(self, machine_id: int, device_id: str, name: str | None):
        device = Device(
            machine_id=machine_id,
            device_id=device_id,
            name=name,
        )
        self.db.add(device)
        self.db.commit()
        self.db.refresh(device)
        return device
    
    def get_by_machine_id(self, machine_id: int):
        return self.db.query(Device).filter(Device.machine_id == machine_id).first()


    def get_by_device_id(self, device_id: str):
        return self.db.query(Device).filter(Device.device_id == device_id).first()

    def update(self, device: Device, data: dict):
        for key, value in data.items():
            setattr(device, key, value)

        self.db.commit()
        self.db.refresh(device)
        return device

    def soft_delete(self, device: Device):
        device.status = "inactive"
        self.db.commit()
        return device

    def get_for_user(self, user):
        if user.role == "admin":
            return self.db.query(Device).all()

        return (
            self.db.query(Device)
            .join(UserDevice, Device.id == UserDevice.device_id)
            .filter(UserDevice.user_id == user.id)
            .all()
        )