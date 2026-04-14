from pydantic import BaseModel


class DeviceCreate(BaseModel):
    machine_id: int
    device_id: str
    name: str | None = None


class DeviceUpdate(BaseModel):
    name: str | None = None
    status: str | None = None


class DeviceRead(BaseModel):
    id: int
    machine_id: int
    device_id: str
    name: str | None
    status: str

    model_config = {"from_attributes": True}