from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.core.dependencies import get_current_user, require_admin
from app.repositories.device_repository import DeviceRepository
from app.schemas.device import DeviceCreate, DeviceUpdate, DeviceRead
from app.models.user import User

router = APIRouter(prefix="/devices", tags=["devices"])

@router.get("", response_model=list[DeviceRead])
def get_devices(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    repo = DeviceRepository(db)
    return repo.get_for_user(user)

@router.post("", response_model=DeviceRead)
def create_device(
    payload: DeviceCreate,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    repo = DeviceRepository(db)

    if repo.get_by_machine_id(payload.machine_id):
        raise HTTPException(
            status_code=400,
            detail="Device with this machine_id already exists"
        )

    if repo.get_by_device_id(payload.device_id):
        raise HTTPException(
            status_code=400,
            detail="Device with this device_id already exists"
        )

    return repo.create(
        machine_id=payload.machine_id,
        device_id=payload.device_id,
        name=payload.name,
    )


@router.patch("/{device_id}", response_model=DeviceRead)
def update_device(
    device_id: int,
    payload: DeviceUpdate,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    repo = DeviceRepository(db)
    device = repo.get_by_device_id(device_id)

    if not device:
        raise HTTPException(status_code=404, detail="Device not found")

    return repo.update(device, payload.dict(exclude_unset=True))


@router.delete("/{device_id}")
def delete_device(
    device_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    repo = DeviceRepository(db)
    device = repo.get_by_device_id(device_id)

    if not device:
        raise HTTPException(status_code=404, detail="Device not found")

    repo.soft_delete(device)
    return {"message": "Device deactivated"}