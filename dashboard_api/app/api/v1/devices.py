from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.core.dependencies import get_current_user
from app.repositories.device_repository import DeviceRepository

router = APIRouter(prefix="/devices", tags=["devices"])


@router.get("")
def get_devices(
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    repo = DeviceRepository(db)
    return repo.get_for_user(user)