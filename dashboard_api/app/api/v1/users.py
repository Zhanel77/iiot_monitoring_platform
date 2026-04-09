from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.dependencies import require_admin
from app.db.session import get_db
from app.models.user import User
from app.repositories.user_repository import UserRepository
from app.schemas.user import UserRead, UserUpdateRoleRequest, UserUpdateStatusRequest

router = APIRouter(prefix="/users", tags=["users"])


@router.get("", response_model=list[UserRead])
def list_users(
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    return UserRepository(db).list_users()


@router.patch("/{user_id}/role", response_model=UserRead)
def update_user_role(
    user_id: int,
    payload: UserUpdateRoleRequest,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    allowed_roles = {"admin", "operator", "viewer"}
    if payload.role not in allowed_roles:
        raise HTTPException(status_code=400, detail="Invalid role")

    repo = UserRepository(db)
    user = repo.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return repo.update_role(user, payload.role)


@router.patch("/{user_id}/status", response_model=UserRead)
def update_user_status(
    user_id: int,
    payload: UserUpdateStatusRequest,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    repo = UserRepository(db)
    user = repo.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return repo.update_status(user, payload.is_active)