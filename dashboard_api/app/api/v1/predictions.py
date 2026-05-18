from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.core.dependencies import get_current_user, require_admin
from app.repositories.prediction_repository import PredictionRepository
from app.schemas.prediction import PredictionCreate, PredictionRead

router = APIRouter(prefix="/predictions", tags=["predictions"])


from fastapi import Query

@router.get("", response_model=list[PredictionRead])
def get_predictions(
    limit: int = Query(50, ge=1, le=1000),
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    repo = PredictionRepository(db)
    return repo.get_for_user(user, limit=limit)


@router.post("", response_model=PredictionRead)
def create_prediction(
    payload: PredictionCreate,
    db: Session = Depends(get_db),
    _: dict = Depends(require_admin),
):
    repo = PredictionRepository(db)
    return repo.create(payload.dict())