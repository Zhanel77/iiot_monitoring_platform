from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.alert import Alert
from app.models.prediction import Prediction

router = APIRouter(prefix="/api/v1/alerts", tags=["alerts"])


@router.get("/open")
def get_open_alerts(db: Session = Depends(get_db)):
    return (
        db.query(Alert)
        .filter(Alert.status == "open")
        .order_by(Alert.created_at.desc())
        .limit(20)
        .all()
    )


@router.get("/{alert_id}")
def get_alert_detail(alert_id: int, db: Session = Depends(get_db)):
    alert = db.query(Alert).filter(Alert.id == alert_id).first()

    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    prediction = None

    if alert.prediction_id:
        prediction = (
            db.query(Prediction)
            .filter(Prediction.id == alert.prediction_id)
            .first()
        )

    cloud_prediction = (
        db.query(Prediction)
        .filter(Prediction.machine_id == alert.machine_id)
        .filter(Prediction.model_type == "cloud")
        .filter(Prediction.top_factors.isnot(None))
        .order_by(Prediction.id.desc())
        .first()
    )

    return {
        "id": alert.id,
        "device_id": alert.device_id,
        "machine_id": alert.machine_id,
        "alert_type": alert.alert_type,
        "severity": alert.severity,
        "message": alert.message,
        "status": alert.status,
        "created_at": alert.created_at,
        "prediction_id": alert.prediction_id,

        "risk_score": prediction.risk_score if prediction else None,
        "risk_level": prediction.risk_level if prediction else None,
        "model_type": prediction.model_type if prediction else None,

        "features_used": cloud_prediction.features_used if cloud_prediction else None,
        "top_factors": cloud_prediction.top_factors if cloud_prediction else None,
        "shap_model_type": "cloud" if cloud_prediction else None,
    }