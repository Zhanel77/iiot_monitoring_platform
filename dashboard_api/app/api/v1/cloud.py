from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.repositories.device_repository import DeviceRepository
from app.services.cloud_predictor import CloudPredictor as CloudOldPredictor

from app.core.config import settings
from app.services.cloud_ml.preprocessing import CloudPreprocessor
from app.services.cloud_ml.predictor import CloudPredictor
from app.services.cloud_ml.shap_explainer import CloudShapExplainer
from app.services.cloud_ml.schemas import CloudPredictRequest, CloudPredictResponse

router = APIRouter(prefix="/cloud", tags=["cloud"])

cloud_predictor = CloudOldPredictor()


@router.post("/predict/legacy")
def predict_cloud(payload: dict, db: Session = Depends(get_db)):
    machine_id = payload.get("machine_id")

    device = None
    if machine_id is not None:
        device = DeviceRepository(db).get_by_machine_id(machine_id)

    result = cloud_predictor.predict(payload, device=device)

    return result

preprocessor = CloudPreprocessor(
    feature_list_path=settings.FEATURE_LIST_PATH,
    normalization_config_path=settings.NORMALIZATION_CONFIG_PATH,
)

predictor = CloudPredictor(
    model_path=settings.MODEL_PATH,
    feature_names=preprocessor.feature_list,
)

explainer = CloudShapExplainer(
    model_path=settings.MODEL_PATH,
    feature_names=preprocessor.feature_list,
)


@router.get("/health")
def cloud_health():
    return {"status": "ok", "service": "cloud_ml_inside_dashboard_api"}


@router.post("/predict", response_model=CloudPredictResponse)
def cloud_predict(payload: CloudPredictRequest):
    try:
        features, features_used = preprocessor.prepare_features(
            payload.model_dump()
        )

        prediction, risk_score = predictor.predict(features)

        should_explain = settings.ENABLE_SHAP_FOR_ALL or (
            settings.SHAP_ONLY_FOR_HIGH_RISK
            and risk_score >= settings.HIGH_RISK_THRESHOLD
        )

        top_factors = []

        if should_explain:
            top_factors = explainer.explain(
                features=features,
                feature_values=features_used,
                top_k=settings.TOP_K_FEATURES,
            )

        prediction_label = "HIGH_RISK" if prediction == 1 else "NORMAL"

        return CloudPredictResponse(
            prediction=prediction,
            prediction_label=prediction_label,
            risk_score=risk_score,
            top_factors=top_factors,
            features_used=features_used,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))