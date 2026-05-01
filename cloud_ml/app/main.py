from fastapi import FastAPI, HTTPException

from app.config import (
    ENABLE_SHAP_FOR_ALL,
    FEATURE_LIST_PATH,
    HIGH_RISK_THRESHOLD,
    MODEL_PATH,
    NORMALIZATION_CONFIG_PATH,
    SHAP_ONLY_FOR_HIGH_RISK,
    TOP_K_FEATURES,
)
from app.shap_explainer import CloudShapExplainer
from app.predictor import CloudPredictor
from app.preprocessing import CloudPreprocessor
from app.schemas import CloudPredictRequest, CloudPredictResponse, ShapFactor

app = FastAPI(title="Cloud ML Service with SHAP")


preprocessor = CloudPreprocessor(
    feature_list_path=FEATURE_LIST_PATH,
    normalization_config_path=NORMALIZATION_CONFIG_PATH,
)

predictor = CloudPredictor(model_path=MODEL_PATH)

explainer = CloudShapExplainer(
    model_path=MODEL_PATH,
    feature_names=preprocessor.feature_list,
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=CloudPredictResponse)
def predict_cloud(payload: CloudPredictRequest):
    try:
        features, features_used = preprocessor.prepare_features(payload.model_dump())
        prediction, risk_score = predictor.predict(features)

        prediction_label = "HIGH_RISK" if prediction == 1 else "NORMAL"

        should_explain = False

        if ENABLE_SHAP_FOR_ALL:
            should_explain = True
        elif SHAP_ONLY_FOR_HIGH_RISK and risk_score >= HIGH_RISK_THRESHOLD:
            should_explain = True

        top_factors = []
        if should_explain:
            shap_rows = explainer.explain(
                features=features,
                feature_values=features_used,
                top_k=TOP_K_FEATURES,
            )
            top_factors = [ShapFactor(**row) for row in shap_rows]

        return CloudPredictResponse(
            prediction=prediction,
            prediction_label=prediction_label,
            risk_score=risk_score,
            top_factors=top_factors,
            features_used=features_used,
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))