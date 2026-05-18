# app/api/v1/cloud.py
import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.repositories.device_repository import DeviceRepository
from app.services.cloud_predictor import CloudPredictor as CloudOldPredictor

from app.core.config import settings
from app.services.cloud_ml.preprocessing import CloudPreprocessor
from app.services.cloud_ml.predictor import CloudPredictor
from app.services.cloud_ml.shap_explainer import CloudShapExplainer
from app.services.cloud_ml.schemas import CloudPredictRequest, CloudPredictResponse, ShapFactor
from app.services.weather_adjuster import WeatherAdjuster
from app.services.device_service import DeviceService
from app.services.weather_service import WeatherService

logger = logging.getLogger(__name__)

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


@router.get("/health")
def cloud_health():
    return {"status": "ok", "service": "cloud_ml_inside_dashboard_api"}


# Инициализация ML компонентов с обработкой ошибок
try:
    logger.info("Initializing ML components...")
    logger.info(f"FEATURE_LIST_PATH: {settings.FEATURE_LIST_PATH}")
    logger.info(f"NORMALIZATION_CONFIG_PATH: {settings.NORMALIZATION_CONFIG_PATH}")
    logger.info(f"MODEL_PATH: {settings.MODEL_PATH}")
    
    preprocessor = CloudPreprocessor(
        feature_list_path=settings.FEATURE_LIST_PATH,
        normalization_config_path=settings.NORMALIZATION_CONFIG_PATH,
    )
    logger.info(f"Preprocessor initialized. Feature list: {preprocessor.feature_list}")
    
    predictor = CloudPredictor(
        model_path=settings.MODEL_PATH,
        feature_names=preprocessor.feature_list,
    )
    logger.info("Predictor initialized successfully")
    
    explainer = CloudShapExplainer(
        model_path=settings.MODEL_PATH,
        feature_names=preprocessor.feature_list,
    )
    logger.info("Explainer initialized successfully")
    
except Exception as e:
    logger.error(f"Failed to initialize ML components: {e}", exc_info=True)
    preprocessor = None
    predictor = None
    explainer = None


@router.post("/predict", response_model=CloudPredictResponse)
async def cloud_predict(
    payload: CloudPredictRequest,
    db: Session = Depends(get_db)
):
    try:
        logger.info(f"=== Cloud prediction request ===")
        logger.info(f"Device ID: {payload.device_id}")
        logger.info(f"Input data: air_temp={payload.air_temperature_k}, process_temp={payload.process_temperature_k}, speed={payload.rotational_speed_rpm}, torque={payload.torque_nm}, wear={payload.tool_wear_min}")
        
        # Проверка инициализации ML компонентов
        if predictor is None:
            logger.error("ML components not initialized")
            raise HTTPException(status_code=503, detail="ML model not loaded. Check logs for details.")
        
        # 1. Получаем weather конфигурацию устройства
        logger.info(f"Getting weather config for device: {payload.device_id}")
        device_service = DeviceService(db)
        logger.info(f"Payload device_id: '{payload.device_id}'")
        weather_config = device_service.get_weather_config(payload.device_id)
        
        logger.info(f"Weather config: {weather_config}")
        
        # 2. Получаем погоду если нужно
        weather_data = None
        weather_factor = 1.0
        weather_explanation = None
        
        if (
            weather_config
            and weather_config.weather_dependent
            and weather_config.latitude is not None
            and weather_config.longitude is not None
        ):
            logger.info(
                f"Getting weather for lat={weather_config.latitude}, "
                f"lon={weather_config.longitude}"
            )

            try:
                weather_service = WeatherService(settings.OPENWEATHER_API_KEY)

                weather_data = await weather_service.get_weather(
                    lat=weather_config.latitude,
                    lon=weather_config.longitude,
                )

                logger.info(f"Weather data received: {weather_data}")

                if weather_data:
                    weather_factor, weather_explanation = (
                        WeatherAdjuster.calculate_weather_factor(
                            weather=weather_data,
                            weather_sensitivity=weather_config.weather_sensitivity or "medium",
                        )
                    )

            except Exception as e:
                logger.error(f"Error getting weather data: {e}", exc_info=True)

        else:
            logger.info("Weather config missing or coordinates are empty")
        # 3. Подготовка признаков для ML модели
        input_data = {
            "air_temperature_k": payload.air_temperature_k,
            "process_temperature_k": payload.process_temperature_k,
            "rotational_speed_rpm": payload.rotational_speed_rpm,
            "torque_nm": payload.torque_nm,
            "tool_wear_min": payload.tool_wear_min,
        }
        
        logger.info(f"Preparing features from input: {input_data}")
        features, features_used = preprocessor.prepare_features(input_data)
        logger.info(f"Features prepared: {features}")
        logger.info(f"Features used: {features_used}")
        
        # 4. ML предсказание
        logger.info("Calling ML predictor...")
        ml_prediction, ml_risk_score = predictor.predict(features)
        logger.info(f"ML prediction: prediction={ml_prediction}, risk_score={ml_risk_score}")
        
        # 5. Применяем weather фактор
        final_risk_score = ml_risk_score * weather_factor
        final_risk_score = min(1.0, max(0.0, final_risk_score))
        final_prediction = 1 if final_risk_score >= 0.5 else 0
        
        logger.info(f"Final result after weather adjustment: prediction={final_prediction}, risk_score={final_risk_score}, weather_factor={weather_factor}")
        
        # 6. SHAP объяснение для ML части
        should_explain = settings.ENABLE_SHAP_FOR_ALL or (
            settings.SHAP_ONLY_FOR_HIGH_RISK
            and final_risk_score >= settings.HIGH_RISK_THRESHOLD
        )
        
        top_factors = []
        if should_explain:
            logger.info(f"Generating SHAP explanations (threshold: {settings.HIGH_RISK_THRESHOLD})")
            top_factors = explainer.explain(
                features=features,
                feature_values=features_used,
                top_k=settings.TOP_K_FEATURES,
            )
            top_factors = [ShapFactor(**factor) for factor in top_factors]
            logger.info(f"SHAP factors: {top_factors}")
        
        # 7. Добавляем weather explanation если применимо
        if weather_explanation:
            logger.info("Adding weather explanation to top factors")
            top_factors.append(ShapFactor(**weather_explanation))
            top_factors.sort(
                key=lambda x: abs(x.shap_value) if x.shap_value else abs(x.feature_value),
                reverse=True
            )
            top_factors = top_factors[:settings.TOP_K_FEATURES]
        
        prediction_label = "CRITICAL" if final_prediction == 1 else "NORMAL"
        
        response = CloudPredictResponse(
            prediction=final_prediction,
            prediction_label=prediction_label,
            risk_score=final_risk_score,
            ml_risk_score=ml_risk_score,
            weather_factor=weather_factor if weather_factor != 1.0 else None,
            top_factors=top_factors,
            features_used={
                **features_used,
                "weather_applied": weather_factor != 1.0,
                "weather_dependent": weather_config is not None,
                "weather_sensitivity": weather_config.weather_sensitivity if weather_config else None
            }
        )
        
        logger.info(f"Response prepared successfully")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in cloud prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")