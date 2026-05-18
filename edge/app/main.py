import json
import logging
from typing import Any

import paho.mqtt.client as mqtt

from app.adapter import adapt_raw_message
from app.config import settings
from app.config_loader import load_normalization_config
from app.feature_engineering import enrich_features, normalize_features
from app.predictor import EdgePredictor, resolve_prediction_label, resolve_risk_level
from app.publisher import publish_prediction
from app.schemas import RawMQTTMessage, EdgePredictionMessage


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("edge-service")


normalization_config = load_normalization_config(settings.NORMALIZATION_CONFIG_PATH)
predictor = EdgePredictor(
    model_path=settings.MODEL_PATH,
    feature_order=settings.MODEL_FEATURE_ORDER
)


def process_payload(payload: str) -> EdgePredictionMessage:
    raw_dict: Any = json.loads(payload)
    raw_message = RawMQTTMessage(**raw_dict)

    adapted = adapt_raw_message(raw_message)

    enriched_features = enrich_features(adapted.features)

    # Убираем Process temperature [K], потому что модели он не нужен напрямую
    features_for_model = {
        "Air temperature [K]": enriched_features["Air temperature [K]"],
        "temp_diff": enriched_features["temp_diff"],
        "Rotational speed [rpm]": enriched_features["Rotational speed [rpm]"],
        "Torque [Nm]": enriched_features["Torque [Nm]"],
        "power_kw": enriched_features["power_kw"],
        "Tool wear [min]": enriched_features["Tool wear [min]"],
    }

    normalized_features = normalize_features(features_for_model, normalization_config)

    prediction, risk_score = predictor.predict(normalized_features)

    risk_level = resolve_risk_level(
        risk_score=risk_score,
        warning_threshold=settings.WARNING_THRESHOLD,
        critical_threshold=settings.CRITICAL_THRESHOLD
    )

    prediction_label = resolve_prediction_label(prediction)

    result = EdgePredictionMessage(
        device_id=adapted.device_id,
        machine_id=adapted.machine_id,
        timestamp=adapted.timestamp,
        source=adapted.source,
        scenario=adapted.scenario,
        prediction=prediction,
        prediction_label=prediction_label,
        risk_score=round(risk_score, 4),
        risk_level=risk_level,
        model_name="edge_model.pkl",
        features_raw=features_for_model,
        features_normalized=normalized_features,
        ground_truth=adapted.ground_truth,
    )

    return result


def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        logger.info("Connected to MQTT broker")
        client.subscribe(settings.MQTT_INPUT_TOPIC)
        logger.info("Subscribed to topic: %s", settings.MQTT_INPUT_TOPIC)
    else:
        logger.error("Failed to connect to MQTT broker. rc=%s", rc)


def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode("utf-8")
        logger.info("Received message from %s: %s", msg.topic, payload)

        result = process_payload(payload)

        publish_prediction(client, settings.MQTT_OUTPUT_TOPIC, result)

        logger.info(
            "Prediction published | machine_id=%s prediction=%s risk_score=%.4f risk_level=%s",
            result.machine_id,
            result.prediction,
            result.risk_score,
            result.risk_level
        )
    except Exception as e:
        logger.exception("Failed to process MQTT message: %s", e)


def main():
    client = mqtt.Client(client_id=settings.MQTT_CLIENT_ID, callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message

    logger.info(
        "Connecting to MQTT broker %s:%s ...",
        settings.MQTT_BROKER_HOST,
        settings.MQTT_BROKER_PORT
    )

    client.connect(settings.MQTT_BROKER_HOST, settings.MQTT_BROKER_PORT, keepalive=60)
    client.loop_forever()


if __name__ == "__main__":
    main()