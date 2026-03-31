import json
import paho.mqtt.client as mqtt
from app.schemas import EdgePredictionMessage


def publish_prediction(client: mqtt.Client, topic: str, message: EdgePredictionMessage) -> None:
    payload = message.model_dump_json()
    result = client.publish(topic, payload)

    if result.rc != mqtt.MQTT_ERR_SUCCESS:
        raise RuntimeError(f"Failed to publish prediction to topic={topic}, rc={result.rc}")