import json
import math
import time
from typing import Any

import paho.mqtt.client as mqtt

from app.config import Settings
from app.influx_writer import InfluxWriter
from app.logger import setup_logger
from app.postgres_writer import PostgresWriter


logger = setup_logger()


class MQTTConsumer:
    def __init__(self) -> None:
        self.client = mqtt.Client(
            client_id=f"{Settings.MQTT_CLIENT_ID}-{time.time()}"
        )   
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        self.pg = PostgresWriter()
        self.influx = InfluxWriter()

    def on_connect(
        self,
        client: mqtt.Client,
        userdata: Any,
        flags: dict,
        rc: int,
    ) -> None:
        if rc == 0:
            logger.info(
                "Connected to MQTT broker | host=%s port=%s",
                Settings.MQTT_BROKER_HOST,
                Settings.MQTT_BROKER_PORT,
            )
            client.subscribe(Settings.MQTT_TOPIC_EDGE)
            logger.info("Subscribed to topic | topic=%s", Settings.MQTT_TOPIC_EDGE)
        else:
            logger.error("Failed to connect to MQTT broker | rc=%s", rc)

    def on_message(
        self,
        client: mqtt.Client,
        userdata: Any,
        msg: mqtt.MQTTMessage,
    ) -> None:
        try:
            payload = msg.payload.decode("utf-8") 
            logger.info("Received message from %s: %s", msg.topic, payload)

            data = json.loads(payload)
            event = self._normalize_event(data)

            self.pg.upsert_device(
                machine_id=event["machine_id"],
                device_id=event["device_id"],
            )

            prediction_id = self.pg.insert_prediction(
                device_id=event["device_id"],
                machine_id=event["machine_id"],
                event_time=event["timestamp"],
                source=event.get("source"),
                scenario=event.get("scenario"),
                prediction=event["prediction"],
                risk_score=event["risk_score"],
                risk_level=event["risk_level"],
                model_type="edge",
            )

            self.influx.write_edge_event(event)

            if self._should_create_alert(event):
                self.pg.insert_alert(
                    device_id=event["device_id"],
                    machine_id=event["machine_id"],
                    prediction_id=prediction_id,
                    alert_type="failure_risk",
                    severity=self._severity_from_score(event["risk_score"]),
                    message=(
                        f"High risk detected for {event['device_id']} "
                        f"(score={event['risk_score']:.4f}, level={event['risk_level']})"
                    ),
                )
                logger.warning(
                    "Alert created | device_id=%s machine_id=%s risk_score=%.4f",
                    event["device_id"],
                    event["machine_id"],
                    event["risk_score"],
                )

            logger.info(
                "Event stored | machine_id=%s device_id=%s prediction=%s risk_score=%.4f risk_level=%s",
                event["machine_id"],
                event["device_id"],
                event["prediction"],
                event["risk_score"],
                event["risk_level"],
            )

        except json.JSONDecodeError:
            logger.exception("Invalid JSON received from topic=%s", msg.topic)
        except Exception:
            logger.exception("Unexpected error while processing message")

    def _normalize_event(self, data: dict) -> dict:
        ground_truth = data.get("ground_truth", {})
        failure_type = ground_truth.get("failure_type", {})

        features_raw = data.get("features_raw", {})

        air_temperature_k = float(features_raw.get("Air temperature [K]", 0.0))
        temp_diff = float(features_raw.get("temp_diff", 0.0))
        rotational_speed_rpm = float(features_raw.get("Rotational speed [rpm]", 0.0))
        torque_nm = float(features_raw.get("Torque [Nm]", 0.0))
        power_kw = float(features_raw.get("power_kw", 0.0))
        tool_wear_min = float(features_raw.get("Tool wear [min]", 0.0))

        process_temperature_k = air_temperature_k + temp_diff

        return {
            "device_id": data["device_id"],
            "machine_id": int(data["machine_id"]),
            "timestamp": data["timestamp"],
            "source": data.get("source"),
            "scenario": data.get("scenario"),

            "air_temperature_k": air_temperature_k,
            "process_temperature_k": process_temperature_k,
            "rotational_speed_rpm": rotational_speed_rpm,
            "torque_nm": torque_nm,
            "tool_wear_min": tool_wear_min,

            "machine_failure": int(ground_truth.get("machine_failure", 0)),
            "twf": int(failure_type.get("twf", 0)),
            "hdf": int(failure_type.get("hdf", 0)),
            "pwf": int(failure_type.get("pwf", 0)),
            "osf": int(failure_type.get("osf", 0)),
            "rnf": int(failure_type.get("rnf", 0)),

            "prediction": int(data.get("prediction", 0)),
            "prediction_label": str(data.get("prediction_label", "UNKNOWN")),
            "risk_score": float(data.get("risk_score", 0.0)),
            "risk_level": str(data.get("risk_level", "UNKNOWN")),
            "model_name": str(data.get("model_name", "unknown")),

            "temp_diff": temp_diff,
            "power_kw": power_kw,
        }

    def _should_create_alert(self, event: dict) -> bool:
        return (
            event["prediction"] == 1
            or event["risk_level"].upper() in {"HIGH", "CRITICAL", "HIGH_RISK"}
            or event["risk_score"] >= Settings.ALERT_RISK_THRESHOLD
        )

    def _severity_from_score(self, score: float) -> str:
        if score >= 0.95:
            return "critical"
        if score >= 0.80:
            return "high"
        if score >= 0.50:
            return "medium"
        return "low"

    def start(self) -> None:
        logger.info(
            "Connecting to MQTT | host=%s port=%s",
            Settings.MQTT_BROKER_HOST,
            Settings.MQTT_BROKER_PORT,
        )

        self.client.reconnect_delay_set(min_delay=1, max_delay=10)

        self.client.connect(
            Settings.MQTT_BROKER_HOST,
            Settings.MQTT_BROKER_PORT,
            keepalive=60,
        )

        self.client.loop_start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping MQTT consumer")
        finally:
            try:
                self.client.loop_stop()
                self.client.disconnect()
            except Exception:
                pass

            try:
                self.pg.close()
            except Exception:
                pass

            try:
                self.influx.close()
            except Exception:
                pass