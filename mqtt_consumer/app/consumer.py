import json
import time
from typing import Any

import paho.mqtt.client as mqtt
import requests

from app.config import Settings
from app.influx_writer import InfluxWriter
from app.logger import setup_logger
from app.postgres_writer import PostgresWriter



logger = setup_logger()

FEATURE_MESSAGES = {
    "tool_wear_min": {
        "increase": "tool wear reached abnormal operating levels",
        "decrease": "tool wear remains within safe operating range",
    },

    "torque_nm": {
        "increase": "high torque load is stressing the machine",
        "decrease": "torque load remains stable",
    },

    "temp_diff": {
        "increase": "thermal imbalance was detected",
        "decrease": "thermal conditions remain stable",
    },

    "rotational_speed_rpm": {
        "increase": "high rotational speed contributes to instability",
        "decrease": "stable rotational speed reduces failure probability",
    },

    "air_temperature_k": {
        "increase": "air temperature contributes to operational stress",
        "decrease": "air temperature remains within acceptable limits",
    },
}


def build_explanation(
    top_factors: list[dict],
    risk_level: str,
) -> str:

    if not top_factors:
        return "No explainability data available."

    explanations = []

    for factor in top_factors[:3]:
        feature = str(factor.get("feature", "")).lower()
        effect = str(factor.get("effect", "")).lower()
        value = factor.get("feature_value")

        feature_messages = FEATURE_MESSAGES.get(feature)

        if feature_messages:
            text = feature_messages.get(effect)

            if value is not None:
                text = f"{text} (value={value})"

            explanations.append(text)

    if not explanations:
        return "Model detected abnormal machine behavior."

    joined = "; ".join(explanations)

    risk_level = risk_level.upper()

    if risk_level in {"CRITICAL", "HIGH_RISK"}:
        return (
            f"Critical failure risk detected because {joined}. "
            f"Immediate inspection is recommended."
        )

    if risk_level == "WARNING":
        return (
            f"Warning threshold exceeded because {joined}. "
            f"Preventive maintenance is recommended."
        )

    return (
        f"Machine operates within expected range because {joined}."
    )


class MQTTConsumer:
    def __init__(self) -> None:
        self.client = mqtt.Client(
            client_id=f"{Settings.MQTT_CLIENT_ID}-{time.time()}"
        )
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        self.pg = PostgresWriter()
        self.influx = InfluxWriter()
        self.http = requests.Session()

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

            edge_prediction_id = self.pg.insert_prediction(
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
                edge_message = (
                    f"Edge model detected high risk for {event['device_id']} "
                    f"(score={event['risk_score']:.4f}, "
                    f"level={event['risk_level']}, model=edge)"
                )

                self.pg.insert_alert(
                    device_id=event["device_id"],
                    machine_id=event["machine_id"],
                    prediction_id=edge_prediction_id,
                    risk_score=float(event["risk_score"]),
                    risk_level=str(event["risk_level"]),
                    features_used=None,
                    top_factors=None,
                    alert_type="edge_fast_alert",
                    severity=self._severity_from_score(float(event["risk_score"])),
                    message=edge_message,
                )

                logger.warning(
                    "Edge alert created | device_id=%s machine_id=%s risk_score=%.4f",
                    event["device_id"],
                    event["machine_id"],
                    event["risk_score"],
                )

            if Settings.ENABLE_CLOUD_PREDICTION:
                cloud_result = self._get_cloud_prediction(event)

                if cloud_result is not None:
                    weather_factor = cloud_result.get("weather_factor")
                    ml_risk_score = cloud_result.get("ml_risk_score")

                    cloud_prediction_id = self.pg.insert_prediction(
                        device_id=event["device_id"],
                        machine_id=event["machine_id"],
                        event_time=event["timestamp"],
                        source=event.get("source"),
                        scenario=event.get("scenario"),
                        prediction=int(cloud_result["prediction"]),
                        risk_score=float(cloud_result["risk_score"]),
                        risk_level=str(cloud_result["prediction_label"]),
                        features_used=cloud_result.get("features_used"),
                        top_factors=cloud_result.get("top_factors"),
                        model_type="cloud",
                        weather_factor=weather_factor,
                        ml_risk_score=ml_risk_score,
                    )

                    self.influx.write_cloud_event(
                        {
                            **event,
                            "prediction": int(cloud_result.get("prediction", 0)),
                            "risk_score": float(cloud_result.get("risk_score", 0.0)),
                            "risk_level": str(
                                cloud_result.get("prediction_label", "NORMAL")
                            ),
                            "model_type": "cloud",
                            "weather_factor": weather_factor or 1.0,
                            "ml_risk_score": ml_risk_score
                            or cloud_result.get("risk_score", 0.0),
                        }
                    )

                    if self._should_create_alert_from_result(cloud_result):
                        top_factors = cloud_result.get("top_factors", [])

                        explanation = (
                            build_explanation(
                                top_factors=top_factors,
                                risk_level=str(cloud_result["prediction_label"]),
                            )
                            if top_factors
                            else "No SHAP explanation available."
                        )

                        if weather_factor and weather_factor != 1.0:
                            base_score = (
                                f"{float(ml_risk_score):.4f}"
                                if ml_risk_score is not None
                                else "unknown"
                            )

                            explanation += (
                                f" Weather factor applied: {float(weather_factor):.2f}. "
                                f"(ML risk: {base_score} -> "
                                f"final: {float(cloud_result['risk_score']):.4f})"
                            )

                        cloud_message = (
                            f"Cloud AI detected high risk for Machine {event['machine_id']}. "
                            f"Risk score={float(cloud_result['risk_score']):.4f}. "
                        )

                        if ml_risk_score is not None:
                            cloud_message += (
                                f"ML base score={float(ml_risk_score):.4f}. "
                            )

                        if weather_factor and weather_factor != 1.0:
                            cloud_message += (
                                f"Weather factor={float(weather_factor):.2f}. "
                            )

                        cloud_message += f"Explanation: {explanation}"

                        self.pg.insert_alert(
                            device_id=event["device_id"],
                            machine_id=event["machine_id"],
                            prediction_id=cloud_prediction_id,
                            risk_score=float(cloud_result["risk_score"]),
                            risk_level=str(cloud_result["prediction_label"]),
                            features_used=cloud_result.get("features_used"),
                            top_factors=cloud_result.get("top_factors"),
                            alert_type="cloud_shap_risk",
                            severity=self._severity_from_score(
                                float(cloud_result["risk_score"])
                            ),
                            message=cloud_message,
                        )

                        logger.warning(
                            "Cloud alert created | device_id=%s machine_id=%s risk_score=%.4f weather_factor=%s",
                            event["device_id"],
                            event["machine_id"],
                            float(cloud_result["risk_score"]),
                            weather_factor if weather_factor else "none",
                        )

                    logger.info(
                        "Cloud prediction stored | device_id=%s machine_id=%s prediction=%s risk_score=%.4f ml_risk=%s weather_factor=%s risk_level=%s",
                        event["device_id"],
                        event["machine_id"],
                        int(cloud_result["prediction"]),
                        float(cloud_result["risk_score"]),
                        ml_risk_score
                        if ml_risk_score is not None
                        else float(cloud_result["risk_score"]),
                        weather_factor if weather_factor else 1.0,
                        str(cloud_result["prediction_label"]),
                    )

            logger.info(
                "Event stored | machine_id=%s device_id=%s edge_prediction=%s edge_risk_score=%.4f edge_risk_level=%s",
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
            "temp_diff": temp_diff,
            "power_kw": power_kw,

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
        }

    def _build_cloud_payload(self, event: dict) -> dict:
        return {
            "device_id": event["device_id"],  # НОВОЕ: передаем device_id
            "air_temperature_k": event["air_temperature_k"],
            "process_temperature_k": event["process_temperature_k"],
            "rotational_speed_rpm": event["rotational_speed_rpm"],
            "torque_nm": event["torque_nm"],
            "tool_wear_min": event["tool_wear_min"],
        }
    
    def _get_cloud_prediction(self, event: dict) -> dict | None:
        payload = self._build_cloud_payload(event)
        
        # Логируем отправку с device_id
        logger.info(
            "Sending cloud prediction request | device_id=%s machine_id=%s",
            event["device_id"],
            event["machine_id"],
        )

        try:
            response = self.http.post(
                Settings.CLOUD_API_URL,
                json=payload,
                timeout=Settings.CLOUD_API_TIMEOUT,
            )
            response.raise_for_status()

            result = response.json()
            
            # Проверяем наличие всех необходимых полей
            required = {"prediction", "risk_score", "prediction_label"}
            if not required.issubset(result):
                logger.error("Invalid cloud API response | body=%s", result)
                return None
            
            # Логируем применение weather фактора если он был
            if result.get("weather_factor") and result["weather_factor"] != 1.0:
                logger.info(
                    "Weather factor applied | device_id=%s weather_factor=%.2f ml_risk=%.4f final_risk=%.4f",
                    event["device_id"],
                    result["weather_factor"],
                    result.get("ml_risk_score", result["risk_score"]),
                    result["risk_score"],
                )
            
            return result

        except requests.RequestException as exc:
            logger.exception("Cloud prediction request failed | device_id=%s error=%s", 
                            event.get("device_id"), exc)
            return None
        except ValueError:
            logger.exception("Cloud prediction response is not valid JSON")
            return None
    def _should_create_alert(self, event: dict) -> bool:
        return (
            event["prediction"] == 1
            or event["risk_level"].upper() in {"HIGH", "CRITICAL", "HIGH_RISK"}
            or event["risk_score"] >= Settings.ALERT_RISK_THRESHOLD
        )

    def _should_create_alert_from_result(self, result: dict) -> bool:
        risk_level = str(result.get("risk_level") or result.get("prediction_label") or "").upper()
        risk_score = float(result.get("risk_score", 0.0))
        prediction = int(result.get("prediction", 0))

        return (
            prediction == 1
            or risk_level in {"HIGH", "CRITICAL", "HIGH_RISK", "WARNING"}
            or risk_score >= Settings.ALERT_RISK_THRESHOLD
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
                self.http.close()
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