import json
import time
from pathlib import Path

import joblib
import pandas as pd
import paho.mqtt.client as mqtt
import psycopg2


MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "machines/sensor-data"

POSTGRES_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "iiot_monitoring",
    "user": "postgres",
    "password": "postgres",
}

is_connected = False


def load_artifacts():
    project_root = Path(__file__).resolve().parents[1]

    model_path = project_root / "ml" / "models" / "best_model.pkl"
    feature_columns_path = project_root / "ml" / "models" / "feature_columns.json"
    metadata_path = project_root / "ml" / "models" / "best_model_metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not feature_columns_path.exists():
        raise FileNotFoundError(f"Feature columns file not found: {feature_columns_path}")

    model = joblib.load(model_path)

    with open(feature_columns_path, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return model, feature_columns, metadata


MODEL, FEATURE_COLUMNS, METADATA = load_artifacts()


def get_db_connection():
    return psycopg2.connect(
        host=POSTGRES_CONFIG["host"],
        port=POSTGRES_CONFIG["port"],
        dbname=POSTGRES_CONFIG["dbname"],
        user=POSTGRES_CONFIG["user"],
        password=POSTGRES_CONFIG["password"],
    )


def save_prediction_to_db(payload: dict, predicted_class: int, failure_probability: float, risk_level: str):
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        query = """
        INSERT INTO predictions (
            machine_id,
            event_time,
            type,
            air_temperature_k,
            process_temperature_k,
            rotational_speed_rpm,
            torque_nm,
            tool_wear_min,
            actual_failure,
            predicted_class,
            failure_probability,
            risk_level
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        values = (
            int(payload.get("machine_id", 0)),
            payload.get("timestamp"),
            int(payload["type"]),
            float(payload["air_temperature_k"]),
            float(payload["process_temperature_k"]),
            float(payload["rotational_speed_rpm"]),
            float(payload["torque_nm"]),
            float(payload["tool_wear_min"]),
            int(payload.get("actual_failure", 0)),
            int(predicted_class),
            float(failure_probability),
            risk_level,
        )

        cursor.execute(query, values)
        conn.commit()
        print("Saved prediction to PostgreSQL.")

    except Exception as e:
        print(f"Database error: {e}")

    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()


def build_input_dataframe(payload: dict) -> pd.DataFrame:
    row = {}
    for col in FEATURE_COLUMNS:
        if col not in payload:
            raise ValueError(f"Missing required feature in payload: '{col}'")
        row[col] = payload[col]

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def classify_risk(probability: float) -> str:
    if probability >= 0.8:
        return "CRITICAL"
    if probability >= 0.5:
        return "HIGH"
    if probability >= 0.2:
        return "MEDIUM"
    return "LOW"


def on_connect(client, userdata, flags, rc):
    global is_connected
    if rc == 0:
        is_connected = True
        print(f"Connected to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}")
        client.subscribe(MQTT_TOPIC)
        print(f"Subscribed to topic: {MQTT_TOPIC}")
        if METADATA:
            print(f"Loaded model: {METADATA.get('best_model_name', 'unknown')}")
            print(f"Feature columns: {FEATURE_COLUMNS}")
    else:
        print(f"Failed to connect to MQTT broker. Return code: {rc}")


def on_disconnect(client, userdata, rc):
    global is_connected
    is_connected = False
    print(f"Disconnected from MQTT broker. Return code: {rc}")


def on_message(client, userdata, msg):
    try:
        raw_message = msg.payload.decode("utf-8")
        print("\n" + "=" * 60)
        print(f"Received message on topic '{msg.topic}':")
        print(raw_message)

        payload = json.loads(raw_message)

        machine_id = payload.get("machine_id", "unknown")
        timestamp = payload.get("timestamp", "unknown")
        actual_failure = payload.get("actual_failure", "unknown")

        input_df = build_input_dataframe(payload)

        prediction = MODEL.predict(input_df)[0]

        if hasattr(MODEL, "predict_proba"):
            failure_probability = float(MODEL.predict_proba(input_df)[0][1])
        else:
            failure_probability = float(prediction)

        risk_level = classify_risk(failure_probability)

        print("\nPrediction result:")
        print(f"Machine ID          : {machine_id}")
        print(f"Timestamp           : {timestamp}")
        print(f"Actual failure      : {actual_failure}")
        print(f"Predicted class     : {int(prediction)}")
        print(f"Failure probability : {failure_probability:.4f}")
        print(f"Risk level          : {risk_level}")

        if int(prediction) == 1:
            print("ALERT: Potential machine failure detected.")
        else:
            print("Status: Machine operating normally.")

        save_prediction_to_db(
            payload=payload,
            predicted_class=int(prediction),
            failure_probability=failure_probability,
            risk_level=risk_level,
        )

    except json.JSONDecodeError:
        print("Error: Received message is not valid JSON.")
    except ValueError as e:
        print(f"Validation error: {e}")
    except Exception as e:
        print(f"Unexpected error while processing message: {e}")


def main():
    global is_connected

    print("Starting MQTT consumer...")

    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message

    print("Connecting to broker...")
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()

    wait_seconds = 0
    while not is_connected and wait_seconds < 10:
        time.sleep(1)
        wait_seconds += 1

    if not is_connected:
        print("Could not connect to MQTT broker.")
        client.loop_stop()
        return

    print("Consumer is running and waiting for messages...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping consumer...")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()