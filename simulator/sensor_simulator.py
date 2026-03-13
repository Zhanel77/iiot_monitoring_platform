import json
import sys
import time
from pathlib import Path

import pandas as pd
import paho.mqtt.client as mqtt


MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "machines/sensor-data"
PUBLISH_DELAY_SECONDS = 1.0
MAX_MESSAGES = 20

is_connected = False


def load_dataset() -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "processed" / "ai4i_clean.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {data_path}")

    return pd.read_csv(data_path)


def get_simulation_mode() -> str:
    """
    Read mode from command line.
    Allowed: normal, failure, mixed
    Default: normal
    """
    if len(sys.argv) < 2:
        return "normal"

    mode = sys.argv[1].strip().lower()

    if mode not in {"normal", "failure", "mixed"}:
        raise ValueError("Mode must be one of: normal, failure, mixed")

    return mode


def select_rows(df: pd.DataFrame, mode: str, max_messages: int) -> pd.DataFrame:
    if "machine_failure" not in df.columns:
        raise ValueError("Column 'machine_failure' not found in dataset.")

    if mode == "normal":
        selected = df[df["machine_failure"] == 0].head(max_messages)

    elif mode == "failure":
        selected = df[df["machine_failure"] == 1].head(max_messages)

    elif mode == "mixed":
        normal_df = df[df["machine_failure"] == 0]
        failure_df = df[df["machine_failure"] == 1]

        normal_count = min(max_messages // 2, len(normal_df))
        failure_count = min(max_messages - normal_count, len(failure_df))

        normal_part = normal_df.sample(n=normal_count, random_state=42)
        failure_part = failure_df.sample(n=failure_count, random_state=42)

        selected = pd.concat([normal_part, failure_part])
        selected = selected.sample(frac=1, random_state=42).reset_index(drop=True)

    else:
        raise ValueError("Mode must be normal, failure, or mixed.")

    if selected.empty:
        raise ValueError(f"No rows found for mode: {mode}")

    return selected.reset_index(drop=True)


def prepare_payload(row: pd.Series, row_index: int) -> dict:
    return {
        "machine_id": int(row_index + 1),
        "timestamp": pd.Timestamp.now().isoformat(),
        "type": int(row["type"]),
        "air_temperature_k": float(row["air_temperature_k"]),
        "process_temperature_k": float(row["process_temperature_k"]),
        "rotational_speed_rpm": float(row["rotational_speed_rpm"]),
        "torque_nm": float(row["torque_nm"]),
        "tool_wear_min": float(row["tool_wear_min"]),
        "actual_failure": int(row["machine_failure"]),
    }


def on_connect(client, userdata, flags, rc):
    global is_connected
    if rc == 0:
        is_connected = True
        print(f"Connected to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}")
    else:
        print(f"Failed to connect to MQTT broker. Return code: {rc}")


def on_disconnect(client, userdata, rc):
    global is_connected
    is_connected = False
    print(f"Disconnected from MQTT broker. Return code: {rc}")


def main():
    global is_connected

    mode = get_simulation_mode()

    print("Loading processed dataset...")
    df = load_dataset()
    print(f"Dataset loaded successfully. Total rows: {len(df)}")

    selected_df = select_rows(df, mode, MAX_MESSAGES)

    print(f"Simulation mode: {mode}")
    print(f"Selected rows: {len(selected_df)}")
    print("Actual failure distribution:")
    print(selected_df["machine_failure"].value_counts())

    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect

    print("Connecting to MQTT broker...")
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()

    wait_seconds = 0
    while not is_connected and wait_seconds < 10:
        time.sleep(1)
        wait_seconds += 1

    if not is_connected:
        print("Could not establish MQTT connection. Exiting.")
        client.loop_stop()
        return

    print(f"Publishing to topic: {MQTT_TOPIC}")
    print(f"Sending up to {len(selected_df)} messages with {PUBLISH_DELAY_SECONDS}s delay...\n")

    sent_count = 0

    for row_index, row in selected_df.iterrows():
        if not is_connected:
            print("MQTT connection lost. Stopping simulation.")
            break

        payload = prepare_payload(row, row_index)
        message = json.dumps(payload)

        result = client.publish(MQTT_TOPIC, message)

        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            print(f"[{sent_count + 1}] Sent message:")
            print(message)
        else:
            print(f"Failed to publish message #{sent_count + 1}. Return code: {result.rc}")

        sent_count += 1
        time.sleep(PUBLISH_DELAY_SECONDS)

    client.loop_stop()
    client.disconnect()

    print("\nSimulation completed.")
    print(f"Total messages attempted: {sent_count}")


if __name__ == "__main__":
    main()