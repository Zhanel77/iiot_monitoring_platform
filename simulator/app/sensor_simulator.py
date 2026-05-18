import csv
import json
import time
from datetime import datetime, timezone

import paho.mqtt.client as mqtt

from app.config import Config
from app.scenarios import apply_scenario

def connect_with_retry(client, host, port, retries=10):
    for i in range(retries):
        try:
            client.connect(host, port, 60)
            print("Connected to MQTT")
            return
        except Exception as e:
            print(f"MQTT not ready, retry {i+1}/{retries}...")
            time.sleep(2)

    raise Exception("Failed to connect to MQTT")

class SensorSimulator:
    def __init__(self) -> None:
        self.client = mqtt.Client()
        connect_with_retry(
            self.client,
            Config.MQTT_BROKER_HOST,
            Config.MQTT_BROKER_PORT,
        )

    def load_rows(self) -> list[dict]:
        rows: list[dict] = []

        with open(Config.CSV_PATH, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                rows.append({
                    "machine_id": int(row["machine_id"]),
                    "Air temperature [K]": float(row["Air temperature [K]"]),
                    "Process temperature [K]": float(row["Process temperature [K]"]),
                    "Rotational speed [rpm]": float(row["Rotational speed [rpm]"]),
                    "Torque [Nm]": float(row["Torque [Nm]"]),
                    "Tool wear [min]": float(row["Tool wear [min]"]),
                    "Machine failure": int(row["Machine failure"]),
                    "TWF": int(row["TWF"]),
                    "HDF": int(row["HDF"]),
                    "PWF": int(row["PWF"]),
                    "OSF": int(row["OSF"]),
                    "RNF": int(row["RNF"]),
                })

        return rows

    def build_payload(self, row: dict) -> dict:
        machine_id = row["machine_id"]

        return {
            "device_id": f"sim_machine_{machine_id}",
            "machine_id": machine_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "simulator",
            "scenario": Config.SCENARIO_NAME,
            "sensors": {
                "air_temperature_k": row["Air temperature [K]"],
                "process_temperature_k": row["Process temperature [K]"],
                "rotational_speed_rpm": row["Rotational speed [rpm]"],
                "torque_nm": row["Torque [Nm]"],
                "tool_wear_min": row["Tool wear [min]"],
            },
            "ground_truth": {
                "machine_failure": row["Machine failure"],
                "failure_type": {
                    "twf": row["TWF"],
                    "hdf": row["HDF"],
                    "pwf": row["PWF"],
                    "osf": row["OSF"],
                    "rnf": row["RNF"],
                },
            },
        }

    def publish_row(self, row: dict) -> None:
        scenario_row = apply_scenario(row, Config.SCENARIO_NAME)
        payload = self.build_payload(scenario_row)

        result = self.client.publish(
            Config.MQTT_TOPIC,
            json.dumps(payload),
        )

        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            print(f"Published to {Config.MQTT_TOPIC}: {json.dumps(payload)}")
        else:
            print(f"Failed to publish message for machine_id={row['machine_id']}")

    def run(self) -> None:
        rows = self.load_rows()

        if not rows:
            print("No rows found in simulation input CSV.")
            return

        print(
            f"Simulator started. Broker={Config.MQTT_BROKER_HOST}:{Config.MQTT_BROKER_PORT}, "
            f"topic={Config.MQTT_TOPIC}, rows={len(rows)}, scenario={Config.SCENARIO_NAME}"
        )

        while True:
            for row in rows:
                self.publish_row(row)
                time.sleep(Config.PUBLISH_INTERVAL_SEC)

            if not Config.LOOP_FOREVER:
                break

        self.client.disconnect()
        print("Simulator stopped.")