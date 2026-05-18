import os


class Config:
    MQTT_BROKER_HOST = os.getenv("MQTT_BROKER_HOST", "mosquitto")
    MQTT_BROKER_PORT = int(os.getenv("MQTT_BROKER_PORT", "1883"))
    MQTT_TOPIC = os.getenv("MQTT_TOPIC", "iiot/raw")

    CSV_PATH = os.getenv("CSV_PATH", "data/simulation_input.csv")
    PUBLISH_INTERVAL_SEC = float(os.getenv("PUBLISH_INTERVAL_SEC", "1.0"))
    LOOP_FOREVER = os.getenv("LOOP_FOREVER", "true").lower() == "true"
    SCENARIO_NAME = os.getenv("SCENARIO_NAME", "normal_run")