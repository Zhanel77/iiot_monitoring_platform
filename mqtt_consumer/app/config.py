import os


class Settings:
    MQTT_BROKER_HOST = os.getenv("MQTT_BROKER_HOST", "mosquitto")
    MQTT_BROKER_PORT = int(os.getenv("MQTT_BROKER_PORT", "1883"))
    MQTT_TOPIC_EDGE = os.getenv("MQTT_TOPIC_EDGE", "iiot/edge")
    MQTT_CLIENT_ID = os.getenv("MQTT_CLIENT_ID", "mqtt-consumer")

    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
    POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB = os.getenv("POSTGRES_DB", "predictive_db")
    POSTGRES_USER = os.getenv("POSTGRES_USER", "admin")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "admin")

    INFLUX_URL = os.getenv("INFLUX_URL", "http://influxdb:8086")
    INFLUX_TOKEN = os.getenv("INFLUX_TOKEN", "my-super-token")
    INFLUX_ORG = os.getenv("INFLUX_ORG", "iiot-org")
    INFLUX_BUCKET = os.getenv("INFLUX_BUCKET", "iiot_data")

    ALERT_RISK_THRESHOLD = float(os.getenv("ALERT_RISK_THRESHOLD", "0.8"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")