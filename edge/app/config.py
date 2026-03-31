import os


class Settings:
    # MQTT
    MQTT_BROKER_HOST = os.getenv("MQTT_BROKER_HOST", "mosquitto")
    MQTT_BROKER_PORT = int(os.getenv("MQTT_BROKER_PORT", "1883"))

    # ВАЖНО: под твой .env
    MQTT_INPUT_TOPIC = os.getenv("MQTT_TOPIC_RAW", "iiot/raw")
    MQTT_OUTPUT_TOPIC = os.getenv("MQTT_TOPIC_EDGE", "iiot/edge")

    MQTT_CLIENT_ID = os.getenv("MQTT_CLIENT_ID", "edge-service")

    # Model
    MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/edge_model.joblib")

    # Normalization config
    NORMALIZATION_CONFIG_PATH = os.getenv(
        "NORMALIZATION_CONFIG_PATH",
        "app/config/training_fallback.json"
    )

    # Thresholds
    WARNING_THRESHOLD = float(os.getenv("WARNING_THRESHOLD", "0.4"))
    CRITICAL_THRESHOLD = float(os.getenv("CRITICAL_THRESHOLD", "0.7"))

    # Feature order (очень важно для модели!)
    MODEL_FEATURE_ORDER = [
        "Air temperature [K]",
        "temp_diff",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "power_kw",
        "Tool wear [min]"
    ]


settings = Settings()