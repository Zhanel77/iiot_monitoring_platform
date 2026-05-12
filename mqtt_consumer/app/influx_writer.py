# app/influx_writer.py
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

from app.config import Settings


class InfluxWriter:
    def __init__(self) -> None:
        self.client = InfluxDBClient(
            url=Settings.INFLUX_URL,
            token=Settings.INFLUX_TOKEN,
            org=Settings.INFLUX_ORG,
        )
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.bucket = Settings.INFLUX_BUCKET
        self.org = Settings.INFLUX_ORG

    def write_edge_event(self, event: dict) -> None:
        point = (
            Point("machine_metrics")
            .tag("device_id", str(event["device_id"]))
            .tag("machine_id", str(event["machine_id"]))
            .tag("source", str(event.get("source", "unknown")))
            .tag("scenario", str(event.get("scenario", "unknown")))
            .field("air_temperature_k", float(event["air_temperature_k"]))
            .field("process_temperature_k", float(event["process_temperature_k"]))
            .field("rotational_speed_rpm", float(event["rotational_speed_rpm"]))
            .field("torque_nm", float(event["torque_nm"]))
            .field("tool_wear_min", float(event["tool_wear_min"]))
            .field("machine_failure", int(event.get("machine_failure", 0)))
            .field("prediction", int(event["prediction"]))
            .field("risk_score", float(event["risk_score"]))
            .field("temp_diff", float(event.get("temp_diff", 0.0)))
            .field("power_kw", float(event.get("power_kw", 0.0)))
            .time(event["timestamp"], WritePrecision.NS)
        )

        self.write_api.write(bucket=self.bucket, org=self.org, record=point)

    def write_cloud_event(self, event: dict) -> None:
        # Безопасное получение weather_factor
        weather_factor = event.get("weather_factor")
        if weather_factor is None:
            weather_factor = 1.0
        else:
            weather_factor = float(weather_factor)
        
        # Безопасное получение ml_risk_score
        ml_risk_score = event.get("ml_risk_score")
        if ml_risk_score is None:
            ml_risk_score = event.get("risk_score", 0.0)
        else:
            ml_risk_score = float(ml_risk_score)
        
        point = (
            Point("cloud_predictions")
            .tag("device_id", str(event["device_id"]))
            .tag("machine_id", str(event["machine_id"]))
            .tag("model_type", "cloud")
            .tag("source", str(event.get("source", "unknown")))
            .tag("scenario", str(event.get("scenario", "unknown")))
            .field("prediction", int(event["prediction"]))
            .field("risk_score", float(event["risk_score"]))
            .field("ml_risk_score", ml_risk_score)
            .field("weather_factor", weather_factor)
            .time(event["timestamp"], WritePrecision.NS)
        )

        self.write_api.write(bucket=self.bucket, org=self.org, record=point)

    def close(self) -> None:
        self.client.close() 