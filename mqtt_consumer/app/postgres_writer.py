from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor, Json

from app.config import Settings


class PostgresWriter:
    def __init__(self) -> None:
        self.conn = psycopg2.connect(
            host=Settings.POSTGRES_HOST,
            port=Settings.POSTGRES_PORT,
            dbname=Settings.POSTGRES_DB,
            user=Settings.POSTGRES_USER,
            password=Settings.POSTGRES_PASSWORD,
            cursor_factory=RealDictCursor,
        )
        self.conn.autocommit = True

    def upsert_device(self, machine_id: int, device_id: str) -> None:
        DEVICE_LOCATIONS = {
            1: {"latitude": 51.1694, "longitude": 71.4491},  # Astana
            2: {"latitude": 43.2220, "longitude": 76.8512},  # Almaty
            3: {"latitude": 47.0945, "longitude": 51.9238},  # Atyrau
            4: {"latitude": 42.3417, "longitude": 69.5901},  # Shymkent
            5: {"latitude": 50.2839, "longitude": 57.1660},  # Aktobe
        }

        location = DEVICE_LOCATIONS.get(
            machine_id,
            {"latitude": 51.1694, "longitude": 71.4491},
        )

        query = """
        INSERT INTO devices (
            machine_id,
            device_id,
            name,
            weather_dependent,
            latitude,
            longitude,
            weather_sensitivity
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (device_id) DO UPDATE
        SET machine_id = EXCLUDED.machine_id,
            name = EXCLUDED.name,
            weather_dependent = EXCLUDED.weather_dependent,
            latitude = EXCLUDED.latitude,
            longitude = EXCLUDED.longitude,
            weather_sensitivity = EXCLUDED.weather_sensitivity
        """

        with self.conn.cursor() as cur:
            cur.execute(
                query,
                (
                    machine_id,
                    device_id,
                    f"Machine {machine_id}",
                    True,
                    location["latitude"],
                    location["longitude"],
                    "medium",
                ),
            )

        self.conn.commit()

    def insert_prediction(
        self,
        device_id: str,
        machine_id: int,
        event_time: str,
        source: Optional[str],
        scenario: Optional[str],
        prediction: int,
        risk_score: float,
        risk_level: str,
        features_used=None,
        top_factors=None,
        model_type: str = "edge",
        weather_factor: Optional[float] = None,  # НОВЫЙ параметр
        ml_risk_score: Optional[float] = None,    # НОВЫЙ параметр
    ) -> int:
        # Обновляем запрос с новыми полями
        query = """
        INSERT INTO predictions (
            device_id,
            machine_id,
            event_time,
            source,
            scenario,
            prediction,
            risk_score,
            features_used,
            top_factors,
            risk_level,
            model_type,
            weather_factor,
            ml_risk_score
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """
        with self.conn.cursor() as cur:
            cur.execute(
                query,
                (
                    device_id,
                    machine_id,
                    event_time,
                    source,
                    scenario,
                    prediction,
                    risk_score,
                    Json(features_used) if features_used is not None else None,
                    Json(top_factors) if top_factors is not None else None,
                    risk_level,
                    model_type,
                    weather_factor,  # НОВОЕ поле
                    ml_risk_score,   # НОВОЕ поле
                ),
            )
            row = cur.fetchone()
            return int(row["id"])
        
    def insert_alert(
        self,
        device_id: str,
        machine_id: int,
        prediction_id: int,
        alert_type: str,
        severity: str,
        message: str,
    ) -> None:
        query = """
        INSERT INTO alerts (
            device_id,
            machine_id,
            prediction_id,
            alert_type,
            severity,
            message
        )
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        with self.conn.cursor() as cur:
            cur.execute(
                query,
                (
                    device_id,
                    machine_id,
                    prediction_id,
                    alert_type,
                    severity,
                    message,
                ),
            )

    def close(self) -> None:
        if self.conn:
            self.conn.close()