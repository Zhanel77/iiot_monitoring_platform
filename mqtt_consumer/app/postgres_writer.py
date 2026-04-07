from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor

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
        query = """
        INSERT INTO devices (machine_id, device_id, name)
        VALUES (%s, %s, %s)
        ON CONFLICT (device_id) DO UPDATE
        SET machine_id = EXCLUDED.machine_id,
            name = EXCLUDED.name
        """
        with self.conn.cursor() as cur:
            cur.execute(query, (machine_id, device_id, f"Machine {machine_id}"))

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
        model_type: str = "edge",
    ) -> int:
        query = """
        INSERT INTO predictions (
            device_id,
            machine_id,
            event_time,
            source,
            scenario,
            prediction,
            risk_score,
            risk_level,
            model_type
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                    risk_level,
                    model_type,
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