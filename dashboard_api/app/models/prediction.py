from datetime import datetime
from sqlalchemy import String, Integer, Float, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from app.db.base import Base
from sqlalchemy.dialects.postgresql import JSONB


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    device_id: Mapped[str] = mapped_column(String(100), nullable=False)
    machine_id: Mapped[int] = mapped_column(Integer, nullable=False)
    event_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    source: Mapped[str | None] = mapped_column(String(50))
    scenario: Mapped[str | None] = mapped_column(String(50))
    prediction: Mapped[int] = mapped_column(Integer, nullable=False)
    risk_score: Mapped[float | None] = mapped_column(Float)
    risk_level: Mapped[str | None] = mapped_column(String(20))
    model_type: Mapped[str] = mapped_column(String(50), default="edge")
    features_used: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    top_factors: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)