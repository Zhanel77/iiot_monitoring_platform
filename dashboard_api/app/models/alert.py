from datetime import datetime
from sqlalchemy import String, Integer, DateTime, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column
from app.db.base import Base
from sqlalchemy import Float
from sqlalchemy.dialects.postgresql import JSONB

class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(primary_key=True)
    device_id: Mapped[str] = mapped_column(String(100), nullable=False)
    machine_id: Mapped[int] = mapped_column(Integer, nullable=False)

    prediction_id: Mapped[int | None] = mapped_column(
        ForeignKey("predictions.id", ondelete="SET NULL")
    )

    risk_score: Mapped[float | None] = mapped_column(Float)

    risk_level: Mapped[str | None] = mapped_column(
        String(20)
    )

    features_used: Mapped[dict | None] = mapped_column(
        JSONB
    )

    top_factors: Mapped[list | None] = mapped_column(
        JSONB
    )

    alert_type: Mapped[str] = mapped_column(String(50), nullable=False)
    severity: Mapped[str] = mapped_column(String(20), nullable=False)
    message: Mapped[str | None] = mapped_column(Text)

    status: Mapped[str] = mapped_column(String(20), default="open")

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))   