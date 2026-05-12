"""add weather fields to devices and predictions

Revision ID: d91f6bbbc027
Revises: 0001
Create Date: 2026-05-04 14:59:57.515769

"""
from alembic import op
import sqlalchemy as sa


revision = 'd91f6bbbc027'
down_revision = '0001'
branch_labels = None
depends_on = None


def upgrade():
    # Добавляем weather поля в таблицу devices
    op.add_column("devices", sa.Column("weather_dependent", sa.Boolean(), nullable=False, server_default="false"))
    op.add_column("devices", sa.Column("latitude", sa.Float(), nullable=True))
    op.add_column("devices", sa.Column("longitude", sa.Float(), nullable=True))
    op.add_column("devices", sa.Column("weather_sensitivity", sa.String(length=20), nullable=False, server_default="medium"))
    
    # Добавляем weather поля в таблицу predictions
    op.add_column("predictions", sa.Column("weather_factor", sa.Float(), nullable=True, server_default="1.0"))
    op.add_column("predictions", sa.Column("ml_risk_score", sa.Float(), nullable=True))
    
    # Создаем индексы для новых полей
    op.create_index("idx_predictions_weather_factor", "predictions", ["weather_factor"])
    op.create_index("idx_predictions_ml_risk_score", "predictions", ["ml_risk_score"])
    op.create_index("idx_devices_weather_dependent", "devices", ["weather_dependent"])
    op.create_index("idx_devices_coordinates", "devices", ["latitude", "longitude"])


def downgrade():
    # Удаляем индексы
    op.drop_index("idx_devices_coordinates", table_name="devices")
    op.drop_index("idx_devices_weather_dependent", table_name="devices")
    op.drop_index("idx_predictions_ml_risk_score", table_name="predictions")
    op.drop_index("idx_predictions_weather_factor", table_name="predictions")
    
    # Удаляем поля из predictions
    op.drop_column("predictions", "ml_risk_score")
    op.drop_column("predictions", "weather_factor")
    
    # Удаляем поля из devices
    op.drop_column("devices", "weather_sensitivity")
    op.drop_column("devices", "longitude")
    op.drop_column("devices", "latitude")
    op.drop_column("devices", "weather_dependent")