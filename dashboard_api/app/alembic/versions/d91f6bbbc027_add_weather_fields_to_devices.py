"""add weather fields to devices

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
    op.add_column("devices", sa.Column("weather_dependent", sa.Boolean(), nullable=False, server_default="false"))
    op.add_column("devices", sa.Column("latitude", sa.Float(), nullable=True))
    op.add_column("devices", sa.Column("longitude", sa.Float(), nullable=True))
    op.add_column("devices", sa.Column("weather_sensitivity", sa.String(length=20), nullable=False, server_default="medium"))


def downgrade():
    op.drop_column("devices", "weather_sensitivity")
    op.drop_column("devices", "longitude")
    op.drop_column("devices", "latitude")
    op.drop_column("devices", "weather_dependent")