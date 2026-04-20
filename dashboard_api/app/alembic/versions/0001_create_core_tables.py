"""create core tables

Revision ID: 0001
Revises:
Create Date: 2026-04-09
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # USERS
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("full_name", sa.String(255), nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("role", sa.String(50), nullable=False, server_default="viewer"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
    )

    op.create_index("ix_users_email", "users", ["email"], unique=True)

    # DEVICES
    op.create_table(
        "devices",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("machine_id", sa.Integer(), nullable=False, unique=True),
        sa.Column("device_id", sa.String(100), nullable=False, unique=True),
        sa.Column("name", sa.String(100)),
        sa.Column("status", sa.String(50), server_default="active"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # PREDICTIONS
    op.create_table(
        "predictions",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("device_id", sa.String(100), nullable=False),
        sa.Column("machine_id", sa.Integer(), nullable=False),
        sa.Column("event_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source", sa.String(50)),
        sa.Column("scenario", sa.String(50)),
        sa.Column("prediction", sa.Integer(), nullable=False),
        sa.Column("risk_score", sa.Float()),
        sa.Column("risk_level", sa.String(20)),
        sa.Column("model_type", sa.String(50), server_default="edge"),
        sa.Column("features_used", JSONB(), nullable=True),
        sa.Column("top_factors", JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ALERTS
    op.create_table(
        "alerts",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("device_id", sa.String(100), nullable=False),
        sa.Column("machine_id", sa.Integer(), nullable=False),
        sa.Column("prediction_id", sa.BigInteger(), nullable=True),
        sa.Column("alert_type", sa.String(50), nullable=False),
        sa.Column("severity", sa.String(20), nullable=False),
        sa.Column("message", sa.Text()),
        sa.Column("status", sa.String(20), server_default="open"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("resolved_at", sa.DateTime(timezone=True)),
        sa.ForeignKeyConstraint(
            ["prediction_id"],
            ["predictions.id"],
            ondelete="SET NULL",
        ),
    )

    op.create_table(
        "user_devices",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("device_id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),

        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["device_id"], ["devices.id"], ondelete="CASCADE"),

        sa.UniqueConstraint("user_id", "device_id", name="uq_user_device"),
    )   


def downgrade():
    op.drop_table("user_devices")
    op.drop_table("alerts")
    op.drop_table("predictions")
    op.drop_table("devices")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")