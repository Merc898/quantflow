"""Initial database schema with TimescaleDB hypertables.

Revision ID: 001_initial
Revises: None
Create Date: 2024-01-01 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create initial database schema."""
    # Create TimescaleDB extension
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")

    # Market data table
    op.create_table(
        "market_data",
        sa.Column("time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("open", sa.Float, nullable=True),
        sa.Column("high", sa.Float, nullable=True),
        sa.Column("low", sa.Float, nullable=True),
        sa.Column("close", sa.Float, nullable=True),
        sa.Column("volume", sa.BigInteger, nullable=True),
        sa.Column("vwap", sa.Float, nullable=True),
        sa.Column("adjusted", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("source", sa.String(50), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("time", "symbol"),
    )
    op.create_index("ix_market_data_symbol_time", "market_data", ["symbol", "time"])

    # Features table
    op.create_table(
        "features",
        sa.Column("time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("feature", sa.String(100), nullable=False),
        sa.Column("value", sa.Float, nullable=False),
        sa.Column("version", sa.Integer, nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("time", "symbol", "feature"),
    )
    op.create_index("ix_features_symbol_time", "features", ["symbol", "time"])
    op.create_index("ix_features_symbol_feature", "features", ["symbol", "feature"])

    # Signals table
    op.create_table(
        "signals",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("model_name", sa.String(100), nullable=False),
        sa.Column("signal", sa.Float, nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("forecast_return", sa.Float, nullable=True),
        sa.Column("forecast_std", sa.Float, nullable=True),
        sa.Column("regime", sa.String(50), nullable=True),
        sa.Column("metadata", postgresql.JSONB, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_signals_time", "signals", ["time"])
    op.create_index("ix_signals_symbol_time", "signals", ["symbol", "time"])
    op.create_index("ix_signals_model_time", "signals", ["model_name", "time"])

    # Recommendations table
    op.create_table(
        "recommendations",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("recommendation", sa.String(20), nullable=False),
        sa.Column("signal_strength", sa.Float, nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("position_size", sa.Float, nullable=False),
        sa.Column("expected_return_21d", sa.Float, nullable=True),
        sa.Column("expected_vol_21d", sa.Float, nullable=True),
        sa.Column("var_95_1d", sa.Float, nullable=True),
        sa.Column("max_drawdown_estimate", sa.Float, nullable=True),
        sa.Column("model_contributions", postgresql.JSONB, server_default="{}"),
        sa.Column("top_bullish_factors", postgresql.JSONB, server_default="[]"),
        sa.Column("top_bearish_factors", postgresql.JSONB, server_default="[]"),
        sa.Column("regime", postgresql.JSONB, server_default="{}"),
        sa.Column("rationale", sa.Text, nullable=True),
        sa.Column("risk_warnings", postgresql.JSONB, server_default="[]"),
        sa.Column("data_quality_score", sa.Float, server_default="1.0"),
        sa.Column("models_used", sa.Integer, server_default="0"),
        sa.Column("models_available", sa.Integer, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_recommendations_time", "recommendations", ["time"])
    op.create_index(
        "ix_recommendations_symbol_time",
        "recommendations",
        ["symbol", "time"],
        unique=True,
    )

    # Users table
    op.create_table(
        "users",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("email", sa.String(255), unique=True, nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("is_active", sa.Boolean, server_default="true"),
        sa.Column("is_verified", sa.Boolean, server_default="false"),
        sa.Column("subscription_tier", sa.String(20), server_default="free"),
        sa.Column("stripe_customer_id", sa.String(100), nullable=True),
        sa.Column("stripe_subscription_id", sa.String(100), nullable=True),
        sa.Column("watchlist", postgresql.JSONB, server_default="[]"),
        sa.Column("alert_settings", postgresql.JSONB, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    # API keys table
    op.create_table(
        "api_keys",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.BigInteger, nullable=False),
        sa.Column("key_hash", sa.String(255), unique=True, nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("is_active", sa.Boolean, server_default="true"),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("requests_count", sa.BigInteger, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_api_keys_user_id", "api_keys", ["user_id"])
    op.create_index("ix_api_keys_key_hash", "api_keys", ["key_hash"], unique=True)

    # Model performance table
    op.create_table(
        "model_performance",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("model_name", sa.String(100), nullable=False),
        sa.Column("symbol", sa.String(20), nullable=True),
        sa.Column("ic", sa.Float, nullable=True),
        sa.Column("icir", sa.Float, nullable=True),
        sa.Column("hit_rate", sa.Float, nullable=True),
        sa.Column("sharpe", sa.Float, nullable=True),
        sa.Column("metadata", postgresql.JSONB, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("date", "model_name", "symbol", name="uq_model_performance"),
    )
    op.create_index("ix_model_performance_date", "model_performance", ["date"])
    op.create_index("ix_model_performance_model_date", "model_performance", ["model_name", "date"])

    # Agent outputs table
    op.create_table(
        "agent_outputs",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("agent_name", sa.String(50), nullable=False),
        sa.Column("sentiment_score", sa.Float, nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("key_events", postgresql.JSONB, server_default="[]"),
        sa.Column("bullish_factors", postgresql.JSONB, server_default="[]"),
        sa.Column("bearish_factors", postgresql.JSONB, server_default="[]"),
        sa.Column("raw_sources", postgresql.JSONB, server_default="[]"),
        sa.Column("metadata", postgresql.JSONB, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_agent_outputs_time", "agent_outputs", ["time"])
    op.create_index("ix_agent_outputs_symbol_time", "agent_outputs", ["symbol", "time"])

    # Create TimescaleDB hypertables
    op.execute("SELECT create_hypertable('market_data', 'time', if_not_exists => TRUE)")
    op.execute("SELECT create_hypertable('features', 'time', if_not_exists => TRUE)")
    op.execute("SELECT create_hypertable('signals', 'time', if_not_exists => TRUE)")


def downgrade() -> None:
    """Drop all tables."""
    op.drop_table("agent_outputs")
    op.drop_table("model_performance")
    op.drop_table("api_keys")
    op.drop_table("users")
    op.drop_table("recommendations")
    op.drop_table("signals")
    op.drop_table("features")
    op.drop_table("market_data")
    op.execute("DROP EXTENSION IF EXISTS timescaledb")
