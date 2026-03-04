"""SQLAlchemy ORM models for QuantFlow database.

Defines tables for market data, features, signals, and user data.
Optimized for TimescaleDB time-series storage.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    type_annotation_map = {
        dict[str, Any]: JSONB,
    }


class TimestampMixin:
    """Mixin for models with created_at and updated_at timestamps."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class MarketData(Base, TimestampMixin):
    """Market data table for OHLCV and related data.

    Stores time-series market data with TimescaleDB hypertable.
    """

    __tablename__ = "market_data"

    time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        primary_key=True,
        nullable=False,
        index=True,
    )
    symbol: Mapped[str] = mapped_column(String(20), primary_key=True, nullable=False)
    open: Mapped[float | None] = mapped_column(Float, nullable=True)
    high: Mapped[float | None] = mapped_column(Float, nullable=True)
    low: Mapped[float | None] = mapped_column(Float, nullable=True)
    close: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    vwap: Mapped[float | None] = mapped_column(Float, nullable=True)
    adjusted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    source: Mapped[str | None] = mapped_column(String(50), nullable=True)

    __table_args__ = (
        Index("ix_market_data_symbol_time", "symbol", "time"),
        {"timescaledb_hypertable": {"time_column_name": "time"}},
    )

    def __repr__(self) -> str:
        return f"<MarketData(symbol={self.symbol}, time={self.time}, close={self.close})>"


class Feature(Base, TimestampMixin):
    """Feature table for computed features.

    Stores engineered features with versioning for reproducibility.
    """

    __tablename__ = "features"

    time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        primary_key=True,
        nullable=False,
    )
    symbol: Mapped[str] = mapped_column(String(20), primary_key=True, nullable=False)
    feature: Mapped[str] = mapped_column(String(100), primary_key=True, nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    __table_args__ = (
        Index("ix_features_symbol_time", "symbol", "time"),
        Index("ix_features_symbol_feature", "symbol", "feature"),
        {"timescaledb_hypertable": {"time_column_name": "time"}},
    )

    def __repr__(self) -> str:
        return f"<Feature(symbol={self.symbol}, feature={self.feature}, value={self.value})>"


class Signal(Base, TimestampMixin):
    """Signal table for model outputs.

    Stores signals from all models with confidence and metadata.
    """

    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    signal: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    forecast_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    forecast_std: Mapped[float | None] = mapped_column(Float, nullable=True)
    regime: Mapped[str | None] = mapped_column(String(50), nullable=True)
    model_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)

    __table_args__ = (
        Index("ix_signals_symbol_time", "symbol", "time"),
        Index("ix_signals_model_time", "model_name", "time"),
        {"timescaledb_hypertable": {"time_column_name": "time"}},
    )

    def __repr__(self) -> str:
        return f"<Signal(symbol={self.symbol}, model={self.model_name}, signal={self.signal})>"


class Recommendation(Base, TimestampMixin):
    """Recommendation table for final Buy/Hold/Sell signals.

    Stores aggregated recommendations with full explainability.
    """

    __tablename__ = "recommendations"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    recommendation: Mapped[str] = mapped_column(String(20), nullable=False)
    signal_strength: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    position_size: Mapped[float] = mapped_column(Float, nullable=False)
    expected_return_21d: Mapped[float | None] = mapped_column(Float, nullable=True)
    expected_vol_21d: Mapped[float | None] = mapped_column(Float, nullable=True)
    var_95_1d: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_drawdown_estimate: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_contributions: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    top_bullish_factors: Mapped[list[str]] = mapped_column(JSONB, default=list)
    top_bearish_factors: Mapped[list[str]] = mapped_column(JSONB, default=list)
    regime: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    risk_warnings: Mapped[list[str]] = mapped_column(JSONB, default=list)
    data_quality_score: Mapped[float] = mapped_column(Float, default=1.0)
    models_used: Mapped[int] = mapped_column(Integer, default=0)
    models_available: Mapped[int] = mapped_column(Integer, default=0)

    __table_args__ = (Index("ix_recommendations_symbol_time", "symbol", "time", unique=True),)

    def __repr__(self) -> str:
        return f"<Recommendation(symbol={self.symbol}, rec={self.recommendation})>"


class User(Base, TimestampMixin):
    """User table for SaaS authentication."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    subscription_tier: Mapped[str] = mapped_column(String(20), default="free", nullable=False)
    stripe_customer_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    stripe_subscription_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    watchlist: Mapped[list[str]] = mapped_column(JSONB, default=list)
    alert_settings: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)

    def __repr__(self) -> str:
        return f"<User(email={self.email}, tier={self.subscription_tier})>"


class ApiKey(Base, TimestampMixin):
    """API key table for programmatic access."""

    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    key_hash: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    requests_count: Mapped[int] = mapped_column(BigInteger, default=0)

    def __repr__(self) -> str:
        return f"<ApiKey(name={self.name}, user_id={self.user_id})>"


class ModelPerformance(Base, TimestampMixin):
    """Model performance tracking for IC-based weighting."""

    __tablename__ = "model_performance"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    symbol: Mapped[str | None] = mapped_column(String(20), nullable=True)  # NULL for universe-wide
    ic: Mapped[float | None] = mapped_column(Float, nullable=True)  # Information Coefficient
    icir: Mapped[float | None] = mapped_column(Float, nullable=True)  # IC / std(IC)
    hit_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    sharpe: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)

    __table_args__ = (
        UniqueConstraint("date", "model_name", "symbol", name="uq_model_performance"),
        Index("ix_model_performance_model_date", "model_name", "date"),
    )

    def __repr__(self) -> str:
        return f"<ModelPerformance(model={self.model_name}, date={self.date}, ic={self.ic})>"


class AgentOutput(Base, TimestampMixin):
    """Agent output for LLM-based market intelligence."""

    __tablename__ = "agent_outputs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    agent_name: Mapped[str] = mapped_column(String(50), nullable=False)
    sentiment_score: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    key_events: Mapped[list[str]] = mapped_column(JSONB, default=list)
    bullish_factors: Mapped[list[str]] = mapped_column(JSONB, default=list)
    bearish_factors: Mapped[list[str]] = mapped_column(JSONB, default=list)
    raw_sources: Mapped[list[str]] = mapped_column(JSONB, default=list)
    model_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)

    __table_args__ = (Index("ix_agent_outputs_symbol_time", "symbol", "time"),)

    def __repr__(self) -> str:
        return f"<AgentOutput(symbol={self.symbol}, agent={self.agent_name})>"
