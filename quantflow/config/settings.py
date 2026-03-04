"""Application settings using pydantic-settings.

All settings are loaded from environment variables with sensible defaults.
Sensitive values (API keys, secrets) must be set via environment variables.

The .env file is resolved relative to this file's location so settings load
correctly regardless of the working directory (tests, CLI, Docker, etc.).
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root is three levels up from quantflow/config/settings.py
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Attributes:
        environment: Current environment (development, staging, production).
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        debug: Enable debug mode.

        Database settings:
        database_url: PostgreSQL connection URL (asyncpg driver).
        redis_url: Redis connection URL.

        API keys for data providers:
        alpha_vantage_api_key: Alpha Vantage API key.
        polygon_api_key: Polygon.io API key.
        fred_api_key: FRED API key.
        quandl_api_key: Quandl/Nasdaq Data Link API key.

        LLM API keys for agentic intelligence:
        openai_api_key: OpenAI API key.
        anthropic_api_key: Anthropic API key.
        perplexity_api_key: Perplexity API key.

        Auth settings for SaaS:
        jwt_secret_key: Secret key for JWT token signing.
        jwt_algorithm: JWT signing algorithm.
        jwt_expire_minutes: JWT token expiration time in minutes.

        Stripe settings for billing:
        stripe_secret_key: Stripe secret key.
        stripe_webhook_secret: Stripe webhook signing secret.

        Model settings:
        mlflow_tracking_uri: MLflow tracking server URI.
        model_cache_ttl_seconds: TTL for model prediction cache.
    """

    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        # Empty string values in .env (e.g. API_KEY=) are treated as unset,
        # so optional SecretStr fields remain None instead of SecretStr("").
        env_ignore_empty=True,
    )

    # Application
    environment: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"
    debug: bool = False

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://quantflow:quantflow@localhost:5432/quantflow",
        description="PostgreSQL connection URL with asyncpg driver",
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL",
    )

    # Data Provider API Keys
    alpha_vantage_api_key: SecretStr | None = None
    polygon_api_key: SecretStr | None = None
    fred_api_key: SecretStr | None = None
    quandl_api_key: SecretStr | None = None

    # LLM API Keys
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    perplexity_api_key: SecretStr | None = None

    # Authentication
    jwt_secret_key: SecretStr = Field(
        default=SecretStr("change-me-in-production-with-secure-random-key"),
        description="Secret key for JWT token signing",
    )
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24 * 7  # 7 days

    # Stripe Billing
    stripe_secret_key: SecretStr | None = None
    stripe_webhook_secret: SecretStr | None = None

    # MLflow
    mlflow_tracking_uri: str = "file:./mlruns"

    # Caching
    model_cache_ttl_seconds: int = 300  # 5 minutes
    feature_cache_ttl_seconds: int = 300  # 5 minutes
    signal_cache_ttl_seconds: int = 60  # 1 minute

    # Rate Limiting (per tier)
    rate_limit_free_requests_per_day: int = 100
    rate_limit_premium_requests_per_day: int = 1000
    rate_limit_institutional_requests_per_day: int = 10000

    # Backtesting
    backtest_commission_per_share: float = 0.005  # $0.005/share
    backtest_slippage_bps: float = 5.0  # 5 basis points
    backtest_market_impact_coefficient: float = 10.0

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is a valid Python logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}, got {v}")
        return v_upper

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment is one of the allowed values."""
        valid_envs = {"development", "staging", "production"}
        if v not in valid_envs:
            raise ValueError(f"environment must be one of {valid_envs}, got {v}")
        return v

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"

    @property
    def is_staging(self) -> bool:
        """Check if running in staging environment."""
        return self.environment == "staging"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns:
        Settings: Application settings loaded from environment.
    """
    return Settings()


# Global settings instance
settings = get_settings()
