"""SQLAlchemy async engine and session management.

Provides async database connectivity with connection pooling and
TimescaleDB support for time-series data.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from quantflow.config.settings import settings


def get_async_engine() -> AsyncEngine:
    """Create an async SQLAlchemy engine.

    Uses NullPool in development for simpler connection management.
    Configures connection pooling for production.

    Returns:
        AsyncEngine: SQLAlchemy async engine instance.
    """
    # Use NullPool (no pooling) in development for simpler debugging.
    # NullPool does not support pool_size / max_overflow, so only pass those
    # arguments when a real connection pool is in use (staging / production).
    use_null_pool = settings.is_development
    pool_kwargs: dict[str, Any] = {
        "echo": settings.debug,
        "future": True,
        "pool_pre_ping": True,
        "poolclass": NullPool if use_null_pool else None,
    }
    if not use_null_pool:
        pool_kwargs["pool_size"] = 10 if settings.is_production else 5
        pool_kwargs["max_overflow"] = 20 if settings.is_production else 10

    engine = create_async_engine(settings.database_url, **pool_kwargs)

    return engine


# Global engine instance
engine: AsyncEngine = get_async_engine()

# Session factory
async_session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)
# Alias used by dependencies.py
AsyncSessionLocal = async_session_factory


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Get an async database session.

    Yields:
        AsyncSession: Database session for the current request.

    Example:
        async for session in get_async_session():
            result = await session.execute(query)
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_session_context() -> AsyncGenerator[AsyncSession, None]:
    """Get an async database session as a context manager.

    Yields:
        AsyncSession: Database session for the current context.

    Example:
        async with get_session_context() as session:
            result = await session.execute(query)
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Initialize database tables and TimescaleDB extensions.

    Creates all tables if they don't exist and sets up TimescaleDB
    hypertables for time-series data.
    """
    from quantflow.db.models import Base

    async with engine.begin() as conn:
        # Create TimescaleDB extension
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb"))

        # Create all tables
        await conn.run_sync(Base.metadata.create_all)

        # Create hypertables for time-series tables
        await _create_hypertables(conn)


async def _create_hypertables(conn: Any) -> None:
    """Create TimescaleDB hypertables for time-series data.

    Args:
        conn: Database connection.
    """
    # Create hypertable for market_data
    await conn.execute(
        text("SELECT create_hypertable('market_data', 'time', if_not_exists => TRUE)")
    )

    # Create hypertable for features
    await conn.execute(text("SELECT create_hypertable('features', 'time', if_not_exists => TRUE)"))

    # Create hypertable for signals
    await conn.execute(text("SELECT create_hypertable('signals', 'time', if_not_exists => TRUE)"))


async def close_db() -> None:
    """Close database connections.

    Should be called during application shutdown.
    """
    await engine.dispose()
