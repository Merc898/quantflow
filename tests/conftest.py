"""Pytest configuration and fixtures."""

import asyncio
from collections.abc import Generator

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Generate sample OHLCV data for testing.

    Returns:
        DataFrame with DatetimeIndex and columns: open, high, low, close, volume
    """
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=252, freq="B", tz="UTC")

    # Generate realistic price data using geometric Brownian motion
    returns = np.random.normal(0.0005, 0.02, len(dates))
    close = 100 * np.cumprod(1 + returns)

    # Generate OHLCV data
    high = close * (1 + np.abs(np.random.normal(0, 0.01, len(dates))))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, len(dates))))
    open_price = close * (1 + np.random.normal(0, 0.005, len(dates)))
    volume = np.random.randint(1_000_000, 10_000_000, len(dates))

    df = pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )

    # Ensure OHLC consistency: high >= max(open, close), low <= min(open, close)
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].min(axis=1)

    return df


@pytest.fixture
def sample_returns() -> pd.Series:
    """Generate sample returns for testing.

    Returns:
        Series of daily returns
    """
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=252, freq="B", tz="UTC")
    returns = pd.Series(np.random.normal(0.0005, 0.02, len(dates)), index=dates, name="returns")
    return returns
