"""Base interface for all market data fetchers.

Defines the contract every fetcher must implement so the pipeline
can use any source interchangeably.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict

from quantflow.config.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Config / schema types
# ---------------------------------------------------------------------------


class FetcherConfig(BaseModel):
    """Configuration for a data fetcher.

    Attributes:
        timeout_seconds: HTTP request timeout in seconds.
        max_retries: Maximum number of retry attempts.
        retry_delay_seconds: Delay between retries in seconds.
    """

    timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_delay_seconds: float = 1.0


class FundamentalsData(BaseModel):
    """Snapshot of key fundamental metrics for a symbol.

    All ratio fields are Optional because not every data source
    provides the full set.
    """

    symbol: str
    as_of_date: datetime

    # Valuation
    pe_ratio: float | None = None
    pb_ratio: float | None = None
    ps_ratio: float | None = None
    ev_ebitda: float | None = None

    # Quality / Profitability
    roe: float | None = None
    roa: float | None = None
    gross_margin: float | None = None
    fcf_yield: float | None = None

    # Leverage
    debt_equity: float | None = None
    current_ratio: float | None = None

    # Growth
    revenue_growth_yoy: float | None = None
    earnings_growth_yoy: float | None = None

    # Earnings surprise (most recent quarter, avg of last 4)
    earnings_surprise_1q: float | None = None
    earnings_surprise_4q_avg: float | None = None

    # Market cap (USD)
    market_cap: float | None = None

    model_config = ConfigDict(extra="ignore")


@dataclass
class ValidationReport:
    """Results of data quality validation.

    Attributes:
        is_valid: True only when all checks pass.
        errors: List of fatal errors (block processing).
        warnings: List of non-fatal warnings (log and continue).
        missing_pct: Fraction of missing close prices.
        row_count: Total number of rows inspected.
    """

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    missing_pct: float = 0.0
    row_count: int = 0


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class DataFetchError(Exception):
    """Raised when a data fetch fails after all retries."""


class DataQualityError(Exception):
    """Raised when fetched data fails quality validation."""


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

IntervalLiteral = Literal["1m", "5m", "1h", "1d"]

_REQUIRED_OHLCV_COLS = {"open", "high", "low", "close", "volume"}


class BaseDataFetcher(ABC):
    """Abstract base class for all QuantFlow data fetchers.

    Every concrete fetcher must:
    - Return timezone-aware (UTC) DataFrames with a DatetimeIndex.
    - Return columns: open, high, low, close, volume (+ optional vwap).
    - Implement :meth:`fetch_ohlcv` and :meth:`fetch_fundamentals`.
    - Call :meth:`validate` before returning data.

    Args:
        config: Fetcher configuration (timeout, retries, …).
    """

    def __init__(self, config: FetcherConfig | None = None) -> None:
        """Initialise the fetcher with optional configuration.

        Args:
            config: Fetcher-level settings.  Defaults to ``FetcherConfig()``.
        """
        self.config = config or FetcherConfig()
        self._logger = get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    async def fetch_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: IntervalLiteral = "1d",
    ) -> pd.DataFrame:
        """Fetch OHLCV data for *symbol* between *start* and *end*.

        Args:
            symbol: Ticker symbol (e.g. "AAPL").
            start: Inclusive start datetime (timezone-aware, UTC).
            end: Exclusive end datetime (timezone-aware, UTC).
            interval: Bar interval — one of "1m", "5m", "1h", "1d".

        Returns:
            DataFrame with DatetimeIndex (UTC) and columns
            ``[open, high, low, close, volume]``.  ``vwap`` is optional.
            Columns are ``np.float64`` except ``volume`` which is ``int64``.

        Raises:
            DataFetchError: If the network request fails after all retries.
            DataQualityError: If the returned data fails validation.
        """
        ...

    @abstractmethod
    async def fetch_fundamentals(self, symbol: str) -> FundamentalsData:
        """Fetch the latest fundamental data for *symbol*.

        Args:
            symbol: Ticker symbol (e.g. "AAPL").

        Returns:
            :class:`FundamentalsData` snapshot as of the most recent filing.

        Raises:
            DataFetchError: If the request fails.
        """
        ...

    # ------------------------------------------------------------------
    # Shared validation helper
    # ------------------------------------------------------------------

    async def validate(self, df: pd.DataFrame) -> ValidationReport:
        """Run data quality checks on an OHLCV DataFrame.

        Checks performed:
        - Required columns present.
        - DatetimeIndex is timezone-aware.
        - No duplicate timestamps.
        - No future timestamps.
        - OHLC consistency: H >= max(O, C) and L <= min(O, C).
        - Volume >= 0.
        - Close price > 0.
        - Missing close price fraction below threshold.

        Args:
            df: OHLCV DataFrame to validate.

        Returns:
            :class:`ValidationReport` with ``is_valid`` flag plus error /
            warning lists.
        """
        from quantflow.config.constants import MAX_MISSING_DATA_PCT, MIN_PRICE

        errors: list[str] = []
        warnings: list[str] = []

        # 1. Required columns
        missing_cols = _REQUIRED_OHLCV_COLS - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            return ValidationReport(
                is_valid=False,
                errors=errors,
                row_count=len(df),
            )

        # 2. Timezone-aware index
        if df.index.tz is None:
            errors.append("DatetimeIndex must be timezone-aware (UTC).")

        # 3. Duplicate timestamps
        n_dups = df.index.duplicated().sum()
        if n_dups > 0:
            errors.append(f"{n_dups} duplicate timestamps found.")

        # 4. Future timestamps (only safe when index is tz-aware; tz-naive
        #    indexes have already been flagged above and can't be compared
        #    against a tz-aware Timestamp without raising TypeError).
        if df.index.tz is not None:
            now_utc = pd.Timestamp.utcnow()
            n_future = (df.index > now_utc).sum()
            if n_future > 0:
                errors.append(f"{n_future} rows have timestamps in the future.")

        # 5. OHLC consistency
        ohlc_bad = df.loc[
            (df["high"] < df["open"])
            | (df["high"] < df["close"])
            | (df["low"] > df["open"])
            | (df["low"] > df["close"])
        ]
        if len(ohlc_bad) > 0:
            warnings.append(f"{len(ohlc_bad)} bars with OHLC inconsistency.")

        # 6. Volume >= 0
        neg_vol = (df["volume"] < 0).sum()
        if neg_vol > 0:
            errors.append(f"{neg_vol} rows have negative volume.")

        # 7. Minimum price
        low_price = (df["close"] < MIN_PRICE).sum()
        if low_price > 0:
            warnings.append(f"{low_price} bars with close below MIN_PRICE ({MIN_PRICE}).")

        # 8. Missing data
        missing_pct = df["close"].isna().mean()
        if missing_pct > MAX_MISSING_DATA_PCT:
            errors.append(
                f"Missing close data {missing_pct:.1%} exceeds threshold "
                f"{MAX_MISSING_DATA_PCT:.1%}."
            )

        is_valid = len(errors) == 0
        report = ValidationReport(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            missing_pct=float(missing_pct),
            row_count=len(df),
        )

        if not is_valid:
            self._logger.error(
                "Data validation failed",
                errors=errors,
                warnings=warnings,
            )
        elif warnings:
            self._logger.warning("Data validation warnings", warnings=warnings)

        return report
