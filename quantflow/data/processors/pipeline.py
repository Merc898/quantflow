"""Data processing pipeline for QuantFlow.

Orchestrates fetching, cleaning, feature engineering, and caching
for a given symbol.  All I/O is async.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel

from quantflow.config.constants import MAX_MISSING_DATA_PCT
from quantflow.config.logging import get_logger
from quantflow.data.features import compute_all_features
from quantflow.data.fetchers.base import (
    DataFetchError,
    DataQualityError,
    FundamentalsData,
)
from quantflow.data.fetchers.yfinance_fetcher import YFinanceFetcher

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Config and output schemas
# ---------------------------------------------------------------------------


class PipelineConfig(BaseModel):
    """Configuration for a single pipeline run.

    Attributes:
        symbol: Ticker symbol to process.
        start: Start date for historical data fetch.
        end: End date (exclusive).
        interval: Bar frequency — "1d" for daily.
        outlier_iqr_threshold: IQR multiplier for outlier removal.
        store_to_db: Whether to persist results to TimescaleDB.
        cache_ttl_seconds: Redis cache TTL.
    """

    symbol: str
    start: datetime
    end: datetime
    interval: str = "1d"
    outlier_iqr_threshold: float = 5.0
    store_to_db: bool = False
    cache_ttl_seconds: int = 300


@dataclass
class ProcessedData:
    """Output of a full pipeline run.

    Attributes:
        symbol: Ticker symbol.
        ohlcv: Cleaned OHLCV DataFrame.
        features: Feature DataFrame (same index as ohlcv).
        fundamentals: Latest fundamental snapshot or None.
        data_quality_score: Fraction of rows passing quality checks
            (0.0 = completely bad, 1.0 = perfect).
        errors: Non-fatal errors encountered during processing.
        warnings: Warnings encountered during processing.
    """

    symbol: str
    ohlcv: pd.DataFrame
    features: pd.DataFrame
    fundamentals: FundamentalsData | None = None
    data_quality_score: float = 1.0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Outlier removal
# ---------------------------------------------------------------------------


def remove_outliers_iqr(df: pd.DataFrame, threshold: float = 5.0) -> pd.DataFrame:
    """Remove rows with extreme price movements using IQR fencing.

    Rows where the absolute daily return exceeds
    ``threshold × IQR`` above Q3 are treated as data errors and removed.
    Volume outliers are winsorised in place rather than dropped.

    Args:
        df: OHLCV DataFrame with a ``close`` column.
        threshold: IQR multiplier for the fence.

    Returns:
        Cleaned DataFrame with outlier rows removed.
    """
    if "close" not in df.columns or len(df) < 10:
        return df

    log_ret = np.log(df["close"] / df["close"].shift(1)).abs().fillna(0)
    q1 = log_ret.quantile(0.25)
    q3 = log_ret.quantile(0.75)
    iqr = q3 - q1
    fence = q3 + threshold * iqr

    mask = log_ret <= fence
    n_removed = (~mask).sum()
    if n_removed > 0:
        logger.warning(
            "Removed outlier rows",
            n_removed=int(n_removed),
            fence=float(fence),
        )
    return df.loc[mask].copy()


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------


async def run_pipeline(
    symbol: str,
    start: datetime | None = None,
    end: datetime | None = None,
    config: PipelineConfig | None = None,
) -> ProcessedData:
    """Run the full data pipeline for a symbol.

    Steps:
    1. Fetch raw OHLCV from YFinance.
    2. Remove outliers.
    3. Validate data quality.
    4. Compute features.
    5. Attempt fundamentals fetch (non-fatal if it fails).

    Args:
        symbol: Ticker symbol (e.g. "AAPL").
        start: Start datetime (UTC).  Defaults to 2 years ago.
        end: End datetime (UTC).  Defaults to today.
        config: Optional full :class:`PipelineConfig` (overrides symbol/start/end
            if provided).

    Returns:
        :class:`ProcessedData` with OHLCV, features, and fundamentals.

    Raises:
        DataFetchError: If the primary data fetch fails.
        DataQualityError: If cleaned data still fails quality checks.
    """
    now = datetime.now(tz=UTC)

    if config is None:
        config = PipelineConfig(
            symbol=symbol,
            start=start or now.replace(year=now.year - 2),
            end=end or now,
        )

    logger.info(
        "Starting data pipeline",
        symbol=config.symbol,
        start=str(config.start),
        end=str(config.end),
    )

    errors: list[str] = []
    warnings: list[str] = []

    # ------------------------------------------------------------------
    # Step 1 — Fetch raw OHLCV
    # ------------------------------------------------------------------
    fetcher = YFinanceFetcher()
    ohlcv = await fetcher.fetch_ohlcv(
        symbol=config.symbol,
        start=config.start,
        end=config.end,
        interval=config.interval,  # type: ignore[arg-type]
    )

    # ------------------------------------------------------------------
    # Step 2 — Outlier removal
    # ------------------------------------------------------------------
    ohlcv = remove_outliers_iqr(ohlcv, threshold=config.outlier_iqr_threshold)

    # ------------------------------------------------------------------
    # Step 3 — Data quality gate
    # ------------------------------------------------------------------
    report = await fetcher.validate(ohlcv)
    if not report.is_valid:
        raise DataQualityError(
            f"Cleaned data for {config.symbol} failed quality checks: {report.errors}"
        )

    errors.extend(report.errors)
    warnings.extend(report.warnings)

    missing_pct = ohlcv["close"].isna().mean()
    data_quality_score = max(0.0, 1.0 - missing_pct / MAX_MISSING_DATA_PCT)

    # ------------------------------------------------------------------
    # Step 4 — Feature engineering
    # ------------------------------------------------------------------
    features = compute_all_features(ohlcv)

    # ------------------------------------------------------------------
    # Step 5 — Fundamentals (optional, best-effort)
    # ------------------------------------------------------------------
    fundamentals: FundamentalsData | None = None
    try:
        fundamentals = await fetcher.fetch_fundamentals(config.symbol)
    except (DataFetchError, Exception) as exc:
        warnings.append(f"Fundamentals fetch failed (non-fatal): {exc}")
        logger.warning(
            "Fundamentals fetch failed",
            symbol=config.symbol,
            error=str(exc),
        )

    logger.info(
        "Data pipeline complete",
        symbol=config.symbol,
        ohlcv_rows=len(ohlcv),
        feature_columns=len(features.columns),
        data_quality_score=round(data_quality_score, 3),
    )

    return ProcessedData(
        symbol=config.symbol,
        ohlcv=ohlcv,
        features=features,
        fundamentals=fundamentals,
        data_quality_score=data_quality_score,
        errors=errors,
        warnings=warnings,
    )
