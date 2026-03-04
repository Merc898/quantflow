"""YFinance data fetcher — free, no API key required.

Uses the yfinance library to download OHLCV and basic fundamental data.
This is the primary fallback source when paid APIs are unavailable.
"""

from datetime import datetime

import numpy as np
import pandas as pd

from quantflow.config.logging import get_logger
from quantflow.data.fetchers.base import (
    BaseDataFetcher,
    DataFetchError,
    DataQualityError,
    FetcherConfig,
    FundamentalsData,
    IntervalLiteral,
    ValidationReport,
)

logger = get_logger(__name__)

# Mapping from QuantFlow interval strings to yfinance interval strings
_INTERVAL_MAP: dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "1h": "1h",
    "1d": "1d",
}

_YFINANCE_OHLCV_COLUMNS = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
}


class YFinanceFetcher(BaseDataFetcher):
    """Fetches OHLCV and fundamental data from Yahoo Finance via yfinance.

    Free, no API key required.  Rate limits apply (~2 000 requests/hour).
    Data is split/dividend adjusted by default.

    Args:
        config: Optional fetcher configuration.
    """

    def __init__(self, config: FetcherConfig | None = None) -> None:
        """Initialise the YFinance fetcher.

        Args:
            config: Fetcher configuration (timeout, retries, etc.).
        """
        super().__init__(config)
        self._logger = get_logger(__name__)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def fetch_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: IntervalLiteral = "1d",
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance.

        Args:
            symbol: Ticker symbol (e.g. "AAPL").
            start: Inclusive start datetime (timezone-aware, UTC).
            end: Exclusive end datetime (timezone-aware, UTC).
            interval: Bar interval — one of "1m", "5m", "1h", "1d".

        Returns:
            DataFrame with UTC DatetimeIndex and columns
            ``[open, high, low, close, volume]`` in float64 / int64.

        Raises:
            DataFetchError: If yfinance returns empty data or an error.
            DataQualityError: If validation fails.
        """
        import yfinance as yf

        yf_interval = _INTERVAL_MAP.get(interval, "1d")
        self._logger.info(
            "Fetching OHLCV from yfinance",
            symbol=symbol,
            start=str(start),
            end=str(end),
            interval=yf_interval,
        )

        try:
            ticker = yf.Ticker(symbol)
            raw: pd.DataFrame = ticker.history(
                start=start,
                end=end,
                interval=yf_interval,
                auto_adjust=True,  # split & dividend adjusted
                actions=False,
            )
        except Exception as exc:
            raise DataFetchError(
                f"yfinance failed to fetch {symbol}: {exc}"
            ) from exc

        if raw.empty:
            raise DataFetchError(
                f"yfinance returned empty DataFrame for {symbol} "
                f"({start} → {end}, interval={interval})"
            )

        df = self._normalise(raw)
        report: ValidationReport = await self.validate(df)
        if not report.is_valid:
            raise DataQualityError(
                f"Data quality check failed for {symbol}: {report.errors}"
            )

        return df

    async def fetch_fundamentals(self, symbol: str) -> FundamentalsData:
        """Fetch basic fundamental data from Yahoo Finance.

        Args:
            symbol: Ticker symbol (e.g. "AAPL").

        Returns:
            :class:`FundamentalsData` populated from yfinance ``.info``.

        Raises:
            DataFetchError: If the request fails.
        """
        import yfinance as yf

        self._logger.info("Fetching fundamentals from yfinance", symbol=symbol)
        try:
            info: dict = yf.Ticker(symbol).info  # type: ignore[assignment]
        except Exception as exc:
            raise DataFetchError(
                f"yfinance failed to fetch fundamentals for {symbol}: {exc}"
            ) from exc

        def _get(key: str) -> float | None:
            val = info.get(key)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return None
            try:
                return float(val)
            except (TypeError, ValueError):
                return None

        return FundamentalsData(
            symbol=symbol,
            as_of_date=datetime.utcnow(),
            pe_ratio=_get("trailingPE"),
            pb_ratio=_get("priceToBook"),
            ps_ratio=_get("priceToSalesTrailing12Months"),
            ev_ebitda=_get("enterpriseToEbitda"),
            roe=_get("returnOnEquity"),
            roa=_get("returnOnAssets"),
            gross_margin=_get("grossMargins"),
            fcf_yield=_get("freeCashflow"),
            debt_equity=_get("debtToEquity"),
            current_ratio=_get("currentRatio"),
            revenue_growth_yoy=_get("revenueGrowth"),
            earnings_growth_yoy=_get("earningsGrowth"),
            market_cap=_get("marketCap"),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _normalise(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Normalise a raw yfinance DataFrame to QuantFlow format.

        Args:
            raw: DataFrame returned by ``yf.Ticker.history()``.

        Returns:
            Normalised DataFrame with lowercase column names, UTC index,
            correct dtypes, and no NaN rows for all OHLCV columns.
        """
        df = raw.copy()

        # Rename columns to lowercase
        df = df.rename(columns=_YFINANCE_OHLCV_COLUMNS)

        # Keep only the columns we care about
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep].copy()

        # Ensure UTC timezone-aware index
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df.index.name = "time"

        # Cast types for numerical stability
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                df[col] = df[col].astype(np.float64)
        if "volume" in df.columns:
            df["volume"] = df["volume"].fillna(0).astype(np.int64)

        # Drop rows where all OHLC are NaN (trading halts etc.)
        df.dropna(subset=["close"], inplace=True)

        # Sort chronologically
        df.sort_index(inplace=True)

        return df
