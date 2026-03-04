"""Unit tests for the QuantFlow data pipeline (Phase 1).

Tests cover:
- BaseDataFetcher validation logic
- YFinanceFetcher normalisation and error handling (mocked)
- Feature engineering correctness and look-ahead-bias safeguards
- run_pipeline happy-path and error branches (mocked)
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quantflow.data.features import (
    compute_all_features,
    compute_atr,
    compute_bollinger_bands,
    compute_macd,
    compute_momentum,
    compute_realized_vol,
    compute_returns,
    compute_reversal,
    compute_rsi,
    compute_volume_features,
    rolling_zscore,
    winsorize,
)
from quantflow.data.fetchers.base import (
    DataFetchError,
    DataQualityError,
    FetcherConfig,
    FundamentalsData,
    ValidationReport,
)
from quantflow.data.fetchers.yfinance_fetcher import YFinanceFetcher
from quantflow.data.processors.pipeline import (
    ProcessedData,
    PipelineConfig,
    remove_outliers_iqr,
    run_pipeline,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _make_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Return a minimal synthetic OHLCV DataFrame (UTC index)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="B", tz="UTC")
    rets = rng.normal(5e-4, 0.015, n)
    close = 100.0 * np.cumprod(1 + rets)
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    volume = rng.integers(1_000_000, 10_000_000, n)

    df = pd.DataFrame(
        {
            "open": open_.astype(np.float64),
            "high": np.maximum(high, open_).astype(np.float64),
            "low": np.minimum(low, open_).astype(np.float64),
            "close": close.astype(np.float64),
            "volume": volume.astype(np.int64),
        },
        index=dates,
    )
    # Enforce OHLC consistency
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].min(axis=1)
    return df


# ===========================================================================
# BaseDataFetcher.validate
# ===========================================================================


class TestBaseDataFetcherValidation:
    """Tests for the shared validate() method in BaseDataFetcher."""

    @pytest.fixture()
    def fetcher(self) -> YFinanceFetcher:
        return YFinanceFetcher()

    @pytest.mark.asyncio
    async def test_valid_data_passes(self, fetcher: YFinanceFetcher) -> None:
        df = _make_ohlcv()
        report: ValidationReport = await fetcher.validate(df)
        assert report.is_valid is True
        assert report.errors == []
        assert report.row_count == len(df)

    @pytest.mark.asyncio
    async def test_missing_column_is_error(self, fetcher: YFinanceFetcher) -> None:
        df = _make_ohlcv().drop(columns=["volume"])
        report = await fetcher.validate(df)
        assert report.is_valid is False
        assert any("volume" in e for e in report.errors)

    @pytest.mark.asyncio
    async def test_naive_index_is_error(self, fetcher: YFinanceFetcher) -> None:
        df = _make_ohlcv()
        df.index = df.index.tz_localize(None)
        report = await fetcher.validate(df)
        assert report.is_valid is False
        assert any("timezone" in e.lower() for e in report.errors)

    @pytest.mark.asyncio
    async def test_negative_volume_is_error(self, fetcher: YFinanceFetcher) -> None:
        df = _make_ohlcv()
        df.loc[df.index[0], "volume"] = -1
        report = await fetcher.validate(df)
        assert report.is_valid is False

    @pytest.mark.asyncio
    async def test_ohlc_inconsistency_produces_warning(
        self, fetcher: YFinanceFetcher
    ) -> None:
        df = _make_ohlcv()
        # Force high < close (inconsistency)
        df.loc[df.index[0], "high"] = df.loc[df.index[0], "close"] - 1.0
        report = await fetcher.validate(df)
        # Should be valid (warning, not error)
        assert report.is_valid is True
        assert len(report.warnings) > 0

    @pytest.mark.asyncio
    async def test_excessive_missing_data_is_error(
        self, fetcher: YFinanceFetcher
    ) -> None:
        df = _make_ohlcv()
        # Introduce 10% missing close prices
        df.loc[df.index[:30], "close"] = np.nan
        report = await fetcher.validate(df)
        assert report.is_valid is False
        assert any("missing" in e.lower() for e in report.errors)


# ===========================================================================
# YFinanceFetcher._normalise
# ===========================================================================


class TestYFinanceFetcherNormalise:
    """Tests for internal normalisation logic (no network required)."""

    @pytest.fixture()
    def fetcher(self) -> YFinanceFetcher:
        return YFinanceFetcher()

    def _make_raw_yf_df(self, n: int = 50) -> pd.DataFrame:
        """Simulate raw yfinance output (capitalised columns, naive index)."""
        rng = np.random.default_rng(0)
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        close = 150.0 + np.cumsum(rng.normal(0, 1, n))
        return pd.DataFrame(
            {
                "Open": close + rng.normal(0, 0.5, n),
                "High": close + np.abs(rng.normal(0, 1, n)),
                "Low": close - np.abs(rng.normal(0, 1, n)),
                "Close": close,
                "Volume": rng.integers(1_000_000, 5_000_000, n),
                "Dividends": np.zeros(n),
                "Stock Splits": np.zeros(n),
            },
            index=dates,
        )

    def test_columns_lowercased(self, fetcher: YFinanceFetcher) -> None:
        raw = self._make_raw_yf_df()
        result = fetcher._normalise(raw)
        assert set(result.columns) >= {"open", "high", "low", "close", "volume"}
        assert "Dividends" not in result.columns

    def test_index_is_utc(self, fetcher: YFinanceFetcher) -> None:
        raw = self._make_raw_yf_df()
        result = fetcher._normalise(raw)
        assert result.index.tz is not None
        assert str(result.index.tz) == "UTC"

    def test_close_dtype_is_float64(self, fetcher: YFinanceFetcher) -> None:
        raw = self._make_raw_yf_df()
        result = fetcher._normalise(raw)
        assert result["close"].dtype == np.float64

    def test_volume_dtype_is_int64(self, fetcher: YFinanceFetcher) -> None:
        raw = self._make_raw_yf_df()
        result = fetcher._normalise(raw)
        assert result["volume"].dtype == np.int64

    def test_sorted_chronologically(self, fetcher: YFinanceFetcher) -> None:
        raw = self._make_raw_yf_df()
        # Shuffle the raw data
        raw = raw.sample(frac=1, random_state=0)
        result = fetcher._normalise(raw)
        assert result.index.is_monotonic_increasing


# ===========================================================================
# Feature engineering
# ===========================================================================


class TestComputeReturns:
    def test_ret_1d_shape(self, sample_ohlcv_data: pd.DataFrame) -> None:
        feats = compute_returns(sample_ohlcv_data["close"])
        assert "ret_1d" in feats.columns
        assert len(feats) == len(sample_ohlcv_data)

    def test_first_ret_1d_is_nan(self, sample_ohlcv_data: pd.DataFrame) -> None:
        feats = compute_returns(sample_ohlcv_data["close"])
        assert np.isnan(feats["ret_1d"].iloc[0])

    def test_log_ret_1d_first_is_nan(self, sample_ohlcv_data: pd.DataFrame) -> None:
        feats = compute_returns(sample_ohlcv_data["close"])
        assert np.isnan(feats["log_ret_1d"].iloc[0])

    def test_no_lookahead_bias(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """ret_1d on day t must only use close[t] and close[t-1]."""
        close = sample_ohlcv_data["close"]
        feats = compute_returns(close)
        # Manually verify day 1
        expected = close.iloc[1] / close.iloc[0] - 1
        assert abs(feats["ret_1d"].iloc[1] - expected) < 1e-10


class TestComputeMomentum:
    def test_columns_present(self, sample_ohlcv_data: pd.DataFrame) -> None:
        feats = compute_momentum(sample_ohlcv_data["close"])
        assert "mom_6_1" in feats.columns
        assert "mom_12_1" in feats.columns

    def test_skips_recent_month(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """mom_6_1 on day t skips the most recent 21 days."""
        close = sample_ohlcv_data["close"]
        feats = compute_momentum(close)
        # At index 126, mom_6_1 = close[126-21] / close[126-126] - 1
        t = 130
        skip = 21
        lookback = 126
        expected = close.iloc[t - skip] / close.iloc[t - lookback] - 1
        assert abs(feats["mom_6_1"].iloc[t] - expected) < 1e-10


class TestComputeRealizedVol:
    def test_annualised(self, sample_ohlcv_data: pd.DataFrame) -> None:
        feats = compute_realized_vol(sample_ohlcv_data["close"])
        assert "realized_vol_21d" in feats.columns
        # Annualised vol should be in a realistic range for synthetic data
        val = feats["realized_vol_21d"].dropna().iloc[-1]
        assert 0.0 < val < 5.0  # 0–500% annual vol


class TestComputeVolumeFeatures:
    def test_columns(self, sample_ohlcv_data: pd.DataFrame) -> None:
        feats = compute_volume_features(
            sample_ohlcv_data["close"], sample_ohlcv_data["volume"]
        )
        assert "dollar_volume_21d" in feats.columns
        assert "amihud_illiquidity" in feats.columns
        assert "volume_z_score" in feats.columns

    def test_dollar_volume_positive(self, sample_ohlcv_data: pd.DataFrame) -> None:
        feats = compute_volume_features(
            sample_ohlcv_data["close"], sample_ohlcv_data["volume"]
        )
        valid = feats["dollar_volume_21d"].dropna()
        assert (valid > 0).all()


class TestComputeRSI:
    def test_rsi_range(self, sample_ohlcv_data: pd.DataFrame) -> None:
        feats = compute_rsi(sample_ohlcv_data["close"])
        valid = feats["rsi_14"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()


class TestComputeMACD:
    def test_columns(self, sample_ohlcv_data: pd.DataFrame) -> None:
        feats = compute_macd(sample_ohlcv_data["close"])
        assert "macd_signal" in feats.columns
        assert "macd_hist" in feats.columns


class TestComputeBollingerBands:
    def test_position_bounded(self, sample_ohlcv_data: pd.DataFrame) -> None:
        feats = compute_bollinger_bands(sample_ohlcv_data["close"])
        # Not strictly [0,1] due to extreme moves but should be finite
        valid = feats["bb_position"].dropna()
        assert np.isfinite(valid).all()


class TestComputeATR:
    def test_atr_positive(self, sample_ohlcv_data: pd.DataFrame) -> None:
        feats = compute_atr(
            sample_ohlcv_data["high"],
            sample_ohlcv_data["low"],
            sample_ohlcv_data["close"],
        )
        valid = feats["atr_14"].dropna()
        assert (valid > 0).all()


class TestStandardisationHelpers:
    def test_winsorize_clips(self) -> None:
        s = pd.Series([-100.0, 0.0, 1.0, 2.0, 100.0])
        w = winsorize(s)
        assert w.min() > -100.0
        assert w.max() < 100.0

    def test_rolling_zscore_clipped(self) -> None:
        rng = np.random.default_rng(0)
        s = pd.Series(rng.normal(0, 1, 500))
        z = rolling_zscore(s, window=100, clip=3.0)
        valid = z.dropna()
        assert (valid >= -3.0).all()
        assert (valid <= 3.0).all()


class TestComputeAllFeatures:
    def test_returns_dataframe(self, sample_ohlcv_data: pd.DataFrame) -> None:
        feats = compute_all_features(sample_ohlcv_data)
        assert isinstance(feats, pd.DataFrame)

    def test_index_matches_ohlcv(self, sample_ohlcv_data: pd.DataFrame) -> None:
        feats = compute_all_features(sample_ohlcv_data)
        assert len(feats) == len(sample_ohlcv_data)

    def test_expected_columns_present(self, sample_ohlcv_data: pd.DataFrame) -> None:
        feats = compute_all_features(sample_ohlcv_data)
        expected_cols = [
            "ret_1d",
            "ret_5d",
            "ret_21d",
            "log_ret_1d",
            "mom_6_1",
            "realized_vol_21d",
            "dollar_volume_21d",
            "rsi_14",
            "macd_hist",
            "bb_width",
            "atr_14",
        ]
        for col in expected_cols:
            assert col in feats.columns, f"Missing expected column: {col}"

    def test_missing_column_raises(self) -> None:
        df = _make_ohlcv().drop(columns=["volume"])
        with pytest.raises(ValueError, match="volume"):
            compute_all_features(df)


# ===========================================================================
# Outlier removal
# ===========================================================================


class TestRemoveOutliersIQR:
    def test_normal_data_unchanged(self) -> None:
        df = _make_ohlcv()
        cleaned = remove_outliers_iqr(df)
        # No outliers in synthetic GBM data → nothing should be dropped
        assert len(cleaned) == len(df)

    def test_extreme_spike_removed(self) -> None:
        df = _make_ohlcv()
        # Insert an extreme price spike (1000% single-day return)
        spike_idx = df.index[100]
        df.loc[spike_idx, "close"] = df["close"].iloc[99] * 10.0
        cleaned = remove_outliers_iqr(df, threshold=3.0)
        assert len(cleaned) < len(df)

    def test_short_series_returned_unchanged(self) -> None:
        df = _make_ohlcv(n=5)
        cleaned = remove_outliers_iqr(df)
        assert len(cleaned) == len(df)


# ===========================================================================
# run_pipeline (mocked)
# ===========================================================================


class TestRunPipeline:
    """Tests for run_pipeline with mocked fetcher to avoid network calls."""

    @pytest.mark.asyncio
    async def test_happy_path(self) -> None:
        ohlcv = _make_ohlcv(n=300)

        mock_fundamentals = FundamentalsData(
            symbol="AAPL",
            as_of_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            pe_ratio=28.5,
            market_cap=3e12,
        )

        with patch(
            "quantflow.data.processors.pipeline.YFinanceFetcher"
        ) as MockFetcher:
            instance = MockFetcher.return_value
            instance.fetch_ohlcv = AsyncMock(return_value=ohlcv)
            instance.validate = AsyncMock(
                return_value=ValidationReport(is_valid=True, row_count=len(ohlcv))
            )
            instance.fetch_fundamentals = AsyncMock(return_value=mock_fundamentals)

            result: ProcessedData = await run_pipeline(
                "AAPL",
                start=datetime(2022, 1, 1, tzinfo=timezone.utc),
                end=datetime(2023, 1, 1, tzinfo=timezone.utc),
            )

        assert result.symbol == "AAPL"
        assert isinstance(result.ohlcv, pd.DataFrame)
        assert isinstance(result.features, pd.DataFrame)
        assert result.fundamentals is not None
        assert result.data_quality_score == 1.0

    @pytest.mark.asyncio
    async def test_fetch_error_propagates(self) -> None:
        with patch(
            "quantflow.data.processors.pipeline.YFinanceFetcher"
        ) as MockFetcher:
            instance = MockFetcher.return_value
            instance.fetch_ohlcv = AsyncMock(
                side_effect=DataFetchError("Network error")
            )
            with pytest.raises(DataFetchError, match="Network error"):
                await run_pipeline(
                    "INVALID_TICKER",
                    start=datetime(2022, 1, 1, tzinfo=timezone.utc),
                    end=datetime(2023, 1, 1, tzinfo=timezone.utc),
                )

    @pytest.mark.asyncio
    async def test_fundamentals_failure_is_non_fatal(self) -> None:
        ohlcv = _make_ohlcv(n=300)

        with patch(
            "quantflow.data.processors.pipeline.YFinanceFetcher"
        ) as MockFetcher:
            instance = MockFetcher.return_value
            instance.fetch_ohlcv = AsyncMock(return_value=ohlcv)
            instance.validate = AsyncMock(
                return_value=ValidationReport(is_valid=True, row_count=len(ohlcv))
            )
            instance.fetch_fundamentals = AsyncMock(
                side_effect=DataFetchError("Fundamentals unavailable")
            )

            result: ProcessedData = await run_pipeline(
                "AAPL",
                start=datetime(2022, 1, 1, tzinfo=timezone.utc),
                end=datetime(2023, 1, 1, tzinfo=timezone.utc),
            )

        # Pipeline should succeed even if fundamentals fail
        assert result.symbol == "AAPL"
        assert result.fundamentals is None
        assert any("non-fatal" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_quality_failure_raises(self) -> None:
        ohlcv = _make_ohlcv(n=300)

        with patch(
            "quantflow.data.processors.pipeline.YFinanceFetcher"
        ) as MockFetcher:
            instance = MockFetcher.return_value
            instance.fetch_ohlcv = AsyncMock(return_value=ohlcv)
            instance.validate = AsyncMock(
                return_value=ValidationReport(
                    is_valid=False,
                    errors=["Too much missing data"],
                    row_count=len(ohlcv),
                )
            )
            with pytest.raises(DataQualityError):
                await run_pipeline(
                    "AAPL",
                    start=datetime(2022, 1, 1, tzinfo=timezone.utc),
                    end=datetime(2023, 1, 1, tzinfo=timezone.utc),
                )
