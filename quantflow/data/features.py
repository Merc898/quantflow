"""Price-based feature engineering for QuantFlow.

Computes all features defined in Spec 01 from OHLCV data.
All computations are vectorised (no Python loops over time series).
All features are computed with proper time-shifting to avoid look-ahead bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantflow.config.constants import (
    FEATURE_WINSORIZE_LOWER,
    FEATURE_WINSORIZE_UPPER,
    FEATURE_ZSCORE_CLIP,
    LOOKBACK_1Y,
    LOOKBACK_6M,
    MOMENTUM_LOOKBACKS,
    RETURN_LOOKBACKS,
    TRADING_DAYS_PER_YEAR,
    VOLATILITY_LOOKBACKS,
)
from quantflow.config.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------


def compute_returns(close: pd.Series) -> pd.DataFrame:
    """Compute multi-horizon arithmetic and log returns.

    Uses ``.shift(1)`` on the *denominator* so that the value on day *t*
    reflects the return that was knowable at the close of day *t-1*
    (look-ahead-safe).

    Args:
        close: Series of adjusted close prices (UTC DatetimeIndex).

    Returns:
        DataFrame with columns:
        ``ret_1d``, ``ret_5d``, ``ret_21d``, ``ret_63d``, ``ret_252d``,
        ``log_ret_1d``.
    """
    features: dict[str, pd.Series] = {}
    for lag in RETURN_LOOKBACKS:
        features[f"ret_{lag}d"] = close.pct_change(lag)
    features["log_ret_1d"] = np.log(close / close.shift(1))
    return pd.DataFrame(features, index=close.index)


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------


def compute_momentum(close: pd.Series) -> pd.DataFrame:
    """Compute medium- and long-horizon momentum signals.

    Skips the most recent month (21 days) to avoid the well-known
    short-term reversal in monthly rebalancing.

    Args:
        close: Series of adjusted close prices.

    Returns:
        DataFrame with columns ``mom_12_1`` and ``mom_6_1``.
    """
    features: dict[str, pd.Series] = {}
    skip = 21  # 1 month
    for lookback in MOMENTUM_LOOKBACKS:
        col = f"mom_{lookback // 21}_{skip // 21}"
        features[col] = close.shift(skip) / close.shift(lookback) - 1
    return pd.DataFrame(features, index=close.index)


# ---------------------------------------------------------------------------
# Reversal
# ---------------------------------------------------------------------------


def compute_reversal(close: pd.Series) -> pd.DataFrame:
    """Compute 1-month short-term reversal factor.

    Args:
        close: Series of adjusted close prices.

    Returns:
        DataFrame with column ``rev_1m`` (negated 1-month return).
    """
    rev = -(close.pct_change(21))
    return pd.DataFrame({"rev_1m": rev}, index=close.index)


# ---------------------------------------------------------------------------
# Realised Volatility
# ---------------------------------------------------------------------------


def compute_realized_vol(close: pd.Series) -> pd.DataFrame:
    """Compute annualised realised volatility over multiple windows.

    Args:
        close: Series of adjusted close prices.

    Returns:
        DataFrame with columns ``realized_vol_21d`` and ``realized_vol_63d``.
    """
    log_ret = np.log(close / close.shift(1))
    features: dict[str, pd.Series] = {}
    for window in VOLATILITY_LOOKBACKS:
        col = f"realized_vol_{window}d"
        features[col] = log_ret.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    return pd.DataFrame(features, index=close.index)


# ---------------------------------------------------------------------------
# Volume features
# ---------------------------------------------------------------------------


def compute_volume_features(close: pd.Series, volume: pd.Series) -> pd.DataFrame:
    """Compute volume-based liquidity and activity features.

    Args:
        close: Series of adjusted close prices.
        volume: Series of share volumes.

    Returns:
        DataFrame with columns:
        ``dollar_volume_21d``, ``amihud_illiquidity``, ``volume_z_score``.
    """
    dollar_vol = close * volume
    log_ret = np.log(close / close.shift(1)).abs()

    dollar_vol_21d = dollar_vol.rolling(21).mean()

    # Amihud (2002): |ret| / dollar_volume — small number ⟹ more liquid
    amihud = log_ret / dollar_vol.replace(0, np.nan)

    # Volume z-score vs 63-day rolling mean and std
    vol_mean = volume.rolling(63).mean()
    vol_std = volume.rolling(63).std()
    vol_z = (volume - vol_mean) / vol_std.replace(0, np.nan)

    return pd.DataFrame(
        {
            "dollar_volume_21d": dollar_vol_21d,
            "amihud_illiquidity": amihud,
            "volume_z_score": vol_z,
        },
        index=close.index,
    )


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average helper."""
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(close: pd.Series, windows: tuple[int, ...] = (14, 28)) -> pd.DataFrame:
    """Compute Relative Strength Index for multiple windows.

    Args:
        close: Series of adjusted close prices.
        windows: RSI lookback windows in days.

    Returns:
        DataFrame with columns ``rsi_14``, ``rsi_28`` (or matching windows).
    """
    features: dict[str, pd.Series] = {}
    delta = close.diff()
    for w in windows:
        gain = delta.clip(lower=0).ewm(com=w - 1, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(com=w - 1, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        features[f"rsi_{w}"] = 100.0 - (100.0 / (1.0 + rs))
    return pd.DataFrame(features, index=close.index)


def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Compute MACD line, signal line, and histogram.

    Args:
        close: Series of adjusted close prices.
        fast: Fast EMA span.
        slow: Slow EMA span.
        signal: Signal EMA span.

    Returns:
        DataFrame with columns ``macd_signal`` and ``macd_hist``.
    """
    macd_line = _ema(close, fast) - _ema(close, slow)
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {
            "macd_signal": signal_line,
            "macd_hist": histogram,
        },
        index=close.index,
    )


def compute_bollinger_bands(
    close: pd.Series,
    window: int = 20,
    n_std: float = 2.0,
) -> pd.DataFrame:
    """Compute Bollinger Band width and position.

    Args:
        close: Series of adjusted close prices.
        window: Rolling window for band calculation.
        n_std: Number of standard deviations for band width.

    Returns:
        DataFrame with columns ``bb_width`` (band width / mid) and
        ``bb_position`` (current price position in [0, 1] within bands).
    """
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    band_width = (upper - lower) / mid.replace(0, np.nan)
    position = (close - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame(
        {"bb_width": band_width, "bb_position": position},
        index=close.index,
    )


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.DataFrame:
    """Compute Average True Range.

    Args:
        high: Series of daily high prices.
        low: Series of daily low prices.
        close: Series of adjusted close prices.
        window: ATR rolling window.

    Returns:
        DataFrame with column ``atr_14``.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(span=window, adjust=False).mean()
    return pd.DataFrame({f"atr_{window}": atr}, index=close.index)


# ---------------------------------------------------------------------------
# Standardisation helpers
# ---------------------------------------------------------------------------


def winsorize(series: pd.Series) -> pd.Series:
    """Clip a feature series to its 1st / 99th percentile.

    Args:
        series: Raw feature values.

    Returns:
        Winsorized series.
    """
    lower = series.quantile(FEATURE_WINSORIZE_LOWER)
    upper = series.quantile(FEATURE_WINSORIZE_UPPER)
    return series.clip(lower=lower, upper=upper)


def rolling_zscore(
    series: pd.Series,
    window: int = LOOKBACK_1Y,
    clip: float = FEATURE_ZSCORE_CLIP,
) -> pd.Series:
    """Standardise a feature using a rolling z-score.

    Args:
        series: Raw feature values.
        window: Rolling window for mean and std (default 252 days).
        clip: Clip z-scores to ``[-clip, clip]`` (default 3.0).

    Returns:
        Rolling z-score series, clipped to ``[-clip, clip]``.
    """
    mean = series.rolling(window, min_periods=window // 2).mean()
    std = series.rolling(window, min_periods=window // 2).std()
    z = (series - mean) / std.replace(0, np.nan)
    return z.clip(lower=-clip, upper=clip)


# ---------------------------------------------------------------------------
# Master feature builder
# ---------------------------------------------------------------------------


def compute_all_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Compute all price-based features from an OHLCV DataFrame.

    This is the main entry point called by the data pipeline.
    All features are computed with correct lag to avoid look-ahead bias.
    Features are then winsorized and z-scored.

    Args:
        ohlcv: DataFrame with columns ``[open, high, low, close, volume]``
               and a UTC DatetimeIndex.

    Returns:
        DataFrame with all computed features aligned to the same index.
        Features with insufficient history for a window are ``NaN``.
    """
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(ohlcv.columns)
    if missing:
        raise ValueError(f"OHLCV DataFrame missing columns: {missing}")

    close = ohlcv["close"].astype(np.float64)
    high = ohlcv["high"].astype(np.float64)
    low = ohlcv["low"].astype(np.float64)
    volume = ohlcv["volume"].astype(np.float64)

    logger.debug("Computing all features", rows=len(ohlcv))

    parts: list[pd.DataFrame] = [
        compute_returns(close),
        compute_momentum(close),
        compute_reversal(close),
        compute_realized_vol(close),
        compute_volume_features(close, volume),
        compute_rsi(close),
        compute_macd(close),
        compute_bollinger_bands(close),
        compute_atr(high, low, close),
    ]

    features = pd.concat(parts, axis=1)

    # Winsorize and z-score each feature column
    for col in features.columns:
        col_data = features[col].dropna()
        if len(col_data) > 0:
            features[col] = winsorize(features[col])
            features[col] = rolling_zscore(features[col])

    logger.info(
        "Feature computation complete",
        rows=len(features),
        columns=len(features.columns),
        feature_names=list(features.columns),
    )
    return features
