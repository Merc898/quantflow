"""Market regime detection from multiple indicator sources.

Synthesizes a composite regime from:
- Return volatility (rolling vs historical percentile)
- VIX level (if available)
- Trend (200-day SMA position)
- Markov switching model state probabilities
- Yield curve slope (if available)

Output: :class:`RegimeState` Pydantic model consumed by the signal aggregator,
recommendation engine, and weight calibrator.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel, Field

from quantflow.config.constants import (
    LOOKBACK_1Y,
    TRADING_DAYS_PER_YEAR,
    VOL_REGIME_EXTREME_THRESHOLD,
    VOL_REGIME_HIGH_THRESHOLD,
    VOL_REGIME_LOW_THRESHOLD,
)
from quantflow.config.logging import get_logger

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)

_TREND_SMA_WINDOW = 200  # Trading-day SMA for trend regime
_SIDEWAYS_BAND = 0.02  # Price within ±2% of SMA → SIDEWAYS
_VIX_CALM = 15.0
_VIX_ELEVATED = 25.0
_VIX_STRESS = 35.0


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


class RegimeState(BaseModel):
    """Composite market regime state.

    Attributes:
        volatility_regime: Volatility environment label.
        trend_regime: Price trend label.
        macro_regime: Macroeconomic cycle phase.
        overall_regime: Combined label (e.g. "BULL_LOW_VOL").
        regime_confidence: Confidence in the overall regime label ``[0, 1]``.
        regime_duration_days: Estimated days in the current regime.
        timestamp: When the regime was detected.
    """

    volatility_regime: Literal["LOW", "MEDIUM", "HIGH", "CRISIS"]
    trend_regime: Literal["BULL", "BEAR", "SIDEWAYS"]
    macro_regime: Literal["EXPANSION", "LATE_CYCLE", "RECESSION", "RECOVERY"]
    overall_regime: str
    regime_confidence: float = Field(..., ge=0.0, le=1.0)
    regime_duration_days: int = Field(..., ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))


# ---------------------------------------------------------------------------
# Regime detector
# ---------------------------------------------------------------------------


class RegimeDetector:
    """Multi-signal regime detector.

    Combines rolling volatility, VIX, trend, Markov probabilities, and
    yield curve into a single :class:`RegimeState`.

    Args:
        vol_lookback: Rolling window for realized volatility (days).
        sma_window: SMA window for trend detection (days).
    """

    def __init__(
        self,
        vol_lookback: int = LOOKBACK_1Y,
        sma_window: int = _TREND_SMA_WINDOW,
    ) -> None:
        self._vol_lookback = vol_lookback
        self._sma_window = sma_window
        self._logger = get_logger(__name__)

        # History of (timestamp, overall_regime) for duration tracking
        self._regime_history: list[tuple[datetime, str]] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def detect(
        self,
        returns: pd.Series,
        prices: pd.Series | None = None,
        vix: float | None = None,
        markov_state_probs: dict[str, float] | None = None,
        yield_curve_slope: float | None = None,
    ) -> RegimeState:
        """Detect the current market regime from multiple signals.

        Args:
            returns: Daily log-returns Series (DatetimeIndex).
            prices: Price Series for SMA trend detection (optional; uses
                cumulative returns from ``returns`` if None).
            vix: Current VIX level (optional).
            markov_state_probs: Dict mapping state_name → probability from
                a fitted Markov switching model (optional).
            yield_curve_slope: 10Y − 2Y Treasury yield spread in percentage
                points (optional).  Negative ⟹ inverted curve.

        Returns:
            :class:`RegimeState` with all regime components.
        """
        clean_ret = returns.dropna().astype(np.float64)

        vol_regime, vol_conf = self._vol_regime(clean_ret, vix)
        trend_regime, trend_conf = self._trend_regime(clean_ret, prices)
        macro_regime, macro_conf = self._macro_regime(yield_curve_slope, clean_ret)

        # Override vol regime with Markov probs if available
        if markov_state_probs:
            vol_regime, vol_conf = self._markov_override(markov_state_probs, vol_regime, vol_conf)

        overall = f"{trend_regime}_{vol_regime}_VOL"
        confidence = float(np.mean([vol_conf, trend_conf, macro_conf]))

        duration = self._regime_duration(overall)
        self._update_history(overall)

        self._logger.info(
            "Regime detected",
            vol_regime=vol_regime,
            trend_regime=trend_regime,
            macro_regime=macro_regime,
            overall=overall,
            confidence=round(confidence, 3),
            duration_days=duration,
        )

        return RegimeState(
            volatility_regime=vol_regime,  # type: ignore[arg-type]
            trend_regime=trend_regime,  # type: ignore[arg-type]
            macro_regime=macro_regime,  # type: ignore[arg-type]
            overall_regime=overall,
            regime_confidence=round(confidence, 4),
            regime_duration_days=duration,
        )

    # ------------------------------------------------------------------
    # Component detectors
    # ------------------------------------------------------------------

    def _vol_regime(
        self,
        returns: pd.Series,
        vix: float | None,
    ) -> tuple[str, float]:
        """Classify the volatility regime.

        Uses the rolling realized volatility percentile against its own
        history.  VIX overrides if it signals a more extreme regime.

        Args:
            returns: Clean daily returns.
            vix: Current VIX level (optional).

        Returns:
            Tuple (regime_label, confidence).
        """
        window = min(self._vol_lookback, len(returns))
        if window < 21:
            return "MEDIUM", 0.40

        recent_ret = returns.iloc[-window:]
        realized_vol = float(recent_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100)

        # Historical vol distribution for percentile calculation
        rolling_vol = (returns.rolling(21).std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100).dropna()

        pct = 50.0 if len(rolling_vol) < 5 else float((rolling_vol < realized_vol).mean() * 100)

        if pct >= VOL_REGIME_EXTREME_THRESHOLD:
            regime, conf = "CRISIS", 0.85
        elif pct >= VOL_REGIME_HIGH_THRESHOLD:
            regime, conf = "HIGH", 0.75
        elif pct <= VOL_REGIME_LOW_THRESHOLD:
            regime, conf = "LOW", 0.75
        else:
            regime, conf = "MEDIUM", 0.65

        # VIX override
        if vix is not None:
            if vix >= _VIX_STRESS and regime != "CRISIS":
                regime, conf = "CRISIS", 0.90
            elif vix >= _VIX_ELEVATED and regime == "LOW":
                regime, conf = "MEDIUM", 0.80
            elif vix < _VIX_CALM and regime in ("HIGH", "CRISIS"):
                regime, conf = "MEDIUM", 0.70

        return regime, conf

    def _trend_regime(
        self,
        returns: pd.Series,
        prices: pd.Series | None,
    ) -> tuple[str, float]:
        """Classify the price trend regime.

        Uses the 200-day SMA position.  Falls back to cumulative returns
        if a price series is not provided.

        Args:
            returns: Daily returns.
            prices: Optional price series (same index as returns).

        Returns:
            Tuple (regime_label, confidence).
        """
        if prices is None:
            # Reconstruct approximate price from cumulative returns
            prices = (1.0 + returns).cumprod()

        clean_prices = prices.dropna()
        sma_window = min(self._sma_window, len(clean_prices) - 1)
        if sma_window < 10:
            return "SIDEWAYS", 0.40

        sma = clean_prices.rolling(sma_window).mean()
        last_price = float(clean_prices.iloc[-1])
        last_sma = float(sma.iloc[-1])

        if last_sma < 1e-8:
            return "SIDEWAYS", 0.40

        deviation = (last_price - last_sma) / last_sma

        if deviation > _SIDEWAYS_BAND:
            # How far above SMA drives confidence
            conf = min(0.90, 0.60 + abs(deviation) * 2.0)
            return "BULL", float(conf)
        elif deviation < -_SIDEWAYS_BAND:
            conf = min(0.90, 0.60 + abs(deviation) * 2.0)
            return "BEAR", float(conf)
        else:
            return "SIDEWAYS", 0.55

    def _macro_regime(
        self,
        yield_slope: float | None,
        returns: pd.Series,
    ) -> tuple[str, float]:
        """Classify the macroeconomic regime.

        Uses yield curve slope (10Y−2Y) if available; otherwise infers
        from recent return momentum as a proxy.

        Args:
            yield_slope: 10Y−2Y spread (positive = normal, negative = inverted).
            returns: Daily return series for momentum fallback.

        Returns:
            Tuple (regime_label, confidence).
        """
        if yield_slope is not None:
            if yield_slope < -0.50:
                return "RECESSION", 0.70
            elif yield_slope < 0.0:
                return "LATE_CYCLE", 0.65
            elif yield_slope < 1.0:
                return "EXPANSION", 0.70
            else:
                return "RECOVERY", 0.65

        # Fallback: use 12-month return momentum
        lookback = min(252, len(returns))
        if lookback < 50:
            return "EXPANSION", 0.40

        annual_ret = float(returns.iloc[-lookback:].sum())
        if annual_ret > 0.15:
            return "EXPANSION", 0.55
        elif annual_ret > 0.0:
            return "LATE_CYCLE", 0.50
        elif annual_ret > -0.15:
            return "RECOVERY", 0.50
        else:
            return "RECESSION", 0.55

    @staticmethod
    def _markov_override(
        markov_probs: dict[str, float],
        vol_regime: str,
        vol_conf: float,
    ) -> tuple[str, float]:
        """Override volatility regime with Markov switching model probabilities.

        Maps high-volatility Markov states to CRISIS/HIGH regimes.

        Args:
            markov_probs: State probabilities from MarkovSwitchingModel.
            vol_regime: Current vol regime from rolling-vol method.
            vol_conf: Current vol regime confidence.

        Returns:
            Tuple (potentially updated regime, confidence).
        """
        # Look for states labelled "high_vol" or "bear"
        high_vol_prob = sum(
            p
            for name, p in markov_probs.items()
            if any(kw in name.lower() for kw in ("high", "bear", "crisis", "2"))
        )
        if high_vol_prob > 0.70 and vol_regime not in ("HIGH", "CRISIS"):
            return "HIGH", max(vol_conf, float(high_vol_prob))
        if high_vol_prob > 0.90:
            return "CRISIS", float(high_vol_prob)
        return vol_regime, vol_conf

    # ------------------------------------------------------------------
    # Duration tracking
    # ------------------------------------------------------------------

    def _regime_duration(self, current_regime: str) -> int:
        """Estimate how many days the current regime has been active.

        Args:
            current_regime: Current overall regime label.

        Returns:
            Integer count of consecutive days in this regime.
        """
        duration = 0
        for _, regime in reversed(self._regime_history):
            if regime == current_regime:
                duration += 1
            else:
                break
        return duration

    def _update_history(self, regime: str) -> None:
        """Append the current regime to the rolling history.

        Args:
            regime: Current overall regime label.
        """
        self._regime_history.append((datetime.now(tz=UTC), regime))
        # Keep last 252 entries
        self._regime_history = self._regime_history[-252:]
