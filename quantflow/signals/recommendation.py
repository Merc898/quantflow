"""Final recommendation engine.

Converts composite signal + confidence → BUY/SELL recommendation string
with Kelly-inspired position sizing and risk gate overrides.

Signal → Recommendation mapping:
  signal > +0.50 AND conf > 0.65 → STRONG_BUY
  signal > +0.20 AND conf > 0.50 → BUY
  signal > +0.05 AND conf > 0.40 → WEAK_BUY
  signal in [-0.05, +0.05]        → HOLD
  signal < -0.05 AND conf > 0.40 → WEAK_SELL
  signal < -0.20 AND conf > 0.50 → SELL
  signal < -0.50 AND conf > 0.65 → STRONG_SELL
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel, Field

from quantflow.config.constants import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    MAX_POSITION_SIZE,
    SIGNAL_BUY_THRESHOLD,
    SIGNAL_SELL_THRESHOLD,
    SIGNAL_STRONG_BUY_THRESHOLD,
    SIGNAL_STRONG_SELL_THRESHOLD,
    SIGNAL_WEAK_BUY_THRESHOLD,
    SIGNAL_WEAK_SELL_THRESHOLD,
    TARGET_ANNUAL_VOLATILITY,
    TRADING_DAYS_PER_YEAR,
)
from quantflow.config.logging import get_logger
from quantflow.signals.regime_detector import RegimeState

if TYPE_CHECKING:
    import pandas as pd

    from quantflow.models.base import ModelOutput
    from quantflow.risk.var_es import RiskReport

logger = get_logger(__name__)

# Risk gate parameters
_VOL_SPIKE_MULTIPLIER = 3.0  # Annualised vol > 3× 252d average → hold
_MAX_DRAWDOWN_GATE = 0.15  # 30-day drawdown > 15% → hold
_EARNINGS_LOOKBACK_DAYS = 3  # Earnings within 3 days → apply caution


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


class FinalRecommendation(BaseModel):
    """Final BUY/HOLD/SELL recommendation with full attribution.

    Attributes:
        symbol: Ticker symbol.
        timestamp: UTC timestamp.
        recommendation: Recommendation label.
        signal_strength: Composite signal in ``[-1, +1]``.
        confidence: Overall confidence in ``[0, 1]``.
        suggested_position_size: Kelly-inspired position size in ``[-max, +max]``.
        expected_return_21d: Expected 21-day return.
        expected_vol_21d: Expected 21-day volatility.
        var_95_1d: 1-day 95% VaR (negative = loss).
        max_drawdown_estimate: Recent 30-day max drawdown.
        model_contributions: Dict mapping model_name → signal contribution.
        top_bullish_factors: Top bullish evidence strings.
        top_bearish_factors: Top bearish evidence strings.
        regime: Market regime at time of recommendation.
        rationale: Human-readable explanation (LLM-generated or template).
        risk_warnings: List of active risk gate warnings.
        data_quality_score: Fraction of expected model signals received.
        models_used: Number of models contributing to this signal.
        models_available: Total models in the system.
    """

    symbol: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    recommendation: Literal[
        "STRONG_BUY", "BUY", "WEAK_BUY", "HOLD", "WEAK_SELL", "SELL", "STRONG_SELL"
    ]
    signal_strength: float = Field(..., ge=-1.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    suggested_position_size: float = Field(..., ge=-1.0, le=1.0)

    # Risk metrics
    expected_return_21d: float = 0.0
    expected_vol_21d: float = 0.0
    var_95_1d: float = 0.0
    max_drawdown_estimate: float = 0.0

    # Attribution
    model_contributions: dict[str, float] = Field(default_factory=dict)
    top_bullish_factors: list[str] = Field(default_factory=list)
    top_bearish_factors: list[str] = Field(default_factory=list)
    regime: RegimeState | None = None

    # Narrative
    rationale: str = ""
    risk_warnings: list[str] = Field(default_factory=list)

    # Meta
    data_quality_score: float = Field(..., ge=0.0, le=1.0)
    models_used: int = Field(..., ge=0)
    models_available: int = Field(..., ge=0)


# ---------------------------------------------------------------------------
# Recommendation engine
# ---------------------------------------------------------------------------


class RecommendationEngine:
    """Convert composite signal to a final BUY/SELL recommendation.

    Applies the signal→recommendation mapping, Kelly position sizing,
    and risk gates that can override any recommendation to HOLD.

    Args:
        models_available: Total number of models expected in the system
            (used for data quality scoring).
        max_position: Maximum allowed position size.
        target_vol: Annualised vol target for Kelly sizing.
    """

    def __init__(
        self,
        models_available: int = 50,
        max_position: float = MAX_POSITION_SIZE,
        target_vol: float = TARGET_ANNUAL_VOLATILITY,
    ) -> None:
        self._models_available = models_available
        self._max_position = max_position
        self._target_vol = target_vol
        self._logger = get_logger(__name__)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(
        self,
        symbol: str,
        composite_signal: float,
        confidence: float,
        model_outputs: list[ModelOutput] | None = None,
        risk_report: RiskReport | None = None,
        regime: RegimeState | None = None,
        returns: pd.Series | None = None,
        bullish_factors: list[str] | None = None,
        bearish_factors: list[str] | None = None,
        rationale: str = "",
    ) -> FinalRecommendation:
        """Generate a final recommendation.

        Args:
            symbol: Ticker symbol.
            composite_signal: Risk-scaled composite signal in ``[-1, +1]``.
            confidence: Overall confidence in ``[0, 1]``.
            model_outputs: Individual model outputs (for contribution analysis).
            risk_report: VaR / ES risk metrics.
            regime: Current market regime.
            returns: Recent daily return series (for risk gates).
            bullish_factors: List of bullish evidence strings.
            bearish_factors: List of bearish evidence strings.
            rationale: Pre-generated narrative (blank if not yet generated).

        Returns:
            :class:`FinalRecommendation`.
        """
        signal = float(np.clip(composite_signal, -1.0, 1.0))
        conf = float(np.clip(confidence, 0.0, 1.0))

        # Risk gates
        warnings = self._check_risk_gates(returns, regime)
        if warnings:
            # Override: all gate triggers force HOLD
            recommendation = "HOLD"
            pos_size = 0.0
            self._logger.info(
                "Risk gates triggered",
                symbol=symbol,
                warnings=warnings,
            )
        else:
            recommendation = self._signal_to_recommendation(signal, conf)
            pos_size = self._kelly_position_size(signal, conf, returns)

        # Model contributions
        contributions = self._compute_contributions(model_outputs or [], signal)

        # Risk metrics from report
        var_95, expected_ret, expected_vol = 0.0, 0.0, 0.0
        if risk_report:
            var_95 = risk_report.var
        if model_outputs:
            expected_ret = float(np.mean([o.forecast_return for o in model_outputs]))
            expected_vol = float(np.mean([o.forecast_std for o in model_outputs]))

        # Max drawdown
        max_dd = self._max_drawdown(returns) if returns is not None else 0.0

        # Data quality
        n_used = len(model_outputs) if model_outputs else 0
        quality = min(1.0, n_used / max(self._models_available, 1))

        self._logger.info(
            "Recommendation generated",
            symbol=symbol,
            recommendation=recommendation,
            signal=round(signal, 4),
            confidence=round(conf, 4),
            position_size=round(pos_size, 4),
            risk_warnings=len(warnings),
        )

        return FinalRecommendation(
            symbol=symbol,
            recommendation=recommendation,  # type: ignore[arg-type]
            signal_strength=round(signal, 6),
            confidence=round(conf, 6),
            suggested_position_size=round(pos_size, 6),
            expected_return_21d=round(expected_ret, 6),
            expected_vol_21d=round(expected_vol, 6),
            var_95_1d=round(var_95, 6),
            max_drawdown_estimate=round(max_dd, 6),
            model_contributions=contributions,
            top_bullish_factors=(bullish_factors or [])[:5],
            top_bearish_factors=(bearish_factors or [])[:5],
            regime=regime,
            rationale=rationale,
            risk_warnings=warnings,
            data_quality_score=round(quality, 4),
            models_used=n_used,
            models_available=self._models_available,
        )

    # ------------------------------------------------------------------
    # Signal → recommendation mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _signal_to_recommendation(signal: float, confidence: float) -> str:
        """Map composite signal + confidence to a recommendation string.

        Args:
            signal: Composite signal in ``[-1, +1]``.
            confidence: Confidence in ``[0, 1]``.

        Returns:
            Recommendation label.
        """
        if signal > SIGNAL_STRONG_BUY_THRESHOLD and confidence > CONFIDENCE_HIGH:
            return "STRONG_BUY"
        if signal > SIGNAL_BUY_THRESHOLD and confidence > CONFIDENCE_MEDIUM:
            return "BUY"
        if signal > SIGNAL_WEAK_BUY_THRESHOLD and confidence > CONFIDENCE_LOW:
            return "WEAK_BUY"
        if signal < SIGNAL_STRONG_SELL_THRESHOLD and confidence > CONFIDENCE_HIGH:
            return "STRONG_SELL"
        if signal < SIGNAL_SELL_THRESHOLD and confidence > CONFIDENCE_MEDIUM:
            return "SELL"
        if signal < SIGNAL_WEAK_SELL_THRESHOLD and confidence > CONFIDENCE_LOW:
            return "WEAK_SELL"
        return "HOLD"

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def _kelly_position_size(
        self,
        signal: float,
        confidence: float,
        returns: pd.Series | None,
    ) -> float:
        """Kelly-inspired position sizing with volatility adjustment.

        position_size = (signal × confidence) × (target_vol / realized_vol)
        Clipped to [-max_position, +max_position].

        Args:
            signal: Composite signal.
            confidence: Signal confidence.
            returns: Return series for realized vol estimation.

        Returns:
            Suggested position size.
        """
        base = signal * confidence

        if returns is not None and len(returns.dropna()) >= 21:
            realized_vol = float(returns.dropna().iloc[-21:].std() * np.sqrt(TRADING_DAYS_PER_YEAR))
            if realized_vol > 1e-6:
                base *= self._target_vol / realized_vol

        return float(np.clip(base, -self._max_position, self._max_position))

    # ------------------------------------------------------------------
    # Risk gates
    # ------------------------------------------------------------------

    def _check_risk_gates(
        self,
        returns: pd.Series | None,
        regime: RegimeState | None,
    ) -> list[str]:
        """Check all risk gate conditions.

        Args:
            returns: Return series for vol and drawdown checks.
            regime: Current regime (for crisis check).

        Returns:
            List of active warning strings (empty = no gates triggered).
        """
        warnings: list[str] = []

        if returns is not None:
            clean = returns.dropna()

            # Gate 1: Vol spike (current vol > 3× trailing-year average)
            if len(clean) >= 252:
                annual_avg_vol = float(clean.iloc[:-21].std() * np.sqrt(TRADING_DAYS_PER_YEAR))
                current_vol = float(clean.iloc[-21:].std() * np.sqrt(TRADING_DAYS_PER_YEAR))
                if annual_avg_vol > 1e-6 and current_vol > _VOL_SPIKE_MULTIPLIER * annual_avg_vol:
                    warnings.append(
                        f"Volatility spike: {current_vol:.0%} > "
                        f"{_VOL_SPIKE_MULTIPLIER}× avg ({annual_avg_vol:.0%})"
                    )

            # Gate 2: Max drawdown gate
            if len(clean) >= 21:
                max_dd = self._max_drawdown(clean.iloc[-30:])
                if abs(max_dd) > _MAX_DRAWDOWN_GATE:
                    warnings.append(
                        f"Max drawdown ({max_dd:.1%}) exceeds {_MAX_DRAWDOWN_GATE:.0%} gate"
                    )

        # Gate 3: Crisis regime
        if regime is not None and regime.volatility_regime == "CRISIS":
            warnings.append("Market in CRISIS volatility regime")

        return warnings

    # ------------------------------------------------------------------
    # Attribution and metrics helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_contributions(
        model_outputs: list[ModelOutput],
        composite_signal: float,
    ) -> dict[str, float]:
        """Compute each model's proportional contribution to the composite.

        Args:
            model_outputs: All model outputs.
            composite_signal: The final composite signal.

        Returns:
            Dict mapping model_name → signed contribution.
        """
        if not model_outputs:
            return {}

        total_abs = sum(abs(o.signal) * o.confidence for o in model_outputs)
        if total_abs < 1e-8:
            n = len(model_outputs)
            return {o.model_name: round(composite_signal / n, 4) for o in model_outputs}

        contributions = {}
        for o in model_outputs:
            weight = (abs(o.signal) * o.confidence) / total_abs
            contributions[o.model_name] = round(weight * o.signal, 6)
        return contributions

    @staticmethod
    def _max_drawdown(returns: pd.Series) -> float:
        """Compute the maximum drawdown from a return series.

        Args:
            returns: Daily log-return series.

        Returns:
            Maximum drawdown (negative float, e.g. -0.12 = 12% drawdown).
        """
        if len(returns) < 2:
            return 0.0
        cum = (1.0 + returns.fillna(0.0)).cumprod()
        running_max = cum.cummax()
        drawdown = (cum - running_max) / running_max.clip(lower=1e-8)
        return float(drawdown.min())
