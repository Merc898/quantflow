"""Two-stage ensemble signal aggregator.

STAGE 1 — Within-category aggregation:
  Statistical models  → stat_composite
  ML models           → ml_composite
  Sentiment           → sentiment_composite

STAGE 2 — Cross-category weighted combination:
  composite = w_stat * stat_composite + w_ml * ml_composite + w_sentiment * sentiment_composite

Risk scaling (volatility targeting):
  final_signal = composite * (target_vol / realized_vol)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from quantflow.config.constants import (
    MAX_POSITION_SIZE,
    TARGET_ANNUAL_VOLATILITY,
    TRADING_DAYS_PER_YEAR,
)
from quantflow.config.logging import get_logger
from quantflow.models.base import ModelOutput

logger = get_logger(__name__)

# Default category weights (updated dynamically by DynamicWeightCalibrator)
_DEFAULT_STAT_WEIGHT = 0.35
_DEFAULT_ML_WEIGHT = 0.45
_DEFAULT_SENTIMENT_WEIGHT = 0.20

# Model name → category mapping
_STAT_MODELS = frozenset({
    "GARCHModel", "ARIMAModel", "KalmanFilterModel",
    "VARModel", "MarkovSwitchingModel", "PCARiskFactorModel",
})
_ML_MODELS = frozenset({
    "GBTSignalModel", "RandomForestModel", "LASSOModel",
    "RidgeModel", "ElasticNetModel", "LSTMSignalModel",
    "TimeSeriesTransformerModel", "DRLPortfolioAgent",
})


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------


class CategoryComposite(BaseModel):
    """Composite signal within a model category.

    Attributes:
        category: Category label (``"statistical"``, ``"ml"``, ``"sentiment"``).
        composite: Weighted average signal for this category.
        n_models: Number of models contributing.
        component_signals: Dict mapping model_name → its signal value.
        weights_used: Dict mapping model_name → normalised weight applied.
    """

    category: str
    composite: float
    n_models: int
    component_signals: dict[str, float] = Field(default_factory=dict)
    weights_used: dict[str, float] = Field(default_factory=dict)


class AggregationResult(BaseModel):
    """Full result of the two-stage signal aggregation.

    Attributes:
        raw_composite: Unscaled composite signal before vol targeting.
        risk_scaled_signal: Final signal after vol targeting, clipped to [-1, 1].
        category_composites: Per-category breakdown.
        category_weights: Weights applied to each category.
        target_vol: Annualised target volatility used for scaling.
        realized_vol: Annualised realized volatility of the underlying.
        vol_scale_factor: Multiplier applied for vol targeting.
        metadata: Additional diagnostics.
    """

    raw_composite: float
    risk_scaled_signal: float = Field(..., ge=-1.0, le=1.0)
    category_composites: dict[str, CategoryComposite] = Field(default_factory=dict)
    category_weights: dict[str, float] = Field(default_factory=dict)
    target_vol: float
    realized_vol: float | None = None
    vol_scale_factor: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Ensemble aggregator
# ---------------------------------------------------------------------------


class EnsembleAggregator:
    """Two-stage ensemble aggregator for quantitative model signals.

    Args:
        stat_weight: Weight for the statistical model category.
        ml_weight: Weight for the ML model category.
        sentiment_weight: Weight for the sentiment category.
        target_vol: Annualised volatility target for risk scaling.
    """

    def __init__(
        self,
        stat_weight: float = _DEFAULT_STAT_WEIGHT,
        ml_weight: float = _DEFAULT_ML_WEIGHT,
        sentiment_weight: float = _DEFAULT_SENTIMENT_WEIGHT,
        target_vol: float = TARGET_ANNUAL_VOLATILITY,
    ) -> None:
        self._w_stat = stat_weight
        self._w_ml = ml_weight
        self._w_sent = sentiment_weight
        self._target_vol = target_vol
        self._logger = get_logger(__name__)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def aggregate(
        self,
        model_outputs: list[ModelOutput],
        sentiment_score: float | None = None,
        realized_vol: float | None = None,
        model_weights: dict[str, float] | None = None,
    ) -> AggregationResult:
        """Aggregate all model signals into a final composite signal.

        Args:
            model_outputs: List of :class:`ModelOutput` objects from all models.
            sentiment_score: Pre-computed composite sentiment (from agent layer)
                in ``[-1, +1]``.  Used for the sentiment category.
            realized_vol: Annualised realized volatility for risk scaling.
                If None, no vol-targeting is applied.
            model_weights: Optional dict mapping model_name → weight override.
                If None, weights are derived from model confidence scores.

        Returns:
            :class:`AggregationResult` with the final signal and breakdown.
        """
        if not model_outputs and sentiment_score is None:
            return self._empty_result()

        # Partition model outputs by category
        stat_outputs = [o for o in model_outputs if o.model_name in _STAT_MODELS]
        ml_outputs = [o for o in model_outputs if o.model_name in _ML_MODELS]
        # Unknown model names go to ML (most likely)
        unknown = [o for o in model_outputs
                   if o.model_name not in _STAT_MODELS and o.model_name not in _ML_MODELS]
        ml_outputs = ml_outputs + unknown

        # Stage 1: within-category composites
        stat_composite = self._category_composite(
            stat_outputs, model_weights, category="statistical"
        )
        ml_composite = self._category_composite(
            ml_outputs, model_weights, category="ml"
        )
        sent_composite = self._sentiment_composite(sentiment_score)

        # Stage 2: cross-category weighted combination
        # Normalise category weights based on which categories have data
        cat_weights = self._normalise_category_weights(
            stat_outputs, ml_outputs, sentiment_score
        )

        raw = (
            cat_weights["statistical"] * stat_composite.composite
            + cat_weights["ml"] * ml_composite.composite
            + cat_weights["sentiment"] * sent_composite.composite
        )
        raw = float(np.clip(raw, -1.0, 1.0))

        # Risk scaling (volatility targeting)
        scale_factor = 1.0
        if realized_vol is not None and realized_vol > 1e-6:
            scale_factor = float(
                np.clip(self._target_vol / realized_vol, 0.1, 3.0)
            )
        risk_scaled = float(np.clip(raw * scale_factor, -1.0, 1.0))

        n_total = len(model_outputs) + (1 if sentiment_score is not None else 0)

        self._logger.info(
            "Signals aggregated",
            n_models=len(model_outputs),
            raw_composite=round(raw, 4),
            risk_scaled=round(risk_scaled, 4),
            vol_scale=round(scale_factor, 3),
        )

        return AggregationResult(
            raw_composite=round(raw, 6),
            risk_scaled_signal=round(risk_scaled, 6),
            category_composites={
                "statistical": stat_composite,
                "ml": ml_composite,
                "sentiment": sent_composite,
            },
            category_weights=cat_weights,
            target_vol=self._target_vol,
            realized_vol=realized_vol,
            vol_scale_factor=round(scale_factor, 4),
            metadata={"n_models_total": n_total},
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _category_composite(
        self,
        outputs: list[ModelOutput],
        model_weights: dict[str, float] | None,
        category: str,
    ) -> CategoryComposite:
        """Compute the confidence-weighted composite for a category.

        Args:
            outputs: Model outputs in this category.
            model_weights: Optional external weight overrides.
            category: Category label.

        Returns:
            :class:`CategoryComposite`.
        """
        if not outputs:
            return CategoryComposite(
                category=category,
                composite=0.0,
                n_models=0,
            )

        signals = {o.model_name: o.signal for o in outputs}
        if model_weights:
            raw_w = {o.model_name: model_weights.get(o.model_name, o.confidence)
                     for o in outputs}
        else:
            raw_w = {o.model_name: max(o.confidence, 1e-4) for o in outputs}

        total_w = sum(raw_w.values())
        if total_w < 1e-8:
            # All overridden weights are zero — use equal weights
            total_w = float(len(raw_w))
        norm_w = {m: w / total_w for m, w in raw_w.items()}

        composite = sum(norm_w[m] * signals[m] for m in signals)
        composite = float(np.clip(composite, -1.0, 1.0))

        return CategoryComposite(
            category=category,
            composite=round(composite, 6),
            n_models=len(outputs),
            component_signals={m: round(v, 4) for m, v in signals.items()},
            weights_used={m: round(v, 4) for m, v in norm_w.items()},
        )

    @staticmethod
    def _sentiment_composite(sentiment_score: float | None) -> CategoryComposite:
        """Build a CategoryComposite for the sentiment category.

        Args:
            sentiment_score: Composite sentiment from the agent layer.

        Returns:
            :class:`CategoryComposite` for sentiment.
        """
        score = float(np.clip(sentiment_score, -1.0, 1.0)) if sentiment_score is not None else 0.0
        return CategoryComposite(
            category="sentiment",
            composite=round(score, 6),
            n_models=1 if sentiment_score is not None else 0,
            component_signals={"AgentSentiment": round(score, 4)},
            weights_used={"AgentSentiment": 1.0} if sentiment_score is not None else {},
        )

    def _normalise_category_weights(
        self,
        stat_outputs: list[ModelOutput],
        ml_outputs: list[ModelOutput],
        sentiment_score: float | None,
    ) -> dict[str, float]:
        """Normalise category weights based on data availability.

        Categories with no data get weight 0, and remaining weights
        are rescaled to sum to 1.

        Args:
            stat_outputs: Statistical model outputs.
            ml_outputs: ML model outputs.
            sentiment_score: Sentiment score (or None).

        Returns:
            Normalised category weight dict.
        """
        raw = {
            "statistical": self._w_stat if stat_outputs else 0.0,
            "ml": self._w_ml if ml_outputs else 0.0,
            "sentiment": self._w_sent if sentiment_score is not None else 0.0,
        }
        total = sum(raw.values())
        if total < 1e-8:
            return {"statistical": 1.0 / 3, "ml": 1.0 / 3, "sentiment": 1.0 / 3}
        return {k: round(v / total, 6) for k, v in raw.items()}

    @staticmethod
    def _empty_result() -> AggregationResult:
        """Return a neutral empty result when no signals are available."""
        return AggregationResult(
            raw_composite=0.0,
            risk_scaled_signal=0.0,
            target_vol=TARGET_ANNUAL_VOLATILITY,
            metadata={"n_models_total": 0},
        )
