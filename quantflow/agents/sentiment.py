"""Multi-source sentiment aggregator.

Combines signals from:
- Agent LLM outputs (OpenAI, Anthropic, Perplexity)
- VADER rule-based sentiment on raw scraped text
- FinBERT (finance-specific BERT) via transformers
- StockTwits structured bullish/bearish ratio

Produces a single composite sentiment score with regime classification
and a contrarian flag at sentiment extremes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from quantflow.agents.schemas import AgentOutput, AggregatedSentiment, RawDocument
from quantflow.config.constants import (
    SENTIMENT_DECAY_FACTOR,
    SENTIMENT_EXTREME_BEAR_THRESHOLD,
    SENTIMENT_EXTREME_BULL_THRESHOLD,
)
from quantflow.config.logging import get_logger

logger = get_logger(__name__)

_FINBERT_MODEL = "ProsusAI/finbert"
_VADER_WEIGHT = 0.20
_FINBERT_WEIGHT = 0.30
_AGENT_WEIGHT = 0.50          # LLM agents combined weight
_HISTORY_WINDOW_DAYS = 30     # Rolling window for z-score baseline


# ---------------------------------------------------------------------------
# VADER wrapper
# ---------------------------------------------------------------------------


def vader_sentiment(text: str) -> float:
    """Compute VADER compound sentiment score for a text.

    Args:
        text: Raw text (news headline, social post, etc.).

    Returns:
        Compound sentiment in ``[-1.0, +1.0]``.
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore[import]

        sia = SentimentIntensityAnalyzer()
        scores = sia.polarity_scores(text)
        return float(scores["compound"])
    except ImportError:
        logger.warning("vaderSentiment not installed, returning 0.0")
        return 0.0


def vader_batch(texts: list[str]) -> float:
    """Compute average VADER sentiment across a batch of texts.

    Args:
        texts: List of text strings.

    Returns:
        Mean VADER compound score.
    """
    if not texts:
        return 0.0
    scores = [vader_sentiment(t) for t in texts]
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# FinBERT wrapper
# ---------------------------------------------------------------------------


def finbert_sentiment(texts: list[str], batch_size: int = 8) -> float:
    """Compute FinBERT sentiment score averaged across texts.

    Falls back to VADER if transformers is not installed or model
    cannot be loaded (e.g. no internet access).

    Args:
        texts: List of text strings (max ~512 tokens each).
        batch_size: Inference batch size.

    Returns:
        Mean sentiment in ``[-1.0, +1.0]`` (positive / negative / neutral mapped).
    """
    if not texts:
        return 0.0
    try:
        from transformers import pipeline  # type: ignore[import]

        classifier = pipeline(
            "text-classification",
            model=_FINBERT_MODEL,
            truncation=True,
            max_length=512,
        )
        # Truncate to avoid OOM on very long texts
        truncated = [t[:1000] for t in texts]
        results = classifier(truncated, batch_size=batch_size)

        label_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
        scores = [
            label_map.get(r["label"].lower(), 0.0) * r["score"]
            for r in results
        ]
        return float(np.mean(scores))

    except Exception as exc:
        logger.warning(
            "FinBERT unavailable, falling back to VADER",
            error=str(exc),
        )
        return vader_batch(texts)


# ---------------------------------------------------------------------------
# Sentiment aggregator
# ---------------------------------------------------------------------------


class SentimentAggregator:
    """Multi-source sentiment aggregator with regime classification.

    Combines LLM agent outputs, VADER, and FinBERT using configurable
    weights.  Maintains a rolling history for z-score baseline computation.

    Args:
        agent_weight: Combined weight for LLM agent outputs.
        vader_weight: Weight for VADER rule-based scores.
        finbert_weight: Weight for FinBERT model scores.
        use_finbert: Whether to run FinBERT (requires HuggingFace model).
    """

    def __init__(
        self,
        agent_weight: float = _AGENT_WEIGHT,
        vader_weight: float = _VADER_WEIGHT,
        finbert_weight: float = _FINBERT_WEIGHT,
        use_finbert: bool = True,
    ) -> None:
        self._agent_weight = agent_weight
        self._vader_weight = vader_weight
        self._finbert_weight = finbert_weight
        self._use_finbert = use_finbert

        # Rolling history: list of (timestamp, composite_score)
        self._history: list[tuple[datetime, float]] = []

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def aggregate(
        self,
        agent_outputs: list[AgentOutput],
        raw_documents: list[RawDocument] | None = None,
    ) -> AggregatedSentiment:
        """Compute composite sentiment from all available signals.

        Args:
            agent_outputs: Normalised outputs from all LLM agents.
            raw_documents: Optional scraped documents for VADER/FinBERT.

        Returns:
            :class:`AggregatedSentiment` with regime and contrarian flag.
        """
        source_scores: dict[str, float] = {}

        # --- Agent sentiment (LLM) ---
        agent_score = self._aggregate_agent_outputs(agent_outputs)
        source_scores["agent_llm"] = round(agent_score, 4)

        # --- VADER on raw documents ---
        vader_score = 0.0
        if raw_documents:
            texts = [d.text for d in raw_documents if d.text][:50]
            vader_score = vader_batch(texts)
        source_scores["vader"] = round(vader_score, 4)

        # --- FinBERT ---
        finbert_score = 0.0
        if raw_documents and self._use_finbert:
            texts = [d.text for d in raw_documents if d.text][:20]
            finbert_score = finbert_sentiment(texts)
        source_scores["finbert"] = round(finbert_score, 4)

        # Normalise weights (they may not sum to 1 when some sources absent)
        w_agent = self._agent_weight if agent_outputs else 0.0
        w_vader = self._vader_weight if raw_documents else 0.0
        w_finbert = self._finbert_weight if (raw_documents and self._use_finbert) else 0.0
        total_w = w_agent + w_vader + w_finbert

        if total_w < 1e-8:
            composite = 0.0
        else:
            composite = float(
                (w_agent * agent_score + w_vader * vader_score + w_finbert * finbert_score)
                / total_w
            )
        composite = float(np.clip(composite, -1.0, 1.0))

        # Update history and compute z-score
        self._update_history(composite)
        z_score = self._compute_zscore(composite)

        # 5-day momentum
        momentum = self._compute_momentum()

        # Regime classification
        regime = self._classify_regime(composite, z_score)

        # Contrarian flag at extremes
        contrarian = abs(z_score) >= abs(SENTIMENT_EXTREME_BULL_THRESHOLD)

        logger.info(
            "Sentiment aggregated",
            composite=round(composite, 4),
            regime=regime,
            z_score=round(z_score, 2),
            contrarian=contrarian,
            source_scores=source_scores,
        )

        return AggregatedSentiment(
            composite_sentiment=round(composite, 6),
            sentiment_regime=regime,
            sentiment_momentum=round(momentum, 6),
            contrarian_signal=contrarian,
            sentiment_zscore=round(z_score, 4),
            source_scores=source_scores,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _aggregate_agent_outputs(self, outputs: list[AgentOutput]) -> float:
        """Weighted average of agent sentiment scores.

        Uses each agent's ``confidence`` as the per-agent weight.
        Applies exponential decay so older outputs count less.

        Args:
            outputs: List of agent outputs (sorted by time, oldest first).

        Returns:
            Weighted average sentiment float.
        """
        if not outputs:
            return 0.0

        scores: list[float] = []
        weights: list[float] = []
        for ao in outputs:
            if not (-1.0 <= ao.sentiment_score <= 1.0):
                continue
            scores.append(ao.sentiment_score)
            weights.append(max(ao.confidence, 0.01))

        if not scores:
            return 0.0

        w_arr = np.array(weights, dtype=np.float64)
        s_arr = np.array(scores, dtype=np.float64)
        return float(np.dot(w_arr, s_arr) / w_arr.sum())

    def _update_history(self, score: float) -> None:
        """Append a new composite score to the rolling history.

        Prunes entries older than ``_HISTORY_WINDOW_DAYS``.

        Args:
            score: Composite sentiment score.
        """
        from datetime import timedelta

        now = datetime.now(tz=timezone.utc)
        self._history.append((now, score))
        cutoff = now - timedelta(days=_HISTORY_WINDOW_DAYS)
        self._history = [(t, s) for t, s in self._history if t >= cutoff]

    def _compute_zscore(self, current: float) -> float:
        """Compute z-score of current sentiment vs rolling 30-day history.

        Args:
            current: Current composite sentiment.

        Returns:
            Z-score (0.0 if insufficient history).
        """
        if len(self._history) < 5:
            return 0.0
        hist_scores = np.array([s for _, s in self._history])
        mean = hist_scores.mean()
        std = hist_scores.std()
        if std < 1e-8:
            return 0.0
        return float((current - mean) / std)

    def _compute_momentum(self, days: int = 5) -> float:
        """Compute change in composite sentiment over recent days.

        Args:
            days: Momentum lookback window.

        Returns:
            Sentiment momentum float (positive = improving sentiment).
        """
        if len(self._history) < 2:
            return 0.0
        recent = [s for _, s in self._history[-days:]]
        older = [s for _, s in self._history[: max(1, len(self._history) - days)]]
        return float(np.mean(recent) - np.mean(older))

    @staticmethod
    def _classify_regime(composite: float, z_score: float) -> str:
        """Map composite score and z-score to a sentiment regime label.

        Args:
            composite: Composite sentiment in ``[-1, +1]``.
            z_score: Sentiment z-score vs 30-day history.

        Returns:
            One of: EXTREME_BEAR / BEAR / NEUTRAL / BULL / EXTREME_BULL.
        """
        if z_score >= SENTIMENT_EXTREME_BULL_THRESHOLD:
            return "EXTREME_BULL"
        elif z_score <= SENTIMENT_EXTREME_BEAR_THRESHOLD:
            return "EXTREME_BEAR"
        elif composite >= 0.20:
            return "BULL"
        elif composite <= -0.20:
            return "BEAR"
        else:
            return "NEUTRAL"
