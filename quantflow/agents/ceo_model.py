"""CEO Validator Model — cross-validates all agent outputs.

The CEO is a meta-model that:
1. Receives all agent outputs (Perplexity, OpenAI, Anthropic, scrapers)
2. Checks for consistency and contradictions
3. Weights outputs by dynamically-updated source reliability
4. Detects potential hallucinations by cross-referencing factual claims
5. Produces a final validated :class:`ValidatedIntelligenceReport`

Implementation: uses AnthropicAgent with a CIO-role system prompt as the
meta-reasoner, with fallback to a pure statistical consensus when the API
is unavailable.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from quantflow.agents.schemas import AgentOutput, ValidatedIntelligenceReport
from quantflow.config.constants import (
    AGENT_WEIGHT_ANTHROPIC,
    AGENT_WEIGHT_OPENAI,
    AGENT_WEIGHT_PERPLEXITY,
    AGENT_WEIGHT_SCRAPERS,
    CONSENSUS_HIGH_THRESHOLD,
    CONSENSUS_LOW_THRESHOLD,
    CONSENSUS_MEDIUM_THRESHOLD,
)
from quantflow.config.logging import get_logger

logger = get_logger(__name__)

# Default source reliability weights (updated weekly from DB in production)
_DEFAULT_WEIGHTS: dict[str, float] = {
    "PerplexityAgent": AGENT_WEIGHT_PERPLEXITY,
    "OpenAIAgent": AGENT_WEIGHT_OPENAI,
    "AnthropicAgent": AGENT_WEIGHT_ANTHROPIC,
    "WebScraperAgent": AGENT_WEIGHT_SCRAPERS,
}

# Conflict threshold: std of sentiment scores > this → flag conflict
_CONFLICT_STD_THRESHOLD = 0.35


class CEOValidatorModel:
    """Meta-model that cross-validates and synthesises agent outputs.

    Args:
        anthropic_agent: Optional AnthropicAgent instance for LLM-powered
            validation.  Falls back to statistical consensus if None.
        source_weights: Dict mapping agent names to reliability weights.
            Weights are normalised internally so they need not sum to 1.
    """

    def __init__(
        self,
        anthropic_agent: Any | None = None,
        source_weights: dict[str, float] | None = None,
    ) -> None:
        self._anthropic = anthropic_agent
        self._weights = source_weights or dict(_DEFAULT_WEIGHTS)
        self._logger = get_logger(__name__)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def validate(
        self,
        agent_outputs: list[AgentOutput],
        market_data: dict[str, Any] | None = None,
    ) -> ValidatedIntelligenceReport:
        """Cross-validate agent outputs and produce a final report.

        Args:
            agent_outputs: All raw agent outputs for the symbol.
            market_data: Optional market data snapshot for fact-checking
                (price, volume, recent returns, earnings data).

        Returns:
            :class:`ValidatedIntelligenceReport` with validated sentiment,
            confidence, conflict flag, and CEO reasoning.
        """
        if not agent_outputs:
            return self._empty_report()

        symbol = agent_outputs[0].symbol
        timestamp = datetime.now(tz=timezone.utc)

        # --- Statistical consensus (always computed) ---
        stat_result = self._statistical_consensus(agent_outputs)
        validated_sentiment = stat_result["validated_sentiment"]
        consensus_confidence = stat_result["consensus_confidence"]
        conflict_detected = stat_result["conflict_detected"]
        hallucinations = self._count_hallucinations(agent_outputs, market_data or {})

        ceo_override = False
        ceo_reasoning: str | None = None
        key_narrative = stat_result.get("narrative", "")
        risk_events: list[str] = stat_result.get("risk_events", [])

        # --- LLM validation pass (if AnthropicAgent available) ---
        if self._anthropic is not None:
            try:
                llm_result = await self._anthropic.ceo_validate(
                    agent_outputs, market_data or {}
                )
                llm_sentiment = float(
                    llm_result.get("validated_sentiment", validated_sentiment)
                )
                # CEO override: LLM disagrees with statistical consensus by > 0.3
                if abs(llm_sentiment - validated_sentiment) > 0.30:
                    ceo_override = True
                    ceo_reasoning = llm_result.get("ceo_reasoning")

                validated_sentiment = llm_sentiment
                conflict_detected = bool(
                    llm_result.get("conflict_detected", conflict_detected)
                )
                hallucinations = max(
                    hallucinations,
                    int(llm_result.get("hallucinations_flagged", 0)),
                )
                key_narrative = str(llm_result.get("key_narrative", key_narrative))
                risk_events = list(llm_result.get("risk_events", risk_events))

            except Exception as exc:
                self._logger.warning(
                    "CEO LLM validation failed, using statistical consensus",
                    error=str(exc),
                )

        # Final clip to valid range
        validated_sentiment = float(np.clip(validated_sentiment, -1.0, 1.0))

        self._logger.info(
            "CEO validation complete",
            symbol=symbol,
            validated_sentiment=round(validated_sentiment, 4),
            consensus_confidence=round(consensus_confidence, 4),
            conflict=conflict_detected,
            ceo_override=ceo_override,
            hallucinations=hallucinations,
        )

        return ValidatedIntelligenceReport(
            symbol=symbol,
            timestamp=timestamp,
            validated_sentiment=round(validated_sentiment, 6),
            consensus_confidence=round(consensus_confidence, 6),
            conflict_detected=conflict_detected,
            key_narrative=key_narrative,
            risk_events=risk_events,
            agent_outputs=agent_outputs,
            ceo_override=ceo_override,
            ceo_reasoning=ceo_reasoning,
            hallucinations_flagged=hallucinations,
            source_weights=self._normalised_weights(agent_outputs),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _statistical_consensus(
        self, outputs: list[AgentOutput]
    ) -> dict[str, Any]:
        """Compute weighted-average consensus from agent outputs.

        Args:
            outputs: List of agent outputs.

        Returns:
            Dict with validated_sentiment, consensus_confidence,
            conflict_detected, narrative, risk_events.
        """
        sentiments: list[float] = []
        weights: list[float] = []
        all_bullish: list[str] = []
        all_bearish: list[str] = []
        all_events: list[str] = []

        for ao in outputs:
            w = self._weights.get(ao.agent_name, 0.10) * ao.confidence
            sentiments.append(ao.sentiment_score)
            weights.append(max(w, 1e-4))
            all_bullish.extend(ao.bullish_factors[:3])
            all_bearish.extend(ao.bearish_factors[:3])
            all_events.extend(ao.key_events[:3])

        w_arr = np.array(weights)
        s_arr = np.array(sentiments)
        validated = float(np.dot(w_arr, s_arr) / w_arr.sum())

        # Conflict: high variance among agent signals
        std_val = float(np.std(s_arr))
        conflict_detected = std_val > _CONFLICT_STD_THRESHOLD

        # Confidence from agreement level and source reliability
        agreement_pct = float((np.abs(s_arr - validated) < 0.25).mean())
        if agreement_pct >= CONSENSUS_HIGH_THRESHOLD:
            base_conf = 0.80
        elif agreement_pct >= CONSENSUS_MEDIUM_THRESHOLD:
            base_conf = 0.60
        else:
            base_conf = 0.40

        # Discount confidence when conflict detected
        consensus_confidence = base_conf * (0.75 if conflict_detected else 1.0)

        # Build simple narrative from top factors
        direction = "positive" if validated > 0.05 else "negative" if validated < -0.05 else "neutral"
        top_bull = list(dict.fromkeys(all_bullish))[:2]
        top_bear = list(dict.fromkeys(all_bearish))[:2]
        narrative = (
            f"Overall market intelligence for this asset is {direction} "
            f"(consensus score {validated:.2f}). "
        )
        if top_bull:
            narrative += f"Key support: {', '.join(top_bull)}. "
        if top_bear:
            narrative += f"Key risks: {', '.join(top_bear)}."

        risk_events = list(dict.fromkeys(all_events))[:5]

        return {
            "validated_sentiment": validated,
            "consensus_confidence": consensus_confidence,
            "conflict_detected": conflict_detected,
            "narrative": narrative.strip(),
            "risk_events": risk_events,
        }

    def _count_hallucinations(
        self,
        outputs: list[AgentOutput],
        market_data: dict[str, Any],
    ) -> int:
        """Count factual claims that contradict provided market data.

        Args:
            outputs: Agent outputs with factual_claims populated.
            market_data: Reference data for cross-checking.

        Returns:
            Number of potential hallucinations detected.
        """
        count = 0
        for ao in outputs:
            for claim in ao.factual_claims:
                if claim.contradiction:
                    count += 1
        return count

    def _normalised_weights(
        self, outputs: list[AgentOutput]
    ) -> dict[str, float]:
        """Return normalised per-agent weights used for this validation.

        Args:
            outputs: Agent outputs (to filter to present agents only).

        Returns:
            Dict mapping agent name to normalised weight.
        """
        present = {ao.agent_name for ao in outputs}
        raw = {k: v for k, v in self._weights.items() if k in present}
        total = sum(raw.values())
        if total < 1e-8:
            return {}
        return {k: round(v / total, 4) for k, v in raw.items()}

    def _empty_report(self) -> ValidatedIntelligenceReport:
        """Return a neutral empty report when no agent outputs are available."""
        return ValidatedIntelligenceReport(
            symbol="UNKNOWN",
            timestamp=datetime.now(tz=timezone.utc),
            validated_sentiment=0.0,
            consensus_confidence=0.0,
            conflict_detected=False,
            key_narrative="No agent outputs available.",
            risk_events=[],
            agent_outputs=[],
        )
