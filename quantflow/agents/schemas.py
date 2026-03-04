"""Shared Pydantic schemas for the QuantFlow agentic intelligence layer.

All agent outputs, intelligence reports, and intermediate data structures
are defined here so every module shares a single source of truth.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Raw scraper output
# ---------------------------------------------------------------------------


class RawDocument(BaseModel):
    """A single document returned by the web scraper.

    Attributes:
        source: Human-readable source name (e.g. "Reuters", "SEC EDGAR").
        url: Canonical URL of the document.
        title: Headline or document title.
        text: Full plain-text body (HTML stripped).
        timestamp: Publication or retrieval timestamp (UTC).
        symbol: Ticker symbol this document relates to (if known).
        relevance_score: Estimated relevance to the symbol [0.0, 1.0].
    """

    source: str
    url: str
    title: str = ""
    text: str
    timestamp: datetime
    symbol: str | None = None
    relevance_score: float = Field(default=1.0, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Factual claim (for hallucination detection)
# ---------------------------------------------------------------------------


class FactualClaim(BaseModel):
    """A verifiable factual claim extracted from an agent output.

    Attributes:
        claim: The claim text (e.g. "EPS was $1.23 in Q3 2024").
        value: Extracted numeric value if applicable.
        verified: True if cross-checked against structured DB data.
        contradiction: True if contradicts known data.
    """

    claim: str
    value: float | None = None
    verified: bool = False
    contradiction: bool = False


# ---------------------------------------------------------------------------
# LLM structured outputs
# ---------------------------------------------------------------------------


class SentimentScore(BaseModel):
    """Structured sentiment result from an LLM agent.

    Attributes:
        score: Normalised sentiment score in ``[-1.0, +1.0]``.
        confidence: Model confidence in the score ``[0.0, 1.0]``.
        reasoning: Short textual justification (1–2 sentences).
        bullish_factors: Key reasons for positive sentiment.
        bearish_factors: Key reasons for negative sentiment.
    """

    score: float = Field(..., ge=-1.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = ""
    bullish_factors: list[str] = Field(default_factory=list)
    bearish_factors: list[str] = Field(default_factory=list)


class EventClassification(BaseModel):
    """Classification of a market event extracted from text.

    Attributes:
        event_type: Category (e.g. "earnings_beat", "M&A", "FDA_approval").
        sentiment_impact: Expected price impact direction ``[-1, +1]``.
        magnitude: Expected magnitude of impact (0=negligible, 1=large).
        confidence: Classification confidence ``[0.0, 1.0]``.
        description: Short description of the event.
    """

    event_type: str
    sentiment_impact: float = Field(default=0.0, ge=-1.0, le=1.0)
    magnitude: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    description: str = ""


class EarningsAnalysis(BaseModel):
    """Structured analysis of an earnings release or filing.

    Attributes:
        eps_surprise_pct: EPS beat/miss as percentage of estimate.
        revenue_surprise_pct: Revenue beat/miss percentage.
        guidance_tone: "raised", "maintained", "lowered", or "none".
        management_tone_score: Tone of management commentary ``[-1, +1]``.
        key_risks: List of identified risk factors.
        key_positives: List of identified positive factors.
        forward_looking_statements: Notable forward guidance statements.
    """

    eps_surprise_pct: float | None = None
    revenue_surprise_pct: float | None = None
    guidance_tone: str = "none"
    management_tone_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    key_risks: list[str] = Field(default_factory=list)
    key_positives: list[str] = Field(default_factory=list)
    forward_looking_statements: list[str] = Field(default_factory=list)


class DocumentAnalysis(BaseModel):
    """Deep document analysis result from AnthropicAgent.

    Attributes:
        summary: 3–5 sentence plain English summary.
        sentiment_score: Overall sentiment of the document ``[-1, +1]``.
        key_findings: List of most important findings.
        risk_factors: Extracted risk factors.
        answers: Answers to the query questions posed to the agent.
    """

    summary: str = ""
    sentiment_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    key_findings: list[str] = Field(default_factory=list)
    risk_factors: list[str] = Field(default_factory=list)
    answers: dict[str, str] = Field(default_factory=dict)


class Synthesis(BaseModel):
    """Cross-agent synthesis result from AnthropicAgent.

    Attributes:
        consensus_sentiment: Weighted consensus sentiment ``[-1, +1]``.
        conflict_detected: Whether agents substantially disagree.
        conflict_description: Explanation of the conflict if detected.
        narrative: 2–3 sentence market narrative.
        recommended_action: Suggested signal direction.
    """

    consensus_sentiment: float = Field(default=0.0, ge=-1.0, le=1.0)
    conflict_detected: bool = False
    conflict_description: str = ""
    narrative: str = ""
    recommended_action: str = "hold"


class PerplexityResponse(BaseModel):
    """Response from the Perplexity real-time search API.

    Attributes:
        answer: Full answer text.
        citations: List of cited URLs.
        search_queries_used: The underlying search queries.
        tokens_used: Total tokens consumed.
    """

    answer: str
    citations: list[str] = Field(default_factory=list)
    search_queries_used: list[str] = Field(default_factory=list)
    tokens_used: int = 0


# ---------------------------------------------------------------------------
# Agent output (normalised across all agents)
# ---------------------------------------------------------------------------


class AgentOutput(BaseModel):
    """Normalised output from any single agent.

    Attributes:
        agent_name: Identifier of the producing agent.
        symbol: Target ticker symbol.
        timestamp: UTC timestamp of output production.
        sentiment_score: Normalised sentiment ``[-1, +1]``.
        confidence: Agent confidence ``[0, 1]``.
        key_events: Notable events identified.
        bullish_factors: Identified positive factors.
        bearish_factors: Identified negative factors.
        raw_sources: URLs or citation strings used.
        factual_claims: Verifiable factual claims extracted.
        metadata: Agent-specific extra data.
    """

    agent_name: str
    symbol: str
    timestamp: datetime
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    key_events: list[str] = Field(default_factory=list)
    bullish_factors: list[str] = Field(default_factory=list)
    bearish_factors: list[str] = Field(default_factory=list)
    raw_sources: list[str] = Field(default_factory=list)
    factual_claims: list[FactualClaim] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("sentiment_score")
    @classmethod
    def sentiment_must_be_finite(cls, v: float) -> float:
        """Ensure sentiment score is a finite number."""
        import math

        if not math.isfinite(v):
            raise ValueError(f"sentiment_score must be finite, got {v}")
        return v


# ---------------------------------------------------------------------------
# Intelligence report (CEO validator output)
# ---------------------------------------------------------------------------


class ValidatedIntelligenceReport(BaseModel):
    """Final intelligence report after CEO cross-validation.

    Attributes:
        symbol: Target ticker symbol.
        timestamp: Report generation UTC timestamp.
        validated_sentiment: CEO-validated composite sentiment ``[-1, +1]``.
        consensus_confidence: Overall confidence ``[0, 1]``.
        conflict_detected: Whether agents substantially disagreed.
        key_narrative: Plain English 2–3 sentence market summary.
        risk_events: Upcoming catalysts / risks identified.
        agent_outputs: All raw agent outputs used.
        ceo_override: True if CEO overrode the raw consensus.
        ceo_reasoning: CEO's explanation if override occurred.
        hallucinations_flagged: Number of factual contradictions found.
        source_weights: Per-agent reliability weights used.
    """

    symbol: str
    timestamp: datetime
    validated_sentiment: float = Field(..., ge=-1.0, le=1.0)
    consensus_confidence: float = Field(..., ge=0.0, le=1.0)
    conflict_detected: bool = False
    key_narrative: str = ""
    risk_events: list[str] = Field(default_factory=list)
    agent_outputs: list[AgentOutput] = Field(default_factory=list)
    ceo_override: bool = False
    ceo_reasoning: str | None = None
    hallucinations_flagged: int = 0
    source_weights: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Sentiment aggregator output
# ---------------------------------------------------------------------------

_SENTIMENT_REGIMES = frozenset(
    {"EXTREME_BEAR", "BEAR", "NEUTRAL", "BULL", "EXTREME_BULL"}
)


class AggregatedSentiment(BaseModel):
    """Output of the multi-source sentiment aggregation.

    Attributes:
        composite_sentiment: Weighted composite score ``[-1, +1]``.
        sentiment_regime: One of EXTREME_BEAR / BEAR / NEUTRAL / BULL / EXTREME_BULL.
        sentiment_momentum: 5-day change in composite sentiment.
        contrarian_signal: True if extreme reading (z > 2 or z < -2).
        sentiment_zscore: Current score vs 30-day rolling baseline.
        source_scores: Per-source breakdown of sentiment scores.
    """

    composite_sentiment: float = Field(..., ge=-1.0, le=1.0)
    sentiment_regime: str
    sentiment_momentum: float = 0.0
    contrarian_signal: bool = False
    sentiment_zscore: float = 0.0
    source_scores: dict[str, float] = Field(default_factory=dict)

    @field_validator("sentiment_regime")
    @classmethod
    def regime_must_be_valid(cls, v: str) -> str:
        """Validate regime label."""
        if v not in _SENTIMENT_REGIMES:
            raise ValueError(f"sentiment_regime must be one of {_SENTIMENT_REGIMES}")
        return v
