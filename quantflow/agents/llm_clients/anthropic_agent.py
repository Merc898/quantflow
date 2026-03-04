"""Anthropic Claude agent for deep document analysis and synthesis.

Responsibilities:
- Long-document analysis (entire 10-K via extended context)
- Cross-agent signal synthesis and conflict resolution
- Generating plain-English recommendation rationale
- Acting as the CEO meta-model for cross-validation
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import httpx

from quantflow.agents.llm_clients.base_client import BaseLLMClient, LLMClientError
from quantflow.agents.schemas import (
    AgentOutput,
    DocumentAnalysis,
    Synthesis,
)
from quantflow.config.logging import get_logger

logger = get_logger(__name__)

_ANTHROPIC_MESSAGES_ENDPOINT = "https://api.anthropic.com/v1/messages"
_MODEL = "claude-sonnet-4-6"
_ANTHROPIC_VERSION = "2023-06-01"
_MAX_TOKENS = 2048
_ANALYSIS_TEMPERATURE = 0.1
_SYNTHESIS_TEMPERATURE = 0.3

_ANALYST_SYSTEM = (
    "You are a critical, evidence-driven quantitative analyst at a top hedge fund. "
    "Your job is to analyze financial documents and extract actionable intelligence. "
    "Be precise, skeptical of hype, and always quantify uncertainty. "
    "When synthesizing conflicting views, explicitly note the contradiction. "
    "Output structured JSON unless otherwise instructed."
)

_CEO_SYSTEM = (
    "You are the Chief Investment Officer reviewing intelligence reports from multiple analysts. "
    "Your task: identify consensus, flag contradictions, detect potential hallucinations by "
    "checking numerical claims against provided market data, and produce a validated final report. "
    "Apply Bayesian skepticism: extraordinary claims require extraordinary evidence. "
    "Output ONLY valid JSON."
)


class AnthropicAgent(BaseLLMClient):
    """Claude agent for deep analysis, synthesis, and validation.

    Args:
        api_key: Anthropic API key.
        model: Claude model identifier.
        timeout_s: Request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str,
        model: str = _MODEL,
        timeout_s: float = 60.0,
    ) -> None:
        super().__init__(api_key=api_key, timeout_s=timeout_s)
        self._model = model

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    async def analyze_long_document(
        self,
        document: str,
        queries: list[str],
    ) -> DocumentAnalysis:
        """Analyse a long financial document against specific queries.

        Args:
            document: Full document text (will be truncated to ~100k chars).
            queries: List of specific questions to answer from the document.

        Returns:
            :class:`DocumentAnalysis` with summary, findings, and answers.
        """
        query_list = "\n".join(f"- {q}" for q in queries)
        prompt = (
            f"Analyse the following financial document and answer these questions:\n{query_list}\n\n"
            "Return JSON with fields: summary, sentiment_score (float -1 to 1), "
            "key_findings (list), risk_factors (list), answers (dict question->answer).\n\n"
            f"Document (truncated to 100k chars):\n{document[:100_000]}"
        )
        response = await self._chat(prompt, system=_ANALYST_SYSTEM)
        data = self.parse_json_from_response(response)
        return DocumentAnalysis(**data)

    async def synthesize_signals(
        self,
        agent_outputs: list[AgentOutput],
    ) -> Synthesis:
        """Synthesise potentially conflicting signals from multiple agents.

        Args:
            agent_outputs: List of normalised agent outputs.

        Returns:
            :class:`Synthesis` with consensus sentiment and conflict detection.
        """
        summaries = []
        for ao in agent_outputs:
            summaries.append(
                f"Agent: {ao.agent_name}, Sentiment: {ao.sentiment_score:.2f}, "
                f"Confidence: {ao.confidence:.2f}, "
                f"Bullish: {ao.bullish_factors[:3]}, Bearish: {ao.bearish_factors[:3]}"
            )
        agent_text = "\n".join(summaries)

        prompt = (
            "Synthesise the following analyst reports into a single consensus view.\n\n"
            f"Reports:\n{agent_text}\n\n"
            "Return JSON with fields:\n"
            "  consensus_sentiment (float -1 to 1),\n"
            "  conflict_detected (bool),\n"
            "  conflict_description (str),\n"
            "  narrative (str, 2-3 sentences),\n"
            "  recommended_action (str: 'buy'|'hold'|'sell')"
        )
        response = await self._chat(prompt, system=_ANALYST_SYSTEM)
        data = self.parse_json_from_response(response)
        return Synthesis(**data)

    async def generate_recommendation_rationale(
        self,
        symbol: str,
        signal: float,
        confidence: float,
        bullish_factors: list[str],
        bearish_factors: list[str],
        regime: str | None = None,
    ) -> str:
        """Generate a plain-English rationale for a trading recommendation.

        Args:
            symbol: Ticker symbol.
            signal: Normalised signal value ``[-1, +1]``.
            confidence: Signal confidence ``[0, 1]``.
            bullish_factors: Key bullish evidence.
            bearish_factors: Key bearish evidence.
            regime: Optional current market regime label.

        Returns:
            2–3 sentence plain English rationale string.
        """
        direction = "BUY" if signal > 0.05 else "SELL" if signal < -0.05 else "HOLD"
        prompt = (
            f"Generate a concise 2-3 sentence investment rationale for {symbol}.\n"
            f"Recommendation: {direction} (signal={signal:.2f}, confidence={confidence:.0%})\n"
            f"Bullish factors: {bullish_factors[:5]}\n"
            f"Bearish factors: {bearish_factors[:5]}\n"
            f"Market regime: {regime or 'unknown'}\n\n"
            "Write in the style of a senior portfolio manager. Be direct and specific. "
            "No JSON — plain text only."
        )
        return await self._chat(prompt, system=_ANALYST_SYSTEM, temperature=0.5)

    async def ceo_validate(
        self,
        agent_outputs: list[AgentOutput],
        market_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Run the CEO cross-validation pass on all agent outputs.

        Args:
            agent_outputs: All raw agent outputs to validate.
            market_data: Current market data snapshot for fact-checking.

        Returns:
            Dict with validated_sentiment, conflict_detected, ceo_override,
            ceo_reasoning, hallucinations_flagged, key_narrative, risk_events.
        """
        reports = []
        for ao in agent_outputs:
            reports.append(
                {
                    "agent": ao.agent_name,
                    "sentiment": ao.sentiment_score,
                    "confidence": ao.confidence,
                    "bullish": ao.bullish_factors[:4],
                    "bearish": ao.bearish_factors[:4],
                    "key_events": ao.key_events[:4],
                    "claims": [c.claim for c in ao.factual_claims[:5]],
                }
            )

        prompt = (
            "You are the CIO reviewing these analyst reports. "
            "Cross-validate them against the market data snapshot.\n\n"
            f"Analyst reports:\n{json.dumps(reports, indent=2)}\n\n"
            f"Market data snapshot:\n{json.dumps(market_data, indent=2)}\n\n"
            "Return JSON with:\n"
            "  validated_sentiment (float -1 to 1) — your best estimate\n"
            "  conflict_detected (bool)\n"
            "  ceo_override (bool) — True if you disagree with naive consensus\n"
            "  ceo_reasoning (str | null) — why you overrode (if you did)\n"
            "  hallucinations_flagged (int) — count of claims contradicting market data\n"
            "  key_narrative (str) — 2-3 sentence plain English summary\n"
            "  risk_events (list[str]) — upcoming catalysts and risks"
        )
        response = await self._chat(prompt, system=_CEO_SYSTEM, temperature=0.1)
        return self.parse_json_from_response(response)

    async def build_agent_output(
        self,
        document: str,
        symbol: str,
        sources: list[str] | None = None,
    ) -> AgentOutput:
        """Build a normalised :class:`AgentOutput` from a document.

        Args:
            document: Financial document text.
            symbol: Ticker symbol.
            sources: Source URLs.

        Returns:
            :class:`AgentOutput`.
        """
        analysis = await self.analyze_long_document(
            document,
            queries=[
                f"What is the overall sentiment for {symbol}?",
                "What are the top 3 bullish factors?",
                "What are the top 3 bearish factors?",
                "What are the key risks?",
            ],
        )
        bullish = analysis.key_findings[:3]
        bearish = analysis.risk_factors[:3]
        return AgentOutput(
            agent_name="AnthropicAgent",
            symbol=symbol,
            timestamp=datetime.now(tz=UTC),
            sentiment_score=analysis.sentiment_score,
            confidence=0.70,
            bullish_factors=bullish,
            bearish_factors=bearish,
            raw_sources=sources or [],
            metadata={"summary": analysis.summary, "model": self._model},
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _chat(
        self,
        user_message: str,
        system: str = _ANALYST_SYSTEM,
        temperature: float = _ANALYSIS_TEMPERATURE,
    ) -> str:
        """Send a single-turn chat message and return the text response.

        Args:
            user_message: User turn content.
            system: System prompt.
            temperature: Sampling temperature.

        Returns:
            Assistant response text.
        """
        payload = {
            "model": self._model,
            "max_tokens": _MAX_TOKENS,
            "temperature": temperature,
            "system": system,
            "messages": [{"role": "user", "content": user_message}],
        }
        response = await self.call_with_retry(payload, _ANTHROPIC_MESSAGES_ENDPOINT)
        try:
            content = response["content"][0]["text"]
            usage = response.get("usage", {})
            self._total_tokens_used += usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            return content
        except (KeyError, IndexError) as exc:
            raise LLMClientError(f"Unexpected Anthropic response: {response}") from exc

    async def _call_api(
        self,
        payload: dict[str, Any],
        endpoint: str,
    ) -> dict[str, Any]:
        """Execute a single async HTTP call to the Anthropic Messages API.

        Args:
            payload: Request body.
            endpoint: API endpoint URL.

        Returns:
            Parsed JSON response.
        """
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                endpoint,
                json=payload,
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": _ANTHROPIC_VERSION,
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            return resp.json()
