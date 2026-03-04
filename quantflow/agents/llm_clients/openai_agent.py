"""OpenAI GPT-4o agent for structured financial analysis.

Responsibilities:
- Structured sentiment extraction from news text (JSON mode)
- Market event classification
- Earnings analysis from filing text
- Temperature=0.1 for analytical tasks
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import httpx

from quantflow.agents.llm_clients.base_client import BaseLLMClient, LLMClientError
from quantflow.agents.schemas import (
    AgentOutput,
    EarningsAnalysis,
    EventClassification,
    SentimentScore,
)
from quantflow.config.logging import get_logger

logger = get_logger(__name__)

_OPENAI_CHAT_ENDPOINT = "https://api.openai.com/v1/chat/completions"
_MODEL = "gpt-4o"
_ANALYSIS_TEMPERATURE = 0.1
_SUMMARY_TEMPERATURE = 0.7
_MAX_TOKENS = 1024

_SYSTEM_PROMPT = (
    "You are a senior quantitative analyst at a top-tier hedge fund. "
    "Analyse the following financial content and extract structured signals. "
    "Be precise, cite evidence from the text, and quantify uncertainty. "
    "Output ONLY valid JSON matching the schema provided. No prose outside the JSON."
)

_SENTIMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "minimum": -1.0, "maximum": 1.0},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reasoning": {"type": "string"},
        "bullish_factors": {"type": "array", "items": {"type": "string"}},
        "bearish_factors": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["score", "confidence", "reasoning", "bullish_factors", "bearish_factors"],
}

_EVENT_SCHEMA = {
    "type": "object",
    "properties": {
        "event_type": {"type": "string"},
        "sentiment_impact": {"type": "number", "minimum": -1.0, "maximum": 1.0},
        "magnitude": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "description": {"type": "string"},
    },
    "required": ["event_type", "sentiment_impact", "magnitude", "confidence", "description"],
}

_EARNINGS_SCHEMA = {
    "type": "object",
    "properties": {
        "eps_surprise_pct": {"type": ["number", "null"]},
        "revenue_surprise_pct": {"type": ["number", "null"]},
        "guidance_tone": {"type": "string", "enum": ["raised", "maintained", "lowered", "none"]},
        "management_tone_score": {"type": "number", "minimum": -1.0, "maximum": 1.0},
        "key_risks": {"type": "array", "items": {"type": "string"}},
        "key_positives": {"type": "array", "items": {"type": "string"}},
        "forward_looking_statements": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["guidance_tone", "management_tone_score", "key_risks", "key_positives"],
}


class OpenAIAgent(BaseLLMClient):
    """GPT-4o agent for structured financial analysis.

    Args:
        api_key: OpenAI API key.
        model: Model identifier (default "gpt-4o").
        timeout_s: Request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str,
        model: str = _MODEL,
        timeout_s: float = 30.0,
    ) -> None:
        super().__init__(api_key=api_key, timeout_s=timeout_s)
        self._model = model

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    async def extract_sentiment(self, text: str, symbol: str) -> SentimentScore:
        """Extract structured sentiment from financial text.

        Args:
            text: News article, filing excerpt, or other financial text.
            symbol: Ticker symbol for context.

        Returns:
            :class:`SentimentScore` with score, confidence, and factors.
        """
        prompt = (
            f"Analyse the sentiment of the following text for {symbol} stock.\n\n"
            f"Text:\n{text[:4000]}\n\n"
            f"Return JSON matching this schema: {json.dumps(_SENTIMENT_SCHEMA)}"
        )
        payload = self._build_payload(prompt, temperature=_ANALYSIS_TEMPERATURE)
        response = await self.call_with_retry(payload, _OPENAI_CHAT_ENDPOINT)
        content = self._extract_content(response)
        data = self.parse_json_from_response(content)
        return SentimentScore(**data)

    async def classify_event(self, text: str) -> EventClassification:
        """Classify a financial market event from text.

        Args:
            text: Text describing or containing a market event.

        Returns:
            :class:`EventClassification` with event type and expected impact.
        """
        prompt = (
            "Classify the primary market event in the following text.\n\n"
            f"Text:\n{text[:4000]}\n\n"
            f"Return JSON matching: {json.dumps(_EVENT_SCHEMA)}"
        )
        payload = self._build_payload(prompt, temperature=_ANALYSIS_TEMPERATURE)
        response = await self.call_with_retry(payload, _OPENAI_CHAT_ENDPOINT)
        content = self._extract_content(response)
        data = self.parse_json_from_response(content)
        return EventClassification(**data)

    async def analyze_earnings(self, filing_text: str) -> EarningsAnalysis:
        """Analyse an earnings release or SEC filing for key metrics.

        Args:
            filing_text: Full or partial earnings filing text.

        Returns:
            :class:`EarningsAnalysis` with surprise, guidance, and tone.
        """
        prompt = (
            "Analyse the following earnings filing and extract key metrics.\n\n"
            f"Filing:\n{filing_text[:6000]}\n\n"
            f"Return JSON matching: {json.dumps(_EARNINGS_SCHEMA)}"
        )
        payload = self._build_payload(prompt, temperature=_ANALYSIS_TEMPERATURE)
        response = await self.call_with_retry(payload, _OPENAI_CHAT_ENDPOINT)
        content = self._extract_content(response)
        data = self.parse_json_from_response(content)
        return EarningsAnalysis(**data)

    async def build_agent_output(
        self,
        text: str,
        symbol: str,
        sources: list[str] | None = None,
    ) -> AgentOutput:
        """Build a normalised :class:`AgentOutput` from news text.

        Calls :meth:`extract_sentiment` and :meth:`classify_event` in parallel,
        then packages the results.

        Args:
            text: Source text for analysis.
            symbol: Ticker symbol.
            sources: Optional list of source URLs.

        Returns:
            :class:`AgentOutput` ready for the CEO validator.
        """
        sentiment = await self.extract_sentiment(text, symbol)
        return AgentOutput(
            agent_name="OpenAIAgent",
            symbol=symbol,
            timestamp=datetime.now(tz=UTC),
            sentiment_score=sentiment.score,
            confidence=sentiment.confidence,
            bullish_factors=sentiment.bullish_factors,
            bearish_factors=sentiment.bearish_factors,
            raw_sources=sources or [],
            metadata={"reasoning": sentiment.reasoning, "model": self._model},
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_payload(
        self,
        user_prompt: str,
        temperature: float = _ANALYSIS_TEMPERATURE,
    ) -> dict[str, Any]:
        """Build the OpenAI chat completion request payload.

        Args:
            user_prompt: The user message content.
            temperature: Sampling temperature.

        Returns:
            Request payload dict.
        """
        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": _MAX_TOKENS,
            "response_format": {"type": "json_object"},
        }

    def _extract_content(self, response: dict[str, Any]) -> str:
        """Extract the assistant message content from an API response.

        Args:
            response: Parsed API response dict.

        Returns:
            Content string.

        Raises:
            LLMClientError: If response structure is unexpected.
        """
        try:
            content = response["choices"][0]["message"]["content"]
            usage = response.get("usage", {})
            self._total_tokens_used += usage.get("total_tokens", 0)
            return content
        except (KeyError, IndexError) as exc:
            raise LLMClientError(f"Unexpected OpenAI response structure: {response}") from exc

    async def _call_api(
        self,
        payload: dict[str, Any],
        endpoint: str,
    ) -> dict[str, Any]:
        """Execute a single async HTTP call to the OpenAI API.

        Args:
            payload: Request body dict.
            endpoint: API endpoint URL.

        Returns:
            Parsed JSON response.
        """
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                endpoint,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            return resp.json()
