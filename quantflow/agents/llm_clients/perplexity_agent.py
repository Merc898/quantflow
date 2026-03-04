"""Perplexity AI agent for grounded real-time web search.

Uses the Perplexity sonar model to answer structured financial queries
with live web citations — giving the most current market intelligence.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx

from quantflow.agents.llm_clients.base_client import BaseLLMClient, LLMClientError
from quantflow.agents.schemas import AgentOutput, PerplexityResponse
from quantflow.config.logging import get_logger

logger = get_logger(__name__)

_PERPLEXITY_ENDPOINT = "https://api.perplexity.ai/chat/completions"
_MODEL = "sonar-pro"
_MAX_TOKENS = 1024
_TEMPERATURE = 0.1

# Standard financial query templates per symbol
_QUERY_TEMPLATES = [
    "What are the latest analyst upgrades or downgrades for {symbol} stock?",
    "What recent news or events could materially affect {symbol} stock price?",
    "What is the current market sentiment among professional investors for {symbol}?",
    "What macro economic risks are most discussed in financial markets today?",
]


class PerplexityAgent(BaseLLMClient):
    """Grounded real-time market intelligence via Perplexity.

    Issues multiple structured queries per symbol and returns
    answers with citations for fact-checking.

    Args:
        api_key: Perplexity API key.
        model: Perplexity model to use.
        timeout_s: Per-request timeout.
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

    async def query(
        self,
        prompt: str,
        symbol: str | None = None,
    ) -> PerplexityResponse:
        """Execute a single grounded search query.

        Args:
            prompt: Natural language question.
            symbol: Optional ticker for contextual logging.

        Returns:
            :class:`PerplexityResponse` with answer and citations.
        """
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a financial research assistant. "
                        "Answer concisely and cite your sources. "
                        "Focus on factual, current information relevant to investors."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": _MAX_TOKENS,
            "temperature": _TEMPERATURE,
            "return_citations": True,
            "return_related_questions": False,
        }

        self._logger.info(
            "Perplexity query",
            symbol=symbol,
            prompt_preview=prompt[:80],
        )
        response = await self.call_with_retry(payload, _PERPLEXITY_ENDPOINT)
        return self._parse_response(response, prompt)

    async def run_symbol_queries(self, symbol: str) -> list[PerplexityResponse]:
        """Run all standard financial queries for a ticker symbol.

        Args:
            symbol: Ticker symbol (e.g. "AAPL").

        Returns:
            List of :class:`PerplexityResponse` objects, one per query template.
        """
        import asyncio

        queries = [tmpl.format(symbol=symbol) for tmpl in _QUERY_TEMPLATES]
        tasks = [self.query(q, symbol=symbol) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid: list[PerplexityResponse] = []
        for q, r in zip(queries, results, strict=False):
            if isinstance(r, Exception):
                self._logger.warning(
                    "Perplexity query failed",
                    symbol=symbol,
                    query=q[:60],
                    error=str(r),
                )
            else:
                valid.append(r)  # type: ignore[arg-type]
        return valid

    async def build_agent_output(
        self,
        symbol: str,
    ) -> AgentOutput:
        """Build a normalised :class:`AgentOutput` from all symbol queries.

        Args:
            symbol: Ticker symbol.

        Returns:
            :class:`AgentOutput` with aggregated Perplexity findings.
        """
        responses = await self.run_symbol_queries(symbol)

        if not responses:
            return AgentOutput(
                agent_name="PerplexityAgent",
                symbol=symbol,
                timestamp=datetime.now(tz=UTC),
                sentiment_score=0.0,
                confidence=0.0,
                metadata={"error": "all_queries_failed"},
            )

        # Combine all answers into a single blob and do basic sentiment scoring
        combined_text = " ".join(r.answer for r in responses)
        all_citations = list({url for r in responses for url in r.citations})
        total_tokens = sum(r.tokens_used for r in responses)

        # Simple keyword-based pre-sentiment (true sentiment via OpenAIAgent)
        sentiment = self._keyword_sentiment(combined_text)

        return AgentOutput(
            agent_name="PerplexityAgent",
            symbol=symbol,
            timestamp=datetime.now(tz=UTC),
            sentiment_score=sentiment,
            confidence=min(0.80, 0.4 + len(all_citations) * 0.02),
            key_events=self._extract_key_phrases(combined_text),
            raw_sources=all_citations[:20],
            metadata={
                "n_queries": len(responses),
                "n_citations": len(all_citations),
                "total_tokens": total_tokens,
                "model": self._model,
            },
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_response(
        self,
        response: dict[str, Any],
        original_prompt: str,
    ) -> PerplexityResponse:
        """Parse a raw Perplexity API response into a typed object.

        Args:
            response: Parsed JSON from the API.
            original_prompt: The query that produced this response.

        Returns:
            :class:`PerplexityResponse`.

        Raises:
            LLMClientError: If the response structure is unexpected.
        """
        try:
            answer = response["choices"][0]["message"]["content"]
            usage = response.get("usage", {})
            self._total_tokens_used += usage.get("total_tokens", 0)

            # Citations may appear in different fields depending on model version
            citations: list[str] = []
            if "citations" in response:
                citations = response["citations"]
            elif "search_results" in response:
                citations = [r.get("url", "") for r in response["search_results"] if "url" in r]

            return PerplexityResponse(
                answer=answer,
                citations=citations,
                search_queries_used=[original_prompt],
                tokens_used=usage.get("total_tokens", 0),
            )
        except (KeyError, IndexError) as exc:
            raise LLMClientError(f"Unexpected Perplexity response structure: {response}") from exc

    def _keyword_sentiment(self, text: str) -> float:
        """Fast keyword-based sentiment score as a Perplexity pre-pass.

        Returns a rough score in ``[-0.5, +0.5]`` — the LLM agents
        will refine this with proper semantic analysis.

        Args:
            text: Combined query answer text.

        Returns:
            Rough sentiment float.
        """
        text_lower = text.lower()
        bullish_kw = [
            "upgrade",
            "beat",
            "outperform",
            "bullish",
            "growth",
            "strong",
            "positive",
            "buy",
            "momentum",
            "record",
            "guidance raised",
        ]
        bearish_kw = [
            "downgrade",
            "miss",
            "underperform",
            "bearish",
            "decline",
            "weak",
            "negative",
            "sell",
            "layoffs",
            "guidance lowered",
            "risk",
        ]
        bull_score = sum(text_lower.count(kw) for kw in bullish_kw)
        bear_score = sum(text_lower.count(kw) for kw in bearish_kw)
        total = bull_score + bear_score
        if total == 0:
            return 0.0
        raw = (bull_score - bear_score) / total
        return float(max(-0.5, min(0.5, raw)))

    def _extract_key_phrases(self, text: str, max_phrases: int = 5) -> list[str]:
        """Extract candidate key event phrases from combined text.

        Args:
            text: Combined query answers.
            max_phrases: Maximum phrases to return.

        Returns:
            List of key phrase strings.
        """
        import re

        # Split into sentences and pick the most information-dense ones
        sentences = re.split(r"[.!?]", text)
        trigger_words = {
            "earnings",
            "revenue",
            "guidance",
            "merger",
            "acquisition",
            "fda",
            "approval",
            "downgrade",
            "upgrade",
            "layoff",
            "recall",
            "settlement",
            "dividend",
            "buyback",
            "ceo",
            "cfo",
            "outlook",
        }
        scored = []
        for s in sentences:
            s_stripped = s.strip()
            if len(s_stripped) < 20:
                continue
            score = sum(1 for w in trigger_words if w in s_stripped.lower())
            if score > 0:
                scored.append((score, s_stripped[:120]))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [phrase for _, phrase in scored[:max_phrases]]

    async def _call_api(
        self,
        payload: dict[str, Any],
        endpoint: str,
    ) -> dict[str, Any]:
        """Execute a single async HTTP call to the Perplexity API.

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
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            return resp.json()
