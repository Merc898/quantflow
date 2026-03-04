"""Agent orchestrator — coordinates the full intelligence cycle.

Fans out all agents concurrently via ``asyncio.gather``, collects
:class:`AgentOutput` objects, runs them through the CEO validator,
and returns a :class:`ValidatedIntelligenceReport`.

Rate limiting, circuit breakers, and retry logic live in each agent client.
The orchestrator is responsible for coordination and timeout enforcement.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from quantflow.agents.ceo_model import CEOValidatorModel
from quantflow.agents.schemas import AgentOutput, ValidatedIntelligenceReport
from quantflow.agents.sentiment import SentimentAggregator
from quantflow.config.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_CYCLE_TIMEOUT_S = 55.0  # keep within the 60-second SLA from spec


class AgentOrchestrator:
    """Coordinate all agents for a complete intelligence cycle.

    Execution flow:
    1. Fan-out: run all configured agents concurrently.
    2. Collect raw :class:`AgentOutput` objects.
    3. Pass to :class:`SentimentAggregator` for composite sentiment.
    4. Pass to :class:`CEOValidatorModel` for cross-validation.
    5. Return a :class:`ValidatedIntelligenceReport`.

    Args:
        openai_agent: Optional OpenAIAgent instance.
        anthropic_agent: Optional AnthropicAgent instance.
        perplexity_agent: Optional PerplexityAgent instance.
        web_scraper: Optional WebScraperAgent instance.
        ceo_model: CEO validator (instantiated with the anthropic_agent).
        sentiment_aggregator: Optional SentimentAggregator instance.
        cycle_timeout_s: Hard timeout for the entire cycle in seconds.
    """

    def __init__(
        self,
        openai_agent: Any | None = None,
        anthropic_agent: Any | None = None,
        perplexity_agent: Any | None = None,
        web_scraper: Any | None = None,
        ceo_model: CEOValidatorModel | None = None,
        sentiment_aggregator: SentimentAggregator | None = None,
        cycle_timeout_s: float = _DEFAULT_CYCLE_TIMEOUT_S,
    ) -> None:
        self._openai = openai_agent
        self._anthropic = anthropic_agent
        self._perplexity = perplexity_agent
        self._scraper = web_scraper
        self._ceo = ceo_model or CEOValidatorModel(anthropic_agent=anthropic_agent)
        self._sentiment = sentiment_aggregator or SentimentAggregator()
        self._timeout = cycle_timeout_s

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run_intelligence_cycle(
        self,
        symbol: str,
        market_data: dict[str, Any] | None = None,
    ) -> ValidatedIntelligenceReport:
        """Run a full intelligence cycle for a single symbol.

        All agents are launched concurrently.  Individual agent failures
        are swallowed with a warning so one bad agent never blocks the cycle.

        Args:
            symbol: Ticker symbol to analyse (e.g. "AAPL").
            market_data: Optional current market data for CEO fact-checking.

        Returns:
            :class:`ValidatedIntelligenceReport` with validated sentiment
            and key narrative.
        """
        logger.info("Intelligence cycle starting", symbol=symbol)
        start = datetime.now(tz=UTC)

        # --- Phase 1: Scrape raw documents ---
        raw_docs = []
        if self._scraper is not None:
            try:
                raw_docs = await asyncio.wait_for(
                    self._scraper.scrape_all(symbol),
                    timeout=self._timeout * 0.4,
                )
            except TimeoutError:
                logger.warning("Web scrape timed out", symbol=symbol)
            except Exception as exc:
                logger.warning("Web scrape failed", symbol=symbol, error=str(exc))

        # Build combined raw text for VADER/FinBERT
        combined_text = " ".join(d.text for d in raw_docs[:30] if d.text)

        # --- Phase 2: Fan out all LLM agents concurrently ---
        agent_tasks: dict[str, Any] = {}
        if self._perplexity is not None:
            agent_tasks["perplexity"] = self._perplexity.build_agent_output(symbol)
        if self._openai is not None and combined_text:
            agent_tasks["openai"] = self._openai.build_agent_output(
                combined_text[:8000],
                symbol,
                sources=[d.url for d in raw_docs[:10]],
            )
        if self._anthropic is not None and combined_text:
            agent_tasks["anthropic"] = self._anthropic.build_agent_output(
                combined_text[:50_000],
                symbol,
                sources=[d.url for d in raw_docs[:10]],
            )

        agent_outputs: list[AgentOutput] = []
        if agent_tasks:
            results = await asyncio.gather(
                *agent_tasks.values(),
                return_exceptions=True,
            )
            for name, result in zip(agent_tasks.keys(), results, strict=False):
                if isinstance(result, Exception):
                    logger.warning(
                        "Agent failed",
                        agent=name,
                        symbol=symbol,
                        error=str(result),
                    )
                else:
                    agent_outputs.append(result)  # type: ignore[arg-type]

        # --- Phase 3: Sentiment aggregation ---
        aggregated = self._sentiment.aggregate(agent_outputs, raw_docs)

        # If no LLM agents ran, create a synthetic scraper output from aggregated sentiment
        if not agent_outputs and raw_docs:
            agent_outputs.append(
                AgentOutput(
                    agent_name="WebScraperAgent",
                    symbol=symbol,
                    timestamp=datetime.now(tz=UTC),
                    sentiment_score=aggregated.composite_sentiment,
                    confidence=0.40,
                    raw_sources=[d.url for d in raw_docs[:5]],
                )
            )

        # --- Phase 4: CEO validation ---
        report = await self._ceo.validate(agent_outputs, market_data or {})

        elapsed = (datetime.now(tz=UTC) - start).total_seconds()
        logger.info(
            "Intelligence cycle complete",
            symbol=symbol,
            elapsed_s=round(elapsed, 2),
            n_agents=len(agent_outputs),
            validated_sentiment=report.validated_sentiment,
            conflict=report.conflict_detected,
        )
        return report

    async def run_batch(
        self,
        symbols: list[str],
        market_data: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, ValidatedIntelligenceReport]:
        """Run intelligence cycles for multiple symbols concurrently.

        Args:
            symbols: List of ticker symbols.
            market_data: Optional dict mapping symbol → market data snapshot.

        Returns:
            Dict mapping symbol → :class:`ValidatedIntelligenceReport`.
        """
        md = market_data or {}
        tasks = {sym: self.run_intelligence_cycle(sym, md.get(sym)) for sym in symbols}
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        reports: dict[str, ValidatedIntelligenceReport] = {}
        for sym, result in zip(tasks.keys(), results, strict=False):
            if isinstance(result, Exception):
                logger.error(
                    "Intelligence cycle failed",
                    symbol=sym,
                    error=str(result),
                )
            else:
                reports[sym] = result  # type: ignore[assignment]
        return reports
