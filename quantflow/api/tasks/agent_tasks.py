"""Celery tasks for scheduled agent intelligence cycles.

Tasks are scheduled via Celery Beat:
- Free-tier symbols: every ``AGENT_INTERVAL_FREE_HOURS`` hours.
- Premium-tier symbols: every ``AGENT_INTERVAL_PREMIUM_HOURS`` hours.

All tasks are idempotent — re-running a task for the same symbol within
its interval window simply overwrites the cached result.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from celery import shared_task
from celery.utils.log import get_task_logger

from quantflow.config.constants import (
    AGENT_INTERVAL_FREE_HOURS,
    AGENT_INTERVAL_PREMIUM_HOURS,
    TIER_FREE,
    TIER_PREMIUM,
)

logger = get_task_logger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_orchestrator() -> Any:
    """Instantiate AgentOrchestrator from environment-configured API keys.

    Returns:
        Configured :class:`AgentOrchestrator` instance.
    """
    from quantflow.agents.llm_clients.anthropic_agent import AnthropicAgent
    from quantflow.agents.llm_clients.openai_agent import OpenAIAgent
    from quantflow.agents.llm_clients.perplexity_agent import PerplexityAgent
    from quantflow.agents.orchestrator import AgentOrchestrator
    from quantflow.agents.scrapers.web_scraper import WebScraperAgent

    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    perplexity_key = os.environ.get("PERPLEXITY_API_KEY")

    openai_agent = OpenAIAgent(api_key=openai_key) if openai_key else None
    anthropic_agent = AnthropicAgent(api_key=anthropic_key) if anthropic_key else None
    perplexity_agent = PerplexityAgent(api_key=perplexity_key) if perplexity_key else None

    return AgentOrchestrator(
        openai_agent=openai_agent,
        anthropic_agent=anthropic_agent,
        perplexity_agent=perplexity_agent,
        web_scraper=WebScraperAgent(),
    )


def _run_async(coro: Any) -> Any:
    """Run an async coroutine synchronously inside a Celery task.

    Celery workers run in synchronous context, so we create a fresh event
    loop for each task invocation.

    Args:
        coro: Awaitable coroutine.

    Returns:
        Coroutine result.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Core intelligence task
# ---------------------------------------------------------------------------


@shared_task(
    name="quantflow.agents.run_intelligence_cycle",
    bind=True,
    max_retries=2,
    default_retry_delay=60,
    acks_late=True,
    soft_time_limit=90,
    time_limit=120,
)
def run_intelligence_cycle_task(
    self: Any,
    symbol: str,
    market_data: dict[str, Any] | None = None,
    tier: str = TIER_FREE,
) -> dict[str, Any]:
    """Run a full agent intelligence cycle for a single ticker symbol.

    Executed by Celery workers; dispatched by Beat for scheduled cycles.
    Result is stored in Celery's result backend (Redis) for consumption
    by the API layer.

    Args:
        symbol: Ticker symbol to analyse (e.g. ``"AAPL"``).
        market_data: Optional current market data snapshot for CEO validation.
        tier: Subscription tier (``"free"`` or ``"premium"``).

    Returns:
        Serialised :class:`ValidatedIntelligenceReport` as a dict.
    """
    logger.info(
        "Starting intelligence cycle task",
        extra={"symbol": symbol, "tier": tier},
    )

    try:
        orchestrator = _build_orchestrator()
        report = _run_async(
            orchestrator.run_intelligence_cycle(symbol, market_data)
        )
        result = report.model_dump(mode="json")
        logger.info(
            "Intelligence cycle task complete",
            extra={
                "symbol": symbol,
                "validated_sentiment": result.get("validated_sentiment"),
                "conflict": result.get("conflict_detected"),
            },
        )
        return result

    except Exception as exc:
        logger.error(
            "Intelligence cycle task failed",
            extra={"symbol": symbol, "error": str(exc)},
            exc_info=True,
        )
        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Batch intelligence task
# ---------------------------------------------------------------------------


@shared_task(
    name="quantflow.agents.run_batch_intelligence",
    bind=True,
    max_retries=1,
    default_retry_delay=120,
    acks_late=True,
    soft_time_limit=300,
    time_limit=360,
)
def run_batch_intelligence_task(
    self: Any,
    symbols: list[str],
    market_data: dict[str, dict[str, Any]] | None = None,
    tier: str = TIER_FREE,
) -> dict[str, dict[str, Any]]:
    """Run intelligence cycles for multiple symbols in a single task.

    Internally uses ``asyncio.gather`` for concurrent execution.

    Args:
        symbols: List of ticker symbols.
        market_data: Optional dict mapping symbol → market data snapshot.
        tier: Subscription tier.

    Returns:
        Dict mapping symbol → serialised :class:`ValidatedIntelligenceReport`.
    """
    logger.info(
        "Starting batch intelligence task",
        extra={"n_symbols": len(symbols), "tier": tier},
    )

    try:
        orchestrator = _build_orchestrator()
        reports = _run_async(
            orchestrator.run_batch(symbols, market_data)
        )
        return {sym: r.model_dump(mode="json") for sym, r in reports.items()}

    except Exception as exc:
        logger.error(
            "Batch intelligence task failed",
            extra={"symbols": symbols, "error": str(exc)},
            exc_info=True,
        )
        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Celery Beat schedule helpers
# ---------------------------------------------------------------------------


def get_beat_schedule() -> dict[str, dict[str, Any]]:
    """Return the Celery Beat periodic task schedule.

    Called from the Celery app configuration to register beat tasks.
    Symbol lists are fetched from the database in production; here we
    provide the schedule template.

    Returns:
        Dict suitable for ``app.conf.beat_schedule``.
    """
    return {
        # Premium symbols run every AGENT_INTERVAL_PREMIUM_HOURS hours
        "premium-intelligence-cycle": {
            "task": "quantflow.agents.run_batch_intelligence",
            "schedule": AGENT_INTERVAL_PREMIUM_HOURS * 3600,
            "kwargs": {"symbols": [], "tier": TIER_PREMIUM},
            "options": {"queue": "intelligence_premium"},
        },
        # Free-tier symbols run every AGENT_INTERVAL_FREE_HOURS hours
        "free-intelligence-cycle": {
            "task": "quantflow.agents.run_batch_intelligence",
            "schedule": AGENT_INTERVAL_FREE_HOURS * 3600,
            "kwargs": {"symbols": [], "tier": TIER_FREE},
            "options": {"queue": "intelligence_free"},
        },
    }
