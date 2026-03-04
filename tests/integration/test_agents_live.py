"""Integration tests for agentic intelligence — real API calls.

These tests hit live external APIs (OpenAI, Anthropic, Perplexity, web).
They are SKIPPED unless the environment variable ``LIVE_APIS=true`` is set.

Usage:
    LIVE_APIS=true pytest tests/integration/test_agents_live.py -v -s

Required env vars (when LIVE_APIS=true):
    OPENAI_API_KEY
    ANTHROPIC_API_KEY
    PERPLEXITY_API_KEY
"""

from __future__ import annotations

import os

import pytest

# ---------------------------------------------------------------------------
# Skip guard — all tests in this module are gated behind LIVE_APIS=true
# ---------------------------------------------------------------------------

_LIVE = os.environ.get("LIVE_APIS", "").lower() in {"true", "1", "yes"}

pytestmark = pytest.mark.skipif(
    not _LIVE,
    reason="Set LIVE_APIS=true to run live API integration tests",
)


# ---------------------------------------------------------------------------
# Helper to check required keys
# ---------------------------------------------------------------------------


def _require_key(env_var: str) -> str:
    """Return the env var value or skip the test if it's not set."""
    value = os.environ.get(env_var)
    if not value:
        pytest.skip(f"{env_var} not set — skipping live test")
    return value


# ---------------------------------------------------------------------------
# WebScraperAgent — no API key required (public endpoints)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_web_scraper_yahoo_rss_live() -> None:
    """Scrape Yahoo Finance RSS for a real ticker."""
    from quantflow.agents.scrapers.web_scraper import WebScraperAgent

    scraper = WebScraperAgent(timeout_s=20.0)
    docs = await scraper._scrape_yahoo_rss("AAPL")

    assert isinstance(docs, list)
    # Yahoo RSS may return 0 items for some symbols — just assert no exception
    for doc in docs:
        assert doc.symbol == "AAPL"
        assert doc.url
        assert doc.title


@pytest.mark.asyncio()
async def test_web_scraper_reddit_live() -> None:
    """Fetch Reddit posts for a ticker (public JSON API)."""
    from quantflow.agents.scrapers.web_scraper import WebScraperAgent

    scraper = WebScraperAgent(timeout_s=20.0)
    docs = await scraper.scrape_reddit("AAPL")

    assert isinstance(docs, list)
    for doc in docs:
        assert doc.symbol == "AAPL"


@pytest.mark.asyncio()
async def test_web_scraper_all_live() -> None:
    """Run full scrape_all pipeline — all sources."""
    from quantflow.agents.scrapers.web_scraper import WebScraperAgent

    scraper = WebScraperAgent(timeout_s=20.0)
    docs = await scraper.scrape_all("MSFT")

    assert isinstance(docs, list)
    # Verify deduplication — no duplicate URLs
    urls = [d.url for d in docs]
    assert len(urls) == len(set(urls))


# ---------------------------------------------------------------------------
# Perplexity live tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_perplexity_query_live() -> None:
    """Single Perplexity query with real API."""
    from quantflow.agents.llm_clients.perplexity_agent import PerplexityAgent

    api_key = _require_key("PERPLEXITY_API_KEY")
    agent = PerplexityAgent(api_key=api_key, timeout_s=30.0)

    result = await agent.query(
        "What is the current analyst consensus for Apple (AAPL) stock?",
        symbol="AAPL",
    )

    assert result.answer
    assert len(result.answer) > 20
    assert result.tokens_used > 0


@pytest.mark.asyncio()
async def test_perplexity_build_agent_output_live() -> None:
    """Full Perplexity agent output for a symbol."""
    from quantflow.agents.llm_clients.perplexity_agent import PerplexityAgent

    api_key = _require_key("PERPLEXITY_API_KEY")
    agent = PerplexityAgent(api_key=api_key, timeout_s=30.0)

    output = await agent.build_agent_output("AAPL")

    assert output.agent_name == "PerplexityAgent"
    assert output.symbol == "AAPL"
    assert -1.0 <= output.sentiment_score <= 1.0
    assert 0.0 <= output.confidence <= 1.0
    assert output.raw_sources  # Should have real citations


# ---------------------------------------------------------------------------
# OpenAI live tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_openai_extract_sentiment_live() -> None:
    """OpenAI GPT-4o sentiment extraction from real text."""
    from quantflow.agents.llm_clients.openai_agent import OpenAIAgent

    api_key = _require_key("OPENAI_API_KEY")
    agent = OpenAIAgent(api_key=api_key, timeout_s=30.0)

    result = await agent.extract_sentiment(
        "Apple reported record Q3 revenue of $120 billion, beating analyst estimates "
        "by 5%. Management raised full-year guidance citing strong iPhone and Services growth. "
        "However, China sales declined 8% amid intensifying local competition.",
        symbol="AAPL",
    )

    assert -1.0 <= result.score <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    assert result.reasoning
    assert isinstance(result.bullish_factors, list)
    assert isinstance(result.bearish_factors, list)


@pytest.mark.asyncio()
async def test_openai_classify_event_live() -> None:
    """OpenAI GPT-4o event classification."""
    from quantflow.agents.llm_clients.openai_agent import OpenAIAgent

    api_key = _require_key("OPENAI_API_KEY")
    agent = OpenAIAgent(api_key=api_key, timeout_s=30.0)

    result = await agent.classify_event(
        "Microsoft announced it will acquire a leading AI startup for $5 billion, "
        "expanding its AI capabilities in the enterprise software segment."
    )

    assert result.event_type
    assert -1.0 <= result.sentiment_impact <= 1.0
    assert 0.0 <= result.magnitude <= 1.0
    assert result.description


@pytest.mark.asyncio()
async def test_openai_build_agent_output_live() -> None:
    """End-to-end OpenAIAgent output for scraped text."""
    from quantflow.agents.llm_clients.openai_agent import OpenAIAgent

    api_key = _require_key("OPENAI_API_KEY")
    agent = OpenAIAgent(api_key=api_key, timeout_s=30.0)

    sample_text = (
        "Apple Inc. delivered another strong quarter, reporting earnings per share of $1.46, "
        "above the consensus estimate of $1.35. Revenue rose 11% year-over-year to $119.6 billion. "
        "The company's Services segment grew 14% to a record $24.2 billion. "
        "CEO Tim Cook highlighted the company's AI strategy and new product pipeline. "
        "The stock rose 4% in after-hours trading."
    )

    output = await agent.build_agent_output(
        text=sample_text,
        symbol="AAPL",
        sources=["https://example.com/earnings-report"],
    )

    assert output.agent_name == "OpenAIAgent"
    assert output.symbol == "AAPL"
    assert -1.0 <= output.sentiment_score <= 1.0
    assert 0.0 <= output.confidence <= 1.0
    assert agent.total_tokens_used > 0


# ---------------------------------------------------------------------------
# Anthropic live tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_anthropic_analyze_long_document_live() -> None:
    """AnthropicAgent deep document analysis."""
    from quantflow.agents.llm_clients.anthropic_agent import AnthropicAgent

    api_key = _require_key("ANTHROPIC_API_KEY")
    agent = AnthropicAgent(api_key=api_key, timeout_s=60.0)

    sample_text = (
        "Apple Inc. Q3 2024 Earnings Release\n\n"
        "Revenue: $119.6 billion (+11% YoY)\n"
        "EPS: $1.46 (vs $1.35 consensus)\n"
        "Services: $24.2 billion (+14% YoY)\n"
        "iPhone revenue: $69.7 billion (+3% YoY)\n"
        "China revenue: $14.7 billion (-8% YoY)\n\n"
        "Management Commentary: Tim Cook noted accelerating AI integration across "
        "the product lineup. CFO Luca Maestri raised full-year guidance citing "
        "Services momentum and the upcoming product cycle refresh. Risk factors "
        "include China regulatory headwinds and competitive pressure in emerging markets.\n"
    )

    result = await agent.analyze_long_document(
        document=sample_text,
        queries=["What drove the revenue beat?", "What are the key risks?"],
    )

    assert result.summary
    assert -1.0 <= result.sentiment_score <= 1.0
    assert isinstance(result.key_findings, list)
    assert isinstance(result.risk_factors, list)


@pytest.mark.asyncio()
async def test_anthropic_build_agent_output_live() -> None:
    """Full AnthropicAgent output pipeline."""
    from quantflow.agents.llm_clients.anthropic_agent import AnthropicAgent

    api_key = _require_key("ANTHROPIC_API_KEY")
    agent = AnthropicAgent(api_key=api_key, timeout_s=60.0)

    sample_text = (
        "Tesla reported Q2 earnings that missed analyst estimates, with EPS of $0.52 "
        "against a consensus of $0.62. Revenue of $25.2 billion came in below the "
        "$25.7 billion expected. Vehicle deliveries of 443,956 were up 5% YoY but "
        "below the 450,000 target. CEO Elon Musk highlighted FSD progress and the "
        "upcoming Robotaxi launch as key growth catalysts. Energy generation revenue "
        "surged 78% to $3.0 billion, a positive surprise. The stock fell 8% after-hours."
    )

    output = await agent.build_agent_output(
        text=sample_text,
        symbol="TSLA",
        sources=["https://example.com/tesla-q2"],
    )

    assert output.agent_name == "AnthropicAgent"
    assert output.symbol == "TSLA"
    assert -1.0 <= output.sentiment_score <= 1.0
    assert 0.0 <= output.confidence <= 1.0


# ---------------------------------------------------------------------------
# CEO Validator live tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_ceo_validator_with_anthropic_live() -> None:
    """Full CEO validation pass using real Anthropic API."""
    from datetime import datetime, timezone

    from quantflow.agents.llm_clients.anthropic_agent import AnthropicAgent
    from quantflow.agents.ceo_model import CEOValidatorModel
    from quantflow.agents.schemas import AgentOutput

    api_key = _require_key("ANTHROPIC_API_KEY")
    anthropic_agent = AnthropicAgent(api_key=api_key, timeout_s=60.0)
    ceo = CEOValidatorModel(anthropic_agent=anthropic_agent)

    now = datetime.now(tz=timezone.utc)
    agent_outputs = [
        AgentOutput(
            agent_name="PerplexityAgent",
            symbol="AAPL",
            timestamp=now,
            sentiment_score=0.45,
            confidence=0.72,
            bullish_factors=["analyst upgrades", "strong revenue"],
            bearish_factors=["macro uncertainty"],
            key_events=["Q3 earnings beat"],
        ),
        AgentOutput(
            agent_name="OpenAIAgent",
            symbol="AAPL",
            timestamp=now,
            sentiment_score=0.38,
            confidence=0.80,
            bullish_factors=["services growth", "margin expansion"],
            bearish_factors=["China headwinds"],
        ),
    ]
    market_data = {
        "price": 185.0,
        "volume": 55_000_000,
        "52w_high": 198.0,
        "52w_low": 143.0,
    }

    report = await ceo.validate(agent_outputs, market_data=market_data)

    assert report.symbol == "AAPL"
    assert -1.0 <= report.validated_sentiment <= 1.0
    assert 0.0 <= report.consensus_confidence <= 1.0
    assert report.key_narrative


# ---------------------------------------------------------------------------
# Full orchestrator integration test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_full_orchestrator_cycle_live() -> None:
    """End-to-end intelligence cycle: scraping + all LLM agents + CEO validation.

    This is the most expensive test — runs all agents with real API calls.
    Symbol: AAPL.
    """
    from quantflow.agents.llm_clients.anthropic_agent import AnthropicAgent
    from quantflow.agents.llm_clients.openai_agent import OpenAIAgent
    from quantflow.agents.llm_clients.perplexity_agent import PerplexityAgent
    from quantflow.agents.orchestrator import AgentOrchestrator
    from quantflow.agents.scrapers.web_scraper import WebScraperAgent
    from quantflow.agents.schemas import ValidatedIntelligenceReport

    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    perplexity_key = os.environ.get("PERPLEXITY_API_KEY")

    if not any([openai_key, anthropic_key, perplexity_key]):
        pytest.skip("No LLM API keys configured — skipping full orchestrator test")

    orchestrator = AgentOrchestrator(
        openai_agent=OpenAIAgent(api_key=openai_key) if openai_key else None,
        anthropic_agent=AnthropicAgent(api_key=anthropic_key) if anthropic_key else None,
        perplexity_agent=PerplexityAgent(api_key=perplexity_key) if perplexity_key else None,
        web_scraper=WebScraperAgent(timeout_s=15.0),
        cycle_timeout_s=90.0,
    )

    report = await orchestrator.run_intelligence_cycle(
        symbol="AAPL",
        market_data={"price": 185.0, "volume": 55_000_000},
    )

    assert isinstance(report, ValidatedIntelligenceReport)
    assert report.symbol == "AAPL"
    assert -1.0 <= report.validated_sentiment <= 1.0
    assert 0.0 <= report.consensus_confidence <= 1.0
    assert isinstance(report.conflict_detected, bool)
    assert isinstance(report.agent_outputs, list)
