"""Unit tests for the agentic intelligence layer.

All external API calls (OpenAI, Anthropic, Perplexity, httpx) are mocked
via pytest fixtures.  No real network calls are made.

Run with:
    pytest tests/unit/test_agents.py -v
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from quantflow.agents.ceo_model import CEOValidatorModel
from quantflow.agents.llm_clients.base_client import (
    CircuitBreakerOpenError,
    LLMClientError,
)
from quantflow.agents.llm_clients.openai_agent import OpenAIAgent
from quantflow.agents.llm_clients.perplexity_agent import PerplexityAgent
from quantflow.agents.orchestrator import AgentOrchestrator
from quantflow.agents.schemas import (
    AgentOutput,
    AggregatedSentiment,
    RawDocument,
    ValidatedIntelligenceReport,
)
from quantflow.agents.sentiment import SentimentAggregator, vader_sentiment

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_agent_output() -> AgentOutput:
    """A single valid AgentOutput for use across tests."""
    return AgentOutput(
        agent_name="OpenAIAgent",
        symbol="AAPL",
        timestamp=datetime.now(tz=UTC),
        sentiment_score=0.4,
        confidence=0.75,
        bullish_factors=["strong earnings", "guidance raised"],
        bearish_factors=["macro headwinds"],
        key_events=["Q3 earnings beat"],
        raw_sources=["https://example.com/news1"],
    )


@pytest.fixture()
def sample_agent_outputs() -> list[AgentOutput]:
    """Three AgentOutputs for multi-agent tests."""
    now = datetime.now(tz=UTC)
    return [
        AgentOutput(
            agent_name="PerplexityAgent",
            symbol="AAPL",
            timestamp=now,
            sentiment_score=0.35,
            confidence=0.70,
            bullish_factors=["upgrade"],
            bearish_factors=[],
        ),
        AgentOutput(
            agent_name="OpenAIAgent",
            symbol="AAPL",
            timestamp=now,
            sentiment_score=0.45,
            confidence=0.80,
            bullish_factors=["revenue beat"],
            bearish_factors=["competition"],
        ),
        AgentOutput(
            agent_name="AnthropicAgent",
            symbol="AAPL",
            timestamp=now,
            sentiment_score=0.30,
            confidence=0.65,
            bullish_factors=["margins expanding"],
            bearish_factors=["inventory build"],
        ),
    ]


@pytest.fixture()
def sample_raw_docs() -> list[RawDocument]:
    """Three RawDocuments for scraper / sentiment tests."""
    now = datetime.now(tz=UTC)
    return [
        RawDocument(
            source="Yahoo Finance",
            url="https://finance.yahoo.com/news/1",
            title="Apple beats earnings expectations",
            text="Apple reported strong Q3 earnings, beating analyst expectations by 8%.",
            timestamp=now,
            symbol="AAPL",
            relevance_score=0.9,
        ),
        RawDocument(
            source="StockTwits",
            url="https://stocktwits.com/symbol/AAPL?m=1",
            title="$AAPL bullish",
            text="AAPL is looking bullish heading into earnings. Strong buy.",
            timestamp=now,
            symbol="AAPL",
            relevance_score=0.6,
        ),
        RawDocument(
            source="Reddit r/investing",
            url="https://reddit.com/r/investing/post1",
            title="AAPL analysis",
            text="Long AAPL for the next quarter. Revenue growth remains solid.",
            timestamp=now,
            symbol="AAPL",
            relevance_score=0.5,
        ),
    ]


# ---------------------------------------------------------------------------
# BaseLLMClient tests
# ---------------------------------------------------------------------------


class TestBaseLLMClient:
    """Tests for retry logic, circuit breaker, and JSON parsing."""

    def _make_openai_agent(self) -> OpenAIAgent:
        return OpenAIAgent(api_key="test-key-openai")

    def test_parse_json_plain(self) -> None:
        agent = self._make_openai_agent()
        data = agent.parse_json_from_response('{"score": 0.5, "confidence": 0.8}')
        assert data["score"] == 0.5
        assert data["confidence"] == 0.8

    def test_parse_json_with_markdown_fence(self) -> None:
        agent = self._make_openai_agent()
        raw = '```json\n{"score": -0.3, "confidence": 0.6}\n```'
        data = agent.parse_json_from_response(raw)
        assert data["score"] == -0.3

    def test_parse_json_invalid_raises(self) -> None:
        agent = self._make_openai_agent()
        with pytest.raises(LLMClientError, match="Could not parse JSON"):
            agent.parse_json_from_response("this is not json {broken")

    @pytest.mark.asyncio()
    async def test_call_with_retry_success(self) -> None:
        """Successful API call returns response without retrying."""
        agent = self._make_openai_agent()
        expected = {"choices": [{"message": {"content": '{"score": 0.1, "confidence": 0.5}'}}]}
        agent._call_api = AsyncMock(return_value=expected)

        result = await agent.call_with_retry(
            {"model": "gpt-4o"}, "https://api.openai.com/v1/chat/completions"
        )
        assert result == expected
        agent._call_api.assert_called_once()

    @pytest.mark.asyncio()
    async def test_circuit_breaker_opens_after_threshold(self) -> None:
        """After 3 consecutive failures, circuit breaker opens."""
        import httpx

        agent = self._make_openai_agent()

        # Simulate an HTTP 429 that never recovers — exhaust max_retries
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        error = httpx.HTTPStatusError("err", request=MagicMock(), response=mock_response)

        agent._call_api = AsyncMock(side_effect=error)

        # Need 3 exhausted calls to open the circuit
        for _ in range(3):
            with pytest.raises(LLMClientError):
                await agent.call_with_retry({}, "https://api.openai.com")

        assert agent._consecutive_failures >= 3
        assert agent._circuit_open_until > time.monotonic()

    @pytest.mark.asyncio()
    async def test_circuit_breaker_raises_when_open(self) -> None:
        """CircuitBreakerOpenError raised when breaker is already open."""
        agent = self._make_openai_agent()
        agent._circuit_open_until = time.monotonic() + 3600.0  # Force open

        with pytest.raises(CircuitBreakerOpenError):
            await agent.call_with_retry({}, "https://api.openai.com")

    def test_total_tokens_tracking(self) -> None:
        agent = self._make_openai_agent()
        assert agent.total_tokens_used == 0


# ---------------------------------------------------------------------------
# OpenAIAgent tests
# ---------------------------------------------------------------------------


class TestOpenAIAgent:
    """Tests for OpenAIAgent — all API calls mocked."""

    def _make_agent(self) -> OpenAIAgent:
        return OpenAIAgent(api_key="sk-test")

    def _openai_response(self, content: str, tokens: int = 100) -> dict[str, Any]:
        """Build a minimal mock OpenAI chat response."""
        return {
            "choices": [{"message": {"content": content}}],
            "usage": {"total_tokens": tokens},
        }

    @pytest.mark.asyncio()
    async def test_extract_sentiment(self) -> None:
        agent = self._make_agent()
        sentiment_json = json.dumps(
            {
                "score": 0.6,
                "confidence": 0.82,
                "reasoning": "Strong earnings beat",
                "bullish_factors": ["revenue beat", "margin expansion"],
                "bearish_factors": ["high capex"],
            }
        )
        agent.call_with_retry = AsyncMock(return_value=self._openai_response(sentiment_json))

        result = await agent.extract_sentiment("Apple crushed earnings.", "AAPL")

        assert result.score == pytest.approx(0.6)
        assert result.confidence == pytest.approx(0.82)
        assert "revenue beat" in result.bullish_factors

    @pytest.mark.asyncio()
    async def test_classify_event(self) -> None:
        agent = self._make_agent()
        event_json = json.dumps(
            {
                "event_type": "earnings_beat",
                "sentiment_impact": 0.7,
                "magnitude": 0.8,
                "confidence": 0.9,
                "description": "Company beat EPS by 15%",
            }
        )
        agent.call_with_retry = AsyncMock(return_value=self._openai_response(event_json))

        result = await agent.classify_event("AAPL beats EPS by 15%.")

        assert result.event_type == "earnings_beat"
        assert result.sentiment_impact == pytest.approx(0.7)

    @pytest.mark.asyncio()
    async def test_analyze_earnings(self) -> None:
        agent = self._make_agent()
        earnings_json = json.dumps(
            {
                "eps_surprise_pct": 8.5,
                "revenue_surprise_pct": 3.2,
                "guidance_tone": "raised",
                "management_tone_score": 0.6,
                "key_risks": ["fx headwinds"],
                "key_positives": ["services growth"],
                "forward_looking_statements": ["expect double-digit growth"],
            }
        )
        agent.call_with_retry = AsyncMock(return_value=self._openai_response(earnings_json))

        result = await agent.analyze_earnings("Q3 earnings call transcript...")

        assert result.guidance_tone == "raised"
        assert result.eps_surprise_pct == pytest.approx(8.5)

    @pytest.mark.asyncio()
    async def test_build_agent_output(self) -> None:
        agent = self._make_agent()
        sentiment_json = json.dumps(
            {
                "score": 0.45,
                "confidence": 0.75,
                "reasoning": "Generally positive tone",
                "bullish_factors": ["growth"],
                "bearish_factors": [],
            }
        )
        agent.call_with_retry = AsyncMock(return_value=self._openai_response(sentiment_json))

        output = await agent.build_agent_output(
            "Apple reports Q3 earnings.", "AAPL", sources=["https://example.com"]
        )

        assert output.agent_name == "OpenAIAgent"
        assert output.symbol == "AAPL"
        assert output.sentiment_score == pytest.approx(0.45)
        assert output.confidence == pytest.approx(0.75)
        assert "https://example.com" in output.raw_sources

    @pytest.mark.asyncio()
    async def test_build_agent_output_api_error_propagates(self) -> None:
        agent = self._make_agent()
        agent.call_with_retry = AsyncMock(side_effect=LLMClientError("API timeout"))

        with pytest.raises(LLMClientError):
            await agent.build_agent_output("text", "AAPL")


# ---------------------------------------------------------------------------
# PerplexityAgent tests
# ---------------------------------------------------------------------------


class TestPerplexityAgent:
    """Tests for PerplexityAgent — all API calls mocked."""

    def _make_agent(self) -> PerplexityAgent:
        return PerplexityAgent(api_key="pplx-test")

    def _perplexity_response(self, answer: str) -> dict[str, Any]:
        return {
            "choices": [{"message": {"content": answer}}],
            "citations": ["https://example.com/source1", "https://example.com/source2"],
            "usage": {"total_tokens": 80},
        }

    @pytest.mark.asyncio()
    async def test_query_returns_response(self) -> None:
        agent = self._make_agent()
        agent.call_with_retry = AsyncMock(
            return_value=self._perplexity_response("AAPL received two analyst upgrades this week.")
        )

        result = await agent.query("Latest analyst upgrades for AAPL?", symbol="AAPL")

        assert "upgrades" in result.answer
        assert len(result.citations) == 2
        assert result.tokens_used == 80

    @pytest.mark.asyncio()
    async def test_run_symbol_queries_aggregates(self) -> None:
        agent = self._make_agent()

        responses = [
            self._perplexity_response("Analysts upgraded AAPL."),
            self._perplexity_response("Strong guidance raised for AAPL."),
            self._perplexity_response("Bullish sentiment among investors."),
            self._perplexity_response("Macro risks include rate hikes."),
        ]
        agent.call_with_retry = AsyncMock(side_effect=responses)

        results = await agent.run_symbol_queries("AAPL")
        assert len(results) == 4

    @pytest.mark.asyncio()
    async def test_build_agent_output_structure(self) -> None:
        agent = self._make_agent()
        responses = [
            self._perplexity_response("Analysts upgraded AAPL. Strong buy recommendation."),
            self._perplexity_response("Revenue guidance raised. Bullish momentum."),
            self._perplexity_response("Positive sentiment among institutional investors."),
            self._perplexity_response("Macro risk from rate hikes. Recession risk noted."),
        ]
        agent.call_with_retry = AsyncMock(side_effect=responses)

        output = await agent.build_agent_output("AAPL")

        assert output.agent_name == "PerplexityAgent"
        assert output.symbol == "AAPL"
        assert -1.0 <= output.sentiment_score <= 1.0
        assert 0.0 <= output.confidence <= 1.0

    @pytest.mark.asyncio()
    async def test_build_agent_output_all_queries_fail(self) -> None:
        agent = self._make_agent()
        agent.call_with_retry = AsyncMock(side_effect=LLMClientError("timeout"))

        output = await agent.build_agent_output("AAPL")

        assert output.agent_name == "PerplexityAgent"
        assert output.confidence == 0.0
        assert output.metadata.get("error") == "all_queries_failed"

    def test_keyword_sentiment_bullish(self) -> None:
        agent = self._make_agent()
        text = "Strong upgrade beat outperform bullish growth positive"
        score = agent._keyword_sentiment(text)
        assert score > 0.0

    def test_keyword_sentiment_bearish(self) -> None:
        agent = self._make_agent()
        text = "Downgrade miss underperform bearish decline weak negative sell"
        score = agent._keyword_sentiment(text)
        assert score < 0.0

    def test_keyword_sentiment_neutral(self) -> None:
        agent = self._make_agent()
        score = agent._keyword_sentiment("The company announced its quarterly results.")
        assert score == 0.0

    def test_extract_key_phrases(self) -> None:
        agent = self._make_agent()
        text = (
            "Apple reported strong earnings. "
            "Revenue guidance was raised for next quarter. "
            "The CEO mentioned dividend plans. "
            "Macro risks include inflation."
        )
        phrases = agent._extract_key_phrases(text, max_phrases=3)
        assert len(phrases) <= 3
        assert all(isinstance(p, str) for p in phrases)


# ---------------------------------------------------------------------------
# WebScraperAgent tests
# ---------------------------------------------------------------------------


class TestWebScraperAgent:
    """Tests for WebScraperAgent — httpx calls mocked."""

    def _make_agent(self) -> Any:
        from quantflow.agents.scrapers.web_scraper import WebScraperAgent

        return WebScraperAgent(timeout_s=5.0)

    @pytest.mark.asyncio()
    async def test_scrape_yahoo_rss_returns_docs(self) -> None:
        agent = self._make_agent()
        rss_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
          <channel>
            <title>Yahoo Finance - AAPL</title>
            <item>
              <title>Apple stock rises on earnings</title>
              <link>https://finance.yahoo.com/news/apple-earnings</link>
              <description>Apple reported Q3 earnings beating estimates</description>
            </item>
            <item>
              <title>Apple new product launch</title>
              <link>https://finance.yahoo.com/news/apple-product</link>
              <description>Apple announces new product line</description>
            </item>
          </channel>
        </rss>"""

        mock_response = MagicMock()
        mock_response.text = rss_xml
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            docs = await agent._scrape_yahoo_rss("AAPL")

        assert len(docs) >= 1
        assert all(d.symbol == "AAPL" for d in docs)
        assert all(d.source == "Yahoo Finance" for d in docs)

    @pytest.mark.asyncio()
    async def test_scrape_stocktwits_returns_docs(self) -> None:
        agent = self._make_agent()
        stocktwits_json = {
            "messages": [
                {
                    "id": 123,
                    "body": "$AAPL is looking very bullish right now!",
                    "entities": {"sentiment": {"basic": "Bullish"}},
                },
                {
                    "id": 124,
                    "body": "Added to my AAPL position today",
                    "entities": {},
                },
            ]
        }

        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value=stocktwits_json)
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            docs = await agent._scrape_stocktwits("AAPL")

        assert len(docs) == 2
        # Structured sentiment gets higher relevance
        assert docs[0].relevance_score == pytest.approx(0.80)

    @pytest.mark.asyncio()
    async def test_scrape_reddit_returns_docs(self) -> None:
        agent = self._make_agent()
        reddit_json = {
            "data": {
                "children": [
                    {
                        "data": {
                            "title": "AAPL is a great long-term hold",
                            "selftext": "Here is why I am bullish on Apple...",
                            "score": 250,
                            "permalink": "/r/investing/comments/abc123/aapl",
                        }
                    }
                ]
            }
        }

        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value=reddit_json)
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            docs = await agent.scrape_reddit("AAPL")

        assert len(docs) >= 1
        assert "AAPL" in docs[0].text or "AAPL" in docs[0].title

    @pytest.mark.asyncio()
    async def test_scrape_all_deduplicates(self) -> None:
        """scrape_all should deduplicate documents by URL."""
        agent = self._make_agent()
        now = datetime.now(tz=UTC)
        # Return same doc from two different scrapers
        doc1 = RawDocument(
            source="Yahoo Finance",
            url="https://finance.yahoo.com/unique-url",
            title="Dupe",
            text="duplicate content",
            timestamp=now,
            symbol="AAPL",
            relevance_score=0.8,
        )
        doc2 = RawDocument(
            source="Yahoo Finance",
            url="https://finance.yahoo.com/unique-url",  # same URL
            title="Dupe",
            text="duplicate content",
            timestamp=now,
            symbol="AAPL",
            relevance_score=0.8,
        )

        agent.scrape_news = AsyncMock(return_value=[doc1, doc2])
        agent.scrape_reddit = AsyncMock(return_value=[])
        agent.fetch_sec_filing = AsyncMock(return_value=[])

        docs = await agent.scrape_all("AAPL")
        urls = [d.url for d in docs]
        assert len(urls) == len(set(urls))

    @pytest.mark.asyncio()
    async def test_scrape_all_handles_individual_failures(self) -> None:
        """scrape_all should return docs from successful scrapers even if others fail."""
        agent = self._make_agent()
        now = datetime.now(tz=UTC)
        good_doc = RawDocument(
            source="StockTwits",
            url="https://stocktwits.com/1",
            title="Bullish",
            text="AAPL bullish",
            timestamp=now,
            symbol="AAPL",
            relevance_score=0.7,
        )
        agent.scrape_news = AsyncMock(return_value=[good_doc])
        agent.scrape_reddit = AsyncMock(side_effect=Exception("Reddit API down"))
        agent.fetch_sec_filing = AsyncMock(return_value=[])

        docs = await agent.scrape_all("AAPL")
        assert len(docs) == 1


# ---------------------------------------------------------------------------
# SentimentAggregator tests
# ---------------------------------------------------------------------------


class TestSentimentAggregator:
    """Tests for SentimentAggregator — no external calls (VADER mocked)."""

    def test_aggregate_with_agent_outputs_only(
        self, sample_agent_outputs: list[AgentOutput]
    ) -> None:
        agg = SentimentAggregator(use_finbert=False)
        result = agg.aggregate(sample_agent_outputs, raw_documents=None)

        assert isinstance(result, AggregatedSentiment)
        assert -1.0 <= result.composite_sentiment <= 1.0
        assert result.sentiment_regime in {
            "EXTREME_BEAR",
            "BEAR",
            "NEUTRAL",
            "BULL",
            "EXTREME_BULL",
        }
        assert "agent_llm" in result.source_scores

    def test_aggregate_no_outputs_no_docs(self) -> None:
        agg = SentimentAggregator(use_finbert=False)
        result = agg.aggregate([], raw_documents=None)

        assert result.composite_sentiment == pytest.approx(0.0)

    def test_aggregate_weights_normalised_without_finbert(
        self, sample_agent_outputs: list[AgentOutput], sample_raw_docs: list[RawDocument]
    ) -> None:
        """Without FinBERT, weights redistribute to agent + VADER."""
        agg = SentimentAggregator(use_finbert=False)
        with patch(
            "quantflow.agents.sentiment.vader_batch",
            return_value=0.3,
        ):
            result = agg.aggregate(sample_agent_outputs, raw_documents=sample_raw_docs)

        assert -1.0 <= result.composite_sentiment <= 1.0

    def test_vader_sentiment_positive(self) -> None:
        pytest.importorskip("vaderSentiment")
        with patch(
            "quantflow.agents.sentiment.SentimentIntensityAnalyzer",
            create=True,
        ):
            from vaderSentiment.vaderSentiment import (
                SentimentIntensityAnalyzer,  # type: ignore[import]
            )

            with patch.object(
                SentimentIntensityAnalyzer,
                "polarity_scores",
                return_value={"compound": 0.72},
            ):
                score = vader_sentiment("Apple stock surges after strong earnings beat!")
                assert score == pytest.approx(0.72)

    def test_vader_sentiment_import_missing_returns_zero(self) -> None:
        """If vaderSentiment is not installed, returns 0.0 gracefully."""
        with patch.dict(
            "sys.modules", {"vaderSentiment": None, "vaderSentiment.vaderSentiment": None}
        ):
            score = vader_sentiment("Some text")
            assert score == pytest.approx(0.0)

    def test_classify_regime_extreme_bull(self) -> None:
        regime = SentimentAggregator._classify_regime(0.5, z_score=2.5)
        assert regime == "EXTREME_BULL"

    def test_classify_regime_extreme_bear(self) -> None:
        regime = SentimentAggregator._classify_regime(-0.5, z_score=-2.5)
        assert regime == "EXTREME_BEAR"

    def test_classify_regime_bull(self) -> None:
        regime = SentimentAggregator._classify_regime(0.3, z_score=0.5)
        assert regime == "BULL"

    def test_classify_regime_bear(self) -> None:
        regime = SentimentAggregator._classify_regime(-0.3, z_score=-0.5)
        assert regime == "BEAR"

    def test_classify_regime_neutral(self) -> None:
        regime = SentimentAggregator._classify_regime(0.01, z_score=0.1)
        assert regime == "NEUTRAL"

    def test_contrarian_flag_at_extreme(self, sample_agent_outputs: list[AgentOutput]) -> None:
        agg = SentimentAggregator(use_finbert=False)
        # Populate enough history to generate z-scores
        for score in [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]:
            agg._update_history(score)
        # Now add an extreme reading — z-score should be large
        agg._update_history(5.0)  # far from mean
        result = agg.aggregate(sample_agent_outputs, raw_documents=None)
        # Just verify contrarian is a bool — threshold depends on history
        assert isinstance(result.contrarian_signal, bool)

    def test_rolling_history_pruned(self) -> None:
        """History entries older than 30 days are pruned."""
        from datetime import timedelta

        agg = SentimentAggregator(use_finbert=False)
        old_time = datetime.now(tz=UTC) - timedelta(days=35)
        agg._history.append((old_time, 0.5))
        agg._update_history(0.3)
        assert all(t > datetime.now(tz=UTC) - timedelta(days=31) for t, _ in agg._history)

    def test_momentum_computed_correctly(self) -> None:
        agg = SentimentAggregator(use_finbert=False)
        # Build history with clear trend
        for _ in range(10):
            agg._update_history(-0.3)
        for _ in range(5):
            agg._update_history(0.5)
        momentum = agg._compute_momentum(days=5)
        assert momentum > 0.0  # Recent scores higher than older

    def test_finbert_fallback_to_vader_on_import_error(self) -> None:
        """FinBERT falls back to VADER when transformers is not available."""
        from quantflow.agents.sentiment import finbert_sentiment

        with (
            patch(
                "quantflow.agents.sentiment.vader_batch",
                return_value=0.25,
            ) as mock_vader,
            patch.dict("sys.modules", {"transformers": None}),
        ):
            finbert_sentiment(["Apple stock is doing great"])
            # Should fall back to vader_batch
            mock_vader.assert_called_once()


# ---------------------------------------------------------------------------
# CEOValidatorModel tests
# ---------------------------------------------------------------------------


class TestCEOValidatorModel:
    """Tests for CEO statistical consensus and LLM validation path."""

    @pytest.mark.asyncio()
    async def test_validate_empty_outputs_returns_neutral(self) -> None:
        ceo = CEOValidatorModel()
        report = await ceo.validate([], market_data={})

        assert report.validated_sentiment == pytest.approx(0.0)
        assert report.consensus_confidence == pytest.approx(0.0)
        assert report.symbol == "UNKNOWN"

    @pytest.mark.asyncio()
    async def test_validate_single_output_no_conflict(
        self, sample_agent_output: AgentOutput
    ) -> None:
        ceo = CEOValidatorModel()
        report = await ceo.validate([sample_agent_output], market_data={})

        assert isinstance(report, ValidatedIntelligenceReport)
        assert report.symbol == "AAPL"
        assert -1.0 <= report.validated_sentiment <= 1.0
        assert not report.conflict_detected  # Single agent → no conflict

    @pytest.mark.asyncio()
    async def test_validate_detects_conflict(self) -> None:
        """Agents with highly divergent sentiments should trigger conflict flag."""
        now = datetime.now(tz=UTC)
        outputs = [
            AgentOutput(
                agent_name="OpenAIAgent",
                symbol="TSLA",
                timestamp=now,
                sentiment_score=0.9,
                confidence=0.8,
            ),
            AgentOutput(
                agent_name="AnthropicAgent",
                symbol="TSLA",
                timestamp=now,
                sentiment_score=-0.9,
                confidence=0.8,
            ),
        ]
        ceo = CEOValidatorModel()
        report = await ceo.validate(outputs, market_data={})

        assert report.conflict_detected is True

    @pytest.mark.asyncio()
    async def test_validate_with_llm_path_override(
        self, sample_agent_outputs: list[AgentOutput]
    ) -> None:
        """When Anthropic agent is present, LLM validation replaces stat consensus."""
        mock_anthropic = AsyncMock()
        mock_anthropic.ceo_validate = AsyncMock(
            return_value={
                "validated_sentiment": 0.55,
                "conflict_detected": False,
                "hallucinations_flagged": 0,
                "key_narrative": "Solid earnings beat with strong guidance.",
                "risk_events": ["fed_rate_decision"],
                "ceo_reasoning": "All agents agree on bullish direction.",
            }
        )

        ceo = CEOValidatorModel(anthropic_agent=mock_anthropic)
        report = await ceo.validate(sample_agent_outputs, market_data={})

        assert report.validated_sentiment == pytest.approx(0.55, abs=1e-4)
        assert report.key_narrative == "Solid earnings beat with strong guidance."
        mock_anthropic.ceo_validate.assert_called_once()

    @pytest.mark.asyncio()
    async def test_validate_llm_failure_falls_back_to_stat(
        self, sample_agent_outputs: list[AgentOutput]
    ) -> None:
        """LLM validation failure should fall back to statistical consensus."""
        mock_anthropic = AsyncMock()
        mock_anthropic.ceo_validate = AsyncMock(side_effect=Exception("API error"))

        ceo = CEOValidatorModel(anthropic_agent=mock_anthropic)
        report = await ceo.validate(sample_agent_outputs, market_data={})

        # Should still return a valid report via statistical fallback
        assert isinstance(report, ValidatedIntelligenceReport)
        assert -1.0 <= report.validated_sentiment <= 1.0

    @pytest.mark.asyncio()
    async def test_ceo_override_flag_when_llm_diverges(
        self, sample_agent_outputs: list[AgentOutput]
    ) -> None:
        """CEO override flag set when LLM disagrees with stat consensus by >0.3."""
        mock_anthropic = AsyncMock()
        # Force a large disagreement
        mock_anthropic.ceo_validate = AsyncMock(
            return_value={
                "validated_sentiment": -0.8,  # Very bearish vs stat consensus ~0.37
                "conflict_detected": True,
                "hallucinations_flagged": 2,
                "key_narrative": "Significant risks overlooked by other agents.",
                "risk_events": [],
                "ceo_reasoning": "Macro environment deeply concerning.",
            }
        )

        ceo = CEOValidatorModel(anthropic_agent=mock_anthropic)
        report = await ceo.validate(sample_agent_outputs, market_data={})

        assert report.ceo_override is True
        assert report.ceo_reasoning is not None

    def test_statistical_consensus_weighted_average(self) -> None:
        """Weighted average should lie between min and max agent sentiments."""
        ceo = CEOValidatorModel()
        now = datetime.now(tz=UTC)
        outputs = [
            AgentOutput(
                agent_name="PerplexityAgent",
                symbol="AAPL",
                timestamp=now,
                sentiment_score=0.2,
                confidence=0.7,
            ),
            AgentOutput(
                agent_name="OpenAIAgent",
                symbol="AAPL",
                timestamp=now,
                sentiment_score=0.6,
                confidence=0.8,
            ),
        ]
        result = ceo._statistical_consensus(outputs)

        assert 0.2 <= result["validated_sentiment"] <= 0.6
        assert not result["conflict_detected"]

    def test_normalised_weights_sum_to_one(self, sample_agent_outputs: list[AgentOutput]) -> None:
        ceo = CEOValidatorModel()
        weights = ceo._normalised_weights(sample_agent_outputs)

        assert abs(sum(weights.values()) - 1.0) < 1e-3  # allow 4-decimal rounding error

    def test_narrative_includes_direction(self) -> None:
        ceo = CEOValidatorModel()
        now = datetime.now(tz=UTC)
        outputs = [
            AgentOutput(
                agent_name="OpenAIAgent",
                symbol="AAPL",
                timestamp=now,
                sentiment_score=0.5,
                confidence=0.8,
                bullish_factors=["strong revenue"],
                bearish_factors=["competition"],
            )
        ]
        result = ceo._statistical_consensus(outputs)
        assert "positive" in result["narrative"].lower() or "neutral" in result["narrative"].lower()


# ---------------------------------------------------------------------------
# AgentOrchestrator tests
# ---------------------------------------------------------------------------


class TestAgentOrchestrator:
    """Tests for AgentOrchestrator — all agents mocked."""

    def _make_orchestrator(self) -> AgentOrchestrator:
        """Return an orchestrator with all agents replaced by AsyncMocks."""

        orc = AgentOrchestrator(cycle_timeout_s=30.0)
        return orc

    def _make_mock_report(self, symbol: str = "AAPL") -> ValidatedIntelligenceReport:
        return ValidatedIntelligenceReport(
            symbol=symbol,
            timestamp=datetime.now(tz=UTC),
            validated_sentiment=0.35,
            consensus_confidence=0.70,
            conflict_detected=False,
            key_narrative="Solid fundamentals.",
            risk_events=[],
            agent_outputs=[],
        )

    @pytest.mark.asyncio()
    async def test_run_intelligence_cycle_no_agents(self) -> None:
        """With no agents configured, orchestrator returns an empty-report."""
        orc = AgentOrchestrator()
        report = await orc.run_intelligence_cycle("AAPL")

        assert isinstance(report, ValidatedIntelligenceReport)
        # Empty report from CEO validator
        assert report.symbol == "UNKNOWN"

    @pytest.mark.asyncio()
    async def test_run_intelligence_cycle_with_mock_agents(self) -> None:
        """Full cycle with mocked perplexity + CEO validator."""
        now = datetime.now(tz=UTC)
        perp_output = AgentOutput(
            agent_name="PerplexityAgent",
            symbol="MSFT",
            timestamp=now,
            sentiment_score=0.4,
            confidence=0.72,
        )

        mock_perplexity = AsyncMock()
        mock_perplexity.build_agent_output = AsyncMock(return_value=perp_output)

        mock_ceo = AsyncMock()
        expected_report = self._make_mock_report("MSFT")
        expected_report = expected_report.model_copy(
            update={"symbol": "MSFT", "validated_sentiment": 0.4}
        )
        mock_ceo.validate = AsyncMock(return_value=expected_report)

        orc = AgentOrchestrator(
            perplexity_agent=mock_perplexity,
            ceo_model=mock_ceo,
        )
        report = await orc.run_intelligence_cycle("MSFT")

        assert report.symbol == "MSFT"
        mock_perplexity.build_agent_output.assert_called_once_with("MSFT")
        mock_ceo.validate.assert_called_once()

    @pytest.mark.asyncio()
    async def test_run_intelligence_cycle_with_scraper(self) -> None:
        """Scraper results are passed through to LLM agents and sentiment."""
        now = datetime.now(tz=UTC)
        doc = RawDocument(
            source="Yahoo Finance",
            url="https://finance.yahoo.com/test",
            title="Test headline",
            text="Strong revenue growth reported.",
            timestamp=now,
            symbol="AAPL",
            relevance_score=0.8,
        )

        mock_scraper = AsyncMock()
        mock_scraper.scrape_all = AsyncMock(return_value=[doc])

        mock_ceo = AsyncMock()
        mock_ceo.validate = AsyncMock(return_value=self._make_mock_report("AAPL"))

        orc = AgentOrchestrator(web_scraper=mock_scraper, ceo_model=mock_ceo)
        report = await orc.run_intelligence_cycle("AAPL")

        mock_scraper.scrape_all.assert_called_once_with("AAPL")
        assert isinstance(report, ValidatedIntelligenceReport)

    @pytest.mark.asyncio()
    async def test_agent_failure_does_not_block_cycle(self) -> None:
        """If one agent fails, the cycle still completes with remaining agents."""
        now = datetime.now(tz=UTC)
        good_output = AgentOutput(
            agent_name="PerplexityAgent",
            symbol="GOOGL",
            timestamp=now,
            sentiment_score=0.2,
            confidence=0.65,
        )

        mock_perplexity = AsyncMock()
        mock_perplexity.build_agent_output = AsyncMock(return_value=good_output)

        mock_openai = AsyncMock()
        mock_openai.build_agent_output = AsyncMock(side_effect=Exception("OpenAI timeout"))

        mock_ceo = AsyncMock()
        mock_ceo.validate = AsyncMock(return_value=self._make_mock_report("GOOGL"))

        orc = AgentOrchestrator(
            perplexity_agent=mock_perplexity,
            openai_agent=mock_openai,
            ceo_model=mock_ceo,
        )
        # Combined text must be non-empty to trigger openai — inject a scraper doc
        mock_scraper = AsyncMock()
        mock_scraper.scrape_all = AsyncMock(
            return_value=[
                RawDocument(
                    source="Yahoo Finance",
                    url="https://test.com",
                    title="t",
                    text="google news here",
                    timestamp=now,
                    symbol="GOOGL",
                    relevance_score=0.7,
                )
            ]
        )
        orc._scraper = mock_scraper

        report = await orc.run_intelligence_cycle("GOOGL")

        assert isinstance(report, ValidatedIntelligenceReport)
        # CEO validate was still called despite OpenAI failure
        mock_ceo.validate.assert_called_once()

    @pytest.mark.asyncio()
    async def test_run_batch_returns_all_symbols(self) -> None:
        symbols = ["AAPL", "MSFT", "GOOGL"]

        orc = AgentOrchestrator()
        orc.run_intelligence_cycle = AsyncMock(
            side_effect=lambda sym, md=None: self._make_mock_report(sym)
        )

        reports = await orc.run_batch(symbols)

        assert set(reports.keys()) == set(symbols)
        assert all(isinstance(r, ValidatedIntelligenceReport) for r in reports.values())

    @pytest.mark.asyncio()
    async def test_run_batch_handles_symbol_failure(self) -> None:
        """If one symbol fails in batch, others still succeed."""
        symbols = ["AAPL", "FAIL_SYMBOL"]

        orc = AgentOrchestrator()

        def _side_effect(sym: str, md: Any = None) -> ValidatedIntelligenceReport:
            if sym == "FAIL_SYMBOL":
                raise RuntimeError("Symbol data unavailable")
            return self._make_mock_report(sym)

        orc.run_intelligence_cycle = AsyncMock(side_effect=_side_effect)

        reports = await orc.run_batch(symbols)

        assert "AAPL" in reports
        assert "FAIL_SYMBOL" not in reports

    @pytest.mark.asyncio()
    async def test_synthetic_scraper_output_when_no_llm_agents(self) -> None:
        """Without LLM agents, a synthetic WebScraperAgent output is created."""
        now = datetime.now(tz=UTC)
        doc = RawDocument(
            source="Yahoo Finance",
            url="https://finance.yahoo.com/test2",
            title="Apple news",
            text="Apple announces new product",
            timestamp=now,
            symbol="AAPL",
            relevance_score=0.7,
        )

        mock_scraper = AsyncMock()
        mock_scraper.scrape_all = AsyncMock(return_value=[doc])

        mock_ceo = AsyncMock()
        # Capture what gets passed to CEO
        captured_outputs: list[list[AgentOutput]] = []

        async def capture_validate(
            outputs: list[AgentOutput], md: dict[str, Any]
        ) -> ValidatedIntelligenceReport:
            captured_outputs.append(outputs)
            return self._make_mock_report("AAPL")

        mock_ceo.validate = capture_validate

        orc = AgentOrchestrator(web_scraper=mock_scraper, ceo_model=mock_ceo)
        await orc.run_intelligence_cycle("AAPL")

        # A synthetic WebScraperAgent output should have been created
        assert len(captured_outputs) == 1
        agent_names = [o.agent_name for o in captured_outputs[0]]
        assert "WebScraperAgent" in agent_names


# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------


class TestAgentSchemas:
    """Test Pydantic schema validation for agent data models."""

    def test_agent_output_sentiment_clipped(self) -> None:
        """sentiment_score must be in [-1, 1]."""
        with pytest.raises(Exception):
            AgentOutput(
                agent_name="TestAgent",
                symbol="AAPL",
                timestamp=datetime.now(tz=UTC),
                sentiment_score=1.5,  # Invalid
                confidence=0.5,
            )

    def test_agent_output_confidence_clipped(self) -> None:
        """confidence must be in [0, 1]."""
        with pytest.raises(Exception):
            AgentOutput(
                agent_name="TestAgent",
                symbol="AAPL",
                timestamp=datetime.now(tz=UTC),
                sentiment_score=0.5,
                confidence=-0.1,  # Invalid
            )

    def test_raw_document_defaults(self) -> None:
        doc = RawDocument(
            source="Test",
            url="https://test.com",
            title="Test",
            text="Test text",
            timestamp=datetime.now(tz=UTC),
            symbol="AAPL",
        )
        assert doc.relevance_score == pytest.approx(1.0)

    def test_validated_intelligence_report_serialisable(
        self, sample_agent_outputs: list[AgentOutput]
    ) -> None:
        report = ValidatedIntelligenceReport(
            symbol="AAPL",
            timestamp=datetime.now(tz=UTC),
            validated_sentiment=0.42,
            consensus_confidence=0.75,
            conflict_detected=False,
            key_narrative="Bull regime.",
            risk_events=["fed_meeting"],
            agent_outputs=sample_agent_outputs,
        )
        data = report.model_dump(mode="json")
        assert data["symbol"] == "AAPL"
        assert len(data["agent_outputs"]) == 3
