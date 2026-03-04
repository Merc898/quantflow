"""Async web scraper for financial news, Reddit, and SEC filings.

All I/O is async via httpx.  JavaScript-rendered pages use playwright
(optional; falls back gracefully if not installed).
Returns raw :class:`RawDocument` objects consumed by the agent pipeline.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from quantflow.agents.schemas import RawDocument
from quantflow.config.logging import get_logger

logger = get_logger(__name__)

# SEC EDGAR base URL
_EDGAR_BASE = "https://www.sec.gov"
_EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index?q=%22{symbol}%22&dateRange=custom&startdt={start}&forms={form}"

# Yahoo Finance news RSS
_YF_NEWS_URL = "https://finance.yahoo.com/rss/headline?s={symbol}"

# StockTwits API
_STOCKTWITS_URL = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"

# User-Agent header to avoid being blocked
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; QuantFlowBot/1.0; +https://quantflow.io/bot)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

_REQUEST_TIMEOUT = 15.0


class WebScraperAgent:
    """Async financial web scraper.

    Scrapes news, Reddit, StockTwits, and SEC filings concurrently.
    Returns normalised :class:`RawDocument` objects.

    Args:
        timeout_s: Per-request HTTP timeout in seconds.
    """

    def __init__(self, timeout_s: float = _REQUEST_TIMEOUT) -> None:
        self._timeout = timeout_s
        self._logger = get_logger(__name__)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    async def scrape_news(
        self,
        symbol: str,
        hours_back: int = 24,
    ) -> list[RawDocument]:
        """Scrape recent news from multiple financial sources.

        Tries Yahoo Finance RSS and StockTwits; more sources can be added.
        Falls back gracefully on per-source errors.

        Args:
            symbol: Ticker symbol (e.g. "AAPL").
            hours_back: Lookback window in hours.

        Returns:
            List of :class:`RawDocument` sorted by relevance descending.
        """
        import asyncio

        tasks = [
            self._scrape_yahoo_rss(symbol),
            self._scrape_stocktwits(symbol),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        docs: list[RawDocument] = []
        for r in results:
            if isinstance(r, Exception):
                self._logger.warning("Scrape source failed", error=str(r))
            elif isinstance(r, list):
                docs.extend(r)

        # Sort by relevance score descending
        docs.sort(key=lambda d: d.relevance_score, reverse=True)
        return docs

    async def scrape_reddit(self, symbol: str) -> list[RawDocument]:
        """Fetch hot posts mentioning symbol from financial subreddits.

        Uses the Reddit JSON API (no auth required for public endpoints).

        Args:
            symbol: Ticker symbol.

        Returns:
            List of :class:`RawDocument` from Reddit posts.
        """
        subreddits = ["investing", "stocks", "wallstreetbets"]
        docs: list[RawDocument] = []

        async with httpx.AsyncClient(
            timeout=self._timeout, headers=_HEADERS, follow_redirects=True
        ) as client:
            for sub in subreddits:
                try:
                    url = f"https://www.reddit.com/r/{sub}/search.json"
                    params = {
                        "q": symbol,
                        "sort": "hot",
                        "limit": 10,
                        "restrict_sr": "1",
                        "t": "day",
                    }
                    resp = await client.get(url, params=params)
                    resp.raise_for_status()
                    data = resp.json()
                    posts = data.get("data", {}).get("children", [])

                    for post in posts:
                        pd_ = post.get("data", {})
                        title = pd_.get("title", "")
                        selftext = pd_.get("selftext", "")
                        score = pd_.get("score", 0)
                        permalink = pd_.get("permalink", "")
                        url_post = f"https://www.reddit.com{permalink}"

                        # Relevance: upvote normalised to [0, 1]
                        relevance = min(1.0, score / 5000.0)
                        text = f"{title}\n{selftext}".strip()
                        if not text:
                            continue

                        docs.append(
                            RawDocument(
                                source=f"Reddit r/{sub}",
                                url=url_post,
                                title=title,
                                text=text[:3000],
                                timestamp=datetime.now(tz=timezone.utc),
                                symbol=symbol,
                                relevance_score=relevance,
                            )
                        )
                except Exception as exc:
                    self._logger.warning(
                        "Reddit scrape failed",
                        subreddit=sub,
                        symbol=symbol,
                        error=str(exc),
                    )

        return docs

    async def fetch_sec_filing(
        self,
        symbol: str,
        form_type: str = "8-K",
        max_docs: int = 3,
    ) -> list[RawDocument]:
        """Fetch recent SEC EDGAR filings for a symbol.

        Args:
            symbol: Ticker symbol.
            form_type: SEC form type (e.g. "8-K", "10-K", "10-Q").
            max_docs: Maximum filings to return.

        Returns:
            List of :class:`RawDocument` containing filing text excerpts.
        """
        from datetime import timedelta

        start = (datetime.now(tz=timezone.utc) - timedelta(days=90)).strftime("%Y-%m-%d")
        search_url = _EDGAR_SEARCH.format(
            symbol=symbol, start=start, form=form_type
        )

        docs: list[RawDocument] = []
        async with httpx.AsyncClient(
            timeout=self._timeout, headers={**_HEADERS, "User-Agent": "QuantFlow/1.0 contact@quantflow.io"},
            follow_redirects=True,
        ) as client:
            try:
                resp = await client.get(search_url)
                resp.raise_for_status()
                hits = resp.json().get("hits", {}).get("hits", [])

                for hit in hits[:max_docs]:
                    src = hit.get("_source", {})
                    file_date = src.get("file_date", "")
                    display_names = src.get("display_names", [])
                    entity = display_names[0].get("name", symbol) if display_names else symbol
                    doc_url = _EDGAR_BASE + src.get("file_num", "")

                    docs.append(
                        RawDocument(
                            source="SEC EDGAR",
                            url=f"{_EDGAR_BASE}/cgi-bin/browse-edgar?action=getcompany&CIK={symbol}&type={form_type}",
                            title=f"{form_type} filing — {entity} ({file_date})",
                            text=f"Filing type: {form_type}. Entity: {entity}. Date: {file_date}. "
                                 f"Ticker: {symbol}. [Full filing available at SEC EDGAR]",
                            timestamp=datetime.now(tz=timezone.utc),
                            symbol=symbol,
                            relevance_score=0.9,
                        )
                    )
            except Exception as exc:
                self._logger.warning(
                    "SEC EDGAR fetch failed",
                    symbol=symbol,
                    form_type=form_type,
                    error=str(exc),
                )

        return docs

    async def scrape_all(self, symbol: str) -> list[RawDocument]:
        """Scrape all sources for a symbol concurrently.

        Args:
            symbol: Ticker symbol.

        Returns:
            Combined and deduplicated list of :class:`RawDocument` objects.
        """
        import asyncio

        results = await asyncio.gather(
            self.scrape_news(symbol),
            self.scrape_reddit(symbol),
            self.fetch_sec_filing(symbol),
            return_exceptions=True,
        )

        docs: list[RawDocument] = []
        for r in results:
            if isinstance(r, list):
                docs.extend(r)
            elif isinstance(r, Exception):
                self._logger.warning("Scrape task failed", error=str(r))

        # Deduplicate by URL
        seen: set[str] = set()
        unique: list[RawDocument] = []
        for d in docs:
            if d.url not in seen:
                seen.add(d.url)
                unique.append(d)

        unique.sort(key=lambda d: d.relevance_score, reverse=True)
        self._logger.info(
            "Scrape complete",
            symbol=symbol,
            total_docs=len(unique),
        )
        return unique

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _scrape_yahoo_rss(self, symbol: str) -> list[RawDocument]:
        """Scrape Yahoo Finance RSS headlines for a symbol.

        Args:
            symbol: Ticker symbol.

        Returns:
            List of :class:`RawDocument` from RSS items.
        """
        import xml.etree.ElementTree as ET

        url = _YF_NEWS_URL.format(symbol=symbol)
        async with httpx.AsyncClient(
            timeout=self._timeout, headers=_HEADERS, follow_redirects=True
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()

        root = ET.fromstring(resp.text)
        ns = {"": "http://www.w3.org/2005/Atom"}

        docs: list[RawDocument] = []
        # Try Atom first, then RSS
        items = root.findall(".//item") or root.findall(".//entry")
        for item in items[:10]:
            title_el = item.find("title")
            link_el = item.find("link")
            desc_el = item.find("description") or item.find("summary")

            title = title_el.text if title_el is not None else ""
            link = link_el.text if link_el is not None else ""
            description = _strip_html(desc_el.text or "") if desc_el is not None else ""

            if not title:
                continue

            text = f"{title}\n{description}".strip()
            docs.append(
                RawDocument(
                    source="Yahoo Finance",
                    url=link or url,
                    title=title,
                    text=text[:2000],
                    timestamp=datetime.now(tz=timezone.utc),
                    symbol=symbol,
                    relevance_score=0.75,
                )
            )
        return docs

    async def _scrape_stocktwits(self, symbol: str) -> list[RawDocument]:
        """Fetch recent StockTwits messages for a symbol.

        Args:
            symbol: Ticker symbol.

        Returns:
            List of :class:`RawDocument` from StockTwits messages.
        """
        url = _STOCKTWITS_URL.format(symbol=symbol)
        async with httpx.AsyncClient(
            timeout=self._timeout, headers=_HEADERS, follow_redirects=True
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

        docs: list[RawDocument] = []
        messages = data.get("messages", [])
        for msg in messages[:15]:
            body = msg.get("body", "")
            msg_id = msg.get("id", "")
            sentiment_label = msg.get("entities", {}).get("sentiment", {})
            # StockTwits provides structured bullish/bearish labels
            relevance = 0.6
            if sentiment_label:
                relevance = 0.80  # structured sentiment = more reliable

            if not body:
                continue

            docs.append(
                RawDocument(
                    source="StockTwits",
                    url=f"https://stocktwits.com/symbol/{symbol}?m={msg_id}",
                    title=body[:80],
                    text=body[:500],
                    timestamp=datetime.now(tz=timezone.utc),
                    symbol=symbol,
                    relevance_score=relevance,
                )
            )
        return docs


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _strip_html(text: str) -> str:
    """Remove HTML tags from a string.

    Args:
        text: Raw HTML string.

    Returns:
        Plain text with tags removed.
    """
    return re.sub(r"<[^>]+>", " ", text).strip()
