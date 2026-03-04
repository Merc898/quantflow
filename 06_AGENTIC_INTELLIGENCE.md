# Spec 06 — Agentic Market Intelligence Layer

## Overview
The agent layer autonomously scrapes the internet, queries multiple LLM APIs,  
and uses a "CEO Model" to cross-validate and synthesize intelligence into  
structured signals consumed by the Signal Fusion Engine.

**File structure:** `quantflow/agents/`

## Architecture

```
AgentOrchestrator
├── WebScraperAgent       → raw HTML/text from news, forums, SEC
├── PerplexityAgent       → real-time web search + citation
├── OpenAIAgent           → GPT-4o for analysis and structuring
├── AnthropicAgent        → Claude for synthesis and validation
├── SentimentAggregator   → combines multi-source sentiment
└── CEOValidatorModel     → cross-validates all agent outputs
```

## 1. Agent Orchestrator
**File:** `orchestrator.py`
```python
class AgentOrchestrator:
    """
    Coordinates all agents with rate limiting, retry logic, and 
    consensus building.
    
    Execution flow:
    1. Trigger on schedule (every 4 hours for Premium, daily for Free)
    2. Fan out: run all agents concurrently via asyncio.gather()
    3. Collect raw outputs → normalize to AgentOutput schema
    4. Pass all outputs to CEOValidatorModel for cross-validation
    5. Return: validated IntelligenceReport
    
    Rate limiting:
    - Per-API rate limiters using token bucket algorithm
    - Exponential backoff with jitter on 429/503 errors
    - Circuit breaker: if >3 consecutive failures, skip agent for 1 hour
    
    Cost tracking:
    - Log token usage per API call to database
    - Alert if daily spend > threshold
    """
    
    async def run_intelligence_cycle(
        self,
        symbols: list[str],
        topics: list[str],  # ["earnings", "macro", "geopolitical", "technical"]
    ) -> IntelligenceReport: ...
```

## 2. Web Scraper Agent
**File:** `scrapers/web_scraper.py`
```python
class WebScraperAgent:
    """
    Sources (all async via httpx + playwright for JS-rendered pages):
    
    Financial News:
    - Reuters, Bloomberg (public), FT, WSJ (headlines)
    - Seeking Alpha, Benzinga, MarketWatch
    - Yahoo Finance news feed
    
    Social / Sentiment:
    - Reddit: r/investing, r/stocks, r/wallstreetbets
      (via asyncpraw, extract hot posts + comments)
    - StockTwits API (structured sentiment: bullish/bearish)
    - Twitter/X API v2 (financial hashtags, verified accounts)
    
    Official / Regulatory:
    - SEC EDGAR: 8-K (earnings, material events), 13F (fund holdings)
    - Fed speeches and minutes (federalreserve.gov)
    - ECB, BIS publications
    - Company IR pages
    
    Data returned: raw text with metadata {source, url, timestamp, relevance_score}
    """
    
    async def scrape_news(self, symbol: str, hours_back: int = 24) -> list[RawDocument]: ...
    async def scrape_reddit(self, symbol: str) -> list[RawDocument]: ...
    async def fetch_sec_filing(self, symbol: str, form_type: str) -> RawDocument: ...
```

## 3. Perplexity Agent
**File:** `llm_clients/perplexity_agent.py`
```python
class PerplexityAgent:
    """
    Uses Perplexity API (sonar-large-32k model) for grounded web search.
    Perplexity returns answers with citations → high quality, recent data.
    
    Queries to run per symbol (build dynamically):
    - "What are the latest analyst upgrades/downgrades for {symbol}?"
    - "What recent news could affect {symbol} stock price?"
    - "What is the current market sentiment for {symbol}?"
    - "What are analysts saying about {sector} sector outlook?"
    - "What macro risks are most discussed in markets today?"
    
    Output: PerplexityResponse with:
      - answer: str
      - citations: list[str]
      - search_queries_used: list[str]
    """
    
    BASE_URL = "https://api.perplexity.ai/chat/completions"
    MODEL = "sonar-large-32k-online"  # real-time web search
    
    async def query(self, prompt: str, symbol: str | None = None) -> PerplexityResponse: ...
```

## 4. OpenAI Agent
**File:** `llm_clients/openai_agent.py`
```python
class OpenAIAgent:
    """
    GPT-4o for:
    1. Structured extraction from raw documents
       (earnings surprises, management tone, forward guidance)
    2. Sentiment scoring on news articles [-1.0, +1.0]
    3. Event classification (earnings beat/miss, M&A, FDA approval, etc.)
    4. Financial statement analysis
    
    System prompt template:
    \"\"\"
    You are a senior quantitative analyst at a top hedge fund.
    Analyze the following financial content and extract structured signals.
    Be precise, cite evidence, and quantify uncertainty.
    Output ONLY valid JSON matching the provided schema.
    \"\"\"
    
    Always use structured output (JSON mode or response_format=json_schema).
    Temperature=0.1 for analytical tasks, 0.7 for summarization.
    """
    
    async def extract_sentiment(self, text: str, symbol: str) -> SentimentScore: ...
    async def classify_event(self, text: str) -> EventClassification: ...
    async def analyze_earnings(self, filing_text: str) -> EarningsAnalysis: ...
```

## 5. Anthropic Agent
**File:** `llm_clients/anthropic_agent.py`
```python
class AnthropicAgent:
    """
    Claude (claude-sonnet-4-20250514) for:
    1. Deep document analysis (long context = entire 10-K)
    2. Cross-checking and fact verification
    3. Synthesizing conflicting signals
    4. Generating natural language rationale for recommendations
    
    Use extended thinking for complex multi-step analysis.
    Use document upload (base64 PDF) for SEC filings.
    """
    
    async def analyze_long_document(self, document: str, queries: list[str]) -> DocumentAnalysis: ...
    async def synthesize_signals(self, agent_outputs: list[AgentOutput]) -> Synthesis: ...
    async def generate_recommendation_rationale(self, signal: FinalSignal) -> str: ...
```

## 6. CEO Validator Model
**File:** `ceo_model.py`
```python
class CEOValidatorModel:
    """
    The "CEO" is a meta-model that:
    1. Receives all agent outputs (Perplexity, OpenAI, Anthropic, scrapers)
    2. Checks for consistency and contradictions
    3. Weights outputs by source reliability (dynamically updated)
    4. Detects hallucinations by cross-referencing factual claims
    5. Produces a final validated IntelligenceReport
    
    Validation logic:
    
    CONSISTENCY CHECK:
    - If Perplexity says "bearish" and OpenAI says "strongly bullish" 
      and scrapers show negative news → flag conflict, investigate
    - Assign confidence score: HIGH (>80% agreement), 
                               MEDIUM (60-80% agreement),
                               LOW (<60% agreement or contradictions)
    
    HALLUCINATION DETECTION:
    - Extract factual claims (numbers, dates, names) from each agent
    - Cross-check against structured data from DB
    - Flag any claim that contradicts verified data
    
    SOURCE RELIABILITY WEIGHTING:
    - Updated weekly based on accuracy of past predictions
    - Stored in DB: {agent_name: reliability_score}
    - Initial weights: Perplexity=0.30, OpenAI=0.30, Anthropic=0.25, Scrapers=0.15
    
    Implementation: Use Claude as the CEO model itself
    (meta-prompt: "You are a critical analyst reviewing these reports...")
    """
    
    async def validate(
        self,
        agent_outputs: list[AgentOutput],
        market_data: dict,
    ) -> ValidatedIntelligenceReport: ...
```

## 7. Sentiment Aggregation
**File:** `sentiment.py`
```python
class SentimentAggregator:
    """
    Combines signals from:
    - Agent outputs (LLM-analyzed sentiment)
    - VADER/TextBlob on raw scraped text (fast, rule-based)
    - FinBERT embeddings (finance-specific BERT)
    - StockTwits structured sentiment (direct bullish/bearish counts)
    - Options put/call ratio (market-implied fear/greed)
    
    Aggregation:
    1. Normalize all scores to [-1, +1]
    2. Compute exponentially weighted average (decay=0.9 per day)
    3. Compare to 30-day historical baseline → sentiment z-score
    4. Extreme sentiment flags:
       z > 2.0: "Extreme Bullish" (potential contrarian sell)
       z < -2.0: "Extreme Bearish" (potential contrarian buy)
    
    Final output:
      composite_sentiment: float [-1, +1]
      sentiment_regime: EXTREME_BEAR / BEAR / NEUTRAL / BULL / EXTREME_BULL
      sentiment_momentum: float (5-day change in sentiment)
      contrarian_signal: bool (if extreme reading)
    """
```

## Data Schema
```python
class AgentOutput(BaseModel):
    agent_name: str
    symbol: str
    timestamp: datetime
    sentiment_score: float          # [-1, +1]
    confidence: float               # [0, 1]
    key_events: list[str]           # ["earnings beat", "CEO departure"]
    bullish_factors: list[str]
    bearish_factors: list[str]
    raw_sources: list[str]          # URLs / citation strings
    factual_claims: list[FactualClaim]

class IntelligenceReport(BaseModel):
    symbol: str
    timestamp: datetime
    validated_sentiment: float
    consensus_confidence: float
    conflict_detected: bool
    key_narrative: str              # 2-3 sentence plain English summary
    risk_events: list[str]         # upcoming catalysts / risks
    agent_outputs: list[AgentOutput]
    ceo_override: bool              # True if CEO model overrode consensus
    ceo_reasoning: str | None
```

## Scheduling
```python
# Celery beat schedule
AGENT_SCHEDULES = {
    "premium_users": {"interval": timedelta(hours=4)},
    "free_users":    {"interval": timedelta(hours=24)},
    "earnings_week": {"interval": timedelta(hours=1)},  # heightened during earnings
}
```
