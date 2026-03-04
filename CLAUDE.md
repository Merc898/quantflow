# QuantFlow — Claude Code Master Instruction File

## Project Identity
**QuantFlow** is an institutional-grade, fully autonomous quantitative research and trading signal platform. It combines state-of-the-art statistical models, machine learning, agentic AI market intelligence, and portfolio optimization into a SaaS product with a **Freemium** and **Premium** tier.

## Mission
Build a production-ready, end-to-end Python application that:
1. Fetches multi-source market data (prices, fundamentals, alternative data)
2. Runs an ensemble of 50+ quantitative models spanning all major quant finance domains
3. Performs agentic internet scraping and cross-validation of market intelligence via LLM APIs
4. Outputs risk-adjusted Buy / Hold / Sell signals with full uncertainty quantification
5. Serves all of this through a FastAPI backend + React frontend as a deployable SaaS product

## Architecture Overview

```
quantflow/
├── CLAUDE.md                        ← You are here
├── specs/                           ← Detailed spec files (read all before coding)
│   ├── 00_ARCHITECTURE.md
│   ├── 01_DATA_PIPELINE.md
│   ├── 02_MODELS_STATISTICAL.md
│   ├── 03_MODELS_ML.md
│   ├── 04_MODELS_RISK_PORTFOLIO.md
│   ├── 05_MODELS_MICROSTRUCTURE.md
│   ├── 06_MODELS_DERIVATIVES.md
│   ├── 07_MODELS_ADVANCED_RESEARCH.md
│   ├── 08_AGENTIC_INTELLIGENCE.md
│   ├── 09_SIGNAL_AGGREGATION.md
│   ├── 10_SAAS_PRODUCT.md
│   └── 11_TESTING_STANDARDS.md
├── quantflow/                       ← Main Python package
│   ├── data/
│   ├── models/
│   ├── agents/
│   ├── signals/
│   ├── risk/
│   ├── portfolio/
│   ├── execution/
│   ├── api/
│   └── utils/
├── frontend/                        ← React/Next.js SaaS frontend
├── tests/
├── notebooks/                       ← Research notebooks
├── docker/
└── docs/
```

## Absolute Standards (Non-Negotiable)

### Code Quality
- **Type hints everywhere** — all functions, all return types, no exceptions
- **Docstrings** — Google-style, every public function and class
- **Pydantic v2** — all data models, configs, and API contracts
- **Async-first** — use `asyncio` / `httpx` for all I/O; never block the event loop
- **Logging** — structured JSON logging via `structlog`; no bare `print()` statements
- **Error handling** — explicit exception hierarchies, never bare `except:`
- **No magic numbers** — all constants in `quantflow/config/constants.py`

### Quantitative Standards
- **Numerical stability** — use `np.float64`, check for NaN/Inf after every computation
- **Vectorized operations** — NumPy/Pandas vectorization; no Python loops over time series
- **Statistical rigor** — always report p-values, confidence intervals, and standard errors
- **Look-ahead bias** — ZERO tolerance; use `.shift(1)` for all lagged features; validate via walk-forward
- **Transaction costs** — all backtest P&L must be net of realistic costs (bid-ask, slippage, commissions)
- **Risk normalization** — all signals must be volatility-normalized before combination
- **Uncertainty** — every prediction must carry a confidence interval or posterior distribution

### Performance
- Pandas operations on >1M rows must use chunked processing or Polars
- Model training jobs >30 seconds must be async with a job queue (Celery + Redis)
- Cache all expensive computations with Redis (TTL appropriate to data staleness)
- Database: TimescaleDB (PostgreSQL extension) for all time-series data

## Reading Order for Implementation
**Read specs in order 00 → 11 before writing any code.**  
Each spec builds on the previous. Implement module by module, run tests, then proceed.

## Key Libraries (Install All)
```
# Core quant
numpy pandas polars scipy statsmodels arch pykalman

# ML
scikit-learn xgboost lightgbm catboost torch torchvision
transformers datasets

# Deep RL
stable-baselines3 gymnasium

# Risk & Portfolio  
cvxpy pyportfolioopt riskfolio-lib quantlib-python

# Time series
sktime darts prophet neuralprophet tsfresh

# Alternative data / NLP
spacy transformers sentence-transformers nltk textblob vaderSentiment

# Market data
yfinance alpha_vantage polygon-api-client fredapi quandl

# Web scraping / Agents
httpx playwright beautifulsoup4 openai anthropic

# Backend
fastapi uvicorn celery redis sqlalchemy asyncpg alembic

# Monitoring
prometheus-client grafana-api structlog sentry-sdk

# Testing
pytest pytest-asyncio pytest-cov hypothesis
```

## Environment Variables Required
```bash
# Data providers
ALPHA_VANTAGE_API_KEY=
POLYGON_API_KEY=
FRED_API_KEY=
QUANDL_API_KEY=

# LLM APIs (for agentic intelligence)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
PERPLEXITY_API_KEY=

# Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/quantflow
REDIS_URL=redis://localhost:6379/0

# Auth (SaaS)
JWT_SECRET_KEY=
STRIPE_SECRET_KEY=
STRIPE_WEBHOOK_SECRET=

# App
ENVIRONMENT=development  # development | staging | production
LOG_LEVEL=INFO
```
