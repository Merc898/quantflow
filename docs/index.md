# QuantFlow

**Institutional-grade quantitative research and trading signal platform.**

QuantFlow combines 50+ quantitative models, agentic AI market intelligence, and portfolio optimization into a production-ready SaaS product with Freemium and Premium tiers.

## Key Features

| Feature | Description |
|---------|-------------|
| **50+ Quant Models** | Statistical (GARCH, Kalman, VAR), ML (XGBoost, LSTM, Transformer), derivatives (Heston), microstructure (Hawkes) |
| **Agentic Intelligence** | Multi-LLM agent pipeline (OpenAI, Claude, Perplexity) with cross-validation CEO model |
| **Signal Fusion** | Two-stage ensemble with regime-aware IC-weighted calibration |
| **Portfolio Optimization** | MVO, Black-Litterman, HRP with realistic transaction cost modelling |
| **Walk-Forward Backtesting** | Zero look-ahead bias, full performance metrics, PDF reporting |
| **Production-Ready** | FastAPI + Celery + TimescaleDB + Redis, Kubernetes deployment, Prometheus/Grafana monitoring |

## Quick Start

```bash
# Clone the repo
git clone https://github.com/your-org/quantflow.git
cd quantflow

# Start the full development stack
docker compose up -d

# Generate a signal for AAPL
curl -X POST http://localhost:8000/api/v1/signals/generate \
  -H "Authorization: Bearer YOUR_JWT" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL"}'
```

## Architecture Overview

```
[Market Data] → [Feature Engine] → [50+ Models] ─┐
[Web Scraping] → [LLM Agents] → [CEO Validator] ─┤
                                                   ↓
                              [Signal Fusion + Regime Detection]
                                                   ↓
                              [Portfolio Optimizer + Risk Gates]
                                                   ↓
                              [Buy / Hold / Sell + Confidence Interval]
                                                   ↓
                              [FastAPI REST + WebSocket] → [React SaaS]
```

## Phase-by-Phase Implementation

| Phase | Status | Description |
|-------|--------|-------------|
| 1 — Foundation | ✅ | Data pipeline, DB, config |
| 2 — Statistical Models | ✅ | GARCH, ARIMA, Kalman, VAR, Markov, PCA |
| 3 — ML Models | ✅ | XGBoost, LSTM, Transformer, DRL |
| 4 — Agent Intelligence | ✅ | Multi-LLM, scraping, CEO validation |
| 5 — Signal Fusion | ✅ | Ensemble, portfolio, recommendations |
| 6 — Advanced Research | ✅ | Neural ODE/SDE, Heston, Hawkes |
| 7 — FastAPI Backend | ✅ | REST + WS, auth, Stripe, rate limiting |
| 8 — Frontend | ✅ | Next.js 14 SaaS dashboard |
| 9 — Backtesting | ✅ | Walk-forward, metrics, PDF reports |
| 10 — Production | ✅ | Docker, k8s, CI/CD, monitoring |
