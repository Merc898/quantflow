# Spec 10 — Implementation Roadmap for Claude Code

## How to Use These Instructions

1. Read ALL spec files (00-09) before writing any code
2. Follow the phases below in order — each phase has clear deliverables
3. Run tests at the end of every phase before proceeding
4. Ask clarifying questions by checking the spec files first

---

## Phase 1: Foundation (Start Here)
**Goal:** Project skeleton, database, data pipeline running end-to-end

### Tasks
1. Create Python package structure: `quantflow/` with all subdirectories
2. Set up `pyproject.toml` with all dependencies (see CLAUDE.md)
3. `quantflow/config/` — Settings (pydantic-settings), constants, logging setup
4. `quantflow/db/` — SQLAlchemy async engine, TimescaleDB schema, Alembic migrations
5. `quantflow/data/fetchers/` — Implement `YFinanceFetcher` first (free, no API key)
6. `quantflow/data/processors/pipeline.py` — Basic OHLCV pipeline
7. `quantflow/data/features.py` — Price-based features (returns, vol, momentum)
8. `tests/unit/test_data_pipeline.py` — Test all of the above
9. `docker-compose.yml` — PostgreSQL+TimescaleDB + Redis

**Success criteria:** `pytest tests/unit/test_data_pipeline.py` passes

---

## Phase 2: Statistical Models
**Goal:** First 8 statistical models implemented and tested

### Tasks (in order)
1. `quantflow/models/base.py` — `BaseQuantModel`, `ModelOutput` schema
2. `quantflow/models/statistical/garch.py` — Start here (most foundational)
3. `quantflow/models/statistical/arima.py`
4. `quantflow/models/statistical/kalman.py`
5. `quantflow/models/statistical/var_vecm.py`
6. `quantflow/models/statistical/markov_switching.py`
7. `quantflow/models/statistical/factor_pca.py`
8. `quantflow/models/ml/base_trainer.py` — Walk-forward evaluator
9. Run walk-forward eval on all 6 models, log to MLflow

**Success criteria:** All 6 models produce `ModelOutput` with valid IC > 0 on SPY

---

## Phase 3: ML Models
**Goal:** Tree models + deep learning pipeline

### Tasks
1. `quantflow/models/ml/gradient_boosting.py` — XGBoost + LightGBM + CatBoost
2. `quantflow/models/ml/classic_ml.py` — Random Forest, LASSO, Ridge
3. `quantflow/models/ml/recurrent.py` — LSTM + GRU
4. `quantflow/models/ml/transformer_ts.py` — PatchTST
5. `quantflow/models/ml/deep_rl.py` — PPO agent (stable-baselines3)
6. Hyperparameter tuning: Optuna integration
7. SHAP values: compute and store for tree models
8. MLflow: all experiments tracked

**Success criteria:** GBT ensemble achieves IC > 0.05 on validation set

---

## Phase 4: Agentic Intelligence
**Goal:** Full agent pipeline with CEO validator

### Tasks
1. `quantflow/agents/llm_clients/openai_agent.py`
2. `quantflow/agents/llm_clients/anthropic_agent.py`
3. `quantflow/agents/llm_clients/perplexity_agent.py`
4. `quantflow/agents/scrapers/web_scraper.py`
5. `quantflow/agents/sentiment.py` — VADER + FinBERT + aggregation
6. `quantflow/agents/ceo_model.py` — Cross-validator
7. `quantflow/agents/orchestrator.py` — Async coordination
8. Celery tasks: schedule agent runs
9. Tests: mock LLM responses to avoid API costs in CI

**Success criteria:** Full intelligence cycle completes for "AAPL" in <60 seconds

---

## Phase 5: Risk, Portfolio & Signal Fusion
**Goal:** Complete signal pipeline producing recommendations

### Tasks
1. `quantflow/risk/var_es.py` — Three VaR methods
2. `quantflow/risk/evt.py` — EVT tail risk
3. `quantflow/risk/stress_tester.py`
4. `quantflow/portfolio/optimizer.py` — MVO + cvxpy
5. `quantflow/portfolio/black_litterman.py`
6. `quantflow/portfolio/hrp.py`
7. `quantflow/signals/normalizer.py`
8. `quantflow/signals/calibrator.py` — IC-based dynamic weights
9. `quantflow/signals/aggregator.py` — Two-stage ensemble
10. `quantflow/signals/regime_detector.py`
11. `quantflow/signals/recommendation.py` — Final signal → BUY/SELL
12. `quantflow/signals/explainer.py` — Claude-generated rationale

**Success criteria:** End-to-end signal generation for any symbol in <10 seconds

---

## Phase 6: Advanced Research Models
**Goal:** Cutting-edge models integrated (can run in background)

### Tasks (parallel with Phase 6, lower priority)
1. `quantflow/models/advanced/neural_ode.py`
2. `quantflow/models/advanced/neural_sde.py`
3. `quantflow/models/advanced/bayesian_nn.py`
4. `quantflow/models/derivatives/heston.py`
5. `quantflow/models/derivatives/vol_surface.py`
6. `quantflow/models/microstructure/hawkes.py`
7. `quantflow/models/microstructure/optimal_execution.py`

---

## Phase 7: FastAPI Backend
**Goal:** Full REST + WebSocket API

### Tasks
1. `quantflow/api/main.py` — App setup, middleware, CORS
2. `quantflow/api/auth/` — JWT auth, user model, registration/login
3. `quantflow/api/routers/signals.py` — Signal endpoints
4. `quantflow/api/routers/portfolio.py` — Portfolio endpoints
5. `quantflow/api/routers/intelligence.py` — Agent endpoints
6. `quantflow/api/routers/subscription.py` — Stripe webhooks
7. `quantflow/api/websockets/` — Real-time streaming
8. Rate limiting per tier
9. Integration tests: `tests/integration/test_api_endpoints.py`

**Success criteria:** All endpoints return valid responses, auth works, tiers enforced

---

## Phase 8: Frontend
**Goal:** Production-ready React/Next.js SaaS app

### Tasks
1. Next.js 14 project setup with TypeScript + Tailwind + shadcn/ui
2. Auth pages: login, register, forgot password
3. Dashboard: portfolio overview + live signal feed
4. Signal deep dive page per symbol
5. Portfolio optimizer UI
6. Screener page
7. Intelligence/AI reports page
8. Pricing page with Stripe checkout
9. Settings: watchlist management, alert configuration
10. Mobile responsive design

---

## Phase 9: Backtesting & Validation
**Goal:** Prove the system works historically

### Tasks
1. `quantflow/backtest/engine.py` — Walk-forward backtester
2. Run full backtest: 2015-2020 train, 2020-2024 test on S&P 500 universe
3. Report all metrics from spec 09
4. Backtest report auto-generated as PDF
5. Comparison: QuantFlow vs S&P 500 buy-and-hold

---

## Phase 10: Production Deployment
### Tasks
1. Dockerfile + docker-compose.yml (development)
2. Kubernetes manifests (production)
3. GitHub Actions CI/CD pipeline
4. Sentry error tracking integration
5. Prometheus metrics + Grafana dashboards
6. Automated alerting (PagerDuty for critical errors)
7. Documentation: API docs auto-generated by FastAPI

---

## Key Principles (Repeat for Emphasis)
- **Never compromise on look-ahead bias** — validate every feature with time-shift tests
- **Always include transaction costs** — gross Sharpe without costs is meaningless
- **Type hints on everything** — run mypy with zero errors
- **Test before moving to next phase** — broken foundations compound
- **Log everything** — structured JSON to make debugging possible
- **The CEO model is critical** — it's the trust mechanism for the agent layer
