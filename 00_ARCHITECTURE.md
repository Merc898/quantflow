# Spec 00 — System Architecture

## High-Level Data Flow

```
[Data Sources] → [Data Pipeline] → [Feature Store]
                                          ↓
[Agentic Intelligence Layer] ──────→ [Signal Fusion Engine]
[50+ Quant Models]          ──────→        ↓
                                   [Risk Adjuster]
                                          ↓
                                   [Portfolio Optimizer]
                                          ↓
                              [Buy/Hold/Sell + Confidence]
                                          ↓
                              [FastAPI] → [React Frontend]
```

## Module Boundaries

### 1. `quantflow/data/` — Data Pipeline
- `fetchers/`: One class per data source, all inherit `BaseDataFetcher`
- `processors/`: Cleaning, resampling, corporate actions adjustment
- `feature_store.py`: Redis-backed feature cache with versioning
- `schemas.py`: Pydantic models for every data type

### 2. `quantflow/models/` — Model Library
Subdirectories mirror spec files 02–07:
- `statistical/` — VAR, Kalman, GARCH, factor models
- `ml/` — tree models, deep learning, RL
- `risk/` — CVaR, EVT, Monte Carlo
- `microstructure/` — Hawkes, optimal execution
- `derivatives/` — Heston, Black-Scholes, numerical solvers
- `advanced/` — Neural ODEs, Neural SDEs, Bayesian DL

All models implement `BaseQuantModel`:
```python
class BaseQuantModel(ABC):
    @abstractmethod
    async def fit(self, data: pd.DataFrame) -> "BaseQuantModel": ...
    
    @abstractmethod
    async def predict(self, data: pd.DataFrame) -> ModelOutput: ...
    
    @abstractmethod
    def get_feature_importance(self) -> dict[str, float]: ...
    
    def validate_no_lookahead(self, data: pd.DataFrame) -> None: ...
```

### 3. `quantflow/agents/` — Agentic Intelligence
- `orchestrator.py`: Coordinates all agents, manages rate limits
- `scrapers/`: Web scrapers per data category
- `llm_clients/`: OpenAI, Anthropic, Perplexity clients
- `ceo_model.py`: Meta-model that cross-validates agent outputs
- `sentiment.py`: Aggregated sentiment signal

### 4. `quantflow/signals/` — Signal Fusion
- `aggregator.py`: Weighted ensemble of all model outputs
- `regime_detector.py`: Current market regime classification
- `calibrator.py`: Probability calibration (Platt, isotonic)
- `recommendation.py`: Final Buy/Hold/Sell + rationale generator

### 5. `quantflow/risk/` — Risk Management
- `position_sizer.py`: Kelly criterion, volatility targeting
- `var_calculator.py`: Historical, parametric, Monte Carlo VaR
- `stress_tester.py`: Scenario and historical stress tests
- `drawdown_monitor.py`: Real-time drawdown tracking

### 6. `quantflow/portfolio/` — Portfolio Construction
- `optimizer.py`: MVO, Black-Litterman, HRP, Robust
- `constraints.py`: Sector limits, turnover, leverage
- `rebalancer.py`: Trade list generation with cost minimization

### 7. `quantflow/api/` — FastAPI Backend
- `routers/`: One router per resource group
- `auth/`: JWT + subscription tier middleware
- `websockets/`: Real-time signal streaming
- `tasks/`: Celery task definitions

### 8. `frontend/` — Next.js 14 App Router
- Dashboard with live signal feed
- Model explainability views (SHAP)
- Portfolio optimizer UI
- Subscription management (Stripe)

## Concurrency Model
```
FastAPI (async) → dispatches to →
  ├── Redis cache (hot path, <10ms)
  ├── TimescaleDB (warm path, <100ms)
  └── Celery workers (cold path, model training/heavy compute)
        └── Results stored in Redis + DB, streamed via WebSocket
```

## Deployment
- Docker Compose for local development
- Kubernetes (k8s) manifests for production
- GitHub Actions CI/CD pipeline
- Sentry error tracking, Prometheus + Grafana metrics
