# Spec 08 — SaaS Product: Backend API + Frontend + Monetization

## Freemium vs Premium Tiers

| Feature | Free | Premium (€49/mo) | Institutional (€299/mo) |
|---------|------|-----------------|------------------------|
| Symbols | 5 | Unlimited | Unlimited |
| Signal refresh | Daily | Every 4 hours | Real-time |
| Models used | 10 basic | All 50+ | All + custom |
| Agentic intelligence | Headlines only | Full AI analysis | Custom agents |
| Historical signals | 30 days | 3 years | Full history |
| Portfolio optimizer | MVO basic | All optimizers | + custom constraints |
| Options/derivatives | ❌ | ✅ | ✅ |
| API access | ❌ | ✅ (1000 req/day) | ✅ (unlimited) |
| SHAP explainability | ❌ | ✅ | ✅ |
| Export (CSV/PDF) | ❌ | ✅ | ✅ |
| Alerts | ❌ | Email | Email + Webhook + SMS |
| Support | Community | Email | Dedicated |

---

## Backend: FastAPI Application

**File:** `quantflow/api/main.py`
```python
app = FastAPI(
    title="QuantFlow API",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# Middleware
app.add_middleware(CORSMiddleware, ...)
app.add_middleware(RateLimitMiddleware, ...)  # per-tier limits
app.add_middleware(RequestLoggingMiddleware, ...)

# Routers
app.include_router(auth_router, prefix="/api/v1/auth")
app.include_router(signals_router, prefix="/api/v1/signals")
app.include_router(portfolio_router, prefix="/api/v1/portfolio")
app.include_router(models_router, prefix="/api/v1/models")
app.include_router(agents_router, prefix="/api/v1/intelligence")
app.include_router(subscription_router, prefix="/api/v1/subscription")
app.include_router(alerts_router, prefix="/api/v1/alerts")
app.include_router(ws_router, prefix="/ws")  # WebSockets
```

### API Endpoints

#### Signals
```
GET  /api/v1/signals/{symbol}              → FinalRecommendation
GET  /api/v1/signals/{symbol}/history      → list[FinalRecommendation]
GET  /api/v1/signals/{symbol}/explain      → ExplainabilityReport
POST /api/v1/signals/batch                 → list[FinalRecommendation]
GET  /api/v1/signals/universe/screener     → ranked list by signal strength
```

#### Portfolio
```
POST /api/v1/portfolio/optimize            → OptimizedPortfolio
GET  /api/v1/portfolio/efficient-frontier  → EfficientFrontier
POST /api/v1/portfolio/stress-test        → StressTestReport
GET  /api/v1/portfolio/risk-report        → RiskReport
```

#### Intelligence
```
GET  /api/v1/intelligence/{symbol}         → IntelligenceReport
GET  /api/v1/intelligence/{symbol}/news    → list[NewsItem]
GET  /api/v1/intelligence/macro            → MacroReport
```

#### WebSocket (real-time)
```
WS /ws/signals/{symbol}                   → stream FinalRecommendation updates
WS /ws/portfolio/{portfolio_id}           → stream portfolio P&L + risk
WS /ws/market                             → stream market regime + macro
```

### Auth System
```python
# JWT-based auth with subscription tier enforcement
class AuthMiddleware:
    async def __call__(self, request: Request, call_next):
        token = extract_bearer_token(request)
        user = await verify_jwt(token)
        subscription = await get_subscription(user.id)
        
        # Inject into request state for route handlers
        request.state.user = user
        request.state.tier = subscription.tier  # "free" | "premium" | "institutional"
        request.state.permissions = TIER_PERMISSIONS[subscription.tier]

# Route-level tier checking
def require_tier(min_tier: str):
    async def dependency(request: Request = Depends(get_current_user)):
        if not has_access(request.state.tier, min_tier):
            raise HTTPException(403, "Upgrade required")
    return dependency
```

---

## Frontend: Next.js 14 Application

**Stack:** Next.js 14 (App Router), TypeScript, Tailwind CSS, shadcn/ui, Recharts, Zustand

**Structure:**
```
frontend/
├── app/
│   ├── (auth)/login/           → Login/signup
│   ├── (auth)/register/
│   ├── dashboard/              → Main dashboard
│   ├── signals/[symbol]/       → Per-symbol deep dive
│   ├── portfolio/              → Portfolio optimizer
│   ├── screener/               → Signal screener
│   ├── intelligence/           → AI market intelligence
│   ├── settings/               → Account settings
│   └── pricing/                → Pricing + upgrade
├── components/
│   ├── charts/
│   │   ├── SignalGauge.tsx      → Buy/sell signal meter
│   │   ├── ConfidenceBar.tsx
│   │   ├── ModelContribChart.tsx → SHAP waterfall
│   │   ├── EfficientFrontier.tsx
│   │   ├── RegimeTimeline.tsx
│   │   └── VolSurface3D.tsx    → Interactive vol surface
│   ├── signals/
│   │   ├── RecommendationCard.tsx
│   │   ├── SignalHistory.tsx
│   │   └── RiskWarnings.tsx
│   └── layout/
│       ├── Sidebar.tsx
│       └── TopBar.tsx
└── lib/
    ├── api.ts                  → typed API client
    └── websocket.ts            → WS connection manager
```

### Key UI Pages

#### Dashboard
- Portfolio overview: total P&L, Sharpe ratio, current drawdown
- Live signal feed: top movers in watchlist
- Market regime indicator (prominent, color-coded)
- Macro sentiment gauge
- Recent AI intelligence highlights

#### Signal Deep Dive (`/signals/AAPL`)
- Recommendation card: STRONG BUY / SELL etc. with confidence %
- Model contribution chart (horizontal bar chart, each model)
- SHAP feature importance waterfall
- Price chart with signal overlay
- Intelligence report (news, sentiment, AI analysis)
- Risk metrics table
- Historical signal accuracy for this symbol

#### Portfolio Optimizer
- Input: holdings or target symbols
- Run optimizer (select: MVO / HRP / Black-Litterman / Robust)
- Output: optimal weights, efficient frontier chart
- Stress test scenarios panel
- "What if" analysis

---

## Stripe Integration
**File:** `quantflow/api/routers/subscription.py`

```python
# Webhook handler for Stripe events
@router.post("/stripe/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig = request.headers.get("stripe-signature")
    event = stripe.Webhook.construct_event(payload, sig, WEBHOOK_SECRET)
    
    match event["type"]:
        case "customer.subscription.created":
            await activate_subscription(event)
        case "customer.subscription.deleted":
            await deactivate_subscription(event)
        case "invoice.payment_failed":
            await handle_payment_failure(event)

# Checkout session creation
@router.post("/create-checkout-session")
async def create_checkout(
    tier: Literal["premium", "institutional"],
    current_user: User = Depends(get_current_user),
) -> CheckoutSession:
    session = stripe.checkout.Session.create(
        mode="subscription",
        payment_method_types=["card"],
        line_items=[{"price": STRIPE_PRICE_IDS[tier], "quantity": 1}],
        success_url=f"{BASE_URL}/dashboard?upgraded=true",
        cancel_url=f"{BASE_URL}/pricing",
        customer_email=current_user.email,
        metadata={"user_id": str(current_user.id)},
    )
    return {"url": session.url}
```

---

## Alerting System
**File:** `quantflow/api/alerts/`
```python
class AlertEngine:
    """
    Alert triggers (Premium+):
    - Signal change: recommendation changes (e.g., HOLD → BUY)
    - Strong signal: composite signal exceeds threshold
    - Risk alert: VaR breached, vol spike detected
    - Intelligence alert: major news event detected by agents
    - Regime change: market regime shift detected
    
    Delivery channels:
    - Email: sendgrid
    - Webhook: POST to user-configured URL
    - SMS: Twilio (Institutional tier)
    - Push notification: Firebase (mobile app)
    """
```

---

## Deployment

### Docker Compose (Development)
```yaml
# docker-compose.yml
services:
  api:        { build: ., command: uvicorn quantflow.api.main:app --reload }
  worker:     { build: ., command: celery -A quantflow.tasks worker -l info }
  beat:       { build: ., command: celery -A quantflow.tasks beat }
  frontend:   { build: ./frontend, command: npm run dev }
  postgres:   { image: timescale/timescaledb:latest-pg16 }
  redis:      { image: redis:7-alpine }
  mlflow:     { image: ghcr.io/mlflow/mlflow }
```

### Production (Kubernetes)
- Horizontal Pod Autoscaler on API pods (scale on CPU >70%)
- Separate node pools for GPU workloads (model training)
- TimescaleDB on managed PostgreSQL (AWS RDS or Supabase)
- Redis Cluster for HA
- CloudFront CDN for frontend static assets
- GitHub Actions for CI/CD with staging environment gate

### CI/CD Pipeline
```yaml
# .github/workflows/deploy.yml
jobs:
  test:
    - pytest with coverage (>80% required)
    - mypy type checking (zero errors)
    - ruff linting
    - backtest regression (Sharpe must not drop >10% vs baseline)
  
  staging:
    - Docker build + push
    - Deploy to staging k8s namespace
    - Run integration tests
    - Run E2E tests (Playwright)
  
  production:
    - Manual approval gate
    - Blue-green deployment
    - Smoke tests
    - Rollback on failure
```
