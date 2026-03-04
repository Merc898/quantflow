# Spec 09 — Testing Standards & Backtesting Framework

## Testing Philosophy
Every quantitative system must prove it works. For each model:
1. **Unit tests**: correctness of individual functions
2. **Statistical tests**: model diagnostics and validity
3. **Backtest**: realistic historical performance evaluation
4. **Walk-forward validation**: out-of-sample robustness

Minimum test coverage: **80%** (enforced in CI). Critical model paths: **95%**.

---

## Unit & Integration Tests (`tests/`)

### Structure
```
tests/
├── unit/
│   ├── test_data_pipeline.py
│   ├── test_models_statistical.py
│   ├── test_models_ml.py
│   ├── test_risk.py
│   ├── test_portfolio.py
│   ├── test_agents.py
│   └── test_signal_fusion.py
├── integration/
│   ├── test_api_endpoints.py
│   ├── test_websockets.py
│   └── test_celery_tasks.py
├── backtest/
│   └── test_strategy_backtest.py
└── conftest.py
```

### Required Tests Per Model
```python
# Example: GARCH model tests
class TestGARCHModel:
    def test_fit_returns_valid_params(self, sample_returns): ...
    def test_forecast_non_negative_variance(self): ...
    def test_no_lookahead_in_rolling_forecast(self): ...
    def test_persistence_less_than_one(self): ...
    def test_arch_effects_detected_correctly(self): ...
    def test_model_output_schema_valid(self): ...
    def test_regime_classification_boundaries(self): ...
    
    # Statistical validity
    def test_ljung_box_residuals_white_noise(self): ...
    def test_standardized_residuals_iid(self): ...
```

### Property-Based Testing (Hypothesis)
```python
from hypothesis import given, strategies as st

@given(
    returns=st.lists(st.floats(min_value=-0.5, max_value=0.5), min_size=252),
    confidence=st.floats(min_value=0.9, max_value=0.999),
)
def test_var_always_negative(returns, confidence):
    """VaR should always be a loss (negative return)."""
    var = compute_var(np.array(returns), confidence)
    assert var <= 0

@given(weights=st.lists(st.floats(0, 1), min_size=2, max_size=20))
def test_portfolio_weights_sum_to_one(weights):
    normalized = normalize_weights(weights)
    assert abs(sum(normalized) - 1.0) < 1e-10
```

---

## Backtesting Framework

**File:** `quantflow/backtest/engine.py`

### Backtest Standards (CRITICAL — Read Carefully)
The following rules are mandatory. Violating them produces misleading results.

```python
BACKTEST_RULES = """
1. LOOK-AHEAD BIAS: ZERO TOLERANCE
   - All features use only data available at signal time
   - Returns computed as close[t] / close[t-n] - 1 (historical, known)
   - Forward returns for evaluation: close[t+h] / close[t] - 1 (shifted)
   - Use pandas .shift() correctly: shift(1) means yesterday's value

2. TRANSACTION COSTS: Always include, never skip
   - Commission: $0.005/share (Interactive Brokers retail)
   - Slippage: 5 bps per trade (assume mid-to-execution spread)
   - Market impact: proportional to trade size / ADV
     impact_bps = 10 * sqrt(trade_size / adv_21d) * realized_vol
   - Short selling cost: LIBOR + 50bps on borrowed value
   - Net P&L = gross P&L - commissions - slippage - market_impact

3. SURVIVORSHIP BIAS: Use point-in-time universe
   - Universe must include delisted stocks for each historical period
   - Use CRSP or Compustat delisting returns for proper handling
   - If these sources unavailable: document limitation clearly

4. DATA SNOOPING: Minimize testing and overfitting
   - Split data: Train (60%) | Validation (20%) | Test (20%)
   - Test set touched ONLY ONCE for final evaluation
   - Report ONLY test set results in user-facing materials
   - Apply Bonferroni or BHY correction for multiple testing

5. REALISTIC CONSTRAINTS
   - Capacity constraint: max 2% of ADV per day
   - Rebalancing: monthly (realistic for most strategies)
   - Short selling: flag which symbols allow shorting
"""
```

### Backtest Engine Implementation
```python
class BacktestEngine:
    def __init__(self, config: BacktestConfig):
        self.config = config  # start, end, universe, initial_capital, costs
    
    async def run(
        self,
        strategy: BaseStrategy,
        universe: list[str],
        start: date,
        end: date,
    ) -> BacktestResult:
        """
        Walk-forward execution:
        1. For each rebalancing date t:
           a. Compute features using only data up to t
           b. Generate signals for all symbols
           c. Run portfolio optimizer with signal as views
           d. Compute trade list vs current positions
           e. Apply transaction costs
           f. Record positions, P&L, risk metrics
        2. Aggregate performance metrics
        """
```

### Performance Metrics (Report All)
```python
class PerformanceMetrics(BaseModel):
    # Returns
    total_return: float
    annualized_return: float
    cagr: float
    
    # Risk-adjusted
    sharpe_ratio: float          # target: > 1.0 net of costs
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float     # vs benchmark (S&P 500)
    
    # Risk
    annualized_vol: float
    max_drawdown: float
    max_drawdown_duration_days: int
    var_95_1d: float
    cvar_95_1d: float
    
    # Signal quality
    ic_mean: float               # target: > 0.03
    ic_std: float
    icir: float                  # target: > 0.3
    hit_rate: float              # % of trades profitable
    
    # Turnover & costs
    annual_turnover: float       # as % of portfolio
    total_costs_bps: float
    gross_vs_net_sharpe: float   # cost drag measurement
    
    # Benchmark comparison
    benchmark_return: float
    alpha: float                 # CAPM alpha (annualized)
    beta: float
    tracking_error: float
    
    # Statistical significance
    sharpe_t_stat: float         # H0: Sharpe = 0; reject if t > 2.0
    ic_t_stat: float
```

### Stress Testing in Backtest
```python
def backtest_stress_periods(result: BacktestResult) -> dict:
    """
    Report performance in known stress periods:
    - GFC 2008-2009
    - COVID crash 2020
    - 2022 rate hike selloff
    - Dot-com bust 2000-2002
    
    For each: return, max drawdown, Sharpe during period
    """
```

---

## Model Validation Checklist
Before any model is deployed to production:
```
□ Unit tests pass (100%)
□ Walk-forward IC > 0.03 on validation set
□ No look-ahead bias (verified by randomizing future data — signal should become random)
□ Residual diagnostics pass (for statistical models)
□ Calibration curve checked (for probability outputs)
□ Backtest Sharpe > 1.0 net of costs on test set
□ Max drawdown < 25%
□ Behavior in 2008, 2020 stress periods documented
□ MLflow run logged with all metrics
□ Model registered in Model Registry
□ Integration test: model runs end-to-end in pipeline
□ Output schema validated by Pydantic
□ Computation time within SLA (<30s for real-time path)
```
