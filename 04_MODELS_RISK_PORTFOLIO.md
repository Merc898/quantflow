# Spec 04 — Risk, Portfolio, Microstructure & Derivatives Models

## PART A: Risk Models (`quantflow/risk/`)

### 1. VaR & Expected Shortfall
**File:** `quantflow/risk/var_es.py`
```python
class RiskCalculator:
    """
    Three methodologies (always compute all three, report range):
    
    1. Historical Simulation
       - Rolling 252-day window of returns
       - VaR_alpha = percentile(returns, alpha)
       - ES_alpha = mean(returns | returns < VaR_alpha)
    
    2. Parametric (Variance-Covariance)
       - Assume returns ~ Normal or Student-t
       - VaR = mu - z_alpha * sigma
       - Fit distribution via MLE; use Student-t if fat tails detected
    
    3. Monte Carlo
       - Simulate 10,000 scenarios from fitted distribution (with correlations)
       - Use copula for joint distribution (see spec 07)
       - VaR / ES from simulated P&L distribution
    
    Confidence levels: 95%, 99%, 99.5%
    Horizons: 1-day, 10-day, 1-month
    
    Backtesting: Kupiec test for VaR model validity
                 Christoffersen test for independence of violations
    """
    
    def compute_portfolio_var(
        self,
        weights: np.ndarray,
        covariance_matrix: np.ndarray,
        method: Literal["historical","parametric","monte_carlo"],
        confidence: float = 0.99,
        horizon: int = 1,
    ) -> RiskReport: ...
```

### 2. Extreme Value Theory (EVT)
**File:** `quantflow/risk/evt.py`
```python
class EVTRiskModel:
    """
    Peaks-Over-Threshold (POT) with Generalized Pareto Distribution.
    
    Steps:
    1. Set threshold u = 95th percentile of loss distribution
    2. Fit GPD to exceedances: loss - u | loss > u
    3. Estimate tail index xi (shape) and scale sigma
    4. Compute extreme quantiles: VaR_99.9%, ES_99.9%
    
    Hill estimator as alternative for tail index.
    Mean excess plot for threshold selection.
    
    Library: scipy.stats (genpareto), or pyextremes
    """
```

### 3. Stress Testing & Scenario Analysis
**File:** `quantflow/risk/stress_tester.py`
```python
HISTORICAL_SCENARIOS = {
    "covid_crash_2020": ("2020-02-19", "2020-03-23"),
    "gfc_2008": ("2008-09-01", "2009-03-09"),
    "dot_com_bust": ("2000-03-10", "2002-10-09"),
    "black_monday_1987": ("1987-10-19", "1987-10-19"),
    "russia_ukraine_2022": ("2022-02-24", "2022-03-15"),
    "svb_crisis_2023": ("2023-03-08", "2023-03-17"),
}

class StressTester:
    def run_historical_scenario(self, portfolio_weights, scenario_name) -> ScenarioResult: ...
    def run_factor_shock(self, shock_dict: dict[str, float]) -> ScenarioResult: ...
    # shock_dict: {"equity_market": -0.20, "credit_spread": +0.02, ...}
    def run_monte_carlo_stress(self, n_scenarios=50_000) -> StressDistribution: ...
```

---

## PART B: Portfolio Optimization (`quantflow/portfolio/`)

### 4. Mean-Variance Optimization (MVO)
**File:** `optimizer.py`
```python
class MVOOptimizer:
    """
    Markowitz 1952 with modern enhancements:
    
    1. Covariance estimation (implement ALL):
       - Sample covariance (baseline)
       - Ledoit-Wolf shrinkage (sklearn)
       - Oracle Approximating Shrinkage (OAS)
       - Random Matrix Theory (RMT): clean eigenvalues
         Filter eigenvalues < lambda_max from Marchenko-Pastur
    
    2. Expected return estimation (implement ALL):
       - Sample mean (shrunk to grand mean)
       - CAPM-implied returns
       - Signal-implied returns (from model ensemble)
       - Black-Litterman posterior (see below)
    
    3. Solve: maximize w'mu - (lambda/2)*w'Sigma*w
       subject to: sum(w)=1, w>=0, w_i<=0.20
       Library: cvxpy
    
    4. Efficient frontier: compute 50 points, store for visualization
    """
```

### 5. Black-Litterman
**File:** `black_litterman.py`
```python
class BlackLittermanOptimizer:
    """
    Steps:
    1. Equilibrium returns: Pi = delta * Sigma * w_market
       (delta = risk aversion, w_market = market cap weights)
    
    2. Views matrix P, view returns Q, uncertainty Omega
       - Views sourced from: model ensemble signals
       - Omega = diag(P * tau * Sigma * P') * confidence_scaling
    
    3. BL posterior:
       mu_BL = [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1 
               * [(tau*Sigma)^-1*Pi + P'*Omega^-1*Q]
    
    4. Optimize with mu_BL as expected returns
    
    tau: typically 1/T (T = estimation window)
    """
```

### 6. Hierarchical Risk Parity (HRP)
**File:** `hrp.py`
```python
class HRPOptimizer:
    """
    Lopez de Prado 2016.
    Steps:
    1. Correlation matrix → distance matrix
    2. Hierarchical clustering (Ward linkage)
    3. Quasi-diagonalization of covariance matrix
    4. Recursive bisection for weight allocation
    
    Advantages: no matrix inversion, stable out-of-sample
    Compare vs MVO: report Sharpe ratio over rolling 252-day test window
    """
```

### 7. Robust Optimization
**File:** `robust_optimizer.py`
```python
class RobustOptimizer:
    """
    Worst-case robust MVO:
    maximize min_{mu in U} w'mu - (lambda/2)*w'Sigma*w
    where U = {mu: ||mu - mu_hat||_2 <= kappa}
    
    Implemented as SOCP via cvxpy.
    kappa calibrated to sample uncertainty of mean estimates.
    """
```

---

## PART C: Market Microstructure (`quantflow/models/microstructure/`)

### 8. Hawkes Processes
**File:** `hawkes.py`
```python
class HawkesProcessModel(BaseQuantModel):
    """
    Self-exciting point process for trade arrival intensity.
    
    lambda(t) = mu + sum_{t_i < t} alpha * exp(-beta*(t - t_i))
    
    Fit via MLE on tick data.
    
    Applications:
    1. Predict short-term order flow imbalance
    2. Detect algorithmic trading activity (clustering)
    3. Market impact estimation
    
    Library: tick (by Alan Hawkes group) or custom MLE with scipy
    """
```

### 9. Almgren-Chriss Optimal Execution
**File:** `optimal_execution.py`
```python
class AlmgrenChrissModel:
    """
    Optimal trade schedule to liquidate X shares over T periods.
    
    Minimize: E[cost] + lambda * Var[cost]
    
    Analytical solution:
    x*(t) = X * sinh(kappa*(T-t)) / sinh(kappa*T)
    where kappa = sqrt(lambda * eta / (sigma^2 * gamma))
    
    Parameters:
      - eta: temporary impact coefficient (estimate from tick data)
      - gamma: permanent impact coefficient
      - sigma: return volatility
      - lambda: risk aversion (set by user or default=1e-6)
    
    Output: optimal execution schedule (shares per interval)
    """
```

---

## PART D: Derivatives Models (`quantflow/models/derivatives/`)

### 10. Black-Scholes + Greeks
**File:** `black_scholes.py`
- Full Greeks: Delta, Gamma, Theta, Vega, Rho, Vanna, Volga
- Implied vol from market prices via Newton-Raphson

### 11. Heston Stochastic Volatility
**File:** `heston.py`
```python
class HestonModel:
    """
    dS = mu*S*dt + sqrt(v)*S*dW_1
    dv = kappa*(theta-v)*dt + xi*sqrt(v)*dW_2
    corr(dW_1, dW_2) = rho
    
    Pricing via Fourier transform (Gil-Pelaez inversion).
    Calibration: minimize sum of squared implied vol errors across strikes/maturities
    Optimizer: scipy.optimize.differential_evolution
    
    Parameters: [kappa, theta, xi, rho, v0]
    """
```

### 12. SABR Model
**File:** `sabr.py`
- Hagan et al. formula for implied vol surface
- Calibrate [alpha, beta, rho, nu] per maturity slice

### 13. Implied Volatility Surface
**File:** `vol_surface.py`
```python
class VolatilitySurface:
    """
    Construct arbitrage-free IV surface:
    1. Collect market option prices across strikes and maturities
    2. Compute raw implied vols (Newton-Raphson per contract)
    3. Fit SVI (Stochastic Volatility Inspired) parametrization:
       w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))
    4. Check: no calendar spread arbitrage, no butterfly arbitrage
    5. Local vol surface via Dupire formula
    
    Output: callable surface (strike, maturity) → implied_vol
    """
```

### 14. Monte Carlo with Variance Reduction
**File:** `monte_carlo.py`
```python
class MonteCarloEngine:
    """
    Techniques (all implemented, select by use case):
    - Antithetic variates
    - Control variates (Black-Scholes as control)
    - Quasi-Monte Carlo (Sobol sequences via scipy.stats.qmc)
    - Stratified sampling
    
    Discretization schemes:
    - Euler-Maruyama (simple, may be biased)
    - Milstein (higher order)
    - Exact simulation for Heston (Broadie-Kaya)
    
    Performance: vectorized numpy, >100k paths/second target
    """
    
    async def price_option(
        self,
        model: Literal["heston","sabr","jump_diffusion"],
        option_type: Literal["call","put","asian","barrier"],
        n_paths: int = 100_000,
        n_steps: int = 252,
        variance_reduction: list[str] = ["antithetic","control_variate"],
    ) -> MCResult: ...
```

### 15. Finite Difference PDE Solver
**File:** `finite_difference.py`
- Crank-Nicolson scheme (implicit, unconditionally stable)
- American option via PSOR (Projected SOR)
- Grid: 200 space steps × 100 time steps (adjustable)
