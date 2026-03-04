# Spec 02 — Statistical & Time-Series Models

All models live in `quantflow/models/statistical/`. Each is a class inheriting `BaseQuantModel`.

## 1. ARIMA / SARIMA
**File:** `arima.py`
```python
class ARIMAModel(BaseQuantModel):
    """
    Auto ARIMA with AIC-based order selection.
    Uses pmdarima for auto order selection.
    Fit on log returns. Forecast horizon: 1, 5, 21 days.
    Report: AIC, BIC, Ljung-Box p-value for residual autocorrelation.
    Output signal: sign(forecast_return) * |forecast_return| / forecast_std
    """
```
- Use `pmdarima.auto_arima` with `seasonal=True`
- Estimate out-of-sample RMSE via rolling 252-day expanding window
- Store residuals for diagnostic plots

## 2. GARCH Family
**File:** `garch.py`
```python
class GARCHModel(BaseQuantModel):
    """
    Implements: GARCH(1,1), EGARCH(1,1), TGARCH(1,1), FIGARCH(1,d,0)
    Library: arch (Kevin Sheppard)
    
    Fit on log returns. Output:
      - Conditional volatility forecast (annualized)
      - Volatility regime: LOW / MEDIUM / HIGH / EXTREME
        (percentile of forecast vol vs 252-day history)
      - Signal contribution: inverse vol weighting
    """
    model_type: Literal["GARCH","EGARCH","TGARCH","FIGARCH"]
```
- Always test for ARCH effects (Engle's LM test) before fitting
- Use `scipy.stats.jarque_bera` to check normality of standardized residuals
- If non-normal, use Student-t or GED distribution
- Report persistence = α + β; warn if > 0.99 (near unit root in variance)

## 3. VAR / VECM
**File:** `var_vecm.py`
```python
class VARModel(BaseQuantModel):
    """
    Vector Autoregression for multivariate return forecasting.
    Cointegration test (Johansen) → switch to VECM if cointegrated.
    
    Variables: target asset + macro factors + sector ETF returns
    Lag selection: AIC/BIC/HQIC, max_lags=10
    Output: 1-step and h-step return forecasts with prediction intervals
    """
```
- Granger causality tests: report which variables predict the target
- Impulse response functions stored for explainability
- Use `statsmodels.tsa.vector_ar`

## 4. State-Space / Kalman Filter
**File:** `kalman.py`
```python
class KalmanFilterModel(BaseQuantModel):
    """
    Applications:
    1. Dynamic beta estimation (time-varying market beta)
    2. Pairs trading spread estimation (cointegration coefficient)
    3. Trend extraction (local level model)
    4. Latent factor estimation
    
    Library: pykalman, or custom NumPy implementation for speed
    """
    application: Literal["beta","spread","trend","factor"]
```
- Use EM algorithm for parameter initialization
- Report filter uncertainty (diagonal of P matrix)
- Dynamic beta signal: if beta_t significantly different from 1, flag

## 5. PCA / Dynamic PCA Factor Models
**File:** `factor_pca.py`
```python
class PCARiskFactorModel(BaseQuantModel):
    """
    Steps:
    1. Compute rolling 252-day PCA on universe return matrix
    2. Extract top K factors (explain >80% variance)
    3. Project target returns onto factor space → get factor exposures
    4. Compute factor-model residual (idiosyncratic return)
    5. Signal: idiosyncratic momentum (alpha vs factors)
    
    Dynamic PCA: re-estimate factors every 21 days
    """
```
- Scree plot data saved for frontend visualization
- Barra-style factor attribution: % return explained per factor

## 6. Regime-Switching / Markov-Switching
**File:** `markov_switching.py`
```python
class MarkovSwitchingModel(BaseQuantModel):
    """
    2-regime and 3-regime Hidden Markov Models on returns.
    Library: hmmlearn, statsmodels.tsa.regime_switching
    
    Regimes: typically map to Bull/Bear or Low-vol/High-vol/Crisis
    Output:
      - Smoothed state probabilities P(regime_t | all data)
      - Regime-conditional expected return and volatility
      - Regime persistence (expected duration)
    Signal: weighted average of regime-conditional forecasts
    """
```
- Cross-validate with BIC for number of regimes (2 to 4)
- Regime labels: assign based on mean return and volatility
- Feed regime probabilities to Signal Fusion Engine as features

## 7. Stochastic Volatility Models
**File:** `stochastic_vol.py`
```python
class StochasticVolModel(BaseQuantModel):
    """
    Univariate SV via MCMC (pymc or numpyro).
    Multivariate SV for correlation estimation.
    
    Model: log(sigma_t^2) = mu + phi*(log(sigma_{t-1}^2) - mu) + eta_t
    Outputs:
      - Posterior mean and credible intervals for latent volatility
      - Leverage effect estimate (correlation between return and vol shocks)
    """
```
- Use No-U-Turn Sampler (NUTS) via `numpyro` for speed
- Report Gelman-Rubin R-hat < 1.01 as convergence criterion

## 8. Dynamic Factor Models (DFM)
**File:** `dynamic_factor.py`
```python
class DynamicFactorModel(BaseQuantModel):
    """
    State-space DFM estimated via MLE (EM algorithm).
    Common factors extracted from large cross-section of returns.
    
    Use case: nowcasting macro regime from cross-asset data.
    Library: statsmodels.tsa.statespace.dynamic_factor_mq
    """
```

## Shared Requirements for All Statistical Models
- Rolling out-of-sample evaluation: always use `TimeSeriesSplit(n_splits=5)`
- Store: fitted parameters, residuals, information criteria, out-of-sample metrics
- Every model outputs a `ModelOutput` Pydantic object:
```python
class ModelOutput(BaseModel):
    model_name: str
    symbol: str
    timestamp: datetime
    signal: float           # normalized [-1, +1]
    confidence: float       # [0, 1]
    forecast_return: float  # point estimate
    forecast_std: float     # uncertainty
    regime: str | None
    metadata: dict          # model-specific diagnostics
```
