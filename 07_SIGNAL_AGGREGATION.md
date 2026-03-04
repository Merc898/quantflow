# Spec 07 — Signal Fusion & Final Recommendation Engine

## Overview
The Signal Fusion Engine combines 50+ model outputs into a single  
risk-adjusted Buy/Hold/Sell recommendation with full explainability.

**File:** `quantflow/signals/`

## 1. Signal Normalization
**File:** `normalizer.py`
```python
class SignalNormalizer:
    """
    Before fusion, ALL signals must be normalized to a common scale.
    
    Steps per signal:
    1. Winsorize at [1%, 99%] to clip outliers
    2. Cross-sectional z-score (if multi-asset):
       z_i = (signal_i - mean_universe) / std_universe
    3. Time-series z-score (rolling 252-day window)
    4. Clip final z-score to [-3, +3]
    5. Optionally apply isotonic regression for monotonicity
    """
```

## 2. Model Weight Calibration
**File:** `calibrator.py`
```python
class DynamicWeightCalibrator:
    """
    Weights assigned to each model based on recent performance.
    
    Metric: Information Coefficient (IC) over trailing 63 days
    IC = Spearman rank correlation(signal_t, return_{t+21})
    
    Weight formula:
    w_i = max(IC_i, 0) / sum(max(IC_j, 0) for all j)
    
    Special cases:
    - If all IC <= 0: use equal weights (all models struggling)
    - Minimum weight floor: 1% (keep all models alive)
    - Maximum weight cap: 30% (no single model dominates)
    
    Regime-conditional weights:
    - In each regime (from Markov model), maintain separate weight vector
    - Rationale: LSTM may outperform in trending markets, 
                 mean-reversion models in choppy markets
    
    Update frequency: daily (lightweight operation)
    """
    
    def compute_weights(
        self,
        model_signals: dict[str, pd.Series],
        realized_returns: pd.Series,
        regime: str,
    ) -> dict[str, float]: ...
```

## 3. Ensemble Aggregation
**File:** `aggregator.py`
```python
class EnsembleAggregator:
    """
    Aggregate all model signals into composite signal.
    
    Two-stage aggregation:
    
    STAGE 1: Within-category aggregation
    Group models by type, aggregate within each group:
      - Statistical models → stat_composite (mean of ARIMA, GARCH, VAR, Kalman signals)
      - ML models → ml_composite
      - Sentiment → sentiment_composite  
      - Risk models → risk_composite (as penalty/scaling factor, not additive)
    
    STAGE 2: Cross-category aggregation
    Weighted combination of group composites:
      composite = w_stat * stat_composite
                + w_ml * ml_composite
                + w_sentiment * sentiment_composite
    
    Risk scaling (applied after aggregation):
      final_signal = composite * (target_vol / current_vol)
      where target_vol = 15% annualized (volatility targeting)
    
    Default category weights (dynamic, updated by calibrator):
      w_stat = 0.35, w_ml = 0.45, w_sentiment = 0.20
    """
```

## 4. Regime Detection
**File:** `regime_detector.py`
```python
class RegimeDetector:
    """
    Synthesizes regime from multiple sources:
    
    Inputs:
    - Markov switching model: state probabilities
    - GARCH volatility regime: LOW/MEDIUM/HIGH/EXTREME
    - VIX level: <15 (calm), 15-25 (elevated), >25 (stress), >35 (crisis)
    - Yield curve: normal / flat / inverted
    - Trend: 200-day SMA position (above = bull, below = bear)
    
    Output regime (composite of inputs):
    {
      "volatility_regime": "LOW" | "MEDIUM" | "HIGH" | "CRISIS",
      "trend_regime": "BULL" | "BEAR" | "SIDEWAYS",
      "macro_regime": "EXPANSION" | "LATE_CYCLE" | "RECESSION" | "RECOVERY",
      "overall_regime": str,  # combination label
      "regime_confidence": float,
      "regime_duration_days": int,  # how long current regime active
    }
    
    Model weights adjusted based on which models historically 
    perform best in current regime.
    """
```

## 5. Final Recommendation Generator
**File:** `recommendation.py`
```python
class RecommendationEngine:
    """
    Converts composite signal to actionable recommendation.
    
    SIGNAL → RECOMMENDATION MAPPING:
    signal > +0.5  AND confidence > 0.65 → STRONG BUY  ★★★
    signal > +0.2  AND confidence > 0.50 → BUY         ★★
    signal > +0.05 AND confidence > 0.40 → WEAK BUY    ★
    signal in [-0.05, +0.05]             → HOLD         ─
    signal < -0.05 AND confidence > 0.40 → WEAK SELL   ↓
    signal < -0.2  AND confidence > 0.50 → SELL        ↓↓
    signal < -0.5  AND confidence > 0.65 → STRONG SELL ↓↓↓
    
    Position sizing (Kelly-inspired):
    position_size = (signal * confidence) * (target_vol / realized_vol)
    position_size = clip(position_size, -max_pos, +max_pos)
    max_pos = 0.20 (20% of portfolio per position)
    
    Additional risk gates (override to HOLD if any triggered):
    - realized_vol > 3 * 252d_average_vol (vol spike)
    - max_drawdown_30d > 15% (drawdown protection)
    - low_liquidity: dollar_volume_21d < threshold
    - earnings within 3 days AND we have no earnings model
    
    Output: FinalRecommendation Pydantic model
    """
    
    async def generate(
        self,
        symbol: str,
        composite_signal: float,
        model_outputs: list[ModelOutput],
        intelligence_report: IntelligenceReport,
        risk_report: RiskReport,
        regime: RegimeState,
    ) -> FinalRecommendation: ...
```

## 6. Explainability
**File:** `explainer.py`
```python
class RecommendationExplainer:
    """
    Generates human-readable explanation for every recommendation.
    
    Components:
    1. Model contribution chart:
       Each model's % contribution to final signal
    
    2. SHAP waterfall chart:
       Top 10 features driving GBT prediction
    
    3. Natural language narrative (generated by Claude):
       "AAPL is rated BUY primarily due to:
        • Strong momentum signal (12-1 month return: +23%)
        • Positive earnings sentiment from recent 10-Q analysis
        • Low volatility regime supports equity exposure
        Risk factors: elevated P/E vs sector median, upcoming Fed decision"
    
    4. Confidence decomposition:
       - Model agreement score: 0.72 (72% of models agree on direction)
       - Data quality score: 0.95 (nearly complete data)
       - Regime clarity: 0.80 (clear bull regime)
    """
```

## Data Schema
```python
class FinalRecommendation(BaseModel):
    symbol: str
    timestamp: datetime
    recommendation: Literal["STRONG_BUY","BUY","WEAK_BUY","HOLD","WEAK_SELL","SELL","STRONG_SELL"]
    signal_strength: float          # [-1, +1] composite signal
    confidence: float               # [0, 1]
    suggested_position_size: float  # [-1, +1] as fraction of max position
    
    # Risk metrics
    expected_return_21d: float
    expected_vol_21d: float
    var_95_1d: float
    max_drawdown_estimate: float
    
    # Attribution
    model_contributions: dict[str, float]  # model_name → contribution
    top_bullish_factors: list[str]
    top_bearish_factors: list[str]
    regime: RegimeState
    
    # Narrative
    rationale: str                  # LLM-generated explanation
    risk_warnings: list[str]        # ["High vol regime", "Earnings in 2 days"]
    
    # Meta
    data_quality_score: float
    models_used: int
    models_available: int
```
