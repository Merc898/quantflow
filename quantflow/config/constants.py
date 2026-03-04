"""Constants used throughout QuantFlow.

All magic numbers are centralized here for easy maintenance and consistency.
"""

from typing import Final

# =============================================================================
# Trading Calendar
# =============================================================================

TRADING_DAYS_PER_YEAR: Final[int] = 252
TRADING_DAYS_PER_MONTH: Final[int] = 21
TRADING_DAYS_PER_WEEK: Final[int] = 5
BUSINESS_DAYS_PER_MONTH: Final[int] = 22

# =============================================================================
# Lookback Periods (in trading days)
# =============================================================================

LOOKBACK_SHORT: Final[int] = 5  # 1 week
LOOKBACK_MEDIUM: Final[int] = 21  # 1 month
LOOKBACK_LONG: Final[int] = 63  # 3 months
LOOKBACK_6M: Final[int] = 126  # 6 months
LOOKBACK_1Y: Final[int] = 252  # 1 year
LOOKBACK_2Y: Final[int] = 504  # 2 years

# Feature engineering lookbacks
RETURN_LOOKBACKS: Final[tuple[int, ...]] = (1, 5, 21, 63, 252)
VOLATILITY_LOOKBACKS: Final[tuple[int, ...]] = (21, 63)
MOMENTUM_LOOKBACKS: Final[tuple[int, ...]] = (126, 252)  # 6M and 12M skipping 1M

# =============================================================================
# Signal Thresholds
# =============================================================================

SIGNAL_STRONG_BUY_THRESHOLD: Final[float] = 0.5
SIGNAL_BUY_THRESHOLD: Final[float] = 0.2
SIGNAL_WEAK_BUY_THRESHOLD: Final[float] = 0.05
SIGNAL_HOLD_RANGE: Final[tuple[float, float]] = (-0.05, 0.05)
SIGNAL_WEAK_SELL_THRESHOLD: Final[float] = -0.05
SIGNAL_SELL_THRESHOLD: Final[float] = -0.2
SIGNAL_STRONG_SELL_THRESHOLD: Final[float] = -0.5

# Confidence thresholds
CONFIDENCE_HIGH: Final[float] = 0.65
CONFIDENCE_MEDIUM: Final[float] = 0.50
CONFIDENCE_LOW: Final[float] = 0.40

# =============================================================================
# Risk Parameters
# =============================================================================

# VaR confidence levels
VAR_CONFIDENCE_LEVELS: Final[tuple[float, ...]] = (0.95, 0.99, 0.995)

# VaR horizons (in days)
VAR_HORIZONS: Final[tuple[int, ...]] = (1, 10, 21)

# Volatility regimes (percentile thresholds)
VOL_REGIME_LOW_THRESHOLD: Final[float] = 25.0  # Below 25th percentile
VOL_REGIME_HIGH_THRESHOLD: Final[float] = 75.0  # Above 75th percentile
VOL_REGIME_EXTREME_THRESHOLD: Final[float] = 95.0  # Above 95th percentile

# Position limits
MAX_POSITION_SIZE: Final[float] = 0.20  # 20% max per position
MAX_SECTOR_WEIGHT: Final[float] = 0.30  # 30% max sector weight
MIN_POSITION_SIZE: Final[float] = 0.01  # 1% minimum position

# Volatility targeting
TARGET_ANNUAL_VOLATILITY: Final[float] = 0.15  # 15% target vol

# Drawdown limits
MAX_DRAWDOWN_WARNING: Final[float] = 0.10  # 10% warning
MAX_DRAWDOWN_STOP: Final[float] = 0.15  # 15% stop trading

# =============================================================================
# Model Parameters
# =============================================================================

# GARCH
GARCH_P: Final[int] = 1
GARCH_Q: Final[int] = 1
GARCH_PERSISTENCE_WARNING: Final[float] = 0.99  # Warn if alpha + beta > 0.99

# ARIMA
ARIMA_MAX_P: Final[int] = 5
ARIMA_MAX_Q: Final[int] = 5
ARIMA_MAX_D: Final[int] = 2
ARIMA_SEASONAL_PERIOD: Final[int] = 5  # Weekly seasonality

# Walk-forward validation
WALK_FORWARD_N_SPLITS: Final[int] = 5
WALK_FORWARD_GAP: Final[int] = 21  # Days between train/test

# Feature standardization
FEATURE_WINSORIZE_LOWER: Final[float] = 0.01  # 1st percentile
FEATURE_WINSORIZE_UPPER: Final[float] = 0.99  # 99th percentile
FEATURE_ZSCORE_CLIP: Final[float] = 3.0  # Clip z-scores to [-3, 3]

# Information Coefficient
IC_MIN_THRESHOLD: Final[float] = 0.03  # Minimum acceptable IC
ICIR_MIN_THRESHOLD: Final[float] = 0.3  # Minimum acceptable ICIR

# =============================================================================
# Data Quality
# =============================================================================

# Missing data tolerance
MAX_MISSING_DATA_PCT: Final[float] = 0.05  # 5% max missing data
STALE_DATA_MULTIPLIER: Final[int] = 2  # Data stale if > 2x expected frequency

# OHLC validation
MIN_PRICE: Final[float] = 0.01  # Minimum valid price

# =============================================================================
# Backtesting
# =============================================================================

# Transaction costs
COMMISSION_PER_SHARE: Final[float] = 0.005  # $0.005/share (IB retail)
SLIPPAGE_BPS: Final[float] = 5.0  # 5 basis points
MARKET_IMPACT_COEFFICIENT: Final[float] = 10.0  # Impact coefficient

# Short selling
SHORT_BORROW_SPREAD_BPS: Final[float] = 50.0  # LIBOR + 50bps

# Capacity
MAX_ADV_PCT_PER_DAY: Final[float] = 0.02  # Max 2% of ADV per day

# =============================================================================
# Portfolio Optimization
# =============================================================================

# MVO
RISK_AVERSION_LAMBDA: Final[float] = 2.5  # Risk aversion parameter
MIN_VARIANCE_WEIGHT: Final[float] = 0.01  # Floor for weights
MAX_VARIANCE_WEIGHT: Final[float] = 0.30  # Cap for weights

# Black-Litterman
BL_TAU: Final[float] = 0.05  # Uncertainty scaling
BL_DELTA: Final[float] = 2.5  # Risk aversion

# HRP
HRP_LINKAGE_METHOD: Final[str] = "ward"  # Hierarchical clustering linkage

# =============================================================================
# Sentiment
# =============================================================================

SENTIMENT_EXTREME_BULL_THRESHOLD: Final[float] = 2.0  # Z-score > 2.0
SENTIMENT_EXTREME_BEAR_THRESHOLD: Final[float] = -2.0  # Z-score < -2.0
SENTIMENT_DECAY_FACTOR: Final[float] = 0.9  # Exponential decay per day

# =============================================================================
# Agent Intelligence
# =============================================================================

# Agent run intervals (in hours)
AGENT_INTERVAL_PREMIUM_HOURS: Final[int] = 4
AGENT_INTERVAL_FREE_HOURS: Final[int] = 24
AGENT_INTERVAL_EARNINGS_HOURS: Final[int] = 1

# Agent reliability weights (initial)
AGENT_WEIGHT_PERPLEXITY: Final[float] = 0.30
AGENT_WEIGHT_OPENAI: Final[float] = 0.30
AGENT_WEIGHT_ANTHROPIC: Final[float] = 0.25
AGENT_WEIGHT_SCRAPERS: Final[float] = 0.15

# Consensus thresholds
CONSENSUS_HIGH_THRESHOLD: Final[float] = 0.80  # >80% agreement
CONSENSUS_MEDIUM_THRESHOLD: Final[float] = 0.60  # 60-80% agreement
CONSENSUS_LOW_THRESHOLD: Final[float] = 0.60  # <60% agreement

# =============================================================================
# Cache TTLs (seconds)
# =============================================================================

CACHE_TTL_SHORT: Final[int] = 60  # 1 minute
CACHE_TTL_MEDIUM: Final[int] = 300  # 5 minutes
CACHE_TTL_LONG: Final[int] = 3600  # 1 hour
CACHE_TTL_DAILY: Final[int] = 86400  # 24 hours

# =============================================================================
# API Rate Limits (requests per day)
# =============================================================================

RATE_LIMIT_FREE: Final[int] = 100
RATE_LIMIT_PREMIUM: Final[int] = 1000
RATE_LIMIT_INSTITUTIONAL: Final[int] = 10000

# =============================================================================
# Historical Stress Periods
# =============================================================================

STRESS_PERIODS: Final[dict[str, tuple[str, str]]] = {
    "covid_crash_2020": ("2020-02-19", "2020-03-23"),
    "gfc_2008": ("2008-09-01", "2009-03-09"),
    "dot_com_bust": ("2000-03-10", "2002-10-09"),
    "black_monday_1987": ("1987-10-19", "1987-10-19"),
    "russia_ukraine_2022": ("2022-02-24", "2022-03-15"),
    "svb_crisis_2023": ("2023-03-08", "2023-03-17"),
}

# =============================================================================
# Subscription Tiers
# =============================================================================

TIER_FREE: Final[str] = "free"
TIER_PREMIUM: Final[str] = "premium"
TIER_INSTITUTIONAL: Final[str] = "institutional"

# Symbols limits per tier
TIER_SYMBOL_LIMITS: Final[dict[str, int]] = {
    TIER_FREE: 5,
    TIER_PREMIUM: 0,  # Unlimited
    TIER_INSTITUTIONAL: 0,  # Unlimited
}

# Signal refresh intervals per tier (hours)
TIER_REFRESH_INTERVALS: Final[dict[str, int]] = {
    TIER_FREE: 24,
    TIER_PREMIUM: 4,
    TIER_INSTITUTIONAL: 0,  # Real-time
}
