# Spec 01 — Data Pipeline

## Data Sources

### Market Data (Priority 1 — always fetch)
| Source | Data Type | Library | Frequency |
|--------|-----------|---------|-----------|
| Polygon.io | OHLCV, tick, order book | `polygon-api-client` | Tick / 1min |
| Alpha Vantage | OHLCV, fundamentals | `alpha_vantage` | Daily |
| yFinance | OHLCV, options chain | `yfinance` | Daily fallback |
| FRED | Macro: rates, CPI, GDP | `fredapi` | Weekly/Monthly |
| Quandl/Nasdaq Data Link | Futures, COT reports | `quandl` | Daily |

### Alternative Data (Priority 2 — async enrichment)
| Source | Data Type | Method |
|--------|-----------|--------|
| SEC EDGAR | 10-K/10-Q filings | `httpx` + parser |
| Reddit (WSB, investing) | Retail sentiment | `asyncpraw` |
| Twitter/X | Market mentions | API v2 |
| Google Trends | Search interest | `pytrends` |
| News APIs | Headlines | NewsAPI, RSS |

## `BaseDataFetcher` Interface
```python
class BaseDataFetcher(ABC):
    def __init__(self, config: FetcherConfig): ...
    
    @abstractmethod
    async def fetch_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: Literal["1m","5m","1h","1d"],
    ) -> pd.DataFrame: ...
    # Returns: DatetimeIndex, columns=[open,high,low,close,volume,vwap]
    # MUST be timezone-aware (UTC)
    # MUST handle missing bars (forward-fill with flag column)
    
    @abstractmethod
    async def fetch_fundamentals(self, symbol: str) -> FundamentalsData: ...
    
    async def validate(self, df: pd.DataFrame) -> ValidationReport: ...
    # Check: no future dates, OHLC consistency (H>=L, H>=O, H>=C),
    #        volume >= 0, no duplicate timestamps, stationarity flags
```

## Data Processing Pipeline
```python
# quantflow/data/processors/pipeline.py

async def run_pipeline(symbol: str, config: PipelineConfig) -> ProcessedData:
    # Step 1: Fetch raw data from all sources concurrently
    raw = await asyncio.gather(
        polygon_fetcher.fetch_ohlcv(symbol, ...),
        av_fetcher.fetch_fundamentals(symbol),
        fred_fetcher.fetch_macro(...),
    )
    
    # Step 2: Corporate actions adjustment (splits, dividends)
    adjusted = apply_corporate_actions(raw.ohlcv, raw.corporate_actions)
    
    # Step 3: Outlier detection and cleaning
    cleaned = remove_outliers_iqr(adjusted, threshold=5.0)
    
    # Step 4: Feature engineering
    features = await feature_engineer.transform(cleaned)
    
    # Step 5: Store to TimescaleDB + Redis cache
    await store(features, ttl_seconds=300)
    
    return features
```

## Feature Engineering — Required Features

### Price-Based
```python
# Returns
"ret_1d", "ret_5d", "ret_21d", "ret_63d", "ret_252d"
"log_ret_1d"  # log(close_t / close_{t-1})

# Momentum
"mom_12_1"    # 12-month return skipping last month
"mom_6_1"     # 6-month return skipping last month

# Reversal
"rev_1m"      # short-term reversal (1-month)

# Volatility
"realized_vol_21d"  # 21-day realized vol (annualized)
"realized_vol_63d"
"garch_vol"         # GARCH(1,1) conditional vol

# Volume
"dollar_volume_21d"
"amihud_illiquidity"  # |ret| / dollar_volume
"volume_z_score"      # Z-score vs 63-day mean

# Technical
"rsi_14", "rsi_28"
"macd_signal", "macd_hist"
"bb_width", "bb_position"  # Bollinger bands
"atr_14"  # Average True Range
```

### Fundamental (if available)
```python
"pe_ratio", "pb_ratio", "ps_ratio", "ev_ebitda"
"roe", "roa", "gross_margin", "fcf_yield"
"debt_equity", "current_ratio"
"earnings_surprise_1q", "earnings_surprise_4q_avg"
"revenue_growth_yoy", "earnings_growth_yoy"
```

### Macro
```python
"vix_level", "vix_change_5d"
"yield_10y", "yield_2y", "yield_spread_10_2"
"fed_funds_rate"
"cpi_yoy", "pce_yoy"
"ism_manufacturing", "ism_services"
```

## Storage Schema (TimescaleDB)
```sql
-- Hypertable partitioned by time
CREATE TABLE market_data (
    time        TIMESTAMPTZ NOT NULL,
    symbol      TEXT NOT NULL,
    open        DOUBLE PRECISION,
    high        DOUBLE PRECISION,
    low         DOUBLE PRECISION,
    close       DOUBLE PRECISION,
    volume      BIGINT,
    vwap        DOUBLE PRECISION,
    adjusted    BOOLEAN DEFAULT FALSE,
    source      TEXT
);
SELECT create_hypertable('market_data', 'time');
CREATE INDEX ON market_data (symbol, time DESC);

CREATE TABLE features (
    time        TIMESTAMPTZ NOT NULL,
    symbol      TEXT NOT NULL,
    feature     TEXT NOT NULL,
    value       DOUBLE PRECISION,
    version     INTEGER DEFAULT 1
);
SELECT create_hypertable('features', 'time');
```

## Data Quality Rules
- Missing data >5% in any window: raise `DataQualityError`, do not proceed with modeling
- Stale data (last update >2x expected frequency): log warning, use cached value
- All prices must be split/dividend adjusted
- All features must be winsorized at 1st/99th percentile before model input
- Features must be standardized (z-score) per rolling 252-day window
