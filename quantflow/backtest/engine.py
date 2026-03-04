"""Walk-forward backtesting engine with realistic transaction costs.

Implements the backtesting framework specified in Spec 09:
- Zero look-ahead bias: all signals use only data up to the rebalancing date
- Realistic transaction costs: commission, slippage, market impact
- Full PerformanceMetrics suite as specified
- Stress period analysis for GFC, COVID, rate-hike regimes
- Monthly (or configurable) rebalancing
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats

from quantflow.config.logging import get_logger
from quantflow.portfolio.hrp import HRPOptimizer

if TYPE_CHECKING:
    from datetime import date

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Named stress periods (start, end — inclusive)
# ---------------------------------------------------------------------------

_STRESS_PERIODS: dict[str, tuple[str, str]] = {
    "COVID_2020": ("2020-02-01", "2020-06-30"),
    "RateHike_2022": ("2022-01-01", "2022-12-31"),
    "GFC_2008": ("2008-09-01", "2009-03-31"),
    "DotCom_2000": ("2000-03-01", "2002-10-31"),
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class BacktestConfig(BaseModel):
    """Configuration for a backtesting run.

    Attributes:
        start_date: First date on which positions can be held.
        end_date: Last date in the backtest (inclusive).
        initial_capital: Starting portfolio value in USD.
        rebalance_freq: Pandas offset alias for rebalancing frequency
            (e.g. ``"ME"`` for month-end, ``"QE"`` for quarter-end).
        max_position_size: Maximum weight for any single asset.
        min_position_size: Minimum weight to hold (smaller allocations are
            dropped and the portfolio re-normalised).
        commission_per_share: IB-style commission in USD per share.
        slippage_bps: Half-spread execution slippage in basis points.
        allow_short: Whether the strategy may take short positions.
        benchmark_symbol: Ticker used as the benchmark (display only; the
            engine accepts a benchmark price series separately).
        signal_horizon_days: Forward-return horizon (trading days) used
            when computing IC from signals.
    """

    start_date: date
    end_date: date
    initial_capital: float = Field(default=1_000_000.0, gt=0)
    rebalance_freq: str = "ME"
    max_position_size: float = Field(default=0.15, gt=0, le=1.0)
    min_position_size: float = Field(default=0.005, ge=0)
    commission_per_share: float = Field(default=0.005, ge=0)
    slippage_bps: float = Field(default=5.0, ge=0)
    allow_short: bool = False
    benchmark_symbol: str = "SPY"
    signal_horizon_days: int = Field(default=21, ge=1)


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------


class Trade(BaseModel):
    """Record of a single executed trade.

    Attributes:
        date: Execution date.
        symbol: Ticker symbol.
        shares: Signed share quantity (positive = buy, negative = sell/short).
        price: Execution price per share.
        side: ``"BUY"`` or ``"SELL"``.
        commission: Brokerage commission in USD.
        slippage_cost: Execution slippage cost in USD.
        market_impact_cost: Market impact cost in USD.
    """

    date: date
    symbol: str
    shares: float
    price: float
    side: str
    commission: float = 0.0
    slippage_cost: float = 0.0
    market_impact_cost: float = 0.0

    @property
    def total_cost(self) -> float:
        """Total execution cost in USD."""
        return self.commission + self.slippage_cost + self.market_impact_cost

    @property
    def trade_value(self) -> float:
        """Absolute notional value of the trade."""
        return abs(self.shares * self.price)


# ---------------------------------------------------------------------------
# Performance metrics (Spec 09)
# ---------------------------------------------------------------------------


class PerformanceMetrics(BaseModel):
    """Full performance metric suite as specified in Spec 09.

    All return and ratio fields are annualised unless otherwise noted.
    """

    # ---- Returns ----
    total_return: float = 0.0
    annualized_return: float = 0.0
    cagr: float = 0.0

    # ---- Risk-adjusted ----
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0

    # ---- Risk ----
    annualized_vol: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration_days: int = 0
    var_95_1d: float = 0.0
    cvar_95_1d: float = 0.0

    # ---- Signal quality ----
    ic_mean: float = 0.0
    ic_std: float = 0.0
    icir: float = 0.0
    hit_rate: float = 0.0

    # ---- Turnover & costs ----
    annual_turnover: float = 0.0
    total_costs_bps: float = 0.0
    gross_vs_net_sharpe: float = 0.0

    # ---- Benchmark comparison ----
    benchmark_return: float = 0.0
    alpha: float = 0.0
    beta: float = 1.0
    tracking_error: float = 0.0

    # ---- Statistical significance ----
    sharpe_t_stat: float = 0.0
    ic_t_stat: float = 0.0


# ---------------------------------------------------------------------------
# Backtest result container
# ---------------------------------------------------------------------------


class BacktestResult(BaseModel):
    """Full backtest result including equity curve, trades, and metrics.

    Attributes:
        config: The configuration used for this run.
        metrics: Full :class:`PerformanceMetrics` for the entire period.
        equity_curve: Daily net portfolio values.
        equity_dates: ISO date strings aligned with ``equity_curve``.
        benchmark_curve: Daily benchmark portfolio values (rebased).
        benchmark_dates: ISO date strings aligned with ``benchmark_curve``.
        gross_equity_curve: Daily gross (pre-cost) portfolio values.
        daily_returns: Daily net portfolio returns.
        trades: All executed trades.
        ic_series: IC per rebalancing period.
        ic_dates: ISO dates aligned with ``ic_series``.
        stress_periods: :class:`PerformanceMetrics` for each stress period.
    """

    config: BacktestConfig
    metrics: PerformanceMetrics
    equity_curve: list[float] = Field(default_factory=list)
    equity_dates: list[str] = Field(default_factory=list)
    benchmark_curve: list[float] = Field(default_factory=list)
    benchmark_dates: list[str] = Field(default_factory=list)
    gross_equity_curve: list[float] = Field(default_factory=list)
    daily_returns: list[float] = Field(default_factory=list)
    trades: list[Trade] = Field(default_factory=list)
    ic_series: list[float] = Field(default_factory=list)
    ic_dates: list[str] = Field(default_factory=list)
    stress_periods: dict[str, PerformanceMetrics] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    def equity_series(self) -> pd.Series:
        """Return the net equity curve as a :class:`pandas.Series`."""
        return pd.Series(
            self.equity_curve,
            index=pd.to_datetime(self.equity_dates),
            name="equity",
        )

    def benchmark_series(self) -> pd.Series:
        """Return the benchmark curve as a :class:`pandas.Series`."""
        return pd.Series(
            self.benchmark_curve,
            index=pd.to_datetime(self.benchmark_dates),
            name="benchmark",
        )

    def returns_series(self) -> pd.Series:
        """Return daily net returns as a :class:`pandas.Series`."""
        return pd.Series(
            self.daily_returns,
            index=pd.to_datetime(self.equity_dates[1:] if len(self.equity_dates) > 1 else []),
            name="daily_returns",
        )


# ---------------------------------------------------------------------------
# Abstract base strategy
# ---------------------------------------------------------------------------


class BaseStrategy(ABC):
    """Abstract base class for all backtest strategies.

    Subclasses implement :meth:`generate_signals` and :meth:`compute_weights`.
    Both methods must use only data that was available at ``as_of``; any
    forward-looking usage constitutes look-ahead bias.
    """

    @abstractmethod
    def generate_signals(
        self,
        prices: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> dict[str, float]:
        """Generate signals for all symbols using data up to and including ``as_of``.

        Args:
            prices: Close-price DataFrame with DatetimeIndex.
                    Contains ONLY data up to ``as_of``.
            as_of: The date signals are generated for.

        Returns:
            Dict mapping symbol → signal in ``[-1, +1]``.
            Positive values indicate a bullish view.
        """
        ...

    @abstractmethod
    def compute_weights(
        self,
        signals: dict[str, float],
        returns: pd.DataFrame,
    ) -> dict[str, float]:
        """Convert signals into portfolio weights.

        Args:
            signals: Symbol → signal map produced by :meth:`generate_signals`.
            returns: Daily returns DataFrame (columns = symbols), historical
                     data only (no look-ahead).

        Returns:
            Dict mapping symbol → portfolio weight.
            Long-only strategies return weights in ``[0, 1]`` summing to ≤ 1.
        """
        ...


# ---------------------------------------------------------------------------
# QuantFlow default strategy
# ---------------------------------------------------------------------------


class QuantFlowStrategy(BaseStrategy):
    """Price-based momentum + mean-reversion strategy with HRP allocation.

    Signal construction (cross-sectional, z-scored):
    - **12M-1M momentum** — Jegadeesh & Titman (1993) skip-one-month momentum.
    - **Short-term mean-reversion** — 5-day vs 20-day moving-average ratio.
    - **Low-volatility tilt** — inverse realised annualised volatility signal.

    Portfolio construction:
    - Hierarchical Risk Parity (HRP) over long-signal symbols.
    - Fallback to equal-weight if HRP fails (too few observations).

    Args:
        mom_lookback: Lookback window for momentum in trading days.
        mom_skip: Skip period (exclude most recent days from momentum).
        rev_fast: Fast moving-average window for mean-reversion.
        rev_slow: Slow moving-average window for mean-reversion.
        vol_window: Window for realised volatility estimation.
        mom_weight: Weight of momentum sub-signal.
        rev_weight: Weight of mean-reversion sub-signal.
        vol_weight: Weight of low-volatility sub-signal.
    """

    def __init__(
        self,
        mom_lookback: int = 252,
        mom_skip: int = 21,
        rev_fast: int = 5,
        rev_slow: int = 21,
        vol_window: int = 63,
        mom_weight: float = 0.60,
        rev_weight: float = 0.20,
        vol_weight: float = 0.20,
    ) -> None:
        self._mom_lookback = mom_lookback
        self._mom_skip = mom_skip
        self._rev_fast = rev_fast
        self._rev_slow = rev_slow
        self._vol_window = vol_window
        self._mom_w = mom_weight
        self._rev_w = rev_weight
        self._vol_w = vol_weight
        self._hrp = HRPOptimizer()
        self._logger = get_logger(__name__)

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        prices: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> dict[str, float]:
        """Generate cross-sectional momentum + mean-reversion signals.

        Args:
            prices: Close prices up to and including ``as_of``.
            as_of: Signal generation date.

        Returns:
            Symbol → signal in ``[-1, +1]`` (cross-sectionally z-scored).
        """
        min_rows = self._mom_lookback + self._mom_skip + self._rev_slow + 5
        if len(prices) < min_rows:
            return dict.fromkeys(prices.columns, 0.0)

        raw_signals: dict[str, float] = {}
        for sym in prices.columns:
            close = prices[sym].dropna()
            if len(close) < min_rows:
                raw_signals[sym] = 0.0
                continue
            try:
                raw_signals[sym] = self._symbol_signal(close)
            except Exception:
                raw_signals[sym] = 0.0

        return self._cross_sectional_zscore(raw_signals)

    def _symbol_signal(self, close: pd.Series) -> float:
        """Compute composite signal for a single symbol.

        Args:
            close: Price series (already trimmed to valid observations).

        Returns:
            Raw composite signal (before cross-sectional normalisation).
        """
        # 12M-1M momentum
        p_skip = float(close.iloc[-self._mom_skip])
        p_base = float(close.iloc[-self._mom_lookback - self._mom_skip])
        mom = float(np.clip((p_skip / p_base - 1.0) / 0.30, -1.0, 1.0))

        # Short-term mean-reversion (fast > slow → overbought → negative)
        fast_ma = float(close.iloc[-self._rev_fast :].mean())
        slow_ma = float(close.iloc[-self._rev_slow :].mean())
        if slow_ma < 1e-8:
            rev = 0.0
        else:
            rev = float(np.clip(-(fast_ma / slow_ma - 1.0) / 0.05, -1.0, 1.0))

        # Low-vol signal: realised < 20% target → positive
        vol_returns = close.pct_change().dropna().iloc[-self._vol_window :]
        realized_vol = float(vol_returns.std() * np.sqrt(252)) if len(vol_returns) > 5 else 0.20
        vol_sig = float(np.clip((0.20 - realized_vol) / 0.20, -1.0, 1.0))

        return self._mom_w * mom + self._rev_w * rev + self._vol_w * vol_sig

    # ------------------------------------------------------------------
    # Portfolio construction
    # ------------------------------------------------------------------

    def compute_weights(
        self,
        signals: dict[str, float],
        returns: pd.DataFrame,
    ) -> dict[str, float]:
        """Compute HRP weights over long-signal symbols.

        Args:
            signals: Symbol → signal map.
            returns: Daily returns (historical only).

        Returns:
            Symbol → portfolio weight (long-only, sum ≤ 1).
        """
        long_syms = [s for s, v in signals.items() if v > 0.0 and s in returns.columns]
        if len(long_syms) < 2:
            n = len(signals)
            return dict.fromkeys(signals, 1.0 / n) if n > 0 else {}

        try:
            ret_sub = returns[long_syms].dropna(how="all")
            if len(ret_sub) < 30:
                raise ValueError("Insufficient data for HRP")
            result = self._hrp.optimize(ret_sub)
            weights = result.weights
            # Zero out any symbol not in the long set
            return {s: float(weights.get(s, 0.0)) for s in signals}
        except Exception:
            n = len(long_syms)
            return {s: (1.0 / n if s in long_syms else 0.0) for s in signals}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cross_sectional_zscore(signals: dict[str, float]) -> dict[str, float]:
        """Z-score signals cross-sectionally and soft-clip via tanh.

        Args:
            signals: Raw symbol → signal map.

        Returns:
            Normalised symbol → signal map in ``[-1, +1]``.
        """
        keys = list(signals.keys())
        vals = np.array([signals[k] for k in keys], dtype=np.float64)
        if len(vals) < 2:
            return signals
        std = float(vals.std())
        if std < 1e-8:
            return dict.fromkeys(keys, 0.0)
        z = (vals - vals.mean()) / std
        z_clipped = np.tanh(z * 0.5)
        return {k: float(z_clipped[i]) for i, k in enumerate(keys)}


# ---------------------------------------------------------------------------
# Transaction cost model
# ---------------------------------------------------------------------------


def _compute_trade(
    symbol: str,
    shares: float,
    price: float,
    adv_value: float,
    realized_vol: float,
    commission_per_share: float,
    slippage_bps: float,
    trade_date: date,
) -> Trade:
    """Build a :class:`Trade` with all cost components filled in.

    Transaction cost model (from Spec 09):
    - **Commission**: ``$0.005 × |shares|``
    - **Slippage**: ``5 bps × |trade_value|``
    - **Market impact**: ``10 × sqrt(|trade_value| / adv) × realized_vol`` bps
      (capped at 5% of trade value)

    Args:
        symbol: Ticker.
        shares: Signed share quantity.
        price: Execution price.
        adv_value: 21-day average daily dollar volume estimate.
        realized_vol: Annualised realised volatility.
        commission_per_share: Per-share commission rate.
        slippage_bps: Slippage in basis points.
        trade_date: Settlement date.

    Returns:
        Populated :class:`Trade`.
    """
    tv = abs(shares * price)
    commission = abs(shares) * commission_per_share
    slippage_cost = tv * slippage_bps / 10_000.0
    if adv_value > 1e-6:
        impact_bps = 10.0 * np.sqrt(tv / adv_value) * realized_vol
        market_impact = tv * min(impact_bps / 10_000.0, 0.05)
    else:
        market_impact = tv * 0.001  # 10 bps fallback

    return Trade(
        date=trade_date,
        symbol=symbol,
        shares=shares,
        price=price,
        side="BUY" if shares > 0 else "SELL",
        commission=commission,
        slippage_cost=slippage_cost,
        market_impact_cost=market_impact,
    )


# ---------------------------------------------------------------------------
# Drawdown helpers
# ---------------------------------------------------------------------------


def _max_drawdown_and_duration(returns: pd.Series) -> tuple[float, int]:
    """Compute maximum drawdown and its duration in calendar days.

    Args:
        returns: Daily return series.

    Returns:
        Tuple of (max_drawdown, duration_days).
        ``max_drawdown`` is negative (e.g. ``-0.15`` = 15% drawdown).
    """
    cum = (1.0 + returns.fillna(0.0)).cumprod()
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max.clip(lower=1e-8)
    max_dd = float(dd.min())

    if abs(max_dd) < 1e-8 or len(dd) < 2:
        return max_dd, 0

    trough_loc = int(dd.values.argmin())
    peak_loc = int(cum.iloc[: trough_loc + 1].values.argmax())
    peak_ts = dd.index[peak_loc]
    trough_ts = dd.index[trough_loc]
    duration = int((trough_ts - peak_ts).days) if hasattr(trough_ts - peak_ts, "days") else 0
    return max_dd, max(0, duration)


# ---------------------------------------------------------------------------
# Performance metric computation
# ---------------------------------------------------------------------------


def compute_performance_metrics(
    net_returns: pd.Series,
    gross_returns: pd.Series,
    benchmark_returns: pd.Series,
    trades: list[Trade],
    ic_values: list[float],
    initial_capital: float,
    n_years: float,
) -> PerformanceMetrics:
    """Compute all :class:`PerformanceMetrics` from daily return series.

    Args:
        net_returns: Daily net-of-cost portfolio returns.
        gross_returns: Daily gross (pre-cost) portfolio returns.
        benchmark_returns: Daily benchmark returns (aligned to net_returns).
        trades: All executed trades.
        ic_values: IC per rebalancing period.
        initial_capital: Starting capital.
        n_years: Backtest duration in years.

    Returns:
        Fully populated :class:`PerformanceMetrics`.
    """
    n = len(net_returns)
    if n < 2:
        return PerformanceMetrics()

    # --- Returns ---
    total_return = float((1.0 + net_returns).prod() - 1.0)
    cagr = float((1.0 + total_return) ** (1.0 / max(n_years, 0.01)) - 1.0)
    ann_vol = float(net_returns.std() * np.sqrt(252))

    # --- Risk-adjusted ---
    sharpe = float(cagr / ann_vol) if ann_vol > 1e-8 else 0.0
    neg = net_returns[net_returns < 0]
    downside_std = float(neg.std() * np.sqrt(252)) if len(neg) > 1 else ann_vol
    sortino = float(cagr / downside_std) if downside_std > 1e-8 else 0.0
    max_dd, dd_days = _max_drawdown_and_duration(net_returns)
    calmar = float(cagr / abs(max_dd)) if abs(max_dd) > 1e-8 else 0.0

    # --- VaR / CVaR ---
    q = float(np.percentile(net_returns.dropna(), 5))
    tail = net_returns[net_returns <= q]
    cvar = float(tail.mean()) if len(tail) > 0 else q

    # --- Benchmark ---
    bench_total = float((1.0 + benchmark_returns).prod() - 1.0)
    bench_cagr = float((1.0 + bench_total) ** (1.0 / max(n_years, 0.01)) - 1.0)
    aligned = pd.DataFrame({"p": net_returns, "b": benchmark_returns}).dropna()
    tracking_error = info_ratio = alpha = 0.0
    beta = 1.0
    if len(aligned) > 20:
        excess = aligned["p"] - aligned["b"]
        te = float(excess.std() * np.sqrt(252))
        tracking_error = te
        info_ratio = (
            float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 1e-8 else 0.0
        )
        try:
            slope, intercept, *_ = stats.linregress(aligned["b"].values, aligned["p"].values)
            beta = float(slope)
            alpha = float(intercept * 252)
        except Exception:
            pass

    # --- Gross vs net Sharpe ---
    gross_total = float((1.0 + gross_returns).prod() - 1.0)
    gross_cagr = float((1.0 + gross_total) ** (1.0 / max(n_years, 0.01)) - 1.0)
    gross_vol = float(gross_returns.std() * np.sqrt(252))
    gross_sharpe = float(gross_cagr / gross_vol) if gross_vol > 1e-8 else 0.0
    cost_drag = gross_sharpe - sharpe

    # --- Costs ---
    total_cost_usd = sum(t.total_cost for t in trades)
    total_costs_bps = total_cost_usd / (initial_capital * max(n_years, 0.01)) * 10_000.0

    # --- Annual turnover ---
    bought = sum(t.trade_value for t in trades if t.side == "BUY")
    annual_turnover = bought / (initial_capital * max(n_years, 0.01))

    # --- Signal quality ---
    ic_arr = np.array([v for v in ic_values if np.isfinite(v)], dtype=np.float64)
    ic_mean = float(ic_arr.mean()) if len(ic_arr) > 0 else 0.0
    ic_std = float(ic_arr.std()) if len(ic_arr) > 1 else 0.0
    icir = float(ic_mean / ic_std) if ic_std > 1e-8 else 0.0
    hit_rate = float((net_returns > 0).mean()) if n > 0 else 0.0

    # --- t-statistics ---
    sharpe_t = sharpe * np.sqrt(n / 252.0)
    ic_t = (ic_mean / ic_std * np.sqrt(len(ic_arr))) if (ic_std > 1e-8 and len(ic_arr) > 1) else 0.0

    return PerformanceMetrics(
        total_return=round(total_return, 6),
        annualized_return=round(cagr, 6),
        cagr=round(cagr, 6),
        sharpe_ratio=round(sharpe, 4),
        sortino_ratio=round(sortino, 4),
        calmar_ratio=round(calmar, 4),
        information_ratio=round(info_ratio, 4),
        annualized_vol=round(ann_vol, 6),
        max_drawdown=round(max_dd, 6),
        max_drawdown_duration_days=dd_days,
        var_95_1d=round(q, 6),
        cvar_95_1d=round(cvar, 6),
        ic_mean=round(ic_mean, 6),
        ic_std=round(ic_std, 6),
        icir=round(icir, 4),
        hit_rate=round(hit_rate, 4),
        annual_turnover=round(annual_turnover, 4),
        total_costs_bps=round(total_costs_bps, 2),
        gross_vs_net_sharpe=round(cost_drag, 4),
        benchmark_return=round(bench_cagr, 6),
        alpha=round(alpha, 6),
        beta=round(beta, 4),
        tracking_error=round(tracking_error, 6),
        sharpe_t_stat=round(sharpe_t, 4),
        ic_t_stat=round(ic_t, 4),
    )


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------


class BacktestEngine:
    """Walk-forward backtesting engine.

    Executes a strategy over historical price data with:

    - **Zero look-ahead bias**: signals use only data up to each rebalancing date
    - **Realistic transaction costs**: commission, slippage, market impact
    - **Monthly rebalancing** (configurable via ``config.rebalance_freq``)
    - **Full PerformanceMetrics** suite covering all Spec 09 requirements
    - **Stress period analysis** for GFC, COVID, and rate-hike regimes

    Args:
        config: :class:`BacktestConfig` controlling engine behaviour.
    """

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config
        self._logger = get_logger(__name__)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        strategy: BaseStrategy,
        prices: pd.DataFrame,
        benchmark_prices: pd.Series | None = None,
    ) -> BacktestResult:
        """Execute the walk-forward backtest.

        Args:
            strategy: Strategy implementing :class:`BaseStrategy`.
            prices: Close-price DataFrame with DatetimeIndex and symbol
                    columns.  Must span at least the backtest period plus
                    a warm-up window (≥ 300 trading days before ``start_date``).
            benchmark_prices: Optional benchmark close-price series.  If
                              ``None``, the first column of *prices* is used.

        Returns:
            :class:`BacktestResult` with full equity curve, trades, and metrics.

        Raises:
            ValueError: If there are fewer than 63 trading days in the
                        specified backtest window.
        """
        prices = prices.sort_index()
        prices.index = pd.to_datetime(prices.index)

        start_ts = pd.Timestamp(self.config.start_date)
        end_ts = pd.Timestamp(self.config.end_date)

        trading_dates = prices.index[(prices.index >= start_ts) & (prices.index <= end_ts)]
        if len(trading_dates) < 63:
            raise ValueError(
                f"Only {len(trading_dates)} trading days in "
                f"[{self.config.start_date}, {self.config.end_date}]. "
                "Need at least 63."
            )

        self._logger.info(
            "Backtest started",
            start=str(self.config.start_date),
            end=str(self.config.end_date),
            n_symbols=len(prices.columns),
            n_trading_days=len(trading_dates),
        )

        rebal_set = self._rebalancing_dates(trading_dates, self.config.rebalance_freq)
        daily_returns = prices.pct_change()

        # ---- State ----
        capital = self.config.initial_capital
        gross_capital = self.config.initial_capital
        current_weights: dict[str, float] = {}
        all_trades: list[Trade] = []
        equity_pts: list[tuple[pd.Timestamp, float]] = []
        gross_pts: list[tuple[pd.Timestamp, float]] = []
        ic_records: list[tuple[pd.Timestamp, float]] = []

        for i, t in enumerate(trading_dates):
            # ---- Rebalance ----
            if t in rebal_set and i > 0:
                prices_up_to_t = prices.loc[:t]
                returns_up_to_t = daily_returns.loc[:t].dropna(how="all")

                signals = strategy.generate_signals(prices_up_to_t, t)
                new_weights = strategy.compute_weights(signals, returns_up_to_t)
                new_weights = self._apply_position_limits(new_weights)

                cost = self._execute_rebalance(
                    current_weights=current_weights,
                    target_weights=new_weights,
                    prices_row=prices.loc[t],
                    capital=capital,
                    daily_returns_df=daily_returns,
                    t=t,
                    all_trades=all_trades,
                )
                capital = max(capital - cost, 1.0)
                current_weights = new_weights

                # IC: correlation of just-generated signals with forward returns
                ic = _compute_ic(
                    signals=signals,
                    prices=prices,
                    signal_date=t,
                    horizon=self.config.signal_horizon_days,
                )
                if np.isfinite(ic):
                    ic_records.append((t, ic))

            # ---- Daily drift ----
            if current_weights and i > 0:
                port_ret = 0.0
                for sym, w in current_weights.items():
                    if sym in daily_returns.columns:
                        r = float(daily_returns.loc[t, sym])
                        if np.isfinite(r):
                            port_ret += w * r
                capital *= 1.0 + port_ret
                gross_capital *= 1.0 + port_ret

            equity_pts.append((t, capital))
            gross_pts.append((t, gross_capital))

        # ---- Assemble series ----
        eq_s = pd.Series(
            [v for _, v in equity_pts],
            index=pd.DatetimeIndex([d for d, _ in equity_pts]),
            name="equity",
        )
        gq_s = pd.Series(
            [v for _, v in gross_pts],
            index=pd.DatetimeIndex([d for d, _ in gross_pts]),
            name="gross_equity",
        )
        net_daily = eq_s.pct_change().dropna()
        gross_daily = gq_s.pct_change().dropna()

        # ---- Benchmark ----
        bench_raw = benchmark_prices if benchmark_prices is not None else prices.iloc[:, 0]
        bench_raw = bench_raw.sort_index()
        bench_raw.index = pd.to_datetime(bench_raw.index)
        bench_in_window = bench_raw.loc[
            (bench_raw.index >= start_ts) & (bench_raw.index <= end_ts)
        ].dropna()
        if len(bench_in_window) > 1:
            bench_daily = bench_in_window.pct_change().dropna()
            bench_curve = (bench_in_window / bench_in_window.iloc[0]) * self.config.initial_capital
        else:
            bench_daily = pd.Series(dtype=float)
            bench_curve = pd.Series(dtype=float)

        bench_ret_aligned = bench_daily.reindex(net_daily.index).fillna(0.0)

        # ---- Stress periods ----
        stress_metrics = self._compute_stress_metrics(
            net_returns=net_daily,
            benchmark_returns=bench_ret_aligned,
            trades=all_trades,
            ic_records=ic_records,
        )

        # ---- Full metrics ----
        n_years = (self.config.end_date - self.config.start_date).days / 365.25
        ic_values = [v for _, v in ic_records]
        metrics = compute_performance_metrics(
            net_returns=net_daily,
            gross_returns=gross_daily,
            benchmark_returns=bench_ret_aligned,
            trades=all_trades,
            ic_values=ic_values,
            initial_capital=self.config.initial_capital,
            n_years=max(n_years, 0.01),
        )

        self._logger.info(
            "Backtest complete",
            sharpe=round(metrics.sharpe_ratio, 3),
            cagr=round(metrics.cagr, 4),
            max_drawdown=round(metrics.max_drawdown, 4),
            n_trades=len(all_trades),
        )

        return BacktestResult(
            config=self.config,
            metrics=metrics,
            equity_curve=[float(v) for v in eq_s.values],
            equity_dates=[str(d.date()) for d in eq_s.index],
            benchmark_curve=[float(v) for v in bench_curve.values] if len(bench_curve) > 0 else [],
            benchmark_dates=(
                [str(d.date()) for d in bench_curve.index] if len(bench_curve) > 0 else []
            ),
            gross_equity_curve=[float(v) for v in gq_s.values],
            daily_returns=[float(v) for v in net_daily.values],
            trades=all_trades,
            ic_series=ic_values,
            ic_dates=[str(d.date()) for d, _ in ic_records],
            stress_periods=stress_metrics,
        )

    # ------------------------------------------------------------------
    # Rebalancing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rebalancing_dates(
        trading_dates: pd.DatetimeIndex,
        freq: str,
    ) -> set[pd.Timestamp]:
        """Compute the set of rebalancing timestamps within *trading_dates*.

        Snaps each calendar period-end to the last valid trading day on or
        before that period end.

        Args:
            trading_dates: All trading days in the backtest window.
            freq: Pandas offset alias (e.g. ``"ME"`` for month-end).

        Returns:
            Set of :class:`pandas.Timestamp` values that are valid rebalancing dates.
        """
        period_ends = pd.date_range(
            start=trading_dates[0],
            end=trading_dates[-1],
            freq=freq,
        )
        rebal: set[pd.Timestamp] = set()
        for pe in period_ends:
            candidates = trading_dates[trading_dates <= pe]
            if len(candidates) > 0:
                rebal.add(candidates[-1])
        return rebal

    def _apply_position_limits(self, weights: dict[str, float]) -> dict[str, float]:
        """Clip individual positions to ``[min_pos, max_pos]`` and re-normalise.

        Uses a lock-and-scale approach: assets that would exceed the cap after
        proportional scaling are locked at ``max_position_size`` and the
        remainder is redistributed to unlocked assets.  If the number of
        assets is too small to reach full investment, the portfolio is held at
        partial investment (cash ≥ 0) to avoid violating individual caps.

        Args:
            weights: Raw target weights from the strategy.

        Returns:
            Clipped and normalised weights (sum ≤ 1; each weight ≤ max_position_size).
        """
        if not weights:
            return {}
        max_pos = self.config.max_position_size
        min_pos = self.config.min_position_size

        # Remove non-positive and below-minimum entries, then normalise
        free = {s: v for s, v in weights.items() if v >= min_pos}
        if not free:
            return {}
        total = sum(free.values())
        if total < 1e-8:
            return {}
        free = {s: v / total for s, v in free.items()}

        locked: dict[str, float] = {}
        for _ in range(len(free) + 1):
            free_total = sum(free.values())
            if free_total < 1e-8:
                break
            remaining = 1.0 - sum(locked.values())
            scale = remaining / free_total
            # Find assets that would exceed max_pos after scaling
            to_lock = [s for s, v in free.items() if v * scale > max_pos + 1e-10]
            if not to_lock:
                # All free assets satisfy the cap — apply scale and finish
                result = dict(locked)
                for s, v in free.items():
                    result[s] = v * scale
                return {s: v for s, v in result.items() if v >= min_pos}
            # Lock those assets at max_pos
            for s in to_lock:
                locked[s] = max_pos
                del free[s]

        # Infeasible (too few assets to sum to 1) — return locked weights as-is
        result = dict(locked)
        for s, v in free.items():
            result[s] = v
        return {s: v for s, v in result.items() if v >= min_pos}

    def _execute_rebalance(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        prices_row: pd.Series,
        capital: float,
        daily_returns_df: pd.DataFrame,
        t: pd.Timestamp,
        all_trades: list[Trade],
    ) -> float:
        """Compute trades needed to move from *current_weights* to *target_weights*,
        record them in *all_trades*, and return the total cost.

        Args:
            current_weights: Existing portfolio weights.
            target_weights: Desired portfolio weights after rebalance.
            prices_row: Close prices for date *t*.
            capital: Current portfolio value.
            daily_returns_df: Full daily-returns DataFrame.
            t: Rebalancing timestamp.
            all_trades: Mutable list to append trades to.

        Returns:
            Total transaction cost in USD.
        """
        all_syms = set(current_weights) | set(target_weights)
        adv_window = daily_returns_df.loc[:t].tail(21)
        total_cost = 0.0

        for sym in all_syms:
            if sym not in prices_row.index or not np.isfinite(prices_row[sym]):
                continue
            price = float(prices_row[sym])
            if price <= 0:
                continue

            delta_w = target_weights.get(sym, 0.0) - current_weights.get(sym, 0.0)
            if abs(delta_w) < 1e-4:
                continue

            shares = (delta_w * capital) / price

            # ADV estimate (rough: scale by capital turnover)
            if sym in adv_window.columns:
                avg_abs_ret = float(adv_window[sym].abs().mean())
                adv_value = capital * avg_abs_ret * 252 * 0.05
            else:
                adv_value = capital * 0.02

            # Realised vol
            if sym in adv_window.columns and len(adv_window[sym].dropna()) > 5:
                realized_vol = float(adv_window[sym].dropna().std() * np.sqrt(252))
            else:
                realized_vol = 0.20

            trade = _compute_trade(
                symbol=sym,
                shares=shares,
                price=price,
                adv_value=max(adv_value, 1.0),
                realized_vol=max(realized_vol, 0.01),
                commission_per_share=self.config.commission_per_share,
                slippage_bps=self.config.slippage_bps,
                trade_date=t.date(),
            )
            all_trades.append(trade)
            total_cost += trade.total_cost

        return total_cost

    # ------------------------------------------------------------------
    # IC computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_stress_metrics(
        net_returns: pd.Series,
        benchmark_returns: pd.Series,
        trades: list[Trade],
        ic_records: list[tuple[pd.Timestamp, float]],
    ) -> dict[str, PerformanceMetrics]:
        """Compute performance metrics for each named stress period.

        Args:
            net_returns: Full net-return series.
            benchmark_returns: Full benchmark-return series.
            trades: All trades (filtered by date per period).
            ic_records: IC values with timestamps.

        Returns:
            Dict mapping stress-period name → :class:`PerformanceMetrics`.
        """
        results: dict[str, PerformanceMetrics] = {}
        for name, (s_str, e_str) in _STRESS_PERIODS.items():
            s_ts = pd.Timestamp(s_str)
            e_ts = pd.Timestamp(e_str)
            period_net = net_returns.loc[(net_returns.index >= s_ts) & (net_returns.index <= e_ts)]
            if len(period_net) < 5:
                continue
            period_bench = benchmark_returns.reindex(period_net.index).fillna(0.0)
            period_trades = [tr for tr in trades if s_ts.date() <= tr.date <= e_ts.date()]
            period_ic = [v for d, v in ic_records if s_ts <= d <= e_ts]
            n_years = (
                min(e_ts, net_returns.index[-1]) - max(s_ts, net_returns.index[0])
            ).days / 365.25
            results[name] = compute_performance_metrics(
                net_returns=period_net,
                gross_returns=period_net,
                benchmark_returns=period_bench,
                trades=period_trades,
                ic_values=period_ic,
                initial_capital=100_000.0,
                n_years=max(n_years, 0.01),
            )
        return results


# ---------------------------------------------------------------------------
# IC helper (module-level, used by engine)
# ---------------------------------------------------------------------------


def _compute_ic(
    signals: dict[str, float],
    prices: pd.DataFrame,
    signal_date: pd.Timestamp,
    horizon: int,
) -> float:
    """Compute Spearman IC between signals and subsequent forward returns.

    Args:
        signals: Symbol → signal map generated at *signal_date*.
        prices: Full price DataFrame (including post-signal dates for evaluation).
        signal_date: Date the signals were generated.
        horizon: Number of trading days for forward-return computation.

    Returns:
        Spearman IC as a float, or ``nan`` if insufficient data.
    """
    future_slice = prices.loc[signal_date:]
    if len(future_slice) <= horizon:
        return float("nan")

    p_t = prices.loc[signal_date] if signal_date in prices.index else None
    p_h = future_slice.iloc[horizon]

    if p_t is None:
        return float("nan")

    forward_returns: dict[str, float] = {}
    for sym in signals:
        if sym not in prices.columns:
            continue
        p0 = float(p_t.get(sym, float("nan")))
        p1 = float(p_h.get(sym, float("nan")))
        if np.isfinite(p0) and np.isfinite(p1) and p0 > 1e-8:
            forward_returns[sym] = (p1 / p0) - 1.0

    common = [s for s in signals if s in forward_returns]
    if len(common) < 5:
        return float("nan")

    sig_arr = np.array([signals[s] for s in common], dtype=np.float64)
    ret_arr = np.array([forward_returns[s] for s in common], dtype=np.float64)

    try:
        ic, _ = stats.spearmanr(sig_arr, ret_arr)
        return float(ic) if np.isfinite(ic) else float("nan")
    except Exception:
        return float("nan")
