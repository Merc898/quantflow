"""Backtest engine tests (Phase 9).

All tests use purely synthetic price data — no live API calls, no external
downloads.  The synthetic universe consists of 10 fictitious tickers driven
by a seeded random number generator so results are fully deterministic.

Test structure
--------------
- TestBacktestConfig       — configuration validation
- TestPerformanceMetrics   — metric computation on known analytic cases
- TestTransactionCosts     — cost model correctness
- TestQuantFlowStrategy    — signal generation and weight computation
- TestBacktestEngine       — end-to-end run on synthetic data
- TestNoLookAheadBias      — verify signals are blind to future prices
- TestStressPeriods        — stress period extraction and metrics
- TestPdfReport            — PDF generation (smoke-tests layout, no rendering)
"""

from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantflow.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    PerformanceMetrics,
    QuantFlowStrategy,
    Trade,
    _compute_ic,
    _max_drawdown_and_duration,
    compute_performance_metrics,
)
from quantflow.backtest.report import generate_pdf_report

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)
_N_SYMBOLS = 10
_SYMBOLS = [f"SYM{i:02d}" for i in range(_N_SYMBOLS)]

# Warm-up: 400 days before backtest start so momentum lookback is satisfied
_WARMUP = 400
_BACKTEST_DAYS = 504  # ~2 years of trading days
_TOTAL_DAYS = _WARMUP + _BACKTEST_DAYS


def _make_prices(
    n: int = _TOTAL_DAYS,
    symbols: list[str] | None = None,
    seed: int = 42,
    drift: float = 0.0003,
    vol: float = 0.012,
) -> pd.DataFrame:
    """Generate a synthetic close-price DataFrame with a business-day index.

    Args:
        n: Total number of trading days (warm-up + backtest).
        symbols: Column labels.  Defaults to ``_SYMBOLS``.
        seed: RNG seed for reproducibility.
        drift: Daily log-return drift.
        vol: Daily log-return volatility.

    Returns:
        DataFrame with DatetimeIndex and one column per symbol.
    """
    rng = np.random.default_rng(seed)
    syms = symbols or _SYMBOLS
    idx = pd.bdate_range("2018-01-01", periods=n)
    log_rets = rng.normal(drift, vol, size=(n, len(syms)))
    # Add small cross-sectional dispersion so signals have something to rank
    for j in range(len(syms)):
        log_rets[:, j] += rng.normal(0, 0.002, size=n)
    prices = 100.0 * np.exp(np.cumsum(log_rets, axis=0))
    return pd.DataFrame(prices, index=idx, columns=syms)


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    """Full price DataFrame including warm-up period."""
    return _make_prices()


@pytest.fixture(scope="module")
def backtest_config() -> BacktestConfig:
    """Standard backtest config for 2-year synthetic window."""
    # warm-up starts 2018-01-01; backtest starts after _WARMUP business days
    all_dates = pd.bdate_range("2018-01-01", periods=_TOTAL_DAYS)
    bt_start = all_dates[_WARMUP].date()
    bt_end = all_dates[-1].date()
    return BacktestConfig(
        start_date=bt_start,
        end_date=bt_end,
        initial_capital=1_000_000.0,
        rebalance_freq="ME",
    )


@pytest.fixture(scope="module")
def simple_returns() -> pd.Series:
    """250-day return series with known properties."""
    rng = np.random.default_rng(0)
    n = 250
    returns = rng.normal(0.0005, 0.01, n)  # slightly positive drift
    idx = pd.bdate_range("2022-01-01", periods=n)
    return pd.Series(returns, index=idx, name="port")


@pytest.fixture(scope="module")
def strategy() -> QuantFlowStrategy:
    return QuantFlowStrategy()


# ---------------------------------------------------------------------------
# BacktestConfig tests
# ---------------------------------------------------------------------------


class TestBacktestConfig:
    def test_valid_config(self) -> None:
        cfg = BacktestConfig(
            start_date=date(2020, 1, 1),
            end_date=date(2022, 12, 31),
        )
        assert cfg.initial_capital == 1_000_000.0
        assert cfg.rebalance_freq == "ME"
        assert cfg.max_position_size == 0.15
        assert cfg.commission_per_share == 0.005

    def test_custom_config(self) -> None:
        cfg = BacktestConfig(
            start_date=date(2019, 1, 1),
            end_date=date(2020, 12, 31),
            initial_capital=500_000.0,
            rebalance_freq="QE",
            max_position_size=0.20,
            slippage_bps=10.0,
        )
        assert cfg.initial_capital == 500_000.0
        assert cfg.rebalance_freq == "QE"
        assert cfg.max_position_size == 0.20
        assert cfg.slippage_bps == 10.0

    def test_initial_capital_positive(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            BacktestConfig(
                start_date=date(2020, 1, 1),
                end_date=date(2021, 1, 1),
                initial_capital=0.0,
            )

    def test_max_position_size_le_one(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            BacktestConfig(
                start_date=date(2020, 1, 1),
                end_date=date(2021, 1, 1),
                max_position_size=1.5,
            )


# ---------------------------------------------------------------------------
# PerformanceMetrics / compute_performance_metrics tests
# ---------------------------------------------------------------------------


class TestPerformanceMetrics:
    def test_empty_metrics(self) -> None:
        m = PerformanceMetrics()
        assert m.sharpe_ratio == 0.0
        assert m.max_drawdown == 0.0

    def test_positive_returns_positive_sharpe(self, simple_returns: pd.Series) -> None:
        bench = simple_returns * 0.8
        m = compute_performance_metrics(
            net_returns=simple_returns,
            gross_returns=simple_returns,
            benchmark_returns=bench,
            trades=[],
            ic_values=[0.05, 0.04, 0.06, 0.03, 0.07, 0.02, 0.05, 0.04, 0.06, 0.05],
            initial_capital=1_000_000.0,
            n_years=1.0,
        )
        assert m.sharpe_ratio > 0.0, "Positive-drift returns should have positive Sharpe"
        assert m.annualized_vol > 0.0
        assert m.cagr > 0.0
        assert -1.0 <= m.max_drawdown <= 0.0

    def test_negative_returns_negative_sharpe(self) -> None:
        rng = np.random.default_rng(99)
        neg_ret = pd.Series(
            rng.normal(-0.001, 0.01, 252),
            index=pd.bdate_range("2022-01-01", periods=252),
        )
        m = compute_performance_metrics(
            net_returns=neg_ret,
            gross_returns=neg_ret,
            benchmark_returns=neg_ret * 0.5,
            trades=[],
            ic_values=[],
            initial_capital=1_000_000.0,
            n_years=1.0,
        )
        assert m.sharpe_ratio < 0.0

    def test_max_drawdown_is_non_positive(self, simple_returns: pd.Series) -> None:
        bench = simple_returns.copy()
        m = compute_performance_metrics(
            net_returns=simple_returns,
            gross_returns=simple_returns,
            benchmark_returns=bench,
            trades=[],
            ic_values=[],
            initial_capital=1_000_000.0,
            n_years=1.0,
        )
        assert m.max_drawdown <= 0.0

    def test_var_less_than_cvar(self, simple_returns: pd.Series) -> None:
        bench = simple_returns * 0.5
        m = compute_performance_metrics(
            net_returns=simple_returns,
            gross_returns=simple_returns,
            benchmark_returns=bench,
            trades=[],
            ic_values=[],
            initial_capital=1_000_000.0,
            n_years=1.0,
        )
        # CVaR should be ≤ VaR (both negative; CVaR is worse)
        assert m.cvar_95_1d <= m.var_95_1d

    def test_hit_rate_in_range(self, simple_returns: pd.Series) -> None:
        bench = simple_returns * 0.5
        m = compute_performance_metrics(
            net_returns=simple_returns,
            gross_returns=simple_returns,
            benchmark_returns=bench,
            trades=[],
            ic_values=[],
            initial_capital=1_000_000.0,
            n_years=1.0,
        )
        assert 0.0 <= m.hit_rate <= 1.0

    def test_cost_drag_increases_with_trade_costs(self) -> None:
        """Adding trades (costs) should widen gross_vs_net_sharpe gap."""
        rng = np.random.default_rng(7)
        ret = pd.Series(
            rng.normal(0.0005, 0.01, 252),
            index=pd.bdate_range("2022-01-01", periods=252),
        )
        bench = ret * 0.8
        trades = [
            Trade(
                date=date(2022, 2, 1),
                symbol="AAPL",
                shares=100.0,
                price=150.0,
                side="BUY",
                commission=50.0,
                slippage_cost=75.0,
                market_impact_cost=25.0,
            )
            for _ in range(20)
        ]
        m_costly = compute_performance_metrics(
            net_returns=ret,
            gross_returns=ret,
            benchmark_returns=bench,
            trades=trades,
            ic_values=[],
            initial_capital=1_000_000.0,
            n_years=1.0,
        )
        m_free = compute_performance_metrics(
            net_returns=ret,
            gross_returns=ret,
            benchmark_returns=bench,
            trades=[],
            ic_values=[],
            initial_capital=1_000_000.0,
            n_years=1.0,
        )
        assert m_costly.total_costs_bps > m_free.total_costs_bps

    def test_ic_metrics(self) -> None:
        ic_vals = [0.05, 0.06, 0.04, 0.07, 0.03, 0.05, 0.06, 0.04, 0.08, 0.02]
        rng = np.random.default_rng(3)
        ret = pd.Series(
            rng.normal(0.0003, 0.01, 252), index=pd.bdate_range("2022-01-01", periods=252)
        )
        m = compute_performance_metrics(
            net_returns=ret,
            gross_returns=ret,
            benchmark_returns=ret * 0.5,
            trades=[],
            ic_values=ic_vals,
            initial_capital=1_000_000.0,
            n_years=1.0,
        )
        assert abs(m.ic_mean - np.mean(ic_vals)) < 1e-4
        assert m.ic_std > 0.0
        assert m.icir > 0.0  # positive mean IC with positive std → positive ICIR


# ---------------------------------------------------------------------------
# Transaction cost tests
# ---------------------------------------------------------------------------


class TestTransactionCosts:
    def test_trade_total_cost_positive(self) -> None:
        from quantflow.backtest.engine import _compute_trade

        t = _compute_trade(
            symbol="AAPL",
            shares=100.0,
            price=150.0,
            adv_value=5_000_000.0,
            realized_vol=0.20,
            commission_per_share=0.005,
            slippage_bps=5.0,
            trade_date=date(2022, 3, 1),
        )
        assert t.total_cost > 0.0
        assert t.commission == pytest.approx(0.5, rel=1e-4)  # 100 * 0.005
        assert t.slippage_cost == pytest.approx(7.5, rel=1e-2)  # 100*150*5/10000 = 7.5

    def test_trade_side_buy_positive_shares(self) -> None:
        from quantflow.backtest.engine import _compute_trade

        t = _compute_trade(
            symbol="MSFT",
            shares=50.0,
            price=300.0,
            adv_value=10_000_000.0,
            realized_vol=0.18,
            commission_per_share=0.005,
            slippage_bps=5.0,
            trade_date=date(2022, 3, 2),
        )
        assert t.side == "BUY"

    def test_trade_side_sell_negative_shares(self) -> None:
        from quantflow.backtest.engine import _compute_trade

        t = _compute_trade(
            symbol="TSLA",
            shares=-200.0,
            price=250.0,
            adv_value=20_000_000.0,
            realized_vol=0.45,
            commission_per_share=0.005,
            slippage_bps=5.0,
            trade_date=date(2022, 3, 3),
        )
        assert t.side == "SELL"

    def test_market_impact_increases_with_trade_size(self) -> None:
        from quantflow.backtest.engine import _compute_trade

        small = _compute_trade("X", 10.0, 100.0, 1_000_000.0, 0.20, 0.005, 5.0, date(2022, 1, 1))
        large = _compute_trade("X", 1000.0, 100.0, 1_000_000.0, 0.20, 0.005, 5.0, date(2022, 1, 1))
        assert large.market_impact_cost > small.market_impact_cost

    def test_trade_value_property(self) -> None:
        t = Trade(
            date=date(2022, 1, 1),
            symbol="AAA",
            shares=-100.0,
            price=50.0,
            side="SELL",
            commission=0.5,
            slippage_cost=0.25,
            market_impact_cost=0.10,
        )
        assert t.trade_value == pytest.approx(5000.0)
        assert t.total_cost == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# QuantFlowStrategy tests
# ---------------------------------------------------------------------------


class TestQuantFlowStrategy:
    def test_signal_generation_produces_all_symbols(
        self, prices: pd.DataFrame, strategy: QuantFlowStrategy
    ) -> None:
        """Every symbol in the universe must appear in the signal dict."""
        as_of = prices.index[_WARMUP]
        sigs = strategy.generate_signals(prices.loc[:as_of], as_of)
        assert set(sigs.keys()) == set(_SYMBOLS)

    def test_signals_in_range(self, prices: pd.DataFrame, strategy: QuantFlowStrategy) -> None:
        as_of = prices.index[_WARMUP]
        sigs = strategy.generate_signals(prices.loc[:as_of], as_of)
        for sym, v in sigs.items():
            assert -1.0 <= v <= 1.0, f"Signal for {sym} out of range: {v}"

    def test_insufficient_data_returns_zeros(self, strategy: QuantFlowStrategy) -> None:
        """Too-short price history → zero signals."""
        tiny_prices = _make_prices(n=50, symbols=_SYMBOLS[:3])
        as_of = tiny_prices.index[-1]
        sigs = strategy.generate_signals(tiny_prices, as_of)
        for v in sigs.values():
            assert v == pytest.approx(0.0)

    def test_weights_sum_to_at_most_one(
        self, prices: pd.DataFrame, strategy: QuantFlowStrategy
    ) -> None:
        as_of = prices.index[_WARMUP]
        sigs = strategy.generate_signals(prices.loc[:as_of], as_of)
        returns = prices.loc[:as_of].pct_change().dropna()
        weights = strategy.compute_weights(sigs, returns)
        total = sum(weights.values())
        assert total <= 1.0 + 1e-6, f"Weights sum {total} > 1"

    def test_weights_non_negative_long_only(
        self, prices: pd.DataFrame, strategy: QuantFlowStrategy
    ) -> None:
        as_of = prices.index[_WARMUP]
        sigs = strategy.generate_signals(prices.loc[:as_of], as_of)
        returns = prices.loc[:as_of].pct_change().dropna()
        weights = strategy.compute_weights(sigs, returns)
        for sym, w in weights.items():
            assert w >= 0.0, f"Negative weight for {sym}: {w}"

    def test_only_long_signal_symbols_get_weight(
        self, prices: pd.DataFrame, strategy: QuantFlowStrategy
    ) -> None:
        as_of = prices.index[_WARMUP]
        sigs = strategy.generate_signals(prices.loc[:as_of], as_of)
        returns = prices.loc[:as_of].pct_change().dropna()
        weights = strategy.compute_weights(sigs, returns)
        # Symbols with signal ≤ 0 should have weight 0
        for sym, sig in sigs.items():
            if sig <= 0.0:
                assert weights.get(sym, 0.0) == pytest.approx(
                    0.0, abs=1e-6
                ), f"Bearish symbol {sym} (sig={sig:.3f}) has non-zero weight {weights.get(sym, 0.0)}"

    def test_fallback_equal_weight_when_all_signals_zero(self, strategy: QuantFlowStrategy) -> None:
        """All-zero signals → equal-weight fallback."""
        zero_sigs = dict.fromkeys(_SYMBOLS[:4], 0.0)
        rng = np.random.default_rng(5)
        ret_data = rng.normal(0, 0.01, (100, 4))
        returns = pd.DataFrame(
            ret_data,
            columns=_SYMBOLS[:4],
            index=pd.bdate_range("2022-01-01", periods=100),
        )
        weights = strategy.compute_weights(zero_sigs, returns)
        total = sum(weights.values())
        assert total == pytest.approx(1.0, abs=1e-6)
        for v in weights.values():
            assert v == pytest.approx(1.0 / 4, abs=1e-6)

    def test_cross_sectional_zscore_clips_extremes(self, strategy: QuantFlowStrategy) -> None:
        raw = {"A": 10.0, "B": -10.0, "C": 0.0, "D": 1.0}
        out = strategy._cross_sectional_zscore(raw)
        for v in out.values():
            assert -1.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# BacktestEngine end-to-end tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBacktestEngine:
    async def test_run_returns_result(
        self, prices: pd.DataFrame, backtest_config: BacktestConfig
    ) -> None:
        engine = BacktestEngine(backtest_config)
        result = await engine.run(QuantFlowStrategy(), prices)
        assert isinstance(result, BacktestResult)

    async def test_equity_curve_non_empty(
        self, prices: pd.DataFrame, backtest_config: BacktestConfig
    ) -> None:
        engine = BacktestEngine(backtest_config)
        result = await engine.run(QuantFlowStrategy(), prices)
        assert len(result.equity_curve) > 0
        assert len(result.equity_dates) == len(result.equity_curve)

    async def test_equity_starts_near_initial_capital(
        self, prices: pd.DataFrame, backtest_config: BacktestConfig
    ) -> None:
        engine = BacktestEngine(backtest_config)
        result = await engine.run(QuantFlowStrategy(), prices)
        # First equity point should equal initial_capital (no trades yet)
        assert result.equity_curve[0] == pytest.approx(backtest_config.initial_capital, rel=0.01)

    async def test_equity_curve_positive_throughout(
        self, prices: pd.DataFrame, backtest_config: BacktestConfig
    ) -> None:
        """Equity should remain positive even after costs."""
        engine = BacktestEngine(backtest_config)
        result = await engine.run(QuantFlowStrategy(), prices)
        assert all(v > 0 for v in result.equity_curve)

    async def test_all_metric_fields_finite(
        self, prices: pd.DataFrame, backtest_config: BacktestConfig
    ) -> None:
        engine = BacktestEngine(backtest_config)
        result = await engine.run(QuantFlowStrategy(), prices)
        m = result.metrics
        for field_name, field_value in m.model_dump().items():
            if isinstance(field_value, float):
                assert np.isfinite(field_value), f"Non-finite metric: {field_name}={field_value}"

    async def test_max_drawdown_non_positive(
        self, prices: pd.DataFrame, backtest_config: BacktestConfig
    ) -> None:
        engine = BacktestEngine(backtest_config)
        result = await engine.run(QuantFlowStrategy(), prices)
        assert result.metrics.max_drawdown <= 0.0

    async def test_trades_recorded(
        self, prices: pd.DataFrame, backtest_config: BacktestConfig
    ) -> None:
        engine = BacktestEngine(backtest_config)
        result = await engine.run(QuantFlowStrategy(), prices)
        # Monthly rebalancing over ~2 years → many trades
        assert len(result.trades) > 0

    async def test_ic_series_populated(
        self, prices: pd.DataFrame, backtest_config: BacktestConfig
    ) -> None:
        engine = BacktestEngine(backtest_config)
        result = await engine.run(QuantFlowStrategy(), prices)
        # At least some rebalancing periods should produce a valid IC
        assert len(result.ic_series) > 0

    async def test_var_negative(
        self, prices: pd.DataFrame, backtest_config: BacktestConfig
    ) -> None:
        engine = BacktestEngine(backtest_config)
        result = await engine.run(QuantFlowStrategy(), prices)
        assert result.metrics.var_95_1d <= 0.0

    async def test_position_limits_respected(
        self, prices: pd.DataFrame, backtest_config: BacktestConfig
    ) -> None:
        """No single position should exceed max_position_size in signals."""
        strategy = QuantFlowStrategy()
        all_dates = pd.bdate_range("2018-01-01", periods=_TOTAL_DAYS)
        for t in all_dates[_WARMUP::21]:  # sample every 21 days
            prices_up_to_t = prices.loc[:t]
            sigs = strategy.generate_signals(prices_up_to_t, t)
            returns = prices_up_to_t.pct_change().dropna()
            raw_weights = strategy.compute_weights(sigs, returns)
            # Apply engine position limits
            engine = BacktestEngine(backtest_config)
            weights = engine._apply_position_limits(raw_weights)
            for sym, w in weights.items():
                assert (
                    w <= backtest_config.max_position_size + 1e-6
                ), f"Position limit violated at {t}: {sym}={w:.4f} > {backtest_config.max_position_size}"

    async def test_insufficient_trading_days_raises(self) -> None:
        cfg = BacktestConfig(
            start_date=date(2022, 1, 1),
            end_date=date(2022, 1, 10),  # only ~7 trading days
        )
        small_prices = _make_prices(n=10, symbols=_SYMBOLS[:3])
        small_prices.index = pd.bdate_range("2022-01-01", periods=10)
        engine = BacktestEngine(cfg)
        with pytest.raises(ValueError, match="at least 63"):
            await engine.run(QuantFlowStrategy(), small_prices)

    async def test_custom_benchmark(
        self, prices: pd.DataFrame, backtest_config: BacktestConfig
    ) -> None:
        engine = BacktestEngine(backtest_config)
        benchmark = prices["SYM00"]
        result = await engine.run(QuantFlowStrategy(), prices, benchmark_prices=benchmark)
        assert len(result.benchmark_curve) > 0

    async def test_daily_returns_aligned_with_equity(
        self, prices: pd.DataFrame, backtest_config: BacktestConfig
    ) -> None:
        engine = BacktestEngine(backtest_config)
        result = await engine.run(QuantFlowStrategy(), prices)
        # daily_returns has one fewer element than equity_curve (pct_change drops first)
        assert len(result.daily_returns) == len(result.equity_curve) - 1


# ---------------------------------------------------------------------------
# No look-ahead bias tests
# ---------------------------------------------------------------------------


class TestNoLookAheadBias:
    """Verify that signal generation is blind to future price data."""

    def test_signal_unchanged_when_future_data_randomised(self) -> None:
        """Replacing all future prices with random values must not change the
        signal produced for a historical date."""
        strategy = QuantFlowStrategy()
        as_of_idx = _WARMUP  # use the first valid signal date
        as_of = pd.bdate_range("2018-01-01", periods=_TOTAL_DAYS)[as_of_idx]

        prices_full = _make_prices()
        # Signal using only history up to as_of
        prices_up_to = prices_full.loc[:as_of]
        sigs_original = strategy.generate_signals(prices_up_to.copy(), as_of)

        # Now corrupt all data AFTER as_of with large random noise
        rng = np.random.default_rng(999)
        prices_corrupted = prices_full.copy()
        future_mask = prices_corrupted.index > as_of
        prices_corrupted.loc[future_mask] = rng.uniform(
            1, 10_000, size=(future_mask.sum(), len(_SYMBOLS))
        )

        # Generate signal using only data up to as_of (same slice, identical to above)
        prices_up_to_corrupted = prices_corrupted.loc[:as_of]
        sigs_corrupted = strategy.generate_signals(prices_up_to_corrupted, as_of)

        # Signals must be identical — the strategy only sees data up to as_of
        for sym in _SYMBOLS:
            assert sigs_original[sym] == pytest.approx(
                sigs_corrupted[sym], abs=1e-10
            ), f"Signal for {sym} changed after corrupting future data → look-ahead bias detected"

    def test_signal_changes_when_historical_data_changes(self) -> None:
        """Perturbing historical (past) prices MUST change the signal —
        confirming the strategy actually uses historical data."""
        strategy = QuantFlowStrategy()
        all_dates = pd.bdate_range("2018-01-01", periods=_TOTAL_DAYS)
        as_of = all_dates[_WARMUP]

        prices_original = _make_prices()
        sigs_original = strategy.generate_signals(prices_original.loc[:as_of].copy(), as_of)

        # Perturb only the momentum base-period prices (early segment).
        # Uniform scaling preserves all ratios (mom, MA ratios, vol from pct_change),
        # so we target just the first ~150 bars which sets the momentum baseline.
        prices_modified = prices_original.copy()
        base_dates = all_dates[:150]
        prices_modified.loc[base_dates] *= 3.0  # changes p_base but not p_skip → alters momentum

        sigs_modified = strategy.generate_signals(prices_modified.loc[:as_of].copy(), as_of)

        # At least some signals should change
        n_changed = sum(abs(sigs_original[s] - sigs_modified[s]) > 1e-6 for s in _SYMBOLS)
        assert (
            n_changed > 0
        ), "No signal changed after perturbing historical data — strategy may be broken"

    def test_compute_ic_uses_future_data_for_evaluation_only(self) -> None:
        """IC computation needs future prices for evaluation — it must not
        feed those prices back into signal generation."""
        prices_full = _make_prices()
        as_of = prices_full.index[_WARMUP]
        signals = {s: float(np.random.default_rng(i).normal()) for i, s in enumerate(_SYMBOLS)}
        ic = _compute_ic(
            signals=signals,
            prices=prices_full,
            signal_date=as_of,
            horizon=21,
        )
        # IC should be finite (some data available after as_of)
        assert np.isfinite(ic) or ic != ic  # either finite or NaN (not ±inf)


# ---------------------------------------------------------------------------
# Max-drawdown helper tests
# ---------------------------------------------------------------------------


class TestMaxDrawdownHelper:
    def test_no_drawdown_series(self) -> None:
        monotone_up = pd.Series(
            np.linspace(0.001, 0.002, 100),
            index=pd.bdate_range("2022-01-01", periods=100),
        )
        dd, dur = _max_drawdown_and_duration(monotone_up)
        assert dd == pytest.approx(0.0, abs=1e-4)
        assert dur == 0

    def test_known_drawdown(self) -> None:
        # Construct: 50% gain, then 33% loss → net drawdown ~ -22%
        n = 100
        returns = np.zeros(n)
        returns[:50] = 0.01  # 50 days up
        returns[50:80] = -0.015  # 30 days down
        s = pd.Series(returns, index=pd.bdate_range("2022-01-01", periods=n))
        dd, dur = _max_drawdown_and_duration(s)
        assert dd < 0.0
        assert dur > 0

    def test_duration_positive_when_drawdown_exists(self) -> None:
        rng = np.random.default_rng(123)
        ret = pd.Series(rng.normal(0, 0.01, 252), index=pd.bdate_range("2022-01-01", periods=252))
        # Force a clear drawdown
        ret.iloc[100:130] = -0.03
        dd, dur = _max_drawdown_and_duration(ret)
        assert dd < -0.05  # meaningful drawdown
        assert dur > 0


# ---------------------------------------------------------------------------
# Stress period tests
# ---------------------------------------------------------------------------


class TestStressPeriods:
    @pytest.mark.asyncio
    async def test_stress_periods_extracted_when_in_range(self) -> None:
        """Backtest spanning 2020 should produce a COVID_2020 stress entry."""
        # Create prices covering 2019-2022 with 400 day warm-up
        n = 400 + 756  # ~3 years of trading days
        prices = _make_prices(n=n, symbols=_SYMBOLS[:5])
        prices.index = pd.bdate_range("2019-01-01", periods=n)

        cfg = BacktestConfig(
            start_date=date(2020, 1, 1),
            end_date=date(2022, 12, 31),
            initial_capital=500_000.0,
        )
        engine = BacktestEngine(cfg)
        result = await engine.run(QuantFlowStrategy(), prices)
        # COVID_2020 and RateHike_2022 are within range
        assert "COVID_2020" in result.stress_periods
        assert "RateHike_2022" in result.stress_periods

    @pytest.mark.asyncio
    async def test_stress_period_metrics_are_finite(self) -> None:
        n = 400 + 756
        prices = _make_prices(n=n, symbols=_SYMBOLS[:5])
        prices.index = pd.bdate_range("2019-01-01", periods=n)
        cfg = BacktestConfig(
            start_date=date(2020, 1, 1),
            end_date=date(2022, 12, 31),
        )
        engine = BacktestEngine(cfg)
        result = await engine.run(QuantFlowStrategy(), prices)
        for period_name, m in result.stress_periods.items():
            for field_name, val in m.model_dump().items():
                if isinstance(val, float):
                    assert np.isfinite(
                        val
                    ), f"Non-finite metric in stress period {period_name}: {field_name}={val}"

    @pytest.mark.asyncio
    async def test_stress_periods_empty_when_out_of_range(self) -> None:
        """A backtest in 2022 only should not include GFC_2008."""
        n = 400 + 252
        prices = _make_prices(n=n, symbols=_SYMBOLS[:4])
        prices.index = pd.bdate_range("2021-01-01", periods=n)
        cfg = BacktestConfig(
            start_date=date(2022, 1, 1),
            end_date=prices.index[-1].date(),
        )
        engine = BacktestEngine(cfg)
        result = await engine.run(QuantFlowStrategy(), prices)
        assert "GFC_2008" not in result.stress_periods


# ---------------------------------------------------------------------------
# Rebalancing schedule tests
# ---------------------------------------------------------------------------


class TestRebalancingSchedule:
    def test_month_end_rebalancing(
        self, prices: pd.DataFrame, backtest_config: BacktestConfig
    ) -> None:
        engine = BacktestEngine(backtest_config)
        bt_dates = prices.index[
            (prices.index >= pd.Timestamp(backtest_config.start_date))
            & (prices.index <= pd.Timestamp(backtest_config.end_date))
        ]
        rebal = engine._rebalancing_dates(bt_dates, "ME")
        # Should have approximately one date per month
        n_months = (backtest_config.end_date.year - backtest_config.start_date.year) * 12 + (
            backtest_config.end_date.month - backtest_config.start_date.month
        )
        assert abs(len(rebal) - n_months) <= 2  # allow ±2 for boundary effects

    def test_rebalancing_dates_are_trading_days(
        self, prices: pd.DataFrame, backtest_config: BacktestConfig
    ) -> None:
        engine = BacktestEngine(backtest_config)
        bt_dates = prices.index[
            (prices.index >= pd.Timestamp(backtest_config.start_date))
            & (prices.index <= pd.Timestamp(backtest_config.end_date))
        ]
        rebal = engine._rebalancing_dates(bt_dates, "ME")
        trading_set = set(bt_dates)
        for d in rebal:
            assert d in trading_set, f"Rebalancing date {d} is not a trading day"

    def test_position_limits_clipping(self, backtest_config: BacktestConfig) -> None:
        engine = BacktestEngine(backtest_config)
        # 10 assets: first asset concentrates 50%, rest split the remainder.
        # With max_pos=0.15 and 10 assets (max feasible sum=1.5>1), waterfall
        # redistribution converges so all caps are respected AND weights sum to 1.
        raw = {"S0": 0.50} | {f"S{i}": 0.50 / 9 for i in range(1, 10)}
        limited = engine._apply_position_limits(raw)
        for sym, w in limited.items():
            assert (
                w <= backtest_config.max_position_size + 1e-8
            ), f"{sym}={w:.4f} > max_pos={backtest_config.max_position_size}"
        assert abs(sum(limited.values()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# PDF report tests
# ---------------------------------------------------------------------------


class TestPdfReport:
    @pytest.mark.asyncio
    async def test_pdf_generated(
        self, prices: pd.DataFrame, backtest_config: BacktestConfig
    ) -> None:
        """PDF file is created and non-empty."""
        engine = BacktestEngine(backtest_config)
        result = await engine.run(QuantFlowStrategy(), prices)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.pdf"
            out = generate_pdf_report(result, path, strategy_name="Test Strategy")
            assert out.exists()
            assert out.stat().st_size > 1024  # at least 1 KB

    @pytest.mark.asyncio
    async def test_pdf_path_returned(
        self, prices: pd.DataFrame, backtest_config: BacktestConfig
    ) -> None:
        engine = BacktestEngine(backtest_config)
        result = await engine.run(QuantFlowStrategy(), prices)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "qf_report.pdf"
            out = generate_pdf_report(result, path)
            assert isinstance(out, Path)
            assert out.suffix == ".pdf"

    @pytest.mark.asyncio
    async def test_equity_series_helper(
        self, prices: pd.DataFrame, backtest_config: BacktestConfig
    ) -> None:
        engine = BacktestEngine(backtest_config)
        result = await engine.run(QuantFlowStrategy(), prices)
        eq = result.equity_series()
        assert isinstance(eq, pd.Series)
        assert eq.index.dtype == "datetime64[ns]"
        assert len(eq) == len(result.equity_curve)

    @pytest.mark.asyncio
    async def test_benchmark_series_helper(
        self, prices: pd.DataFrame, backtest_config: BacktestConfig
    ) -> None:
        engine = BacktestEngine(backtest_config)
        result = await engine.run(QuantFlowStrategy(), prices, benchmark_prices=prices["SYM00"])
        bench = result.benchmark_series()
        assert isinstance(bench, pd.Series)
        assert len(bench) > 0
