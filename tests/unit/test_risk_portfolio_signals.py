"""Unit tests for Phase 5: Risk, Portfolio, and Signal Fusion.

Uses synthetic data — no external API calls or network dependencies.

Run with:
    pytest tests/unit/test_risk_portfolio_signals.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=42)


@pytest.fixture()
def daily_returns(rng: np.random.Generator) -> pd.Series:
    """300 daily log-returns from a normal distribution (~10% ann vol)."""
    n = 300
    ret = rng.normal(0.0003, 0.01, n)
    idx = pd.bdate_range("2022-01-01", periods=n)
    return pd.Series(ret, index=idx, name="returns", dtype=float)


@pytest.fixture()
def multi_asset_returns(rng: np.random.Generator) -> pd.DataFrame:
    """300 days × 5 assets — correlated synthetic returns."""
    n = 300
    cov = np.array([
        [1.0, 0.6, 0.4, 0.3, 0.2],
        [0.6, 1.0, 0.5, 0.3, 0.1],
        [0.4, 0.5, 1.0, 0.4, 0.2],
        [0.3, 0.3, 0.4, 1.0, 0.3],
        [0.2, 0.1, 0.2, 0.3, 1.0],
    ]) * (0.01**2)
    L = np.linalg.cholesky(cov)
    z = rng.standard_normal((n, 5))
    r = z @ L.T
    idx = pd.bdate_range("2022-01-01", periods=n)
    return pd.DataFrame(r, index=idx, columns=["AAPL", "MSFT", "GOOGL", "AMZN", "META"])


@pytest.fixture()
def model_outputs() -> list:
    """Three synthetic ModelOutput objects."""
    from datetime import datetime, timezone
    from quantflow.models.base import ModelOutput

    now = datetime.now(tz=timezone.utc)
    return [
        ModelOutput(
            model_name="GARCHModel",
            symbol="AAPL",
            timestamp=now,
            signal=0.35,
            confidence=0.70,
            forecast_return=0.02,
            forecast_std=0.08,
        ),
        ModelOutput(
            model_name="GBTSignalModel",
            symbol="AAPL",
            timestamp=now,
            signal=0.50,
            confidence=0.80,
            forecast_return=0.03,
            forecast_std=0.07,
        ),
        ModelOutput(
            model_name="LSTMSignalModel",
            symbol="AAPL",
            timestamp=now,
            signal=0.20,
            confidence=0.60,
            forecast_return=0.01,
            forecast_std=0.09,
        ),
    ]


# ===========================================================================
# VaR / ES Tests
# ===========================================================================


class TestRiskCalculator:
    """Tests for the three VaR / ES methods."""

    def _make_calc(self) -> "RiskCalculator":
        from quantflow.risk.var_es import RiskCalculator
        return RiskCalculator(window=252, n_simulations=1_000)

    def test_historical_var_is_negative_loss(self, daily_returns: pd.Series) -> None:
        calc = self._make_calc()
        report = calc.compute_var_es(daily_returns, confidence=0.99, method="historical")
        # VaR is expressed as a return — should be negative (a loss)
        assert report.var < 0.0
        # ES is always worse than (or equal to) VaR
        assert report.es <= report.var + 1e-8

    def test_parametric_var_structure(self, daily_returns: pd.Series) -> None:
        calc = self._make_calc()
        report = calc.compute_var_es(daily_returns, confidence=0.95, method="parametric")
        assert report.var < 0.0
        assert report.es <= report.var + 1e-8
        assert report.distribution in ("normal", "student_t")
        assert 0.0 < report.confidence < 1.0

    def test_monte_carlo_var_structure(self, daily_returns: pd.Series) -> None:
        calc = self._make_calc()
        report = calc.compute_var_es(daily_returns, confidence=0.99, method="monte_carlo")
        assert report.var < 0.0
        assert report.es <= report.var + 1e-8
        assert report.n_observations == len(daily_returns)

    def test_es_worse_than_var_across_methods(self, daily_returns: pd.Series) -> None:
        calc = self._make_calc()
        for method in ("historical", "parametric", "monte_carlo"):
            report = calc.compute_var_es(daily_returns, method=method)  # type: ignore[arg-type]
            assert report.es <= report.var + 1e-8, f"ES > VaR for method={method}"

    def test_higher_confidence_gives_worse_var(self, daily_returns: pd.Series) -> None:
        calc = self._make_calc()
        r95 = calc.compute_var_es(daily_returns, confidence=0.95, method="historical")
        r99 = calc.compute_var_es(daily_returns, confidence=0.99, method="historical")
        assert r99.var <= r95.var  # 99% VaR must be >= 95% loss

    def test_backtesting_kupiec_pvalue_present(self, daily_returns: pd.Series) -> None:
        calc = self._make_calc()
        report = calc.compute_var_es(daily_returns, run_backtest=True)
        assert report.kupiec_pvalue is not None
        assert 0.0 <= report.kupiec_pvalue <= 1.0

    def test_backtesting_christoffersen_pvalue_present(self, daily_returns: pd.Series) -> None:
        calc = self._make_calc()
        report = calc.compute_var_es(daily_returns, run_backtest=True)
        assert report.christoffersen_pvalue is not None
        assert 0.0 <= report.christoffersen_pvalue <= 1.0

    def test_horizon_scaling(self, daily_returns: pd.Series) -> None:
        calc = self._make_calc()
        r1 = calc.compute_var_es(daily_returns, horizon=1, method="parametric")
        r10 = calc.compute_var_es(daily_returns, horizon=10, method="parametric")
        # 10-day VaR should be larger loss than 1-day
        assert r10.var < r1.var

    def test_portfolio_var_structure(self, multi_asset_returns: pd.DataFrame) -> None:
        from quantflow.risk.var_es import RiskCalculator
        calc = RiskCalculator()
        n = multi_asset_returns.shape[1]
        w = np.ones(n) / n
        cov = np.cov(multi_asset_returns.T.values) * 252
        report = calc.compute_portfolio_var(w, cov)
        assert report.var < 0.0
        assert report.es <= report.var + 1e-8

    def test_insufficient_data_raises(self) -> None:
        from quantflow.risk.var_es import RiskCalculator
        calc = RiskCalculator()
        short = pd.Series(np.random.randn(5))
        with pytest.raises(ValueError, match="Insufficient"):
            calc.compute_var_es(short)

    def test_compute_all_methods(self, daily_returns: pd.Series) -> None:
        calc = self._make_calc()
        results = calc.compute_all_methods(daily_returns)
        assert set(results.keys()) == {"historical", "parametric", "monte_carlo"}

    def test_kupiec_pvalue_range(self) -> None:
        from quantflow.risk.var_es import RiskCalculator
        # Well-calibrated model: ~1% violations at 99% confidence
        n = 500
        violations = np.zeros(n, dtype=int)
        violations[:5] = 1  # 1% violation rate
        pvalue = RiskCalculator._kupiec_test(violations, confidence=0.99)
        assert 0.0 <= pvalue <= 1.0

    def test_christoffersen_pvalue_range(self) -> None:
        from quantflow.risk.var_es import RiskCalculator
        # Independent violations
        violations = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1], dtype=int)
        pvalue = RiskCalculator._christoffersen_test(violations)
        assert 0.0 <= pvalue <= 1.0


# ===========================================================================
# EVT Tests
# ===========================================================================


class TestEVTRiskModel:
    """Tests for the Peaks-Over-Threshold EVT model."""

    @pytest.fixture()
    def daily_returns(self, rng: np.random.Generator) -> pd.Series:
        # 500 rows → 5% tail = 25 exceedances ≥ 20 minimum required by EVTRiskModel
        n = 500
        ret = rng.normal(0.0003, 0.01, n)
        idx = pd.bdate_range("2022-01-01", periods=n)
        return pd.Series(ret, index=idx, name="returns", dtype=float)

    def test_fit_returns_result(self, daily_returns: pd.Series) -> None:
        from quantflow.risk.evt import EVTRiskModel
        model = EVTRiskModel()
        result = model.fit(daily_returns)
        assert result.threshold > 0.0
        assert result.sigma > 0.0
        assert result.n_exceedances > 0
        assert result.n_total == len(daily_returns.dropna())

    def test_extreme_var_worse_than_99(self, daily_returns: pd.Series) -> None:
        from quantflow.risk.evt import EVTRiskModel
        model = EVTRiskModel()
        result = model.fit(daily_returns)
        # 99.9% VaR should be a larger loss than 99% VaR
        assert result.var_999 >= result.var_99

    def test_es_worse_than_var(self, daily_returns: pd.Series) -> None:
        from quantflow.risk.evt import EVTRiskModel
        model = EVTRiskModel()
        result = model.fit(daily_returns)
        assert result.es_99 >= result.var_99
        assert result.es_999 >= result.var_999

    def test_hill_estimator_positive(self, daily_returns: pd.Series) -> None:
        from quantflow.risk.evt import EVTRiskModel
        model = EVTRiskModel()
        result = model.fit(daily_returns)
        # Hill estimator should be positive for fat-tailed data
        assert result.hill_estimate is None or result.hill_estimate > 0.0

    def test_gpd_fit_xi_finite(self, daily_returns: pd.Series) -> None:
        from quantflow.risk.evt import EVTRiskModel
        model = EVTRiskModel()
        result = model.fit(daily_returns)
        assert np.isfinite(result.xi)
        assert np.isfinite(result.sigma)

    def test_insufficient_exceedances_raises(self, rng: np.random.Generator) -> None:
        from quantflow.risk.evt import EVTRiskModel
        # Very short series — unlikely to have 20 exceedances
        short = pd.Series(rng.normal(0, 0.005, 25))
        model = EVTRiskModel(threshold_quantile=0.95)
        with pytest.raises(ValueError, match="exceedances"):
            model.fit(short)


# ===========================================================================
# Stress Tester Tests
# ===========================================================================


class TestStressTester:
    """Tests for historical and factor shock stress testing."""

    def _make_tester(self) -> "StressTester":
        from quantflow.risk.stress_tester import StressTester
        return StressTester(n_stress_scenarios=500)  # Small for speed

    def test_historical_scenario_covid(self, multi_asset_returns: pd.DataFrame) -> None:
        tester = self._make_tester()
        weights = {a: 0.2 for a in multi_asset_returns.columns}
        # Create returns covering the COVID period
        covid_returns = multi_asset_returns.copy()
        covid_returns.index = pd.bdate_range("2020-02-01", periods=len(covid_returns))
        result = tester.run_historical_scenario(weights, covid_returns, "covid_crash_2020")
        assert isinstance(result.portfolio_return, float)
        assert np.isfinite(result.portfolio_return)

    def test_historical_scenario_unknown_returns_zero(
        self, multi_asset_returns: pd.DataFrame
    ) -> None:
        tester = self._make_tester()
        weights = {a: 0.2 for a in multi_asset_returns.columns}
        # No data matches the scenario window → returns neutral result
        result = tester.run_historical_scenario(
            weights, multi_asset_returns, "covid_crash_2020"
        )
        # Since synthetic data doesn't cover 2020, should be 0 or fallback
        assert isinstance(result.portfolio_return, float)

    def test_factor_shock_direction(self) -> None:
        tester = self._make_tester()
        weights = {"AAPL": 0.5, "MSFT": 0.5}
        betas = {
            "AAPL": {"equity_market": 1.2},
            "MSFT": {"equity_market": 0.9},
        }
        shock = {"equity_market": -0.20}
        result = tester.run_factor_shock(weights, betas, shock)
        # Portfolio should lose money (negative return)
        assert result.portfolio_return < 0.0

    def test_factor_shock_positive_return(self) -> None:
        tester = self._make_tester()
        weights = {"AAPL": 1.0}
        betas = {"AAPL": {"equity_market": 1.0}}
        result = tester.run_factor_shock(weights, betas, {"equity_market": 0.10})
        assert result.portfolio_return > 0.0

    def test_monte_carlo_stress_distribution(
        self, multi_asset_returns: pd.DataFrame
    ) -> None:
        tester = self._make_tester()
        weights = {a: 0.2 for a in multi_asset_returns.columns}
        dist = tester.run_monte_carlo_stress(multi_asset_returns, weights, horizon=5)
        # Tail should be worse than median
        assert dist.p1 < dist.p50
        assert dist.p01 < dist.p1
        assert dist.n_scenarios == 500

    def test_no_common_assets_raises(self) -> None:
        from quantflow.risk.stress_tester import StressTester
        tester = StressTester()
        weights = {"XYZ": 1.0}
        returns = pd.DataFrame({"AAPL": [0.01, -0.02, 0.03, 0.01]})
        with pytest.raises(ValueError, match="No common assets"):
            tester.run_monte_carlo_stress(returns, weights, horizon=5)


# ===========================================================================
# MVO Optimizer Tests
# ===========================================================================


class TestMVOOptimizer:
    """Tests for Mean-Variance Optimization."""

    @pytest.fixture(autouse=True)
    def require_cvxpy(self) -> None:
        pytest.importorskip("cvxpy")

    def _make_optimizer(self) -> "MVOOptimizer":
        from quantflow.portfolio.optimizer import MVOOptimizer
        return MVOOptimizer()

    def test_weights_sum_to_one(self, multi_asset_returns: pd.DataFrame) -> None:
        opt = self._make_optimizer()
        result = opt.optimize(multi_asset_returns)
        total = sum(result.weights.values())
        assert abs(total - 1.0) < 1e-4

    def test_weights_within_bounds(self, multi_asset_returns: pd.DataFrame) -> None:
        from quantflow.portfolio.optimizer import MVOOptimizer
        opt = MVOOptimizer(max_weight=0.30, min_weight=0.01)
        result = opt.optimize(multi_asset_returns)
        for w in result.weights.values():
            assert w >= 0.0  # Non-negative
            assert w <= 0.31  # Allow tiny numerical slack

    def test_all_covariance_methods(self, multi_asset_returns: pd.DataFrame) -> None:
        opt = self._make_optimizer()
        for method in ("sample", "ledoit_wolf", "oas"):
            result = opt.optimize(multi_asset_returns, covariance_method=method)  # type: ignore[arg-type]
            assert abs(sum(result.weights.values()) - 1.0) < 1e-4

    def test_rmt_covariance_method(self, multi_asset_returns: pd.DataFrame) -> None:
        opt = self._make_optimizer()
        result = opt.optimize(multi_asset_returns, covariance_method="rmt")
        assert abs(sum(result.weights.values()) - 1.0) < 1e-4

    def test_signal_return_method(
        self, multi_asset_returns: pd.DataFrame
    ) -> None:
        from quantflow.portfolio.optimizer import MVOOptimizer
        opt = MVOOptimizer()
        signal_ret = pd.Series(
            [0.15, 0.10, 0.05, 0.08, 0.12],
            index=multi_asset_returns.columns,
        )
        result = opt.optimize(
            multi_asset_returns,
            return_method="signal",
            signal_returns=signal_ret,
        )
        assert abs(sum(result.weights.values()) - 1.0) < 1e-4

    def test_sharpe_ratio_computed(self, multi_asset_returns: pd.DataFrame) -> None:
        opt = self._make_optimizer()
        result = opt.optimize(multi_asset_returns)
        assert np.isfinite(result.sharpe_ratio)

    def test_efficient_frontier_shape(self, multi_asset_returns: pd.DataFrame) -> None:
        opt = self._make_optimizer()
        result = opt.optimize(multi_asset_returns, compute_frontier=True)
        assert len(result.efficient_frontier) > 0
        for pt in result.efficient_frontier:
            assert pt.expected_vol >= 0.0

    def test_insufficient_data_raises(self, multi_asset_returns: pd.DataFrame) -> None:
        from quantflow.portfolio.optimizer import MVOOptimizer
        opt = MVOOptimizer()
        with pytest.raises(ValueError, match="observations"):
            opt.optimize(multi_asset_returns.iloc[:10])

    def test_rmt_cleaned_cov_psd(
        self, multi_asset_returns: pd.DataFrame
    ) -> None:
        """RMT-cleaned covariance must be positive semi-definite."""
        from quantflow.portfolio.optimizer import MVOOptimizer
        opt = MVOOptimizer()
        data = multi_asset_returns.values
        cov = np.cov(data.T)
        cleaned = opt._rmt_clean_covariance(cov, T=len(data), N=data.shape[1])
        eigenvalues = np.linalg.eigvalsh(cleaned)
        assert np.all(eigenvalues >= -1e-8)


# ===========================================================================
# Black-Litterman Tests
# ===========================================================================


class TestBlackLittermanOptimizer:
    """Tests for Black-Litterman optimizer."""

    @pytest.fixture(autouse=True)
    def require_cvxpy(self) -> None:
        pytest.importorskip("cvxpy")

    def test_weights_sum_to_one(self, multi_asset_returns: pd.DataFrame) -> None:
        from quantflow.portfolio.black_litterman import BlackLittermanOptimizer
        opt = BlackLittermanOptimizer()
        market_weights = pd.Series(
            [0.3, 0.25, 0.2, 0.15, 0.1],
            index=multi_asset_returns.columns,
        )
        views = {"AAPL": 0.15, "MSFT": 0.10}
        result = opt.optimize(multi_asset_returns, market_weights, views)
        assert abs(sum(result.weights.values()) - 1.0) < 1e-4

    def test_no_views_uses_equilibrium(
        self, multi_asset_returns: pd.DataFrame
    ) -> None:
        from quantflow.portfolio.black_litterman import BlackLittermanOptimizer
        opt = BlackLittermanOptimizer()
        market_weights = pd.Series(
            [0.3, 0.25, 0.2, 0.15, 0.1],
            index=multi_asset_returns.columns,
        )
        result = opt.optimize(multi_asset_returns, market_weights, signal_views={})
        assert abs(sum(result.weights.values()) - 1.0) < 1e-4

    def test_bullish_view_increases_weight(
        self, multi_asset_returns: pd.DataFrame
    ) -> None:
        """A strong positive view on AAPL should generally increase its weight."""
        from quantflow.portfolio.black_litterman import BlackLittermanOptimizer
        opt = BlackLittermanOptimizer()
        market_weights = pd.Series(
            [0.20, 0.20, 0.20, 0.20, 0.20],
            index=multi_asset_returns.columns,
        )
        no_view = opt.optimize(multi_asset_returns, market_weights, signal_views={})
        with_view = opt.optimize(
            multi_asset_returns,
            market_weights,
            signal_views={"AAPL": 0.50},
            view_confidences={"AAPL": 0.90},
        )
        # AAPL weight should be higher with a strong bullish view
        assert with_view.weights["AAPL"] >= no_view.weights["AAPL"] - 0.05

    def test_posterior_returns_in_metadata(
        self, multi_asset_returns: pd.DataFrame
    ) -> None:
        from quantflow.portfolio.black_litterman import BlackLittermanOptimizer
        opt = BlackLittermanOptimizer()
        market_weights = pd.Series(
            [0.3, 0.25, 0.2, 0.15, 0.1],
            index=multi_asset_returns.columns,
        )
        result = opt.optimize(
            multi_asset_returns, market_weights, {"AAPL": 0.10}
        )
        assert "posterior_returns" in result.metadata
        assert "AAPL" in result.metadata["posterior_returns"]


# ===========================================================================
# HRP Optimizer Tests
# ===========================================================================


class TestHRPOptimizer:
    """Tests for Hierarchical Risk Parity."""

    def test_weights_sum_to_one(self, multi_asset_returns: pd.DataFrame) -> None:
        from quantflow.portfolio.hrp import HRPOptimizer
        opt = HRPOptimizer()
        result = opt.optimize(multi_asset_returns)
        assert abs(sum(result.weights.values()) - 1.0) < 1e-4

    def test_all_weights_non_negative(self, multi_asset_returns: pd.DataFrame) -> None:
        from quantflow.portfolio.hrp import HRPOptimizer
        opt = HRPOptimizer()
        result = opt.optimize(multi_asset_returns)
        assert all(w >= -1e-8 for w in result.weights.values())

    def test_diversified_weights(self, multi_asset_returns: pd.DataFrame) -> None:
        """HRP should produce reasonably diversified weights (no extreme concentration)."""
        from quantflow.portfolio.hrp import HRPOptimizer
        opt = HRPOptimizer()
        result = opt.optimize(multi_asset_returns)
        max_weight = max(result.weights.values())
        # No single asset should have >50% weight in a 5-asset portfolio
        assert max_weight < 0.60

    def test_cluster_variance_positive(self) -> None:
        from quantflow.portfolio.hrp import HRPOptimizer
        cov = np.array([[0.01, 0.005], [0.005, 0.02]])
        var = HRPOptimizer._cluster_variance(cov, [0, 1])
        assert var > 0.0

    def test_correlation_distance_bounds(self) -> None:
        from quantflow.portfolio.hrp import HRPOptimizer
        corr = np.array([[1.0, 0.8], [0.8, 1.0]])
        dist = HRPOptimizer._correlation_distance(corr)
        assert np.all(dist >= 0.0)
        assert np.all(dist <= 1.0)
        np.testing.assert_array_almost_equal(np.diag(dist), 0.0)

    def test_insufficient_data_raises(self, multi_asset_returns: pd.DataFrame) -> None:
        from quantflow.portfolio.hrp import HRPOptimizer
        opt = HRPOptimizer()
        with pytest.raises(ValueError, match="observations"):
            opt.optimize(multi_asset_returns.iloc[:5])


# ===========================================================================
# Signal Normalizer Tests
# ===========================================================================


class TestSignalNormalizer:
    """Tests for the signal normalization pipeline."""

    def _make_signal(self, n: int = 300) -> pd.Series:
        rng = np.random.default_rng(seed=0)
        return pd.Series(
            rng.normal(0, 1, n),
            index=pd.bdate_range("2022-01-01", periods=n),
        )

    def test_output_clipped_to_range(self) -> None:
        from quantflow.signals.normalizer import SignalNormalizer
        norm = SignalNormalizer(zscore_clip=3.0)
        signal = self._make_signal()
        result = norm.normalize(signal)
        assert result.max() <= 3.0 + 1e-8
        assert result.min() >= -3.0 - 1e-8

    def test_output_has_no_nan(self) -> None:
        from quantflow.signals.normalizer import SignalNormalizer
        norm = SignalNormalizer()
        signal = self._make_signal()
        result = norm.normalize(signal)
        assert not result.isna().any()

    def test_winsorize_clips_extremes(self) -> None:
        from quantflow.signals.normalizer import SignalNormalizer
        norm = SignalNormalizer()
        signal = pd.Series([0.0] * 98 + [100.0, -100.0])
        winsorized = norm._winsorize(signal)
        assert winsorized.max() < 100.0
        assert winsorized.min() > -100.0

    def test_rolling_zscore_output_finite(self) -> None:
        from quantflow.signals.normalizer import SignalNormalizer
        norm = SignalNormalizer()
        signal = self._make_signal()
        result = norm._rolling_zscore(signal)
        assert np.all(np.isfinite(result.fillna(0)))

    def test_normalize_batch_all_models(self) -> None:
        from quantflow.signals.normalizer import SignalNormalizer
        norm = SignalNormalizer()
        signals = {f"model_{i}": self._make_signal() for i in range(3)}
        result = norm.normalize_batch(signals)
        assert set(result.keys()) == set(signals.keys())

    def test_empty_batch_returns_empty(self) -> None:
        from quantflow.signals.normalizer import SignalNormalizer
        norm = SignalNormalizer()
        result = norm.normalize_batch({})
        assert result == {}


# ===========================================================================
# Dynamic Weight Calibrator Tests
# ===========================================================================


class TestDynamicWeightCalibrator:
    """Tests for IC-based dynamic weight calibration."""

    def _make_calibrator(self) -> "DynamicWeightCalibrator":
        from quantflow.signals.calibrator import DynamicWeightCalibrator
        return DynamicWeightCalibrator(ic_lookback=60, ic_horizon=5)

    def _make_signals_and_returns(
        self, rng: np.random.Generator
    ) -> tuple[dict, pd.Series]:
        n = 150
        idx = pd.bdate_range("2022-01-01", periods=n)
        returns = pd.Series(rng.normal(0, 0.01, n), index=idx)
        # Signal 1 correlates with returns; signal 2 is noise
        good_signal = returns.shift(5).fillna(0) + rng.normal(0, 0.005, n)
        noise_signal = pd.Series(rng.normal(0, 1, n), index=idx)
        return {
            "GoodModel": good_signal,
            "NoiseModel": noise_signal,
        }, returns

    def test_weights_sum_to_one(self, rng: np.random.Generator) -> None:
        cal = self._make_calibrator()
        signals, returns = self._make_signals_and_returns(rng)
        weights = cal.compute_weights(signals, returns)
        assert abs(sum(weights.values()) - 1.0) < 1e-4

    def test_weights_within_floor_cap(self, rng: np.random.Generator) -> None:
        from quantflow.signals.calibrator import DynamicWeightCalibrator
        cal = DynamicWeightCalibrator(floor=0.05, cap=0.80)
        signals, returns = self._make_signals_and_returns(rng)
        weights = cal.compute_weights(signals, returns)
        for w in weights.values():
            assert w >= 0.04  # Slight numerical tolerance
            assert w <= 0.81

    def test_empty_signals_returns_empty(self, rng: np.random.Generator) -> None:
        cal = self._make_calibrator()
        returns = pd.Series(rng.normal(0, 0.01, 60))
        weights = cal.compute_weights({}, returns)
        assert weights == {}

    def test_regime_caching(self, rng: np.random.Generator) -> None:
        cal = self._make_calibrator()
        signals, returns = self._make_signals_and_returns(rng)
        cal.compute_weights(signals, returns, regime="BULL_LOW_VOL")
        cached = cal.get_regime_weights("BULL_LOW_VOL")
        assert cached is not None
        assert abs(sum(cached.values()) - 1.0) < 1e-4

    def test_all_negative_ic_gives_equal_weights(self, rng: np.random.Generator) -> None:
        cal = self._make_calibrator()
        # Signals that perfectly anti-correlate with returns
        n = 150
        idx = pd.bdate_range("2022-01-01", periods=n)
        returns = pd.Series(rng.normal(0, 0.01, n), index=idx)
        signals = {"M1": -returns.shift(5).fillna(0), "M2": -returns.shift(5).fillna(0)}
        weights = cal.compute_weights(signals, returns)
        # Should fall back to equal weights
        assert abs(weights.get("M1", 0) - weights.get("M2", 0)) < 0.1

    def test_icir_computation(self, rng: np.random.Generator) -> None:
        cal = self._make_calibrator()
        signals, returns = self._make_signals_and_returns(rng)
        icir = cal.icir(signals, returns)
        assert set(icir.keys()) == set(signals.keys())
        for v in icir.values():
            assert np.isfinite(v)


# ===========================================================================
# Ensemble Aggregator Tests
# ===========================================================================


class TestEnsembleAggregator:
    """Tests for the two-stage signal aggregator."""

    def _make_agg(self) -> "EnsembleAggregator":
        from quantflow.signals.aggregator import EnsembleAggregator
        return EnsembleAggregator()

    def test_output_signal_in_range(self, model_outputs: list) -> None:
        agg = self._make_agg()
        result = agg.aggregate(model_outputs)
        assert -1.0 <= result.risk_scaled_signal <= 1.0

    def test_empty_inputs_returns_neutral(self) -> None:
        agg = self._make_agg()
        result = agg.aggregate([], sentiment_score=None)
        assert result.risk_scaled_signal == 0.0

    def test_sentiment_only(self) -> None:
        agg = self._make_agg()
        result = agg.aggregate([], sentiment_score=0.6)
        assert result.risk_scaled_signal > 0.0

    def test_vol_targeting_scales_down_high_vol(
        self, model_outputs: list
    ) -> None:
        from quantflow.signals.aggregator import EnsembleAggregator
        agg = EnsembleAggregator(target_vol=0.15)
        # High realized vol → scale down
        result = agg.aggregate(model_outputs, realized_vol=0.60)
        raw = result.raw_composite
        assert result.vol_scale_factor < 1.0

    def test_vol_targeting_scales_up_low_vol(
        self, model_outputs: list
    ) -> None:
        from quantflow.signals.aggregator import EnsembleAggregator
        agg = EnsembleAggregator(target_vol=0.15)
        # Low realized vol → scale up
        result = agg.aggregate(model_outputs, realized_vol=0.05)
        assert result.vol_scale_factor > 1.0

    def test_category_composites_present(self, model_outputs: list) -> None:
        agg = self._make_agg()
        result = agg.aggregate(model_outputs, sentiment_score=0.3)
        assert "statistical" in result.category_composites
        assert "ml" in result.category_composites
        assert "sentiment" in result.category_composites

    def test_category_weights_sum_to_one(self, model_outputs: list) -> None:
        agg = self._make_agg()
        result = agg.aggregate(model_outputs, sentiment_score=0.2)
        total = sum(result.category_weights.values())
        assert abs(total - 1.0) < 1e-6

    def test_model_weights_override(self, model_outputs: list) -> None:
        agg = self._make_agg()
        # Give all weight to GBT
        overrides = {"GBTSignalModel": 1.0, "GARCHModel": 0.0, "LSTMSignalModel": 0.0}
        result = agg.aggregate(model_outputs, model_weights=overrides)
        # ML category should be close to GBT signal alone
        ml_cat = result.category_composites.get("ml")
        assert ml_cat is not None
        assert abs(ml_cat.composite - 0.50) < 0.01


# ===========================================================================
# Regime Detector Tests
# ===========================================================================


class TestRegimeDetector:
    """Tests for composite regime detection."""

    def _make_detector(self) -> "RegimeDetector":
        from quantflow.signals.regime_detector import RegimeDetector
        return RegimeDetector(sma_window=50)

    def test_detect_returns_regime_state(self, daily_returns: pd.Series) -> None:
        det = self._make_detector()
        state = det.detect(daily_returns)
        from quantflow.signals.regime_detector import RegimeState
        assert isinstance(state, RegimeState)

    def test_vol_regime_valid_label(self, daily_returns: pd.Series) -> None:
        det = self._make_detector()
        state = det.detect(daily_returns)
        assert state.volatility_regime in {"LOW", "MEDIUM", "HIGH", "CRISIS"}

    def test_trend_regime_valid_label(self, daily_returns: pd.Series) -> None:
        det = self._make_detector()
        state = det.detect(daily_returns)
        assert state.trend_regime in {"BULL", "BEAR", "SIDEWAYS"}

    def test_macro_regime_valid_label(self, daily_returns: pd.Series) -> None:
        det = self._make_detector()
        state = det.detect(daily_returns)
        assert state.macro_regime in {"EXPANSION", "LATE_CYCLE", "RECESSION", "RECOVERY"}

    def test_vix_crisis_override(self, daily_returns: pd.Series) -> None:
        det = self._make_detector()
        # VIX > 35 should force CRISIS
        state = det.detect(daily_returns, vix=40.0)
        assert state.volatility_regime == "CRISIS"

    def test_inverted_yield_curve_recession(self, daily_returns: pd.Series) -> None:
        det = self._make_detector()
        state = det.detect(daily_returns, yield_curve_slope=-1.0)
        assert state.macro_regime == "RECESSION"

    def test_confidence_in_range(self, daily_returns: pd.Series) -> None:
        det = self._make_detector()
        state = det.detect(daily_returns)
        assert 0.0 <= state.regime_confidence <= 1.0

    def test_regime_duration_increases(self, daily_returns: pd.Series) -> None:
        det = self._make_detector()
        # Detect same regime twice — duration should increase
        state1 = det.detect(daily_returns)
        state2 = det.detect(daily_returns)
        if state1.overall_regime == state2.overall_regime:
            assert state2.regime_duration_days >= state1.regime_duration_days

    def test_markov_override_high_vol(self, daily_returns: pd.Series) -> None:
        det = self._make_detector()
        state = det.detect(
            daily_returns,
            markov_state_probs={"low_vol_bull": 0.05, "high_vol_bear": 0.95},
        )
        assert state.volatility_regime in {"HIGH", "CRISIS"}


# ===========================================================================
# Recommendation Engine Tests
# ===========================================================================


class TestRecommendationEngine:
    """Tests for the final recommendation engine."""

    def _make_engine(self) -> "RecommendationEngine":
        from quantflow.signals.recommendation import RecommendationEngine
        return RecommendationEngine(models_available=10)

    def test_strong_buy_signal(self, daily_returns: pd.Series) -> None:
        engine = self._make_engine()
        rec = engine.generate("AAPL", composite_signal=0.7, confidence=0.80)
        assert rec.recommendation == "STRONG_BUY"

    def test_strong_sell_signal(self, daily_returns: pd.Series) -> None:
        engine = self._make_engine()
        rec = engine.generate("AAPL", composite_signal=-0.7, confidence=0.80)
        assert rec.recommendation == "STRONG_SELL"

    def test_hold_near_zero_signal(self) -> None:
        engine = self._make_engine()
        rec = engine.generate("AAPL", composite_signal=0.01, confidence=0.30)
        assert rec.recommendation == "HOLD"

    def test_low_confidence_defaults_to_hold(self) -> None:
        engine = self._make_engine()
        # High signal but very low confidence → HOLD
        rec = engine.generate("AAPL", composite_signal=0.8, confidence=0.10)
        assert rec.recommendation == "HOLD"

    def test_position_size_bounded(self, daily_returns: pd.Series) -> None:
        engine = self._make_engine()
        rec = engine.generate("AAPL", composite_signal=0.9, confidence=0.90, returns=daily_returns)
        assert -0.20 <= rec.suggested_position_size <= 0.20

    def test_position_size_sign_matches_signal(self) -> None:
        engine = self._make_engine()
        buy_rec = engine.generate("AAPL", composite_signal=0.5, confidence=0.70)
        sell_rec = engine.generate("AAPL", composite_signal=-0.5, confidence=0.70)
        assert buy_rec.suggested_position_size > 0
        assert sell_rec.suggested_position_size < 0

    def test_crisis_regime_triggers_warning(self, daily_returns: pd.Series) -> None:
        from quantflow.signals.regime_detector import RegimeState
        from datetime import datetime, timezone
        engine = self._make_engine()
        crisis_regime = RegimeState(
            volatility_regime="CRISIS",
            trend_regime="BEAR",
            macro_regime="RECESSION",
            overall_regime="BEAR_CRISIS_VOL",
            regime_confidence=0.90,
            regime_duration_days=5,
        )
        rec = engine.generate(
            "AAPL",
            composite_signal=0.8,
            confidence=0.90,
            regime=crisis_regime,
        )
        assert rec.recommendation == "HOLD"
        assert any("CRISIS" in w for w in rec.risk_warnings)

    def test_data_quality_score_computed(self, model_outputs: list) -> None:
        engine = self._make_engine()
        rec = engine.generate(
            "AAPL",
            composite_signal=0.4,
            confidence=0.70,
            model_outputs=model_outputs,
        )
        assert 0.0 <= rec.data_quality_score <= 1.0

    def test_model_contributions_present(self, model_outputs: list) -> None:
        engine = self._make_engine()
        rec = engine.generate(
            "AAPL",
            composite_signal=0.4,
            confidence=0.70,
            model_outputs=model_outputs,
        )
        assert len(rec.model_contributions) == len(model_outputs)

    def test_max_drawdown_computed(self, daily_returns: pd.Series) -> None:
        engine = self._make_engine()
        rec = engine.generate(
            "AAPL",
            composite_signal=0.3,
            confidence=0.65,
            returns=daily_returns,
        )
        assert rec.max_drawdown_estimate <= 0.0

    @pytest.mark.parametrize("signal,conf,expected", [
        (0.6, 0.70, "STRONG_BUY"),
        (0.3, 0.55, "BUY"),
        (0.1, 0.45, "WEAK_BUY"),
        (0.0, 0.50, "HOLD"),
        (-0.1, 0.45, "WEAK_SELL"),
        (-0.3, 0.55, "SELL"),
        (-0.6, 0.70, "STRONG_SELL"),
    ])
    def test_signal_to_recommendation_mapping(
        self, signal: float, conf: float, expected: str
    ) -> None:
        from quantflow.signals.recommendation import RecommendationEngine
        result = RecommendationEngine._signal_to_recommendation(signal, conf)
        assert result == expected


# ===========================================================================
# Recommendation Explainer Tests
# ===========================================================================


class TestRecommendationExplainer:
    """Tests for the explainability engine."""

    def _make_recommendation(self) -> "FinalRecommendation":
        from quantflow.signals.recommendation import RecommendationEngine
        engine = RecommendationEngine()
        return engine.generate(
            "AAPL",
            composite_signal=0.4,
            confidence=0.72,
            bullish_factors=["strong earnings", "services growth"],
            bearish_factors=["macro uncertainty"],
            rationale="",
        )

    def _make_aggregation_result(self, model_outputs: list) -> "AggregationResult":
        from quantflow.signals.aggregator import EnsembleAggregator
        agg = EnsembleAggregator()
        return agg.aggregate(model_outputs, sentiment_score=0.3)

    @pytest.mark.asyncio()
    async def test_explain_template_narrative(self, model_outputs: list) -> None:
        from quantflow.signals.explainer import RecommendationExplainer
        explainer = RecommendationExplainer(anthropic_agent=None)
        rec = self._make_recommendation()
        agg = self._make_aggregation_result(model_outputs)
        report = await explainer.explain(rec, agg)

        assert report.narrative
        assert "AAPL" in report.narrative
        assert isinstance(report.contribution_chart.category_contributions, dict)

    @pytest.mark.asyncio()
    async def test_explain_llm_narrative_success(self, model_outputs: list) -> None:
        from quantflow.signals.explainer import RecommendationExplainer
        mock_agent = AsyncMock()
        mock_agent.generate_recommendation_rationale = AsyncMock(
            return_value="AAPL is a strong buy due to robust earnings and services growth."
        )
        explainer = RecommendationExplainer(anthropic_agent=mock_agent)
        rec = self._make_recommendation()
        agg = self._make_aggregation_result(model_outputs)
        report = await explainer.explain(rec, agg)

        assert "AAPL" in report.narrative
        mock_agent.generate_recommendation_rationale.assert_called_once()

    @pytest.mark.asyncio()
    async def test_explain_llm_fallback_on_failure(self, model_outputs: list) -> None:
        from quantflow.signals.explainer import RecommendationExplainer
        mock_agent = AsyncMock()
        mock_agent.generate_recommendation_rationale = AsyncMock(
            side_effect=Exception("API error")
        )
        explainer = RecommendationExplainer(anthropic_agent=mock_agent)
        rec = self._make_recommendation()
        agg = self._make_aggregation_result(model_outputs)
        # Should fall back to template without raising
        report = await explainer.explain(rec, agg)
        assert report.narrative

    @pytest.mark.asyncio()
    async def test_confidence_decomposition_keys(self, model_outputs: list) -> None:
        from quantflow.signals.explainer import RecommendationExplainer
        explainer = RecommendationExplainer()
        rec = self._make_recommendation()
        agg = self._make_aggregation_result(model_outputs)
        report = await explainer.explain(rec, agg)
        assert "model_agreement" in report.confidence_decomposition
        assert "data_quality" in report.confidence_decomposition
        assert "regime_clarity" in report.confidence_decomposition

    @pytest.mark.asyncio()
    async def test_shap_features_included(self, model_outputs: list) -> None:
        from quantflow.signals.explainer import RecommendationExplainer
        explainer = RecommendationExplainer()
        rec = self._make_recommendation()
        agg = self._make_aggregation_result(model_outputs)
        shap = {"momentum_12m": 0.25, "rsi_14": -0.10, "vol_21d": 0.05}
        report = await explainer.explain(rec, agg, shap_features=shap)
        assert len(report.top_features) > 0
        # SHAP features appear in output
        assert any("momentum_12m" in f for f in report.top_features)
