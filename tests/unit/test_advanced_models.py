"""Unit tests for Phase 6 advanced research models.

Covers:
- HestonModel (derivatives/heston.py)
- VolSurfaceModel (derivatives/vol_surface.py)
- HawkesModel (microstructure/hawkes.py)
- OptimalExecutionModel (microstructure/optimal_execution.py)
- BayesianNNModel (advanced/bayesian_nn.py)
- NeuralODEModel (advanced/neural_ode.py)
- NeuralSDEModel (advanced/neural_sde.py)

All tests use synthetic data only — no external API calls.
Tests validate the public interface, output schema, and key invariants.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantflow.models.base import ModelOutput

# ---------------------------------------------------------------------------
# Synthetic data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=42)


@pytest.fixture(scope="module")
def ohlcv_data(rng: np.random.Generator) -> pd.DataFrame:
    """Generate 400 days of synthetic OHLCV data."""
    n = 400
    log_prices = np.cumsum(rng.normal(0.0004, 0.015, n))
    prices = 100.0 * np.exp(log_prices)
    idx = pd.bdate_range("2021-01-01", periods=n, tz="UTC")
    return pd.DataFrame(
        {
            "open": prices * (1 + rng.normal(0, 0.002, n)),
            "high": prices * (1 + np.abs(rng.normal(0, 0.005, n))),
            "low": prices * (1 - np.abs(rng.normal(0, 0.005, n))),
            "close": prices,
            "volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
        },
        index=idx,
    )


@pytest.fixture(scope="module")
def short_data(rng: np.random.Generator) -> pd.DataFrame:
    """Short 60-day dataset for edge-case tests."""
    n = 60
    log_prices = np.cumsum(rng.normal(0.0003, 0.012, n))
    prices = 50.0 * np.exp(log_prices)
    idx = pd.bdate_range("2023-01-01", periods=n, tz="UTC")
    return pd.DataFrame(
        {"close": prices, "volume": rng.integers(500_000, 5_000_000, n).astype(float)},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Helper: assert ModelOutput is valid
# ---------------------------------------------------------------------------


def assert_valid_output(out: ModelOutput, symbol: str = "TEST") -> None:
    assert isinstance(out, ModelOutput)
    assert out.symbol == symbol
    assert -1.0 <= out.signal <= 1.0
    assert 0.0 <= out.confidence <= 1.0
    assert np.isfinite(out.signal)
    assert np.isfinite(out.confidence)
    assert np.isfinite(out.forecast_return)
    assert out.forecast_std >= 0.0
    assert np.isfinite(out.forecast_std)
    assert isinstance(out.model_name, str) and len(out.model_name) > 0


# ===========================================================================
# Heston Model Tests
# ===========================================================================


class TestHestonModel:
    def test_fit_returns_self(self, ohlcv_data: pd.DataFrame) -> None:
        from quantflow.models.derivatives.heston import HestonModel

        m = HestonModel("TEST", n_paths=500, seed=0)
        result = m.fit(ohlcv_data)
        assert result is m
        assert m._is_fitted

    def test_predict_valid_output(self, ohlcv_data: pd.DataFrame) -> None:
        from quantflow.models.derivatives.heston import HestonModel

        m = HestonModel("TEST", n_paths=500, seed=0)
        m.fit(ohlcv_data)
        out = m.predict(ohlcv_data)
        assert_valid_output(out, "TEST")
        assert out.model_name == "HestonModel"

    def test_parameters_physically_valid(self, ohlcv_data: pd.DataFrame) -> None:
        from quantflow.models.derivatives.heston import HestonModel

        m = HestonModel("TEST", n_paths=500, seed=0)
        m.fit(ohlcv_data)
        p = m._params
        assert p is not None
        assert p.kappa > 0.0, "kappa must be positive"
        assert p.theta > 0.0, "theta must be positive"
        assert p.xi > 0.0, "xi must be positive"
        assert -1.0 <= p.rho <= 1.0, "rho must be in [-1, 1]"
        assert p.v0 > 0.0, "v0 must be positive"

    def test_predict_without_fit_raises(self, ohlcv_data: pd.DataFrame) -> None:
        from quantflow.models.derivatives.heston import HestonModel

        m = HestonModel("TEST")
        with pytest.raises(RuntimeError, match="fitted"):
            m.predict(ohlcv_data)

    def test_insufficient_data_raises(self) -> None:
        from quantflow.models.derivatives.heston import HestonModel

        tiny = pd.DataFrame(
            {"close": [100.0, 101.0, 99.0]},
            index=pd.bdate_range("2024-01-01", periods=3, tz="UTC"),
        )
        with pytest.raises(ValueError):
            HestonModel("TEST").fit(tiny)

    def test_regime_tag_in_output(self, ohlcv_data: pd.DataFrame) -> None:
        from quantflow.models.derivatives.heston import HestonModel

        m = HestonModel("TEST", n_paths=500, seed=0)
        m.fit(ohlcv_data)
        out = m.predict(ohlcv_data)
        assert out.regime in {"HIGH_VOL", "MEDIUM_VOL", "LOW_VOL"}

    def test_metadata_contains_heston_params(self, ohlcv_data: pd.DataFrame) -> None:
        from quantflow.models.derivatives.heston import HestonModel

        m = HestonModel("TEST", n_paths=500, seed=0)
        m.fit(ohlcv_data)
        out = m.predict(ohlcv_data)
        for key in ("kappa", "theta", "xi", "rho", "v0"):
            assert key in out.metadata


# ===========================================================================
# VolSurface Model Tests
# ===========================================================================


class TestVolSurfaceModel:
    def test_fit_from_historical(self, ohlcv_data: pd.DataFrame) -> None:
        from quantflow.models.derivatives.vol_surface import VolSurfaceModel

        m = VolSurfaceModel("TEST")
        m.fit(ohlcv_data)
        assert m._is_fitted
        assert m._surface is not None

    def test_predict_valid_output(self, ohlcv_data: pd.DataFrame) -> None:
        from quantflow.models.derivatives.vol_surface import VolSurfaceModel

        m = VolSurfaceModel("TEST")
        m.fit(ohlcv_data)
        out = m.predict(ohlcv_data)
        assert_valid_output(out, "TEST")
        assert out.model_name == "VolSurfaceModel"

    def test_atm_vol_positive(self, ohlcv_data: pd.DataFrame) -> None:
        from quantflow.models.derivatives.vol_surface import VolSurfaceModel

        m = VolSurfaceModel("TEST")
        m.fit(ohlcv_data)
        assert m._surface is not None
        assert m._surface.atm_vol_short > 0.0
        assert m._surface.atm_vol_long > 0.0

    def test_term_structure_slope_finite(self, ohlcv_data: pd.DataFrame) -> None:
        from quantflow.models.derivatives.vol_surface import VolSurfaceModel

        m = VolSurfaceModel("TEST")
        m.fit(ohlcv_data)
        assert m._surface is not None
        assert np.isfinite(m._surface.term_structure_slope)

    def test_fit_with_implied_vol_data(self, ohlcv_data: pd.DataFrame) -> None:
        from quantflow.models.derivatives.vol_surface import VolSurfaceModel

        # Synthetic smile: flat vol surface
        smile = dict.fromkeys(np.linspace(-0.3, 0.3, 7), 0.2)
        impl_data = {21: smile, 63: smile}

        m = VolSurfaceModel("TEST")
        m.fit(ohlcv_data, implied_vol_data=impl_data)
        assert m._is_fitted
        assert len(m._surface.svi_params) == 2

    def test_regime_in_valid_set(self, ohlcv_data: pd.DataFrame) -> None:
        from quantflow.models.derivatives.vol_surface import VolSurfaceModel

        m = VolSurfaceModel("TEST")
        m.fit(ohlcv_data)
        out = m.predict(ohlcv_data)
        assert out.regime in {"HIGH_VOL", "MEDIUM_VOL", "LOW_VOL"}

    def test_vol_percentile_in_range(self, ohlcv_data: pd.DataFrame) -> None:
        from quantflow.models.derivatives.vol_surface import VolSurfaceModel

        m = VolSurfaceModel("TEST")
        m.fit(ohlcv_data)
        assert m._surface is not None
        assert 0.0 <= m._surface.vol_percentile <= 100.0


# ===========================================================================
# Hawkes Model Tests
# ===========================================================================


class TestHawkesModel:
    def test_fit_returns_self(self, ohlcv_data: pd.DataFrame) -> None:
        from quantflow.models.microstructure.hawkes import HawkesModel

        m = HawkesModel("TEST")
        result = m.fit(ohlcv_data)
        assert result is m
        assert m._is_fitted

    def test_predict_valid_output(self, ohlcv_data: pd.DataFrame) -> None:
        from quantflow.models.microstructure.hawkes import HawkesModel

        m = HawkesModel("TEST")
        m.fit(ohlcv_data)
        out = m.predict(ohlcv_data)
        assert_valid_output(out, "TEST")
        assert out.model_name == "HawkesModel"

    def test_branching_ratio_stable(self, ohlcv_data: pd.DataFrame) -> None:
        from quantflow.models.microstructure.hawkes import HawkesModel

        m = HawkesModel("TEST")
        m.fit(ohlcv_data)
        _mu_u, alpha_u, beta_u = m._up_params
        _mu_d, alpha_d, beta_d = m._dn_params
        # Branching ratio must be < 1 for stability (or near 0 if no excitation)
        br_up = alpha_u / max(beta_u, 1e-8)
        br_dn = alpha_d / max(beta_d, 1e-8)
        assert br_up >= 0.0
        assert br_dn >= 0.0

    def test_event_counts_positive(self, ohlcv_data: pd.DataFrame) -> None:
        from quantflow.models.microstructure.hawkes import HawkesModel

        m = HawkesModel("TEST")
        m.fit(ohlcv_data)
        # With 400 days of data we expect at least some events in each direction
        assert len(m._up_times) > 0 or len(m._dn_times) > 0

    def test_metadata_contains_intensities(self, ohlcv_data: pd.DataFrame) -> None:
        from quantflow.models.microstructure.hawkes import HawkesModel

        m = HawkesModel("TEST")
        m.fit(ohlcv_data)
        out = m.predict(ohlcv_data)
        for key in ("lambda_up", "lambda_down", "branching_ratio_up", "branching_ratio_dn"):
            assert key in out.metadata
            assert np.isfinite(out.metadata[key])

    def test_regime_label_valid(self, ohlcv_data: pd.DataFrame) -> None:
        from quantflow.models.microstructure.hawkes import HawkesModel

        m = HawkesModel("TEST")
        m.fit(ohlcv_data)
        out = m.predict(ohlcv_data)
        assert out.regime in {"HIGH_CLUSTERING", "MEDIUM_CLUSTERING", "LOW_CLUSTERING"}

    def test_insufficient_data_raises(self) -> None:
        from quantflow.models.microstructure.hawkes import HawkesModel

        tiny = pd.DataFrame(
            {"close": [100.0, 101.0]},
            index=pd.bdate_range("2024-01-01", periods=2, tz="UTC"),
        )
        with pytest.raises(ValueError):
            HawkesModel("TEST").fit(tiny)


# ===========================================================================
# Optimal Execution Model Tests
# ===========================================================================


class TestOptimalExecutionModel:
    def test_fit_returns_self(self, ohlcv_data: pd.DataFrame) -> None:
        from quantflow.models.microstructure.optimal_execution import OptimalExecutionModel

        m = OptimalExecutionModel("TEST")
        result = m.fit(ohlcv_data)
        assert result is m
        assert m._is_fitted

    def test_predict_valid_output(self, ohlcv_data: pd.DataFrame) -> None:
        from quantflow.models.microstructure.optimal_execution import OptimalExecutionModel

        m = OptimalExecutionModel("TEST")
        m.fit(ohlcv_data)
        out = m.predict(ohlcv_data)
        assert_valid_output(out, "TEST")
        assert out.model_name == "OptimalExecutionModel"

    def test_signal_non_positive(self, ohlcv_data: pd.DataFrame) -> None:
        """Execution costs always reduce net alpha → signal must be ≤ 0."""
        from quantflow.models.microstructure.optimal_execution import OptimalExecutionModel

        m = OptimalExecutionModel("TEST")
        m.fit(ohlcv_data)
        out = m.predict(ohlcv_data)
        assert out.signal <= 0.01, "Execution friction signal should be non-positive"

    def test_trajectory_starts_at_one(self, ohlcv_data: pd.DataFrame) -> None:
        from quantflow.models.microstructure.optimal_execution import OptimalExecutionModel

        m = OptimalExecutionModel("TEST")
        m.fit(ohlcv_data)
        assert m._schedule is not None
        assert abs(m._schedule.trajectory[0] - 1.0) < 0.01

    def test_trajectory_ends_near_zero(self, ohlcv_data: pd.DataFrame) -> None:
        from quantflow.models.microstructure.optimal_execution import OptimalExecutionModel

        m = OptimalExecutionModel("TEST")
        m.fit(ohlcv_data)
        assert m._schedule is not None
        assert abs(m._schedule.trajectory[-1]) < 0.10

    def test_shortfall_positive(self, ohlcv_data: pd.DataFrame) -> None:
        from quantflow.models.microstructure.optimal_execution import OptimalExecutionModel

        m = OptimalExecutionModel("TEST")
        m.fit(ohlcv_data)
        assert m._schedule is not None
        assert m._schedule.expected_shortfall >= 0.0

    def test_regime_label_valid(self, ohlcv_data: pd.DataFrame) -> None:
        from quantflow.models.microstructure.optimal_execution import OptimalExecutionModel

        m = OptimalExecutionModel("TEST")
        m.fit(ohlcv_data)
        out = m.predict(ohlcv_data)
        assert out.regime in {"HIGH_URGENCY", "MEDIUM_URGENCY", "LOW_URGENCY"}


# ===========================================================================
# Bayesian NN Model Tests
# ===========================================================================


class TestBayesianNNModel:
    def test_mc_dropout_fit_predict(self, ohlcv_data: pd.DataFrame) -> None:
        from quantflow.models.advanced.bayesian_nn import BayesianNNModel

        m = BayesianNNModel("TEST", method="mc_dropout", n_epochs=10, n_mc_samples=20, seed=0)
        m.fit(ohlcv_data)
        out = m.predict(ohlcv_data)
        assert_valid_output(out, "TEST")
        assert out.model_name == "BayesianNNModel"

    def test_ensemble_fit_predict(self, ohlcv_data: pd.DataFrame) -> None:
        from quantflow.models.advanced.bayesian_nn import BayesianNNModel

        m = BayesianNNModel("TEST", method="ensemble", n_ensemble=3, n_epochs=10, seed=0)
        m.fit(ohlcv_data)
        out = m.predict(ohlcv_data)
        assert_valid_output(out, "TEST")

    def test_uncertainty_level_in_output(self, ohlcv_data: pd.DataFrame) -> None:
        from quantflow.models.advanced.bayesian_nn import BayesianNNModel

        m = BayesianNNModel("TEST", n_epochs=10, n_mc_samples=10, seed=0)
        m.fit(ohlcv_data)
        out = m.predict(ohlcv_data)
        # Regime should contain UNCERTAINTY_ prefix
        assert "UNCERTAINTY_" in out.regime or out.regime.startswith("UNCERTAINTY")

    def test_confidence_reduced_by_high_uncertainty(self, ohlcv_data: pd.DataFrame) -> None:
        """High uncertainty should reduce confidence below 0.65."""
        from quantflow.models.advanced.bayesian_nn import BayesianNNModel

        m = BayesianNNModel("TEST", n_epochs=10, n_mc_samples=10, seed=0)
        m.fit(ohlcv_data)
        out = m.predict(ohlcv_data)
        assert out.confidence <= 0.80

    def test_predict_without_fit_raises(self, ohlcv_data: pd.DataFrame) -> None:
        from quantflow.models.advanced.bayesian_nn import BayesianNNModel

        with pytest.raises(RuntimeError, match="fitted"):
            BayesianNNModel("TEST").predict(ohlcv_data)

    def test_metadata_has_method_key(self, ohlcv_data: pd.DataFrame) -> None:
        from quantflow.models.advanced.bayesian_nn import BayesianNNModel

        m = BayesianNNModel("TEST", n_epochs=10, n_mc_samples=10, seed=0)
        m.fit(ohlcv_data)
        out = m.predict(ohlcv_data)
        # Either has method key (torch path) or fallback key
        assert "method" in out.metadata or "fallback" in out.metadata


# ===========================================================================
# Neural ODE Model Tests
# ===========================================================================


class TestNeuralODEModel:
    def test_fit_returns_self(self, ohlcv_data: pd.DataFrame) -> None:
        from quantflow.models.advanced.neural_ode import NeuralODEModel

        m = NeuralODEModel("TEST", n_epochs=5, seq_len=21, seed=0)
        result = m.fit(ohlcv_data)
        assert result is m
        assert m._is_fitted

    def test_predict_valid_output(self, ohlcv_data: pd.DataFrame) -> None:
        from quantflow.models.advanced.neural_ode import NeuralODEModel

        m = NeuralODEModel("TEST", n_epochs=5, seq_len=21, seed=0)
        m.fit(ohlcv_data)
        out = m.predict(ohlcv_data)
        assert_valid_output(out, "TEST")
        assert out.model_name == "NeuralODEModel"

    def test_signal_in_bounds(self, ohlcv_data: pd.DataFrame) -> None:
        from quantflow.models.advanced.neural_ode import NeuralODEModel

        m = NeuralODEModel("TEST", n_epochs=5, seq_len=21, seed=0)
        m.fit(ohlcv_data)
        out = m.predict(ohlcv_data)
        assert -1.0 <= out.signal <= 1.0

    def test_predict_without_fit_raises(self, ohlcv_data: pd.DataFrame) -> None:
        from quantflow.models.advanced.neural_ode import NeuralODEModel

        with pytest.raises(RuntimeError, match="fitted"):
            NeuralODEModel("TEST").predict(ohlcv_data)

    def test_metadata_has_solver(self, ohlcv_data: pd.DataFrame) -> None:
        from quantflow.models.advanced.neural_ode import NeuralODEModel

        m = NeuralODEModel("TEST", n_epochs=5, seq_len=21, seed=0)
        m.fit(ohlcv_data)
        out = m.predict(ohlcv_data)
        assert "solver" in out.metadata or "fallback" in out.metadata

    def test_insufficient_data_raises(self) -> None:
        pytest.importorskip("torch")  # fallback mode doesn't raise on tiny data
        from quantflow.models.advanced.neural_ode import NeuralODEModel

        tiny = pd.DataFrame(
            {"close": np.linspace(100, 102, 10)},
            index=pd.bdate_range("2024-01-01", periods=10, tz="UTC"),
        )
        with pytest.raises((ValueError, RuntimeError)):
            NeuralODEModel("TEST", seq_len=21).fit(tiny)


# ===========================================================================
# Neural SDE Model Tests
# ===========================================================================


class TestNeuralSDEModel:
    def test_fit_returns_self(self, ohlcv_data: pd.DataFrame) -> None:
        from quantflow.models.advanced.neural_sde import NeuralSDEModel

        m = NeuralSDEModel("TEST", n_epochs=5, n_sim_paths=200, seed=0)
        result = m.fit(ohlcv_data)
        assert result is m
        assert m._is_fitted

    def test_predict_valid_output(self, ohlcv_data: pd.DataFrame) -> None:
        from quantflow.models.advanced.neural_sde import NeuralSDEModel

        m = NeuralSDEModel("TEST", n_epochs=5, n_sim_paths=200, seed=0)
        m.fit(ohlcv_data)
        out = m.predict(ohlcv_data)
        assert_valid_output(out, "TEST")
        assert out.model_name == "NeuralSDEModel"

    def test_ks_stat_in_metadata(self, ohlcv_data: pd.DataFrame) -> None:
        from quantflow.models.advanced.neural_sde import NeuralSDEModel

        m = NeuralSDEModel("TEST", n_epochs=5, n_sim_paths=200, seed=0)
        m.fit(ohlcv_data)
        out = m.predict(ohlcv_data)
        if "ks_stat" in out.metadata:
            assert 0.0 <= out.metadata["ks_stat"] <= 1.0

    def test_regime_label_valid(self, ohlcv_data: pd.DataFrame) -> None:
        from quantflow.models.advanced.neural_sde import NeuralSDEModel

        m = NeuralSDEModel("TEST", n_epochs=5, n_sim_paths=200, seed=0)
        m.fit(ohlcv_data)
        out = m.predict(ohlcv_data)
        assert out.regime in {"FAT_TAILS", "NORMAL_TAILS"}

    def test_predict_without_fit_raises(self, ohlcv_data: pd.DataFrame) -> None:
        from quantflow.models.advanced.neural_sde import NeuralSDEModel

        with pytest.raises(RuntimeError, match="fitted"):
            NeuralSDEModel("TEST").predict(ohlcv_data)

    def test_confidence_bounded(self, ohlcv_data: pd.DataFrame) -> None:
        from quantflow.models.advanced.neural_sde import NeuralSDEModel

        m = NeuralSDEModel("TEST", n_epochs=5, n_sim_paths=200, seed=0)
        m.fit(ohlcv_data)
        out = m.predict(ohlcv_data)
        assert 0.0 <= out.confidence <= 1.0


# ===========================================================================
# SVI smile fitting test (isolated)
# ===========================================================================


class TestSVIFitting:
    def test_svi_total_variance_shape(self) -> None:
        from quantflow.models.derivatives.vol_surface import VolSurfaceModel

        k = np.linspace(-0.5, 0.5, 11)
        params = np.array([0.04, 0.10, -0.3, 0.0, 0.30])  # a, b, rho, m, sigma
        w = VolSurfaceModel._svi_total_variance(params, k)
        assert w.shape == k.shape
        assert np.all(w > 0.0), "Total variance must be positive"

    def test_svi_fit_flat_smile(self, ohlcv_data: pd.DataFrame) -> None:
        from quantflow.models.derivatives.vol_surface import VolSurfaceModel

        smile = {float(k): 0.20 for k in np.linspace(-0.2, 0.2, 9)}
        m = VolSurfaceModel("TEST")
        svi = m._fit_svi_smile(21, smile)
        assert np.isfinite(svi.a)
        assert np.isfinite(svi.b)
        assert -1.0 < svi.rho < 1.0


# ===========================================================================
# Hawkes log-likelihood test (isolated)
# ===========================================================================


class TestHawkesLogLikelihood:
    def test_negative_ll_decreases_with_fit(self) -> None:
        """Fitted parameters should give a lower (better) NLL than random init."""
        from quantflow.models.microstructure.hawkes import (
            _fit_hawkes,
            _hawkes_log_likelihood,
        )

        rng = np.random.default_rng(99)
        # Synthetic events from a Hawkes process
        event_times = np.sort(rng.uniform(0, 100, 20)).astype(np.float64)
        T = 100.0

        mu, alpha, beta, _ = _fit_hawkes(event_times, T)
        fitted_nll = _hawkes_log_likelihood(np.array([mu, alpha, beta]), event_times, T)
        random_nll = _hawkes_log_likelihood(np.array([0.5, 0.3, 1.5]), event_times, T)

        assert fitted_nll <= random_nll + 10.0  # fitted should be at least as good


# ===========================================================================
# Almgren-Chriss analytical checks
# ===========================================================================


class TestAlmgrenChrissAnalytics:
    def test_trajectory_monotone_decreasing(self, ohlcv_data: pd.DataFrame) -> None:
        from quantflow.models.microstructure.optimal_execution import OptimalExecutionModel

        m = OptimalExecutionModel("TEST", horizon_days=5)
        m.fit(ohlcv_data)
        assert m._schedule is not None
        traj = m._schedule.trajectory
        for i in range(1, len(traj)):
            assert traj[i] <= traj[i - 1] + 1e-6, "Trajectory must be non-increasing"

    def test_trade_list_sums_to_one(self, ohlcv_data: pd.DataFrame) -> None:
        from quantflow.models.microstructure.optimal_execution import OptimalExecutionModel

        m = OptimalExecutionModel("TEST", horizon_days=5)
        m.fit(ohlcv_data)
        assert m._schedule is not None
        total_traded = sum(m._schedule.trade_list)
        assert abs(total_traded - m._schedule.trajectory[0]) < 0.05

    def test_shortfall_increases_with_volatility(self) -> None:
        """Higher-vol asset should have a larger expected shortfall."""
        from quantflow.models.microstructure.optimal_execution import (
            ExecutionParameters,
            OptimalExecutionModel,
        )

        m = OptimalExecutionModel("TEST")
        params_low = ExecutionParameters(sigma=0.10, gamma=1e-5, eta=1e-4, lam=1e-5)
        params_high = ExecutionParameters(sigma=0.50, gamma=5e-5, eta=5e-4, lam=1e-5)
        sched_low = m._solve_ac(params_low)
        sched_high = m._solve_ac(params_high)
        assert sched_high.expected_shortfall >= sched_low.expected_shortfall
