"""Unit tests for Phase 2 statistical models.

Tests use the synthetic OHLCV fixture from conftest.py to ensure
models can be fitted and produce valid ModelOutput objects.
"""

from __future__ import annotations

from datetime import UTC

import numpy as np
import pandas as pd
import pytest

from quantflow.models.base import BaseQuantModel, ModelOutput
from quantflow.models.ml.base_trainer import WalkForwardEvaluator, WalkForwardResult
from quantflow.models.statistical.factor_pca import PCARiskFactorModel
from quantflow.models.statistical.garch import GARCHModel
from quantflow.models.statistical.kalman import KalmanFilterModel

# ===========================================================================
# Helpers
# ===========================================================================


def _make_ohlcv(n: int = 400, seed: int = 99) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    rets = rng.normal(4e-4, 0.012, n)
    close = 100.0 * np.cumprod(1 + rets)
    high = close * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, n)))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    volume = rng.integers(1_000_000, 10_000_000, n).astype(np.int64)
    df = pd.DataFrame(
        {
            "open": open_.astype(np.float64),
            "high": np.maximum(high, open_).astype(np.float64),
            "low": np.minimum(low, open_).astype(np.float64),
            "close": close.astype(np.float64),
            "volume": volume,
        },
        index=dates,
    )
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].min(axis=1)
    return df


# ===========================================================================
# ModelOutput schema
# ===========================================================================


class TestModelOutput:
    def test_valid_output_constructed(self) -> None:
        from datetime import datetime

        output = ModelOutput(
            model_name="test",
            symbol="AAPL",
            timestamp=datetime.now(tz=UTC),
            signal=0.5,
            confidence=0.6,
            forecast_return=0.01,
            forecast_std=0.02,
        )
        assert output.signal == 0.5
        assert output.confidence == 0.6

    def test_signal_out_of_range_raises(self) -> None:
        from datetime import datetime

        with pytest.raises(Exception):
            ModelOutput(
                model_name="test",
                symbol="AAPL",
                timestamp=datetime.now(tz=UTC),
                signal=1.5,  # out of [-1, 1]
                confidence=0.5,
                forecast_return=0.0,
                forecast_std=0.01,
            )

    def test_negative_forecast_std_raises(self) -> None:
        from datetime import datetime

        with pytest.raises(Exception):
            ModelOutput(
                model_name="test",
                symbol="AAPL",
                timestamp=datetime.now(tz=UTC),
                signal=0.0,
                confidence=0.5,
                forecast_return=0.0,
                forecast_std=-0.01,  # negative → invalid
            )


# ===========================================================================
# BaseQuantModel utilities
# ===========================================================================


class TestBaseQuantModelUtilities:
    """Test shared utilities without needing a concrete model."""

    def test_normalise_signal_zero(self) -> None:
        assert BaseQuantModel.normalise_signal(0.0) == pytest.approx(0.0, abs=1e-10)

    def test_normalise_signal_large_positive(self) -> None:
        s = BaseQuantModel.normalise_signal(100.0)
        assert 0.99 < s <= 1.0

    def test_normalise_signal_large_negative(self) -> None:
        s = BaseQuantModel.normalise_signal(-100.0)
        assert -1.0 <= s < -0.99

    def test_normalise_signal_nan_returns_zero(self) -> None:
        assert BaseQuantModel.normalise_signal(float("nan")) == 0.0

    def test_compute_ic_perfect_correlation(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        ic = BaseQuantModel.compute_ic(s, s)
        assert ic == pytest.approx(1.0, abs=1e-6)

    def test_compute_ic_empty_returns_zero(self) -> None:
        ic = BaseQuantModel.compute_ic(pd.Series([], dtype=float), pd.Series([], dtype=float))
        assert ic == 0.0

    def test_check_numeric_clean_passes(self) -> None:
        BaseQuantModel.check_numeric(np.array([1.0, 2.0, 3.0]), "test")

    def test_check_numeric_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            BaseQuantModel.check_numeric(np.array([1.0, np.nan, 3.0]), "test")


# ===========================================================================
# GARCH model
# ===========================================================================


class TestGARCHModel:
    @pytest.fixture()
    def data(self) -> pd.DataFrame:
        return _make_ohlcv(n=400)

    def test_fit_returns_self(self, data: pd.DataFrame) -> None:
        model = GARCHModel("AAPL")
        result = model.fit(data)
        assert result is model

    def test_is_fitted_after_fit(self, data: pd.DataFrame) -> None:
        model = GARCHModel("AAPL")
        model.fit(data)
        assert model._is_fitted is True

    def test_predict_returns_model_output(self, data: pd.DataFrame) -> None:
        model = GARCHModel("AAPL")
        model.fit(data)
        output = model.predict()
        assert isinstance(output, ModelOutput)

    def test_signal_in_range(self, data: pd.DataFrame) -> None:
        model = GARCHModel("AAPL")
        model.fit(data)
        output = model.predict()
        assert -1.0 <= output.signal <= 1.0

    def test_confidence_in_range(self, data: pd.DataFrame) -> None:
        model = GARCHModel("AAPL")
        model.fit(data)
        output = model.predict()
        assert 0.0 <= output.confidence <= 1.0

    def test_regime_is_valid_string(self, data: pd.DataFrame) -> None:
        model = GARCHModel("AAPL")
        model.fit(data)
        output = model.predict()
        assert output.regime in {"LOW_VOL", "MEDIUM_VOL", "HIGH_VOL", "EXTREME_VOL"}

    def test_forecast_std_positive(self, data: pd.DataFrame) -> None:
        model = GARCHModel("AAPL")
        model.fit(data)
        output = model.predict()
        assert output.forecast_std > 0.0

    def test_predict_without_fit_raises(self) -> None:
        model = GARCHModel("AAPL")
        with pytest.raises(RuntimeError):
            model.predict()

    def test_egarch_variant(self, data: pd.DataFrame) -> None:
        model = GARCHModel("AAPL", model_type="EGARCH")
        model.fit(data)
        output = model.predict()
        assert isinstance(output, ModelOutput)

    def test_metadata_contains_aic(self, data: pd.DataFrame) -> None:
        model = GARCHModel("AAPL")
        model.fit(data)
        output = model.predict()
        assert "aic" in output.metadata


# ===========================================================================
# Kalman Filter model
# ===========================================================================


class TestKalmanFilterModel:
    @pytest.fixture()
    def data(self) -> pd.DataFrame:
        return _make_ohlcv(n=300)

    def test_fit_trend_mode(self, data: pd.DataFrame) -> None:
        model = KalmanFilterModel("AAPL", application="trend")
        model.fit(data)
        assert model._is_fitted is True

    def test_predict_returns_output(self, data: pd.DataFrame) -> None:
        model = KalmanFilterModel("AAPL", application="trend")
        model.fit(data)
        output = model.predict()
        assert isinstance(output, ModelOutput)

    def test_signal_in_range(self, data: pd.DataFrame) -> None:
        model = KalmanFilterModel("AAPL", application="trend")
        model.fit(data)
        output = model.predict()
        assert -1.0 <= output.signal <= 1.0

    def test_confidence_in_range(self, data: pd.DataFrame) -> None:
        model = KalmanFilterModel("AAPL", application="trend")
        model.fit(data)
        output = model.predict()
        assert 0.0 <= output.confidence <= 1.0

    def test_beta_mode_metadata(self, data: pd.DataFrame) -> None:
        model = KalmanFilterModel("AAPL", application="beta")
        model.fit(data)
        output = model.predict()
        assert "dynamic_beta" in output.metadata


# ===========================================================================
# PCA Factor model
# ===========================================================================


class TestPCARiskFactorModel:
    @pytest.fixture()
    def data(self) -> pd.DataFrame:
        return _make_ohlcv(n=350)

    def test_fit_and_predict(self, data: pd.DataFrame) -> None:
        model = PCARiskFactorModel("AAPL")
        model.fit(data)
        output = model.predict()
        assert isinstance(output, ModelOutput)

    def test_n_factors_at_least_one(self, data: pd.DataFrame) -> None:
        model = PCARiskFactorModel("AAPL")
        model.fit(data)
        assert model._n_factors >= 1

    def test_signal_in_range(self, data: pd.DataFrame) -> None:
        model = PCARiskFactorModel("AAPL")
        model.fit(data)
        output = model.predict()
        assert -1.0 <= output.signal <= 1.0

    def test_metadata_scree_present(self, data: pd.DataFrame) -> None:
        model = PCARiskFactorModel("AAPL")
        model.fit(data)
        output = model.predict()
        assert "scree" in output.metadata


# ===========================================================================
# Walk-forward evaluator
# ===========================================================================


class TestWalkForwardEvaluator:
    @pytest.fixture()
    def data(self) -> pd.DataFrame:
        return _make_ohlcv(n=600)

    def test_evaluate_garch(self, data: pd.DataFrame) -> None:
        evaluator = WalkForwardEvaluator(n_splits=3, min_train_size=200)
        result: WalkForwardResult = evaluator.evaluate(
            model_factory=lambda: GARCHModel("AAPL"),
            data=data,
            forecast_horizon=21,
        )
        assert isinstance(result, WalkForwardResult)
        assert result.model_name == "GARCH_GARCH"
        assert result.symbol == "AAPL"

    def test_ic_series_has_correct_length(self, data: pd.DataFrame) -> None:
        evaluator = WalkForwardEvaluator(n_splits=3, min_train_size=200)
        result = evaluator.evaluate(
            model_factory=lambda: GARCHModel("AAPL"),
            data=data,
        )
        # At least some folds should have been evaluated
        assert len(result.ic_series) > 0

    def test_hit_rate_in_range(self, data: pd.DataFrame) -> None:
        evaluator = WalkForwardEvaluator(n_splits=3, min_train_size=200)
        result = evaluator.evaluate(
            model_factory=lambda: GARCHModel("AAPL"),
            data=data,
        )
        assert 0.0 <= result.hit_rate <= 1.0

    def test_outputs_list_populated(self, data: pd.DataFrame) -> None:
        evaluator = WalkForwardEvaluator(n_splits=3, min_train_size=200)
        result = evaluator.evaluate(
            model_factory=lambda: GARCHModel("AAPL"),
            data=data,
        )
        assert len(result.outputs) > 0
        for o in result.outputs:
            assert isinstance(o, ModelOutput)

    def test_passes_threshold_property(self, data: pd.DataFrame) -> None:
        evaluator = WalkForwardEvaluator(n_splits=3, min_train_size=200)
        result = evaluator.evaluate(
            model_factory=lambda: GARCHModel("AAPL"),
            data=data,
        )
        # These are just bool flags — must exist
        assert isinstance(result.passes_ic_threshold, bool)
        assert isinstance(result.passes_icir_threshold, bool)

    def test_kalman_evaluator(self, data: pd.DataFrame) -> None:
        evaluator = WalkForwardEvaluator(n_splits=2, min_train_size=200)
        result = evaluator.evaluate(
            model_factory=lambda: KalmanFilterModel("SPY", application="trend"),
            data=data,
        )
        assert isinstance(result, WalkForwardResult)

    def test_pca_evaluator(self, data: pd.DataFrame) -> None:
        evaluator = WalkForwardEvaluator(n_splits=2, min_train_size=200)
        result = evaluator.evaluate(
            model_factory=lambda: PCARiskFactorModel("MSFT"),
            data=data,
        )
        assert isinstance(result, WalkForwardResult)
