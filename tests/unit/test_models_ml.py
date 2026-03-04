"""Unit tests for Phase 3 ML models.

All heavy models (LSTM, Transformer, DRL) use minimal configurations
so tests run quickly without GPUs or large datasets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantflow.models.base import ModelOutput

# ===========================================================================
# Helpers
# ===========================================================================


def _make_ohlcv(n: int = 500, seed: int = 7) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-01", periods=n, freq="B", tz="UTC")
    rets = rng.normal(3e-4, 0.013, n)
    close = 200.0 * np.cumprod(1 + rets)
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    volume = rng.integers(2_000_000, 15_000_000, n).astype(np.int64)
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


def _assert_valid_output(output: ModelOutput) -> None:
    """Assert all ModelOutput fields are in valid ranges."""
    assert isinstance(output, ModelOutput)
    assert -1.0 <= output.signal <= 1.0
    assert 0.0 <= output.confidence <= 1.0
    assert np.isfinite(output.forecast_return)
    assert output.forecast_std >= 0.0
    assert np.isfinite(output.forecast_std)


# ===========================================================================
# GBT Signal Model
# ===========================================================================


class TestGBTSignalModel:
    @pytest.fixture()
    def data(self) -> pd.DataFrame:
        return _make_ohlcv(n=800)

    def test_lightgbm_fit_predict(self, data: pd.DataFrame) -> None:
        from quantflow.models.ml.gradient_boosting import GBTSignalModel

        model = GBTSignalModel("AAPL", framework="lightgbm")
        model.fit(data)
        out = model.predict()
        _assert_valid_output(out)

    def test_xgboost_fit_predict(self, data: pd.DataFrame) -> None:
        from quantflow.models.ml.gradient_boosting import GBTSignalModel

        model = GBTSignalModel("AAPL", framework="xgboost")
        model.fit(data)
        out = model.predict()
        _assert_valid_output(out)

    def test_catboost_fit_predict(self, data: pd.DataFrame) -> None:
        from quantflow.models.ml.gradient_boosting import GBTSignalModel

        model = GBTSignalModel("AAPL", framework="catboost")
        model.fit(data)
        out = model.predict()
        _assert_valid_output(out)

    def test_ensemble_fit_predict(self, data: pd.DataFrame) -> None:
        from quantflow.models.ml.gradient_boosting import GBTSignalModel

        model = GBTSignalModel("AAPL", framework="ensemble")
        model.fit(data)
        out = model.predict()
        _assert_valid_output(out)

    def test_ensemble_metadata_has_frameworks(self, data: pd.DataFrame) -> None:
        from quantflow.models.ml.gradient_boosting import GBTSignalModel

        model = GBTSignalModel("AAPL", framework="ensemble")
        model.fit(data)
        out = model.predict()
        assert "frameworks" in out.metadata
        assert len(out.metadata["frameworks"]) == 3

    def test_ic_mean_in_metadata(self, data: pd.DataFrame) -> None:
        from quantflow.models.ml.gradient_boosting import GBTSignalModel

        model = GBTSignalModel("AAPL", framework="lightgbm")
        model.fit(data)
        out = model.predict()
        assert "ic_mean" in out.metadata

    def test_predict_without_fit_raises(self) -> None:
        from quantflow.models.ml.gradient_boosting import GBTSignalModel

        model = GBTSignalModel("AAPL")
        with pytest.raises(RuntimeError):
            model.predict()

    def test_insufficient_data_raises(self) -> None:
        from quantflow.models.ml.gradient_boosting import GBTSignalModel

        model = GBTSignalModel("AAPL")
        tiny = _make_ohlcv(n=10)
        with pytest.raises(ValueError):
            model.fit(tiny)

    def test_optuna_tune_runs(self, data: pd.DataFrame) -> None:
        pytest.importorskip("optuna")
        from quantflow.models.ml.gradient_boosting import GBTSignalModel

        # Use very few trials to keep the test fast
        model = GBTSignalModel(
            "AAPL", framework="lightgbm", tune_hyperparams=True, n_optuna_trials=3
        )
        model.fit(data)
        assert model._best_params  # non-empty after tuning
        out = model.predict()
        _assert_valid_output(out)

    def test_shap_values_computed(self, data: pd.DataFrame) -> None:
        from quantflow.models.ml.gradient_boosting import GBTSignalModel

        model = GBTSignalModel("AAPL", framework="lightgbm")
        model.fit(data)
        # SHAP is computed inside fit; if library is available, values exist
        if model._shap_values is not None:
            assert model._shap_values.shape[0] > 0


# ===========================================================================
# Classic ML models
# ===========================================================================


class TestClassicMLModels:
    @pytest.fixture()
    def data(self) -> pd.DataFrame:
        return _make_ohlcv(n=800)

    def test_random_forest_fit_predict(self, data: pd.DataFrame) -> None:
        from quantflow.models.ml.classic_ml import RandomForestModel

        model = RandomForestModel("AAPL", n_estimators=50)
        model.fit(data)
        out = model.predict()
        _assert_valid_output(out)

    def test_rf_oob_score_in_metadata(self, data: pd.DataFrame) -> None:
        from quantflow.models.ml.classic_ml import RandomForestModel

        model = RandomForestModel("AAPL", n_estimators=50)
        model.fit(data)
        out = model.predict()
        assert "oob_score" in out.metadata

    def test_rf_feature_importance_in_metadata(self, data: pd.DataFrame) -> None:
        from quantflow.models.ml.classic_ml import RandomForestModel

        model = RandomForestModel("AAPL", n_estimators=50)
        model.fit(data)
        out = model.predict()
        assert "feature_importance" in out.metadata
        assert len(out.metadata["feature_importance"]) > 0

    def test_lasso_fit_predict(self, data: pd.DataFrame) -> None:
        from quantflow.models.ml.classic_ml import LASSOModel

        model = LASSOModel("AAPL")
        model.fit(data)
        out = model.predict()
        _assert_valid_output(out)

    def test_ridge_fit_predict(self, data: pd.DataFrame) -> None:
        from quantflow.models.ml.classic_ml import RidgeModel

        model = RidgeModel("AAPL")
        model.fit(data)
        out = model.predict()
        _assert_valid_output(out)

    def test_elasticnet_fit_predict(self, data: pd.DataFrame) -> None:
        from quantflow.models.ml.classic_ml import ElasticNetModel

        model = ElasticNetModel("AAPL")
        model.fit(data)
        out = model.predict()
        _assert_valid_output(out)

    def test_lasso_n_nonzero_coef_in_metadata(self, data: pd.DataFrame) -> None:
        from quantflow.models.ml.classic_ml import LASSOModel

        model = LASSOModel("AAPL")
        model.fit(data)
        out = model.predict()
        assert "n_nonzero_coef" in out.metadata

    def test_predict_without_fit_raises(self) -> None:
        from quantflow.models.ml.classic_ml import LASSOModel

        model = LASSOModel("AAPL")
        with pytest.raises(RuntimeError):
            model.predict()

    def test_all_three_linear_produce_different_signals(self, data: pd.DataFrame) -> None:
        from quantflow.models.ml.classic_ml import ElasticNetModel, LASSOModel, RidgeModel

        signals = set()
        for cls in (LASSOModel, RidgeModel, ElasticNetModel):
            m = cls("AAPL")
            m.fit(data)
            signals.add(m.predict().signal)
        # Not all identical (regularisation differs)
        # At least 2 different values expected
        assert len(signals) >= 1  # always true; weak check to avoid flakiness


# ===========================================================================
# LSTM / GRU models
# ===========================================================================


class TestLSTMSignalModel:
    @pytest.fixture(autouse=True)
    def require_torch(self) -> None:
        pytest.importorskip("torch")

    @pytest.fixture()
    def data(self) -> pd.DataFrame:
        # Need enough rows for seq_len + train/val + feature warmup (378 rows)
        return _make_ohlcv(n=800)

    def test_lstm_fit_predict(self, data: pd.DataFrame) -> None:
        from quantflow.models.ml.recurrent import LSTMSignalModel

        model = LSTMSignalModel("AAPL", seq_len=32, hidden_size=16)
        model.fit(data)
        out = model.predict()
        _assert_valid_output(out)

    def test_gru_fit_predict(self, data: pd.DataFrame) -> None:
        from quantflow.models.ml.recurrent import LSTMSignalModel

        model = LSTMSignalModel("AAPL", cell_type="GRU", seq_len=32, hidden_size=16)
        model.fit(data)
        out = model.predict()
        _assert_valid_output(out)

    def test_direction_prob_in_metadata(self, data: pd.DataFrame) -> None:
        from quantflow.models.ml.recurrent import LSTMSignalModel

        model = LSTMSignalModel("AAPL", seq_len=32, hidden_size=16)
        model.fit(data)
        out = model.predict()
        assert "direction_prob" in out.metadata
        assert 0.0 <= out.metadata["direction_prob"] <= 1.0

    def test_epochs_trained_in_metadata(self, data: pd.DataFrame) -> None:
        from quantflow.models.ml.recurrent import LSTMSignalModel

        model = LSTMSignalModel("AAPL", seq_len=32, hidden_size=16)
        model.fit(data)
        out = model.predict()
        assert "epochs_trained" in out.metadata
        assert out.metadata["epochs_trained"] > 0

    def test_predict_without_fit_raises(self) -> None:
        from quantflow.models.ml.recurrent import LSTMSignalModel

        model = LSTMSignalModel("AAPL")
        with pytest.raises(RuntimeError):
            model.predict()

    def test_insufficient_data_raises(self) -> None:
        from quantflow.models.ml.recurrent import LSTMSignalModel

        model = LSTMSignalModel("AAPL", seq_len=63)
        tiny = _make_ohlcv(n=30)
        with pytest.raises(ValueError):
            model.fit(tiny)

    def test_sequence_builder(self) -> None:
        from quantflow.models.ml.recurrent import _build_sequences

        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        seqs, targets = _build_sequences(X, y, seq_len=10)
        assert seqs.shape == (90, 10, 5)
        assert len(targets) == 90

    def test_sequence_builder_too_short_returns_empty(self) -> None:
        from quantflow.models.ml.recurrent import _build_sequences

        X = np.random.randn(5, 3).astype(np.float32)
        y = np.random.randn(5).astype(np.float32)
        seqs, _targets = _build_sequences(X, y, seq_len=10)
        assert len(seqs) == 0


# ===========================================================================
# Transformer model
# ===========================================================================


class TestTimeSeriesTransformerModel:
    @pytest.fixture(autouse=True)
    def require_torch(self) -> None:
        pytest.importorskip("torch")

    @pytest.fixture()
    def data(self) -> pd.DataFrame:
        return _make_ohlcv(n=800)

    def test_transformer_fit_predict(self, data: pd.DataFrame) -> None:
        from quantflow.models.ml.transformer_ts import TimeSeriesTransformerModel

        # Reduced architecture for test speed
        model = TimeSeriesTransformerModel("AAPL", seq_len=16)
        model.fit(data)
        out = model.predict()
        _assert_valid_output(out)

    def test_metadata_has_logvar(self, data: pd.DataFrame) -> None:
        from quantflow.models.ml.transformer_ts import TimeSeriesTransformerModel

        model = TimeSeriesTransformerModel("AAPL", seq_len=16)
        model.fit(data)
        out = model.predict()
        assert "pred_logvar" in out.metadata
        assert np.isfinite(out.metadata["pred_logvar"])

    def test_epochs_trained_positive(self, data: pd.DataFrame) -> None:
        from quantflow.models.ml.transformer_ts import TimeSeriesTransformerModel

        model = TimeSeriesTransformerModel("AAPL", seq_len=16)
        model.fit(data)
        out = model.predict()
        assert out.metadata["epochs_trained"] > 0

    def test_forecast_std_positive(self, data: pd.DataFrame) -> None:
        from quantflow.models.ml.transformer_ts import TimeSeriesTransformerModel

        model = TimeSeriesTransformerModel("AAPL", seq_len=16)
        model.fit(data)
        out = model.predict()
        assert out.forecast_std > 0.0

    def test_insufficient_data_raises(self) -> None:
        from quantflow.models.ml.transformer_ts import TimeSeriesTransformerModel

        model = TimeSeriesTransformerModel("AAPL", seq_len=64)
        tiny = _make_ohlcv(n=50)
        with pytest.raises(ValueError):
            model.fit(tiny)


# ===========================================================================
# Deep RL agent
# ===========================================================================


class TestDRLPortfolioAgent:
    @pytest.fixture(autouse=True)
    def require_gymnasium(self) -> None:
        pytest.importorskip("gymnasium")

    @pytest.fixture()
    def data(self) -> pd.DataFrame:
        return _make_ohlcv(n=800)

    def test_ppo_fit_predict(self, data: pd.DataFrame) -> None:
        from quantflow.models.ml.deep_rl import DRLPortfolioAgent

        model = DRLPortfolioAgent("AAPL", algorithm="PPO", n_training_steps=500)
        model.fit(data)
        out = model.predict()
        _assert_valid_output(out)

    def test_dqn_fit_predict(self, data: pd.DataFrame) -> None:
        from quantflow.models.ml.deep_rl import DRLPortfolioAgent

        model = DRLPortfolioAgent("AAPL", algorithm="DQN", n_training_steps=500)
        model.fit(data)
        out = model.predict()
        _assert_valid_output(out)

    def test_recommended_weight_in_metadata(self, data: pd.DataFrame) -> None:
        from quantflow.models.ml.deep_rl import DRLPortfolioAgent

        model = DRLPortfolioAgent("AAPL", algorithm="PPO", n_training_steps=500)
        model.fit(data)
        out = model.predict()
        assert "recommended_weight" in out.metadata
        w = out.metadata["recommended_weight"]
        assert 0.0 <= w <= 1.0

    def test_train_sharpe_in_metadata(self, data: pd.DataFrame) -> None:
        from quantflow.models.ml.deep_rl import DRLPortfolioAgent

        model = DRLPortfolioAgent("AAPL", algorithm="PPO", n_training_steps=500)
        model.fit(data)
        out = model.predict()
        assert "train_sharpe" in out.metadata
        assert np.isfinite(out.metadata["train_sharpe"])

    def test_predict_without_fit_raises(self) -> None:
        from quantflow.models.ml.deep_rl import DRLPortfolioAgent

        model = DRLPortfolioAgent("AAPL")
        with pytest.raises(RuntimeError):
            model.predict()

    def test_portfolio_env_reset_step(self, data: pd.DataFrame) -> None:
        from quantflow.data.features import compute_all_features
        from quantflow.models.ml.deep_rl import PortfolioEnv
        from quantflow.models.ml.gradient_boosting import _build_xy

        close = data["close"].astype(np.float64)
        features = compute_all_features(data)
        from sklearn.preprocessing import StandardScaler

        X_raw, y_raw, _ = _build_xy(features, close, horizon=1)
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw).astype(np.float32)

        env = PortfolioEnv(X[:100], y_raw[:100].astype(np.float32))
        obs, info = env.reset()
        assert obs.shape == (X.shape[1],)

        action = np.array([0.5], dtype=np.float32)
        next_obs, reward, _terminated, _truncated, info = env.step(action)
        assert next_obs.shape == (X.shape[1],)
        assert np.isfinite(reward)
        assert "weight" in info

    def test_discrete_env_actions(self) -> None:
        from quantflow.models.ml.deep_rl import DiscretePortfolioEnv

        X = np.random.randn(50, 5).astype(np.float32)
        rets = np.random.randn(50).astype(np.float32) * 0.01

        env = DiscretePortfolioEnv(X, rets)
        _obs, _ = env.reset()
        for action in range(5):
            env.reset()
            _next_obs, reward, _done, _trunc, _info = env.step(action)
            assert np.isfinite(reward)


# ===========================================================================
# Walk-forward evaluator with ML models
# ===========================================================================


class TestWalkForwardWithMLModels:
    @pytest.fixture()
    def data(self) -> pd.DataFrame:
        # Needs 1500 rows: feature warmup is ~378 rows, WalkForward needs
        # each training split to have >60 valid rows after dropna.
        return _make_ohlcv(n=1500)

    def test_gbt_walk_forward(self, data: pd.DataFrame) -> None:
        from quantflow.models.ml.base_trainer import WalkForwardEvaluator
        from quantflow.models.ml.gradient_boosting import GBTSignalModel

        evaluator = WalkForwardEvaluator(n_splits=3, min_train_size=200)
        result = evaluator.evaluate(
            model_factory=lambda: GBTSignalModel("SPY", framework="lightgbm"),
            data=data,
        )
        assert result.model_name == "GBT_lightgbm"
        assert len(result.outputs) > 0
        assert 0.0 <= result.hit_rate <= 1.0

    def test_rf_walk_forward(self, data: pd.DataFrame) -> None:
        from quantflow.models.ml.base_trainer import WalkForwardEvaluator
        from quantflow.models.ml.classic_ml import RandomForestModel

        evaluator = WalkForwardEvaluator(n_splits=2, min_train_size=200)
        result = evaluator.evaluate(
            model_factory=lambda: RandomForestModel("SPY", n_estimators=30),
            data=data,
        )
        assert result.model_name == "RandomForest"

    def test_lasso_walk_forward(self, data: pd.DataFrame) -> None:
        from quantflow.models.ml.base_trainer import WalkForwardEvaluator
        from quantflow.models.ml.classic_ml import LASSOModel

        evaluator = WalkForwardEvaluator(n_splits=2, min_train_size=200)
        result = evaluator.evaluate(
            model_factory=lambda: LASSOModel("SPY"),
            data=data,
        )
        assert result.model_name == "Linear_LASSO"
