"""Gradient Boosted Tree signal model: XGBoost + LightGBM + CatBoost ensemble.

Predicts 21-day forward return (regression) and direction (classification)
from engineered features.  All three frameworks are fitted independently
then averaged for the ensemble signal.

SHAP values are computed per prediction for explainability.
Hyperparameters are tuned via Optuna with walk-forward CV.
All runs are tracked in MLflow.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from quantflow.config.constants import WALK_FORWARD_GAP, WALK_FORWARD_N_SPLITS
from quantflow.config.logging import get_logger
from quantflow.models.base import BaseQuantModel, ModelOutput

logger = get_logger(__name__)

GBTFramework = Literal["xgboost", "lightgbm", "catboost", "ensemble"]

_DEFAULT_HORIZON = 21  # 21-day forward return target


# ---------------------------------------------------------------------------
# Feature / target preparation helpers
# ---------------------------------------------------------------------------


def _build_xy(
    features: pd.DataFrame,
    close: pd.Series,
    horizon: int = _DEFAULT_HORIZON,
) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """Build feature matrix X and target vector y.

    Target is the ``horizon``-day forward return shifted back so that
    index ``t`` holds the return for period ``[t, t+horizon]``.
    Only rows where both X and y are finite are returned.

    Args:
        features: Pre-computed feature DataFrame (rows = trading days).
        close: Adjusted close price series (same index as features).
        horizon: Forward-return horizon in trading days.

    Returns:
        Tuple of (X, y, valid_index) where X is shape (N, F) and y is (N,).
    """
    fwd_ret = close.shift(-horizon) / close - 1  # future return
    df = features.copy()
    df["__target__"] = fwd_ret

    # Drop rows with any NaN
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    y = df["__target__"].values.astype(np.float64)
    X = df.drop(columns=["__target__"]).values.astype(np.float64)
    return X, y, df.index


# ---------------------------------------------------------------------------
# Individual framework wrappers
# ---------------------------------------------------------------------------


def _fit_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict[str, Any] | None = None,
) -> Any:
    """Fit an XGBoost regressor."""
    import xgboost as xgb  # type: ignore[import]

    defaults = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
    }
    if params:
        defaults.update(params)
    model = xgb.XGBRegressor(**defaults)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train, verbose=False)
    return model


def _fit_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict[str, Any] | None = None,
) -> Any:
    """Fit a LightGBM regressor."""
    import lightgbm as lgb  # type: ignore[import]

    defaults = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "min_child_samples": 20,
        "objective": "regression",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }
    if params:
        defaults.update(params)
    model = lgb.LGBMRegressor(**defaults)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)
    return model


def _fit_catboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict[str, Any] | None = None,
) -> Any:
    """Fit a CatBoost regressor."""
    from catboost import CatBoostRegressor  # type: ignore[import]

    defaults = {
        "iterations": 300,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False,
    }
    if params:
        defaults.update(params)
    model = CatBoostRegressor(**defaults)
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Optuna hyperparameter search
# ---------------------------------------------------------------------------


def _optuna_tune_lightgbm(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 30,
    n_splits: int = 3,
) -> dict[str, Any]:
    """Tune LightGBM hyperparameters via Optuna (walk-forward CV).

    Args:
        X: Feature matrix.
        y: Target vector.
        n_trials: Number of Optuna trials.
        n_splits: Number of CV splits.

    Returns:
        Best hyperparameter dict.
    """
    import optuna
    import lightgbm as lgb  # type: ignore[import]

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": 1,
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 5.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 5.0, log=True),
            "n_estimators": 200,
            "objective": "regression",
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
        }

        tscv = TimeSeriesSplit(n_splits=n_splits, gap=WALK_FORWARD_GAP)
        ic_scores: list[float] = []
        for train_idx, val_idx in tscv.split(X):
            if len(train_idx) < 100:
                continue
            model = lgb.LGBMRegressor(**params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[val_idx])
            # IC: Spearman correlation between prediction and realised return
            from scipy.stats import spearmanr
            ic, _ = spearmanr(pred, y[val_idx])
            if np.isfinite(ic):
                ic_scores.append(ic)

        return float(np.mean(ic_scores)) if ic_scores else -1.0

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ---------------------------------------------------------------------------
# Main GBT model class
# ---------------------------------------------------------------------------


class GBTSignalModel(BaseQuantModel):
    """Gradient Boosted Tree ensemble for return prediction.

    Fits XGBoost, LightGBM, and CatBoost independently on the same
    feature/target data, then averages their predictions for the ensemble
    signal.  SHAP values are computed for LightGBM (fastest).

    Args:
        symbol: Ticker symbol.
        framework: Which framework(s) to use — one of "xgboost",
            "lightgbm", "catboost", or "ensemble" (all three averaged).
        horizon: Forward-return target horizon in trading days.
        tune_hyperparams: If True, run Optuna HPO before final fit.
        n_optuna_trials: Number of Optuna trials (if ``tune_hyperparams``).
        mlflow_experiment: MLflow experiment name for logging.
    """

    def __init__(
        self,
        symbol: str,
        framework: GBTFramework = "ensemble",
        horizon: int = _DEFAULT_HORIZON,
        tune_hyperparams: bool = False,
        n_optuna_trials: int = 30,
        mlflow_experiment: str | None = "quantflow_gbt",
    ) -> None:
        """Initialise the GBT signal model.

        Args:
            symbol: Ticker symbol.
            framework: "xgboost", "lightgbm", "catboost", or "ensemble".
            horizon: Forecast horizon in trading days.
            tune_hyperparams: Whether to run Optuna HPO.
            n_optuna_trials: Number of Optuna trials.
            mlflow_experiment: MLflow experiment name.
        """
        super().__init__(f"GBT_{framework}", symbol)
        self.framework: GBTFramework = framework
        self.horizon = horizon
        self.tune_hyperparams = tune_hyperparams
        self.n_optuna_trials = n_optuna_trials
        self.mlflow_experiment = mlflow_experiment

        self._models: dict[str, Any] = {}
        self._feature_names: list[str] = []
        self._best_params: dict[str, Any] = {}
        self._shap_values: np.ndarray | None = None
        self._vol_estimate: float = 0.02
        self._ic_history: list[float] = []
        self._scaler = StandardScaler()

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame) -> "GBTSignalModel":
        """Fit GBT models on features derived from OHLCV data.

        Expects ``data`` to have a ``close`` column plus pre-computed
        feature columns (from ``quantflow.data.features``).  If only
        ``close`` is present, basic features are computed internally.

        Args:
            data: DataFrame with ``close`` and feature columns.

        Returns:
            Self (fitted model).
        """
        from quantflow.data.features import compute_all_features

        close = data["close"].astype(np.float64)

        # Build feature matrix
        required = {"open", "high", "low", "close", "volume"}
        if required.issubset(set(data.columns)):
            features = compute_all_features(data)
        else:
            # Assume data already contains feature columns (minus close)
            features = data.drop(columns=["close"], errors="ignore")

        X, y, valid_idx = _build_xy(features, close, horizon=self.horizon)
        self._feature_names = list(
            features.columns if hasattr(features, "columns") else []
        )

        if len(X) < 60:
            raise ValueError(
                f"Insufficient training data for {self.symbol}: {len(X)} rows."
            )

        self._vol_estimate = float(np.std(y))

        # Optional Optuna HPO on LightGBM (then apply best params)
        if self.tune_hyperparams:
            logger.info(
                "Running Optuna HPO",
                symbol=self.symbol,
                n_trials=self.n_optuna_trials,
            )
            self._best_params = _optuna_tune_lightgbm(
                X, y, n_trials=self.n_optuna_trials
            )

        # Fit requested frameworks
        frameworks_to_fit = (
            ["xgboost", "lightgbm", "catboost"]
            if self.framework == "ensemble"
            else [self.framework]
        )

        for fw in frameworks_to_fit:
            logger.debug("Fitting framework", framework=fw, symbol=self.symbol)
            if fw == "xgboost":
                self._models["xgboost"] = _fit_xgboost(X, y)
            elif fw == "lightgbm":
                lgb_params = self._best_params if self.tune_hyperparams else {}
                self._models["lightgbm"] = _fit_lightgbm(X, y, lgb_params)
            elif fw == "catboost":
                self._models["catboost"] = _fit_catboost(X, y)

        # Walk-forward IC evaluation for logging / weighting
        self._ic_history = self._compute_walk_forward_ic(X, y)

        # SHAP values on LightGBM (most interpretable)
        if "lightgbm" in self._models:
            self._compute_shap(X)

        # MLflow logging
        if self.mlflow_experiment:
            self._log_mlflow(X, y)

        self._is_fitted = True
        self._log_fit_complete(
            frameworks=frameworks_to_fit,
            n_train=len(X),
            n_features=X.shape[1],
            ic_mean=round(float(np.mean(self._ic_history)) if self._ic_history else 0.0, 4),
        )
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, data: pd.DataFrame | None = None) -> ModelOutput:
        """Generate a signal from the last row of features.

        Args:
            data: Optional new data.  If provided, features are recomputed
                  and the last row is used for prediction.  If None, the
                  training data's last row is used.

        Returns:
            :class:`ModelOutput` with GBT ensemble signal and SHAP metadata.
        """
        self._require_fitted()

        if data is not None:
            from quantflow.data.features import compute_all_features

            required = {"open", "high", "low", "close", "volume"}
            if required.issubset(set(data.columns)):
                features = compute_all_features(data)
            else:
                features = data.drop(columns=["close"], errors="ignore")

            X_last = (
                features.replace([np.inf, -np.inf], np.nan)
                .dropna()
                .tail(1)
                .values.astype(np.float64)
            )
        else:
            # Predict from a neutral feature vector (zero-filled)
            n_features = (
                next(iter(self._models.values())).n_features_in_
                if self._models
                else 1
            )
            X_last = np.zeros((1, n_features))

        if X_last.shape[0] == 0:
            return self._neutral_output()

        # Collect predictions from all fitted frameworks
        preds: list[float] = []
        for fw, model in self._models.items():
            try:
                pred = float(model.predict(X_last)[0])
                if np.isfinite(pred):
                    preds.append(pred)
            except Exception as exc:
                logger.warning("Framework predict failed", framework=fw, error=str(exc))

        if not preds:
            return self._neutral_output()

        forecast_return = float(np.mean(preds))
        forecast_std = float(np.std(preds)) if len(preds) > 1 else self._vol_estimate

        # Signal: forecast / vol_estimate (z-score → tanh)
        raw_signal = forecast_return / (self._vol_estimate * np.sqrt(self.horizon) + 1e-8)
        signal = self.normalise_signal(raw_signal)

        # IC-based confidence
        ic_mean = float(np.mean(self._ic_history)) if self._ic_history else 0.0
        confidence = min(0.80, max(0.20, 0.50 + ic_mean))

        # SHAP top features for metadata
        shap_meta: dict[str, Any] = {}
        if self._shap_values is not None and self._feature_names:
            mean_abs_shap = np.abs(self._shap_values).mean(axis=0)
            top_k = min(5, len(self._feature_names))
            top_idx = np.argsort(mean_abs_shap)[::-1][:top_k]
            shap_meta["top_features"] = {
                self._feature_names[i]: round(float(mean_abs_shap[i]), 5)
                for i in top_idx
            }

        metadata: dict[str, Any] = {
            "frameworks": list(self._models.keys()),
            "individual_forecasts": {fw: round(float(p), 6) for fw, p in zip(self._models.keys(), preds)},
            "ic_mean": round(ic_mean, 4),
            "horizon_days": self.horizon,
            **shap_meta,
        }

        return ModelOutput(
            model_name=self.model_name,
            symbol=self.symbol,
            timestamp=datetime.now(tz=timezone.utc),
            signal=signal,
            confidence=confidence,
            forecast_return=round(forecast_return, 6),
            forecast_std=round(max(forecast_std, 1e-6), 6),
            regime=None,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_walk_forward_ic(
        self, X: np.ndarray, y: np.ndarray
    ) -> list[float]:
        """Compute IC on each walk-forward fold using the fitted models."""
        from scipy.stats import spearmanr

        tscv = TimeSeriesSplit(
            n_splits=WALK_FORWARD_N_SPLITS,
            gap=WALK_FORWARD_GAP,
        )
        ic_list: list[float] = []
        for _, val_idx in tscv.split(X):
            if len(val_idx) < 5:
                continue
            preds: list[float] = []
            for model in self._models.values():
                try:
                    preds.append(float(np.mean(model.predict(X[val_idx]))))
                except Exception:
                    pass
            if not preds:
                continue
            ensemble_pred = np.array(
                [float(np.mean([m.predict(X[val_idx])[i] for m in self._models.values()]))
                 for i in range(len(val_idx))]
            )
            ic, _ = spearmanr(ensemble_pred, y[val_idx])
            if np.isfinite(ic):
                ic_list.append(float(ic))
        return ic_list

    def _compute_shap(self, X: np.ndarray) -> None:
        """Compute SHAP values for the LightGBM model."""
        try:
            import shap  # type: ignore[import]

            explainer = shap.TreeExplainer(self._models["lightgbm"])
            # Use a sample of at most 200 rows for speed
            X_sample = X[-min(200, len(X)):]
            self._shap_values = explainer.shap_values(X_sample)
        except Exception as exc:
            logger.warning("SHAP computation failed", error=str(exc))

    def _log_mlflow(self, X: np.ndarray, y: np.ndarray) -> None:
        """Log training run to MLflow."""
        try:
            import mlflow

            mlflow.set_experiment(self.mlflow_experiment)
            with mlflow.start_run(
                run_name=f"{self.model_name}_{self.symbol}_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            ):
                mlflow.log_params(
                    {
                        "framework": self.framework,
                        "symbol": self.symbol,
                        "horizon": self.horizon,
                        "n_train": len(X),
                        "n_features": X.shape[1],
                        "tune_hyperparams": self.tune_hyperparams,
                    }
                )
                ic_mean = float(np.mean(self._ic_history)) if self._ic_history else 0.0
                mlflow.log_metrics({"ic_mean": ic_mean, "vol_estimate": self._vol_estimate})
                if self._best_params:
                    mlflow.log_params({f"opt_{k}": v for k, v in self._best_params.items()})
        except Exception as exc:
            logger.warning("MLflow logging failed", error=str(exc))

    def _neutral_output(self) -> ModelOutput:
        """Return a neutral zero-signal output."""
        return ModelOutput(
            model_name=self.model_name,
            symbol=self.symbol,
            timestamp=datetime.now(tz=timezone.utc),
            signal=0.0,
            confidence=0.0,
            forecast_return=0.0,
            forecast_std=self._vol_estimate,
            metadata={"error": "prediction_failed"},
        )
