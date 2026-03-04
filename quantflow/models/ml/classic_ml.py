"""Classic ML signal models: Random Forest, LASSO, Ridge, ElasticNet.

All models use rolling standardisation before fitting and TimeSeriesSplit
for cross-validation.  Each inherits BaseQuantModel and returns ModelOutput.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from quantflow.config.constants import WALK_FORWARD_GAP, WALK_FORWARD_N_SPLITS
from quantflow.config.logging import get_logger
from quantflow.models.base import BaseQuantModel, ModelOutput
from quantflow.models.ml.gradient_boosting import _build_xy

logger = get_logger(__name__)

_DEFAULT_HORIZON = 21


# ---------------------------------------------------------------------------
# Shared CV helper
# ---------------------------------------------------------------------------


def _cv_ic(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = WALK_FORWARD_N_SPLITS,
) -> list[float]:
    """Compute per-fold IC using walk-forward CV.

    Args:
        estimator: A scikit-learn regressor (already fitted or cloned per fold).
        X: Feature matrix.
        y: Target vector.
        n_splits: Number of folds.

    Returns:
        List of IC values (Spearman correlation per fold).
    """
    from scipy.stats import spearmanr
    from sklearn.base import clone

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=WALK_FORWARD_GAP)
    ics: list[float] = []
    for train_idx, val_idx in tscv.split(X):
        if len(train_idx) < 50:
            continue
        model_clone = clone(estimator)
        model_clone.fit(X[train_idx], y[train_idx])
        preds = model_clone.predict(X[val_idx])
        ic, _ = spearmanr(preds, y[val_idx])
        if np.isfinite(ic):
            ics.append(float(ic))
    return ics


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------


class RandomForestModel(BaseQuantModel):
    """Random Forest regressor for return prediction.

    Uses OOB score as an internal validation proxy.
    Feature importance is stored for explainability.

    Args:
        symbol: Ticker symbol.
        n_estimators: Number of trees.
        horizon: Forward-return horizon in trading days.
    """

    def __init__(
        self,
        symbol: str,
        n_estimators: int = 500,
        horizon: int = _DEFAULT_HORIZON,
    ) -> None:
        """Initialise the Random Forest model.

        Args:
            symbol: Ticker symbol.
            n_estimators: Number of trees.
            horizon: Forecast horizon in days.
        """
        super().__init__("RandomForest", symbol)
        self.horizon = horizon
        self._model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_features="sqrt",
            oob_score=True,
            n_jobs=-1,
            random_state=42,
        )
        self._scaler = StandardScaler()
        self._feature_names: list[str] = []
        self._vol_estimate: float = 0.02
        self._ic_history: list[float] = []

    def fit(self, data: pd.DataFrame) -> "RandomForestModel":
        """Fit the Random Forest on features derived from OHLCV data.

        Args:
            data: DataFrame with ``close`` and optional feature columns.

        Returns:
            Self (fitted model).
        """
        from quantflow.data.features import compute_all_features

        close = data["close"].astype(np.float64)
        required = {"open", "high", "low", "close", "volume"}
        features = (
            compute_all_features(data)
            if required.issubset(set(data.columns))
            else data.drop(columns=["close"], errors="ignore")
        )
        X_raw, y, _ = _build_xy(features, close, self.horizon)
        self._feature_names = list(features.columns)
        self._vol_estimate = float(np.std(y))

        X = self._scaler.fit_transform(X_raw)
        self._model.fit(X, y)
        self._ic_history = _cv_ic(
            RandomForestRegressor(n_estimators=100, max_features="sqrt", random_state=42),
            X,
            y,
        )
        self._is_fitted = True
        oob = round(float(self._model.oob_score_), 4) if self._model.oob_score else 0.0
        self._log_fit_complete(oob_score=oob, n_train=len(y))
        return self

    def predict(self, data: pd.DataFrame | None = None) -> ModelOutput:
        """Predict forward return and generate signal.

        Args:
            data: Optional new data for prediction.

        Returns:
            :class:`ModelOutput` with RF signal.
        """
        self._require_fitted()
        X_last = self._last_feature_row(data)
        if X_last is None:
            return self._neutral_output()

        X_scaled = self._scaler.transform(X_last)
        # Predict from each tree for uncertainty estimate
        preds = np.array([tree.predict(X_scaled)[0] for tree in self._model.estimators_])
        forecast_return = float(preds.mean())
        forecast_std = float(preds.std())

        raw_signal = forecast_return / (self._vol_estimate * np.sqrt(self.horizon) + 1e-8)
        signal = self.normalise_signal(raw_signal)
        ic_mean = float(np.mean(self._ic_history)) if self._ic_history else 0.0
        confidence = min(0.75, max(0.20, 0.50 + ic_mean))

        # Top-5 feature importances
        importances = self._model.feature_importances_
        top_k = min(5, len(self._feature_names))
        top_idx = np.argsort(importances)[::-1][:top_k]
        feature_imp = {self._feature_names[i]: round(float(importances[i]), 5) for i in top_idx}

        return ModelOutput(
            model_name=self.model_name,
            symbol=self.symbol,
            timestamp=datetime.now(tz=timezone.utc),
            signal=signal,
            confidence=confidence,
            forecast_return=round(forecast_return, 6),
            forecast_std=round(max(forecast_std, 1e-6), 6),
            metadata={"oob_score": round(float(self._model.oob_score_), 4), "feature_importance": feature_imp, "ic_mean": round(ic_mean, 4)},
        )

    def _last_feature_row(self, data: pd.DataFrame | None) -> np.ndarray | None:
        """Extract the last feature row from data or return zeros."""
        if data is None:
            n = self._model.n_features_in_
            return np.zeros((1, n))
        from quantflow.data.features import compute_all_features

        required = {"open", "high", "low", "close", "volume"}
        features = (
            compute_all_features(data)
            if required.issubset(set(data.columns))
            else data.drop(columns=["close"], errors="ignore")
        )
        X = features.replace([np.inf, -np.inf], np.nan).dropna().tail(1).values.astype(np.float64)
        return X if X.shape[0] > 0 else None

    def _neutral_output(self) -> ModelOutput:
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


# ---------------------------------------------------------------------------
# Regularised linear models (LASSO, Ridge, ElasticNet)
# ---------------------------------------------------------------------------

LinearModelType = Literal["lasso", "ridge", "elasticnet"]


class LinearSignalModel(BaseQuantModel):
    """Regularised linear model for return prediction.

    Supports LASSO (L1), Ridge (L2), and ElasticNet (L1+L2).
    Alpha and l1_ratio selected via walk-forward CV.

    Args:
        symbol: Ticker symbol.
        model_type: One of "lasso", "ridge", "elasticnet".
        horizon: Forecast horizon in trading days.
    """

    def __init__(
        self,
        symbol: str,
        model_type: LinearModelType = "lasso",
        horizon: int = _DEFAULT_HORIZON,
    ) -> None:
        """Initialise the linear signal model.

        Args:
            symbol: Ticker symbol.
            model_type: "lasso", "ridge", or "elasticnet".
            horizon: Forecast horizon in days.
        """
        super().__init__(f"Linear_{model_type.upper()}", symbol)
        self.model_type = model_type
        self.horizon = horizon

        self._model: Any = None
        self._scaler = StandardScaler()
        self._feature_names: list[str] = []
        self._vol_estimate: float = 0.02
        self._ic_history: list[float] = []
        self._alpha: float = 1e-3
        self._l1_ratio: float = 0.5

    def fit(self, data: pd.DataFrame) -> "LinearSignalModel":
        """Fit the regularised linear model with CV-selected alpha.

        Args:
            data: DataFrame with ``close`` and optional feature columns.

        Returns:
            Self (fitted model).
        """
        from quantflow.data.features import compute_all_features

        close = data["close"].astype(np.float64)
        required = {"open", "high", "low", "close", "volume"}
        features = (
            compute_all_features(data)
            if required.issubset(set(data.columns))
            else data.drop(columns=["close"], errors="ignore")
        )
        X_raw, y, _ = _build_xy(features, close, self.horizon)
        self._feature_names = list(features.columns)
        self._vol_estimate = float(np.std(y))

        X = self._scaler.fit_transform(X_raw)
        self._alpha, self._l1_ratio = self._select_alpha(X, y)
        self._model = self._make_model(self._alpha, self._l1_ratio)
        self._model.fit(X, y)
        self._ic_history = _cv_ic(self._make_model(self._alpha, self._l1_ratio), X, y)
        self._is_fitted = True
        self._log_fit_complete(alpha=self._alpha, model_type=self.model_type, n_train=len(y))
        return self

    def predict(self, data: pd.DataFrame | None = None) -> ModelOutput:
        """Predict forward return and generate signal.

        Args:
            data: Optional new data for prediction.

        Returns:
            :class:`ModelOutput` with linear model signal.
        """
        self._require_fitted()
        assert self._model is not None

        if data is not None:
            from quantflow.data.features import compute_all_features

            required = {"open", "high", "low", "close", "volume"}
            features = (
                compute_all_features(data)
                if required.issubset(set(data.columns))
                else data.drop(columns=["close"], errors="ignore")
            )
            X_raw = features.replace([np.inf, -np.inf], np.nan).dropna().tail(1).values.astype(np.float64)
        else:
            n = len(self._model.coef_)
            X_raw = np.zeros((1, n))

        if X_raw.shape[0] == 0:
            return self._neutral_output()

        X = self._scaler.transform(X_raw)
        forecast_return = float(self._model.predict(X)[0])
        raw_signal = forecast_return / (self._vol_estimate * np.sqrt(self.horizon) + 1e-8)
        signal = self.normalise_signal(raw_signal)
        ic_mean = float(np.mean(self._ic_history)) if self._ic_history else 0.0
        confidence = min(0.65, max(0.20, 0.45 + ic_mean))

        # Non-zero coefficients (L1 sparsity)
        coef = self._model.coef_
        n_nonzero = int(np.sum(np.abs(coef) > 1e-6))

        return ModelOutput(
            model_name=self.model_name,
            symbol=self.symbol,
            timestamp=datetime.now(tz=timezone.utc),
            signal=signal,
            confidence=confidence,
            forecast_return=round(forecast_return, 6),
            forecast_std=round(self._vol_estimate * np.sqrt(self.horizon), 6),
            metadata={
                "alpha": round(self._alpha, 6),
                "l1_ratio": round(self._l1_ratio, 3),
                "n_nonzero_coef": n_nonzero,
                "ic_mean": round(ic_mean, 4),
            },
        )

    def _make_model(self, alpha: float, l1_ratio: float) -> Any:
        """Instantiate the sklearn estimator for the chosen model type."""
        tscv = TimeSeriesSplit(n_splits=3, gap=WALK_FORWARD_GAP)
        if self.model_type == "lasso":
            return Lasso(alpha=alpha, max_iter=5000, random_state=42)
        elif self.model_type == "ridge":
            return Ridge(alpha=alpha)
        else:
            return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000, random_state=42)

    def _select_alpha(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[float, float]:
        """Grid-search alpha and l1_ratio via walk-forward CV."""
        from sklearn.model_selection import cross_val_score

        best_alpha = 1e-3
        best_l1 = 0.5
        best_score = -np.inf

        alphas = [1e-4, 1e-3, 1e-2, 0.1, 1.0]
        l1_ratios = [0.1, 0.5, 0.9] if self.model_type == "elasticnet" else [0.5]

        tscv = TimeSeriesSplit(n_splits=3, gap=WALK_FORWARD_GAP)
        for alpha in alphas:
            for l1 in l1_ratios:
                model = self._make_model(alpha, l1)
                try:
                    scores = cross_val_score(model, X, y, cv=tscv, scoring="r2")
                    mean_score = float(np.mean(scores))
                    if mean_score > best_score:
                        best_score = mean_score
                        best_alpha = alpha
                        best_l1 = l1
                except Exception:
                    pass

        return best_alpha, best_l1

    def _neutral_output(self) -> ModelOutput:
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


# ---------------------------------------------------------------------------
# Convenience aliases matching spec names
# ---------------------------------------------------------------------------


class LASSOModel(LinearSignalModel):
    """LASSO (L1-regularised) linear signal model."""

    def __init__(self, symbol: str, horizon: int = _DEFAULT_HORIZON) -> None:
        super().__init__(symbol, model_type="lasso", horizon=horizon)


class RidgeModel(LinearSignalModel):
    """Ridge (L2-regularised) linear signal model."""

    def __init__(self, symbol: str, horizon: int = _DEFAULT_HORIZON) -> None:
        super().__init__(symbol, model_type="ridge", horizon=horizon)


class ElasticNetModel(LinearSignalModel):
    """ElasticNet (L1+L2) linear signal model."""

    def __init__(self, symbol: str, horizon: int = _DEFAULT_HORIZON) -> None:
        super().__init__(symbol, model_type="elasticnet", horizon=horizon)
