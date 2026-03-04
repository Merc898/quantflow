"""Bayesian deep learning for uncertainty-aware return forecasting.

Implements two complementary approaches:

A) MC Dropout (Gal & Ghahramani 2016):
   Keep dropout active at inference; run T=100 forward passes.
   Mean = prediction, Std = epistemic uncertainty estimate.

B) Deep Ensembles (Lakshminarayanan 2017):
   Train M=5 independent networks with different random initialisations.
   Output = Gaussian mixture; mean and variance aggregated across members.

The predictive uncertainty is mapped to a confidence modifier:
   uncertainty ∈ {LOW, MEDIUM, HIGH} (by percentile vs rolling history).
   Higher uncertainty → reduced confidence in the signal.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np
import pandas as pd

from quantflow.config.constants import TRADING_DAYS_PER_YEAR
from quantflow.config.logging import get_logger
from quantflow.models.base import BaseQuantModel, ModelOutput

logger = get_logger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Network architecture
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class _BayesianMLP(nn.Module):
        """Two-hidden-layer MLP with dropout for MC Dropout inference.

        Args:
            input_dim: Number of input features.
            hidden_dim: Hidden layer width.
            dropout_p: Dropout probability (kept active at inference).
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 64,
            dropout_p: float = 0.20,
        ) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Dropout(p=dropout_p),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Dropout(p=dropout_p),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------


def _make_features(returns: pd.Series, lookbacks: tuple[int, ...] = (5, 21, 63)) -> np.ndarray:
    """Build a feature matrix from lagged return statistics.

    Features per lookback: mean, std, skew, last-return.

    Args:
        returns: Daily log-return series.
        lookbacks: Rolling window sizes.

    Returns:
        Feature array of shape (n_rows, n_features).
    """
    df = pd.DataFrame(index=returns.index)
    for lb in lookbacks:
        roll = returns.rolling(lb, min_periods=max(lb // 2, 2))
        df[f"mean_{lb}"] = roll.mean()
        df[f"std_{lb}"] = roll.std().fillna(1e-8)
        df[f"skew_{lb}"] = roll.skew()
        df[f"ret_{lb}"] = returns.shift(1).rolling(lb).mean()
    df["last_ret"] = returns.shift(1)
    df = df.ffill().bfill().fillna(0.0)
    arr = df.to_numpy(dtype=np.float32)
    # Clip features to prevent extreme values
    arr = np.clip(arr, -5.0, 5.0)
    return arr


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class BayesianNNModel(BaseQuantModel):
    """Uncertainty-aware neural network for return forecasting.

    Args:
        symbol: Ticker symbol.
        method: ``"mc_dropout"`` or ``"ensemble"``.
        hidden_dim: Hidden layer width.
        n_mc_samples: MC Dropout forward passes at inference.
        n_ensemble: Number of ensemble members (``method="ensemble"`` only).
        dropout_p: Dropout probability.
        n_epochs: Training epochs per network.
        lr: Learning rate.
        horizon: Forecast horizon in days.
        seed: RNG seed for reproducibility.
    """

    def __init__(
        self,
        symbol: str,
        method: Literal["mc_dropout", "ensemble"] = "mc_dropout",
        hidden_dim: int = 64,
        n_mc_samples: int = 100,
        n_ensemble: int = 5,
        dropout_p: float = 0.20,
        n_epochs: int = 100,
        lr: float = 1e-3,
        horizon: int = 21,
        seed: int | None = 42,
    ) -> None:
        super().__init__("BayesianNNModel", symbol)
        self._method = method
        self._hidden_dim = hidden_dim
        self._n_mc = n_mc_samples
        self._n_ensemble = n_ensemble
        self._dropout_p = dropout_p
        self._n_epochs = n_epochs
        self._lr = lr
        self._horizon = horizon
        self._seed = seed

        # Fitted state
        self._networks: list[Any] = []
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None
        self._input_dim: int = 0
        self._uncertainty_history: list[float] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame) -> "BayesianNNModel":
        """Train Bayesian network(s) on historical data.

        Args:
            data: OHLCV DataFrame (UTC DatetimeIndex).

        Returns:
            Self.
        """
        if not _TORCH_AVAILABLE:
            self._fallback_fit(data)
            return self

        log_returns = np.log(data["close"] / data["close"].shift(1)).dropna()
        if len(log_returns) < 63:
            raise ValueError(f"BayesianNNModel requires ≥63 observations; got {len(log_returns)}")

        X = _make_features(log_returns)
        y = (log_returns.shift(-self._horizon).reindex(log_returns.index)
             .to_numpy(dtype=np.float32))

        # Align
        valid = ~np.isnan(y) & np.all(np.isfinite(X), axis=1)
        X, y = X[valid], y[valid]
        if len(X) < 30:
            raise ValueError("Insufficient valid samples after feature construction.")

        # Standardise features
        self._feature_mean = X.mean(axis=0)
        self._feature_std = np.where(X.std(axis=0) > 1e-8, X.std(axis=0), 1.0)
        X = (X - self._feature_mean) / self._feature_std
        self._input_dim = X.shape[1]

        if self._method == "mc_dropout":
            net = self._train_single_network(X, y, seed=self._seed)
            self._networks = [net]
        else:
            self._networks = [
                self._train_single_network(X, y, seed=(self._seed or 0) + i)
                for i in range(self._n_ensemble)
            ]

        self._is_fitted = True
        self._log_fit_complete(
            method=self._method,
            n_networks=len(self._networks),
            n_train_samples=len(X),
        )
        return self

    def predict(self, data: pd.DataFrame) -> ModelOutput:
        """Generate a Bayesian-uncertainty-aware signal.

        Args:
            data: OHLCV DataFrame.

        Returns:
            :class:`ModelOutput` with uncertainty-adjusted confidence.
        """
        self._require_fitted()

        if not _TORCH_AVAILABLE or not self._networks:
            return self._fallback_predict(data)

        log_returns = np.log(data["close"] / data["close"].shift(1)).dropna()

        # Build features for the most recent row
        X_full = _make_features(log_returns)
        assert self._feature_mean is not None and self._feature_std is not None
        X_norm = (X_full - self._feature_mean) / self._feature_std
        x_last = torch.tensor(X_norm[-1:], dtype=torch.float32)  # type: ignore[name-defined]

        if self._method == "mc_dropout":
            pred_mean, pred_std = self._mc_dropout_predict(self._networks[0], x_last)
        else:
            pred_mean, pred_std = self._ensemble_predict(self._networks, x_last)

        # Track uncertainty history
        self._uncertainty_history.append(float(pred_std))
        if len(self._uncertainty_history) > 252:
            self._uncertainty_history = self._uncertainty_history[-252:]

        # Uncertainty level
        unc_level = self._uncertainty_level(pred_std)

        # Signal: normalise expected 21d return by vol
        ann_vol = float(log_returns.iloc[-21:].std() * np.sqrt(TRADING_DAYS_PER_YEAR))
        horizon_vol = ann_vol * np.sqrt(self._horizon / TRADING_DAYS_PER_YEAR)
        signal = self.normalise_signal(pred_mean / max(horizon_vol, 1e-6))

        # Confidence: base 0.60, penalised by uncertainty
        unc_penalty = {
            "LOW": 0.0, "MEDIUM": 0.10, "HIGH": 0.25
        }.get(unc_level, 0.10)
        confidence = float(np.clip(0.60 - unc_penalty, 0.20, 0.80))

        regime = f"UNCERTAINTY_{unc_level}"

        return ModelOutput(
            model_name=self.model_name,
            symbol=self.symbol,
            timestamp=datetime.now(tz=timezone.utc),
            signal=round(signal, 6),
            confidence=round(confidence, 6),
            forecast_return=round(float(pred_mean) * (TRADING_DAYS_PER_YEAR / self._horizon), 6),
            forecast_std=round(float(pred_std), 6),
            regime=regime,
            metadata={
                "method": self._method,
                "pred_mean_21d": round(float(pred_mean), 6),
                "pred_std": round(float(pred_std), 6),
                "uncertainty_level": unc_level,
                "n_networks": len(self._networks),
            },
        )

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def _train_single_network(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seed: int | None,
    ) -> "torch.nn.Module":
        """Train one MLP network.

        Args:
            X: Feature matrix (standardised).
            y: Target returns.
            seed: RNG seed for this member.

        Returns:
            Trained PyTorch module.
        """
        if seed is not None:
            torch.manual_seed(seed)  # type: ignore[attr-defined]

        net = _BayesianMLP(
            input_dim=X.shape[1],
            hidden_dim=self._hidden_dim,
            dropout_p=self._dropout_p,
        )
        opt = optim.Adam(net.parameters(), lr=self._lr, weight_decay=1e-4)  # type: ignore[attr-defined]
        criterion = nn.MSELoss()  # type: ignore[attr-defined]

        X_t = torch.tensor(X, dtype=torch.float32)  # type: ignore[attr-defined]
        y_t = torch.tensor(y, dtype=torch.float32)  # type: ignore[attr-defined]

        net.train()
        for _ in range(self._n_epochs):
            opt.zero_grad()
            pred = net(X_t)
            loss = criterion(pred, y_t)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)  # type: ignore[attr-defined]
            opt.step()

        return net

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _mc_dropout_predict(
        self,
        net: "torch.nn.Module",
        x: "torch.Tensor",
    ) -> tuple[float, float]:
        """Run MC Dropout inference.

        Args:
            net: Trained network (with Dropout layers).
            x: Single-sample input tensor, shape (1, input_dim).

        Returns:
            Tuple of (mean_prediction, std_prediction).
        """
        net.train()  # Keep dropout active
        with torch.no_grad():  # type: ignore[attr-defined]
            preds = torch.stack(  # type: ignore[attr-defined]
                [net(x) for _ in range(self._n_mc)]
            ).squeeze().numpy()
        return float(preds.mean()), float(preds.std())

    @staticmethod
    def _ensemble_predict(
        networks: list["torch.nn.Module"],
        x: "torch.Tensor",
    ) -> tuple[float, float]:
        """Aggregate predictions across ensemble members.

        Args:
            networks: List of trained networks.
            x: Single-sample input tensor.

        Returns:
            Tuple of (mean_prediction, std_prediction).
        """
        means = []
        for net in networks:
            net.eval()
            with torch.no_grad():  # type: ignore[attr-defined]
                pred = net(x).item()
            means.append(pred)
        return float(np.mean(means)), float(np.std(means))

    # ------------------------------------------------------------------
    # Uncertainty classification
    # ------------------------------------------------------------------

    def _uncertainty_level(self, pred_std: float) -> str:
        """Classify predictive uncertainty as LOW / MEDIUM / HIGH.

        Args:
            pred_std: Current predictive standard deviation.

        Returns:
            Uncertainty label.
        """
        if len(self._uncertainty_history) < 10:
            return "MEDIUM"
        hist = np.array(self._uncertainty_history)
        pct = float((hist < pred_std).mean() * 100)
        if pct > 75:
            return "HIGH"
        if pct < 25:
            return "LOW"
        return "MEDIUM"

    # ------------------------------------------------------------------
    # Fallback (no PyTorch)
    # ------------------------------------------------------------------

    def _fallback_fit(self, data: pd.DataFrame) -> None:
        """Fit a simple linear model when PyTorch is unavailable."""
        log_returns = np.log(data["close"] / data["close"].shift(1)).dropna()
        self._feature_mean = np.array([float(log_returns.mean())])
        self._feature_std = np.array([max(float(log_returns.std()), 1e-8)])
        self._networks = [{"mean": float(log_returns.mean()), "std": float(log_returns.std())}]
        self._is_fitted = True
        logger.warning("PyTorch not available; using linear fallback for BayesianNNModel")

    def _fallback_predict(self, data: pd.DataFrame) -> ModelOutput:
        """Predict using simple linear model."""
        log_returns = np.log(data["close"] / data["close"].shift(1)).dropna()
        recent_ret = float(log_returns.iloc[-self._horizon:].mean() * self._horizon)
        ann_vol = float(log_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
        signal = self.normalise_signal(recent_ret / max(ann_vol / np.sqrt(TRADING_DAYS_PER_YEAR / self._horizon), 1e-6))
        return ModelOutput(
            model_name=self.model_name,
            symbol=self.symbol,
            timestamp=datetime.now(tz=timezone.utc),
            signal=round(signal, 6),
            confidence=0.30,
            forecast_return=round(recent_ret * (TRADING_DAYS_PER_YEAR / self._horizon), 6),
            forecast_std=round(ann_vol, 6),
            regime="UNCERTAINTY_MEDIUM",
            metadata={"fallback": True, "reason": "torch_unavailable"},
        )
