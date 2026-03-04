"""Kalman Filter model for dynamic beta, spread, and trend estimation.

Implements a time-varying linear state-space model for four applications:
  - beta: dynamic market-beta estimation
  - spread: cointegration spread for pairs trading
  - trend: local-level (random-walk + noise) trend extraction
  - factor: latent factor estimation

Uses pykalman's EM algorithm for parameter initialisation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np
import pandas as pd

from quantflow.config.logging import get_logger
from quantflow.models.base import BaseQuantModel, ModelOutput

logger = get_logger(__name__)

KalmanApplication = Literal["beta", "spread", "trend", "factor"]


class KalmanFilterModel(BaseQuantModel):
    """Time-varying Kalman filter for multiple financial applications.

    Args:
        symbol: Ticker symbol for the target asset.
        application: Which Kalman application to run.
        benchmark_col: Column name for the benchmark returns (used for
            ``beta`` and ``spread`` modes).
        n_iter_em: Number of EM iterations for parameter estimation.
    """

    def __init__(
        self,
        symbol: str,
        application: KalmanApplication = "beta",
        benchmark_col: str = "benchmark",
        n_iter_em: int = 10,
    ) -> None:
        """Initialise the Kalman filter model.

        Args:
            symbol: Ticker symbol.
            application: One of "beta", "spread", "trend", "factor".
            benchmark_col: Column name for benchmark series.
            n_iter_em: EM iterations for parameter initialisation.
        """
        super().__init__(f"KalmanFilter_{application}", symbol)
        self.application = application
        self.benchmark_col = benchmark_col
        self.n_iter_em = n_iter_em

        self._filtered_state: np.ndarray | None = None
        self._filtered_cov: np.ndarray | None = None
        self._kf: Any = None
        self._last_obs: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame) -> "KalmanFilterModel":
        """Fit the Kalman filter via EM and run forward filtering pass.

        Args:
            data: DataFrame with ``close`` column (and ``benchmark`` column
                  for beta/spread applications).

        Returns:
            Self (fitted model).
        """
        from pykalman import KalmanFilter

        obs = self._prepare_observations(data)
        self._last_obs = obs[-1:].copy()

        n_dim_obs = obs.shape[1] if obs.ndim > 1 else 1
        n_dim_state = n_dim_obs  # one state per observation dimension

        kf = KalmanFilter(
            n_dim_obs=n_dim_obs,
            n_dim_state=n_dim_state,
        )

        # EM initialisation
        kf = kf.em(obs, n_iter=self.n_iter_em)

        # Forward filtering
        means, covs = kf.filter(obs)
        self._filtered_state = means
        self._filtered_cov = covs
        self._kf = kf

        self._is_fitted = True
        self._log_fit_complete(
            application=self.application,
            obs_dim=n_dim_obs,
            n_obs=len(obs),
        )
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, data: pd.DataFrame | None = None) -> ModelOutput:
        """Generate signal from the current filtered state.

        Args:
            data: Optional new observations for online updating.

        Returns:
            :class:`ModelOutput` whose semantics depend on ``application``:
            - ``beta``: signal positive if beta < 1 (defensive), negative if > 1.
            - ``spread``: signal from mean-reversion of the spread.
            - ``trend``: signal follows the latent trend direction.
            - ``factor``: signal from latent factor loading.
        """
        self._require_fitted()
        assert self._filtered_state is not None
        assert self._filtered_cov is not None

        state = self._filtered_state[-1]
        cov = self._filtered_cov[-1]
        state_std = float(np.sqrt(np.diag(cov).mean()))

        if self.application == "beta":
            signal, meta = self._beta_signal(state, state_std)
        elif self.application == "spread":
            signal, meta = self._spread_signal(state, state_std)
        elif self.application == "trend":
            signal, meta = self._trend_signal(state, state_std)
        else:
            signal, meta = self._factor_signal(state, state_std)

        # Confidence from state uncertainty (higher uncertainty → lower confidence)
        confidence = min(0.8, max(0.2, 1.0 / (1.0 + state_std)))

        return ModelOutput(
            model_name=self.model_name,
            symbol=self.symbol,
            timestamp=datetime.now(tz=timezone.utc),
            signal=signal,
            confidence=confidence,
            forecast_return=float(state.mean() if state.ndim > 0 else state),
            forecast_std=state_std,
            regime=None,
            metadata=meta,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _prepare_observations(self, data: pd.DataFrame) -> np.ndarray:
        """Convert a DataFrame to an observation array for the Kalman filter."""
        close = data["close"].astype(np.float64)
        log_ret = np.log(close / close.shift(1)).fillna(0)

        if self.application in ("beta", "spread"):
            if self.benchmark_col in data.columns:
                bench = data[self.benchmark_col].astype(np.float64)
                bench_ret = np.log(bench / bench.shift(1)).fillna(0)
                obs = np.column_stack([log_ret.values, bench_ret.values])
            else:
                obs = log_ret.values.reshape(-1, 1)
        else:
            obs = log_ret.values.reshape(-1, 1)

        return obs.astype(np.float64)

    def _beta_signal(
        self, state: np.ndarray, state_std: float
    ) -> tuple[float, dict[str, Any]]:
        """Signal: deviation of dynamic beta from 1."""
        beta = float(state[0]) if state.ndim > 0 else float(state)
        raw = -(beta - 1.0)  # low beta → bullish (defensive)
        signal = self.normalise_signal(raw * 2.0)
        meta: dict[str, Any] = {
            "dynamic_beta": round(beta, 4),
            "state_std": round(state_std, 4),
        }
        return signal, meta

    def _spread_signal(
        self, state: np.ndarray, state_std: float
    ) -> tuple[float, dict[str, Any]]:
        """Signal: mean-reversion of cointegration spread."""
        spread = float(state[0]) if state.ndim > 0 else float(state)
        raw = -spread / (state_std + 1e-8)
        signal = self.normalise_signal(raw)
        meta: dict[str, Any] = {
            "spread": round(spread, 4),
            "state_std": round(state_std, 4),
        }
        return signal, meta

    def _trend_signal(
        self, state: np.ndarray, state_std: float
    ) -> tuple[float, dict[str, Any]]:
        """Signal: direction of latent local-level trend."""
        trend = float(state[0]) if state.ndim > 0 else float(state)
        signal = self.normalise_signal(trend / (state_std + 1e-8))
        meta: dict[str, Any] = {
            "trend": round(trend, 6),
            "state_std": round(state_std, 4),
        }
        return signal, meta

    def _factor_signal(
        self, state: np.ndarray, state_std: float
    ) -> tuple[float, dict[str, Any]]:
        """Signal: latent factor loading direction."""
        factor = float(state.mean()) if state.ndim > 0 else float(state)
        signal = self.normalise_signal(factor / (state_std + 1e-8))
        meta: dict[str, Any] = {
            "factor_loading": round(factor, 4),
            "state_std": round(state_std, 4),
        }
        return signal, meta
