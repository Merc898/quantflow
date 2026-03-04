"""Heston stochastic volatility model.

Calibrates model parameters from historical return data and generates
trading signals based on mean-reverting variance dynamics.

  dS = (r - q) S dt + sqrt(V) S dW1
  dV = kappa*(theta - V) dt + xi * sqrt(V) dW2
  Corr(dW1, dW2) = rho

Signal logic:
- Drift signal: MC-simulated expected return normalised by expected vol.
- Vol mean-reversion: if V0 > theta, vol will fall → equities tend to
  rally when vol reverts down (leverage effect), contributing a bearish
  impulse that unwinds.
- Combined: 0.6 * drift_signal - 0.4 * vol_mr_component.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from quantflow.config.constants import TRADING_DAYS_PER_YEAR
from quantflow.config.logging import get_logger
from quantflow.models.base import BaseQuantModel, ModelOutput

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Parameter schema
# ---------------------------------------------------------------------------


class HestonParameters(BaseModel):
    """Estimated Heston model parameters.

    Attributes:
        kappa: Mean-reversion speed of variance (annualised).
        theta: Long-run variance (annualised).
        xi: Vol-of-vol (annualised).
        rho: Correlation between return and variance Brownian motions.
        v0: Initial (current) variance.
        r: Risk-free rate (annualised, default 0).
    """

    kappa: float = Field(..., gt=0.0)
    theta: float = Field(..., gt=0.0)
    xi: float = Field(..., gt=0.0)
    rho: float = Field(..., ge=-1.0, le=1.0)
    v0: float = Field(..., gt=0.0)
    r: float = 0.0


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class HestonModel(BaseQuantModel):
    """Heston stochastic vol model for return-signal generation.

    Args:
        symbol: Ticker symbol.
        n_paths: Number of Monte Carlo paths for signal estimation.
        horizon: Forecast horizon in trading days.
        seed: Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        symbol: str,
        n_paths: int = 5_000,
        horizon: int = 21,
        seed: int | None = 42,
    ) -> None:
        super().__init__("HestonModel", symbol)
        self._n_paths = n_paths
        self._horizon = horizon
        self._seed = seed
        self._params: HestonParameters | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame) -> "HestonModel":
        """Fit Heston parameters from OHLCV data.

        Args:
            data: DataFrame with a ``close`` column (UTC DatetimeIndex).

        Returns:
            Self.
        """
        log_returns = np.log(data["close"] / data["close"].shift(1)).dropna()
        if len(log_returns) < 42:
            raise ValueError(
                f"HestonModel requires at least 42 observations; got {len(log_returns)}"
            )
        self._params = self._estimate_parameters(log_returns)
        self._is_fitted = True
        self._log_fit_complete(
            kappa=round(self._params.kappa, 4),
            theta=round(self._params.theta, 4),
            xi=round(self._params.xi, 4),
            rho=round(self._params.rho, 4),
            v0=round(self._params.v0, 6),
        )
        return self

    def predict(self, data: pd.DataFrame) -> ModelOutput:
        """Generate a Heston-based trading signal.

        Args:
            data: DataFrame with a ``close`` column.

        Returns:
            :class:`ModelOutput`.
        """
        self._require_fitted()
        assert self._params is not None  # type narrowing

        log_returns = np.log(data["close"] / data["close"].shift(1)).dropna()

        sim_returns, sim_ann_vol = self._simulate_paths(
            self._params, self._n_paths, self._horizon
        )

        expected_ret_21d = float(np.mean(sim_returns))
        expected_ann_vol = float(np.mean(sim_ann_vol))

        # Drift signal: Sharpe-like ratio from simulated paths
        drift_signal = self.normalise_signal(
            expected_ret_21d / max(expected_ann_vol / np.sqrt(TRADING_DAYS_PER_YEAR / self._horizon), 1e-6)
        )

        # Vol mean-reversion signal
        # If V0 > theta → vol will compress → typically equity-bullish (risk-on),
        # but in our framework high V0 also implies more uncertainty → bearish tilt.
        # Net effect: negative contribution when V0 >> theta.
        v0, theta = self._params.v0, self._params.theta
        vol_gap = float(np.tanh((v0 - theta) / max(theta, 1e-6)))
        combined = float(np.clip(0.6 * drift_signal - 0.4 * vol_gap, -1.0, 1.0))

        # Confidence: deteriorates with high xi (noisy vol-of-vol)
        vol_of_vol_ratio = self._params.xi / max(np.sqrt(self._params.theta), 1e-6)
        confidence = float(np.clip(1.0 - vol_of_vol_ratio / 5.0, 0.20, 0.85))

        # Regime tag
        recent_vol = float(log_returns.iloc[-21:].std() * np.sqrt(TRADING_DAYS_PER_YEAR))
        regime = (
            "HIGH_VOL" if recent_vol > 0.25
            else ("LOW_VOL" if recent_vol < 0.12 else "MEDIUM_VOL")
        )

        return ModelOutput(
            model_name=self.model_name,
            symbol=self.symbol,
            timestamp=datetime.now(tz=timezone.utc),
            signal=round(combined, 6),
            confidence=round(confidence, 6),
            forecast_return=round(expected_ret_21d * (TRADING_DAYS_PER_YEAR / self._horizon), 6),
            forecast_std=round(expected_ann_vol, 6),
            regime=regime,
            metadata={
                "kappa": self._params.kappa,
                "theta": self._params.theta,
                "xi": self._params.xi,
                "rho": self._params.rho,
                "v0": self._params.v0,
                "drift_signal": drift_signal,
                "vol_gap_signal": vol_gap,
                "n_paths": self._n_paths,
                "horizon_days": self._horizon,
            },
        )

    # ------------------------------------------------------------------
    # Parameter estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_parameters(returns: pd.Series) -> HestonParameters:
        """Estimate Heston parameters via method of moments.

        Args:
            returns: Daily log-return series.

        Returns:
            :class:`HestonParameters`.
        """
        ann = TRADING_DAYS_PER_YEAR

        # Realized variance time series
        realized_var = returns.rolling(21).var() * ann
        realized_var = realized_var.dropna()

        theta = float(realized_var.mean())
        theta = max(theta, 1e-6)

        v0 = float(realized_var.iloc[-1]) if len(realized_var) > 0 else theta
        v0 = max(v0, 1e-6)

        # kappa: daily AR(1) on variance → annualise
        kappa = 2.0  # default
        if len(realized_var) >= 21:
            lagged = realized_var.shift(1).dropna()
            current = realized_var.loc[lagged.index]
            aligned = pd.concat([current, lagged], axis=1).dropna()
            if len(aligned) >= 10:
                cov = np.cov(aligned.values.T)
                if cov[1, 1] > 1e-10:
                    ar1 = float(cov[0, 1] / cov[1, 1])
                    ar1 = float(np.clip(ar1, 1e-4, 0.9999))
                    kappa = float(-np.log(ar1) * ann)
        kappa = float(np.clip(kappa, 0.1, 50.0))

        # xi: vol-of-vol (std of variance changes, annualised)
        xi = 0.4  # default
        if len(realized_var) >= 22:
            dv = realized_var.diff().dropna()
            xi = float(dv.std() * np.sqrt(ann))
        xi = float(np.clip(xi, 1e-4, 10.0))

        # rho: correlation of returns with variance changes (leverage effect)
        rho = -0.30
        if len(returns) >= 22:
            dv = (returns.rolling(21).var() * ann).diff().dropna()
            common = returns.index.intersection(dv.index)
            if len(common) >= 10:
                corr = float(returns.loc[common].corr(dv.loc[common]))
                if np.isfinite(corr):
                    rho = float(np.clip(corr, -0.99, 0.99))

        return HestonParameters(kappa=kappa, theta=theta, xi=xi, rho=rho, v0=v0)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _simulate_paths(
        self,
        params: HestonParameters,
        n_paths: int,
        horizon: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate Heston SDE via Euler-Maruyama (full-truncation scheme).

        Args:
            params: Heston parameters.
            n_paths: Number of Monte Carlo paths.
            horizon: Steps to simulate (trading days).

        Returns:
            Tuple of (cumulative_log_returns, annualised_avg_vol) arrays,
            each of shape (n_paths,).
        """
        rng = np.random.default_rng(self._seed)
        dt = 1.0 / TRADING_DAYS_PER_YEAR

        kappa, theta, xi, rho, v0, r = (
            params.kappa, params.theta, params.xi,
            params.rho, params.v0, params.r,
        )
        sqrt_dt = np.sqrt(dt)

        # Cholesky for correlated Brownians
        chol = np.array([[1.0, 0.0], [rho, np.sqrt(max(1.0 - rho ** 2, 0.0))]])

        log_s = np.zeros(n_paths)
        v = np.full(n_paths, v0)
        vol_sum = np.zeros(n_paths)

        for _ in range(horizon):
            z = rng.standard_normal((2, n_paths))
            w1 = z[0]
            w2 = chol[1, 0] * z[0] + chol[1, 1] * z[1]

            v_pos = np.maximum(v, 0.0)
            sqrt_v = np.sqrt(v_pos)

            # Full-truncation Euler for variance
            v_new = v_pos + kappa * (theta - v_pos) * dt + xi * sqrt_v * sqrt_dt * w2
            v = np.maximum(v_new, 0.0)

            log_s += (r - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * w1
            vol_sum += np.sqrt(v_pos)

        avg_daily_vol = vol_sum / horizon
        ann_vol = avg_daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)

        return log_s, ann_vol
