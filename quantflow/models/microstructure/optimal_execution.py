"""Almgren-Chriss optimal execution model.

Computes the optimal liquidation trajectory and expected implementation
shortfall for a given position, based on market impact parameters
calibrated from historical return and volume data.

Model (Almgren & Chriss 2001):
  - Permanent impact:  g(v) = gamma * v  (linear)
  - Temporary impact:  h(v) = eta * v    (linear)
  - Risk aversion:     lambda (trader's aversion to execution variance)
  - Optimal trajectory: x_n = X * sinh(kappa*(T - n*tau)) / sinh(kappa*T)
    where kappa = sqrt(lambda * sigma^2 / eta)

Signal: negative shortfall expressed as a fraction of expected daily
move.  High transaction costs relative to signal alpha ⟹ negative
friction signal.  Useful as a liquidity/cost-of-carry signal modifier
within the ensemble.

Output signal = -tanh(E[shortfall] / (sigma * sqrt(horizon)))
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, Field

from quantflow.config.constants import (
    MARKET_IMPACT_COEFFICIENT,
    SLIPPAGE_BPS,
    TRADING_DAYS_PER_YEAR,
)
from quantflow.config.logging import get_logger
from quantflow.models.base import BaseQuantModel, ModelOutput

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Parameter schema
# ---------------------------------------------------------------------------


class ExecutionParameters(BaseModel):
    """Calibrated Almgren-Chriss market impact parameters.

    Attributes:
        sigma: Annualised daily return volatility.
        gamma: Permanent impact coefficient (price shift per unit traded).
        eta: Temporary impact coefficient (per-step execution cost).
        lam: Risk aversion (execution variance weight).
    """

    sigma: float = Field(..., gt=0.0)
    gamma: float = Field(..., ge=0.0)
    eta: float = Field(..., ge=0.0)
    lam: float = Field(default=1e-5, gt=0.0)


class ExecutionSchedule(BaseModel):
    """Result of Almgren-Chriss optimisation.

    Attributes:
        trajectory: Remaining inventory at each step (normalised to 1.0 initial).
        trade_list: Shares traded each step (normalised).
        expected_shortfall: E[cost] as fraction of initial position value.
        expected_variance: Var[cost] as fraction of initial position value squared.
        kappa: Urgency parameter sqrt(lambda*sigma^2/eta).
        horizon_days: Liquidation horizon used.
    """

    trajectory: list[float]
    trade_list: list[float]
    expected_shortfall: float
    expected_variance: float
    kappa: float
    horizon_days: int


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class OptimalExecutionModel(BaseQuantModel):
    """Almgren-Chriss optimal execution and shortfall estimation.

    Calibrates permanent and temporary market impact from historical
    data and computes the expected execution shortfall for a hypothetical
    1% ADV position.  The shortfall is converted to a friction signal
    that penalises illiquid, high-volatility assets.

    Args:
        symbol: Ticker symbol.
        horizon_days: Liquidation horizon (trading days).
        position_size_adv: Position as fraction of average daily volume.
        risk_aversion: Trader risk aversion (lambda in AC model).
    """

    def __init__(
        self,
        symbol: str,
        horizon_days: int = 10,
        position_size_adv: float = 0.01,
        risk_aversion: float = 1e-5,
    ) -> None:
        super().__init__("OptimalExecutionModel", symbol)
        self._horizon = horizon_days
        self._position_adv = position_size_adv
        self._lam = risk_aversion
        self._params: ExecutionParameters | None = None
        self._schedule: ExecutionSchedule | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame) -> OptimalExecutionModel:
        """Calibrate market impact parameters.

        Args:
            data: OHLCV DataFrame with at least a ``close`` column.
                  Optional ``volume`` column used for ADV-based impact.

        Returns:
            Self.
        """
        log_returns = np.log(data["close"] / data["close"].shift(1)).dropna()
        if len(log_returns) < 21:
            raise ValueError(
                f"OptimalExecutionModel requires ≥21 observations; got {len(log_returns)}"
            )

        self._params = self._calibrate(data, log_returns)
        self._schedule = self._solve_ac(self._params)
        self._is_fitted = True

        self._log_fit_complete(
            sigma=round(self._params.sigma, 4),
            gamma=round(self._params.gamma, 6),
            eta=round(self._params.eta, 6),
            expected_shortfall=round(self._schedule.expected_shortfall, 6),
            kappa=round(self._schedule.kappa, 4),
        )
        return self

    def predict(self, data: pd.DataFrame) -> ModelOutput:
        """Generate an execution-friction signal.

        Args:
            data: OHLCV DataFrame.

        Returns:
            :class:`ModelOutput` where signal < 0 indicates high
            execution cost relative to expected alpha.
        """
        self._require_fitted()
        assert self._params is not None
        assert self._schedule is not None

        params = self._params
        sched = self._schedule

        # Friction signal: high shortfall → harder to profit after costs
        horizon_vol = params.sigma * np.sqrt(self._horizon / TRADING_DAYS_PER_YEAR)
        shortfall_ratio = sched.expected_shortfall / max(horizon_vol, 1e-6)

        # Signal is negative (execution cost is a drag)
        signal = -float(np.tanh(shortfall_ratio * 5.0))
        signal = float(np.clip(signal, -1.0, 1.0))

        # Confidence: higher when we have more data to calibrate
        log_returns = np.log(data["close"] / data["close"].shift(1)).dropna()
        n = len(log_returns)
        confidence = float(np.clip(0.30 + 0.40 * min(n / 252, 1.0), 0.30, 0.70))

        # Regime based on kappa urgency
        if sched.kappa > 1.0:
            regime = "HIGH_URGENCY"
        elif sched.kappa > 0.3:
            regime = "MEDIUM_URGENCY"
        else:
            regime = "LOW_URGENCY"

        return ModelOutput(
            model_name=self.model_name,
            symbol=self.symbol,
            timestamp=datetime.now(tz=UTC),
            signal=round(signal, 6),
            confidence=round(confidence, 6),
            forecast_return=0.0,
            forecast_std=round(params.sigma, 6),
            regime=regime,
            metadata={
                "expected_shortfall_pct": round(sched.expected_shortfall * 100, 4),
                "expected_variance": round(sched.expected_variance, 8),
                "kappa": round(sched.kappa, 4),
                "gamma": round(params.gamma, 6),
                "eta": round(params.eta, 6),
                "horizon_days": self._horizon,
                "trajectory": sched.trajectory[:5],  # first 5 steps
            },
        )

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _calibrate(self, data: pd.DataFrame, log_returns: pd.Series) -> ExecutionParameters:
        """Calibrate market impact parameters from historical data.

        Args:
            data: Full OHLCV DataFrame.
            log_returns: Daily log-return series.

        Returns:
            :class:`ExecutionParameters`.
        """
        sigma_daily = float(log_returns.std())
        sigma_ann = sigma_daily * np.sqrt(TRADING_DAYS_PER_YEAR)

        # Gamma (permanent impact): approximated from SLIPPAGE_BPS and volatility.
        # In practice: gamma = 2 * bid-ask spread / (ADV * price).
        # Here we use a simplified proxy based on vol-normalised impact.
        slippage = SLIPPAGE_BPS * 1e-4  # convert bps to fraction
        gamma = slippage * sigma_daily / max(self._position_adv, 1e-6)
        gamma = max(gamma, 1e-8)

        # Eta (temporary impact): larger for illiquid / high-vol assets.
        # Proxy: eta = MARKET_IMPACT_COEFFICIENT * sigma^2 * position_size
        eta = MARKET_IMPACT_COEFFICIENT * sigma_daily**2 * self._position_adv
        eta = max(eta, 1e-8)

        return ExecutionParameters(
            sigma=sigma_ann,
            gamma=gamma,
            eta=eta,
            lam=self._lam,
        )

    # ------------------------------------------------------------------
    # Almgren-Chriss optimisation
    # ------------------------------------------------------------------

    def _solve_ac(self, params: ExecutionParameters) -> ExecutionSchedule:
        """Compute optimal execution trajectory.

        Args:
            params: Calibrated execution parameters.

        Returns:
            :class:`ExecutionSchedule`.
        """
        N = self._horizon
        T = N  # discrete steps (tau = 1 day each)
        sigma_d = params.sigma / np.sqrt(TRADING_DAYS_PER_YEAR)  # daily vol

        # Urgency parameter kappa
        kappa_sq = params.lam * sigma_d**2 / max(params.eta, 1e-12)
        kappa = float(np.sqrt(max(kappa_sq, 0.0)))

        # Optimal trajectory x_n (normalised, starting at X=1)
        ns = np.arange(N + 1)
        denom = np.sinh(kappa * T) if kappa * T > 1e-6 else 1.0
        if denom < 1e-10:
            # Flat TWAP when kappa ≈ 0
            trajectory = list(1.0 - ns / N)
        else:
            trajectory = [float(np.sinh(kappa * (T - n)) / denom) for n in ns]

        # Trade list (shares sold at each step)
        trade_list = [trajectory[n] - trajectory[n + 1] for n in range(N)]

        # Expected shortfall (normalised, as fraction of initial position value)
        # E[S] = 0.5*gamma*X^2 + eta * sum(v_n^2) / X
        # With X=1: E[S] = 0.5*gamma + eta * sum(trade_list^2)
        expected_shortfall = 0.5 * params.gamma + params.eta * float(
            np.sum(np.array(trade_list) ** 2)
        )

        # Expected variance (market risk of the trajectory)
        # Var[S] = sigma^2 * tau * sum(x_n^2)  (tau=1 day)
        expected_variance = sigma_d**2 * float(np.sum(np.array(trajectory[:-1]) ** 2))

        return ExecutionSchedule(
            trajectory=[round(x, 6) for x in trajectory],
            trade_list=[round(v, 6) for v in trade_list],
            expected_shortfall=round(expected_shortfall, 8),
            expected_variance=round(expected_variance, 8),
            kappa=round(kappa, 6),
            horizon_days=N,
        )
