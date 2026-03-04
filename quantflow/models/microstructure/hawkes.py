"""Hawkes process model for market event clustering.

Models the self-exciting arrival of large price moves using a univariate
exponential Hawkes process:

  lambda(t) = mu + alpha * sum_{t_i < t} exp(-beta * (t - t_i))

Events are defined as daily returns whose absolute value exceeds a
threshold (default 1.5 * rolling standard deviation).

Separate processes are fit for upside (bullish) and downside (bearish)
events.  The signal is derived from the relative intensity differential:

  signal = tanh( (lambda_up - lambda_down) / (lambda_up + lambda_down + eps) )

Branching ratio:  n* = alpha / beta  (must be < 1 for stability)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize

from quantflow.config.constants import TRADING_DAYS_PER_YEAR
from quantflow.config.logging import get_logger
from quantflow.models.base import BaseQuantModel, ModelOutput

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)

_MIN_EVENTS = 10  # minimum events to fit a stable Hawkes process
_EVENT_THRESHOLD = 1.5  # multiples of rolling std to define an event
_ROLLING_WINDOW = 63  # window for rolling std estimation


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _hawkes_log_likelihood(
    params: np.ndarray,
    event_times: np.ndarray,
    T: float,
) -> float:
    """Negative log-likelihood of a univariate exponential Hawkes process.

    Args:
        params: [mu, alpha, beta] — baseline intensity and kernel params.
        event_times: Sorted event times in [0, T].
        T: Observation window length.

    Returns:
        Negative log-likelihood (to be minimised).
    """
    mu, alpha, beta = params
    if mu <= 0 or alpha <= 0 or beta <= 0 or alpha >= beta:
        return 1e10

    n = len(event_times)
    if n == 0:
        return mu * T  # likelihood of no events

    # Recursive computation of the compensator integral via A_i recursion
    # A_i = sum_{j < i} exp(-beta*(t_i - t_j))
    A = np.zeros(n)
    for i in range(1, n):
        A[i] = np.exp(-beta * (event_times[i] - event_times[i - 1])) * (1.0 + A[i - 1])

    lambdas = mu + alpha * A
    log_lhood = (
        np.sum(np.log(np.maximum(lambdas, 1e-12)))
        - mu * T
        - (alpha / beta) * np.sum(1.0 - np.exp(-beta * (T - event_times)))
    )
    return -log_lhood


def _fit_hawkes(
    event_times: np.ndarray,
    T: float,
) -> tuple[float, float, float, bool]:
    """Fit Hawkes parameters via MLE.

    Args:
        event_times: Sorted event arrival times.
        T: Total observation period.

    Returns:
        Tuple of (mu, alpha, beta, converged).
    """
    if len(event_times) < _MIN_EVENTS:
        # Fallback: pure Poisson (no self-excitation)
        mu = len(event_times) / T
        return mu, 0.0, 1.0, False

    # Initialise from empirical rate
    mu0 = len(event_times) / T * 0.5
    alpha0 = mu0 * 0.5
    beta0 = mu0 * 2.0
    x0 = np.array([mu0, alpha0, beta0])

    bounds = [(1e-6, None), (1e-6, None), (1e-4, None)]

    try:
        res = minimize(
            _hawkes_log_likelihood,
            x0,
            args=(event_times, T),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-9},
        )
        mu, alpha, beta = res.x
        converged = res.success
    except Exception:
        mu, alpha, beta = float(x0[0]), float(x0[1]), float(x0[2])
        converged = False

    return float(mu), float(alpha), float(beta), converged


def _current_intensity(
    mu: float,
    alpha: float,
    beta: float,
    event_times: np.ndarray,
    t_now: float,
) -> float:
    """Compute conditional intensity at t_now.

    Args:
        mu: Baseline intensity.
        alpha: Excitation amplitude.
        beta: Decay rate.
        event_times: Past event times.
        t_now: Current time.

    Returns:
        Conditional intensity lambda(t_now).
    """
    excitation = float(np.sum(alpha * np.exp(-beta * (t_now - event_times[event_times < t_now]))))
    return mu + excitation


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class HawkesModel(BaseQuantModel):
    """Hawkes process model for self-exciting event clustering.

    Fits separate upside and downside Hawkes processes to large-move
    events extracted from daily returns.  The signal reflects whether
    bullish or bearish events are clustering more intensely.

    Args:
        symbol: Ticker symbol.
        event_threshold: Return threshold in units of rolling std.
        rolling_window: Window (days) for rolling-std event detection.
    """

    def __init__(
        self,
        symbol: str,
        event_threshold: float = _EVENT_THRESHOLD,
        rolling_window: int = _ROLLING_WINDOW,
    ) -> None:
        super().__init__("HawkesModel", symbol)
        self._threshold = event_threshold
        self._rolling_window = rolling_window

        # Fitted parameters
        self._up_params: tuple[float, float, float] = (0.0, 0.0, 1.0)
        self._dn_params: tuple[float, float, float] = (0.0, 0.0, 1.0)
        self._up_times: np.ndarray = np.array([])
        self._dn_times: np.ndarray = np.array([])
        self._T: float = 1.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame) -> HawkesModel:
        """Fit Hawkes processes for upside and downside events.

        Args:
            data: OHLCV DataFrame (UTC DatetimeIndex).

        Returns:
            Self.
        """
        returns = np.log(data["close"] / data["close"].shift(1)).dropna()
        if len(returns) < 30:
            raise ValueError(f"HawkesModel requires ≥30 observations; got {len(returns)}")

        up_times, dn_times = self._extract_events(returns)
        T = float(len(returns))

        mu_u, alpha_u, beta_u, _ = _fit_hawkes(up_times, T)
        mu_d, alpha_d, beta_d, _ = _fit_hawkes(dn_times, T)

        self._up_params = (mu_u, alpha_u, beta_u)
        self._dn_params = (mu_d, alpha_d, beta_d)
        self._up_times = up_times
        self._dn_times = dn_times
        self._T = T
        self._is_fitted = True

        br_up = alpha_u / max(beta_u, 1e-8)
        br_dn = alpha_d / max(beta_d, 1e-8)
        self._log_fit_complete(
            n_up_events=len(up_times),
            n_dn_events=len(dn_times),
            branching_ratio_up=round(br_up, 4),
            branching_ratio_dn=round(br_dn, 4),
        )
        return self

    def predict(self, data: pd.DataFrame) -> ModelOutput:
        """Generate a Hawkes-intensity-based trading signal.

        Args:
            data: OHLCV DataFrame.

        Returns:
            :class:`ModelOutput`.
        """
        self._require_fitted()

        t_now = self._T  # query at end of training window

        mu_u, alpha_u, beta_u = self._up_params
        mu_d, alpha_d, beta_d = self._dn_params

        lam_u = _current_intensity(mu_u, alpha_u, beta_u, self._up_times, t_now)
        lam_d = _current_intensity(mu_d, alpha_d, beta_d, self._dn_times, t_now)

        # Directional intensity imbalance
        total = lam_u + lam_d + 1e-10
        imbalance = (lam_u - lam_d) / total  # in [-1, 1]
        signal = float(np.tanh(imbalance * 3.0))

        # Branching ratios for confidence and regime
        br_up = alpha_u / max(beta_u, 1e-8)
        br_dn = alpha_d / max(beta_d, 1e-8)
        avg_br = (br_up + br_dn) / 2.0

        # Low branching ratio → weak clustering → lower confidence
        # High branching ratio (close to 1) → strong clustering → high confidence
        confidence = float(np.clip(0.30 + 0.50 * min(avg_br, 1.0), 0.30, 0.80))

        # Regime
        if avg_br > 0.7:
            regime = "HIGH_CLUSTERING"
        elif avg_br > 0.3:
            regime = "MEDIUM_CLUSTERING"
        else:
            regime = "LOW_CLUSTERING"

        log_returns = np.log(data["close"] / data["close"].shift(1)).dropna()
        recent_vol = float(log_returns.iloc[-21:].std() * np.sqrt(TRADING_DAYS_PER_YEAR))

        return ModelOutput(
            model_name=self.model_name,
            symbol=self.symbol,
            timestamp=datetime.now(tz=UTC),
            signal=round(signal, 6),
            confidence=round(confidence, 6),
            forecast_return=0.0,
            forecast_std=round(recent_vol, 6),
            regime=regime,
            metadata={
                "lambda_up": round(lam_u, 6),
                "lambda_down": round(lam_d, 6),
                "branching_ratio_up": round(br_up, 4),
                "branching_ratio_dn": round(br_dn, 4),
                "n_up_events": len(self._up_times),
                "n_dn_events": len(self._dn_times),
                "imbalance": round(imbalance, 4),
            },
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_events(
        self,
        returns: pd.Series,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Classify days as upside / downside events.

        An event is defined as a return whose absolute value exceeds
        ``self._threshold`` times the rolling standard deviation.

        Args:
            returns: Daily log-return series.

        Returns:
            Tuple of (up_event_times, dn_event_times) as integer day indices.
        """
        rolling_std = returns.rolling(self._rolling_window, min_periods=10).std()
        threshold_series = self._threshold * rolling_std

        up_mask = returns > threshold_series
        dn_mask = returns < -threshold_series

        # Use integer positions as event times
        positions = np.arange(len(returns), dtype=np.float64)
        up_times = positions[up_mask.values]
        dn_times = positions[dn_mask.values]

        return up_times, dn_times
