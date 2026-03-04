"""Extreme Value Theory (EVT) risk model.

Implements Peaks-Over-Threshold (POT) with Generalised Pareto Distribution
for modelling tail risk beyond what parametric methods capture.

Also provides the Hill estimator as an alternative tail index estimator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.stats as stats
from pydantic import BaseModel, Field

from quantflow.config.logging import get_logger

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)

_MIN_EXCEEDANCES = 20  # GPD fit degrades below this
_DEFAULT_THRESHOLD_QUANTILE = 0.95


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


class EVTResult(BaseModel):
    """Results of an EVT / GPD tail fit.

    Attributes:
        threshold: Loss threshold u (95th percentile of losses by default).
        xi: GPD shape parameter (tail index).  Positive ⟹ heavy tail.
        sigma: GPD scale parameter.
        n_total: Total observations used.
        n_exceedances: Number of losses exceeding the threshold.
        var_99: 99% VaR from GPD extrapolation.
        var_999: 99.9% VaR from GPD extrapolation.
        es_99: 99% Expected Shortfall from GPD.
        es_999: 99.9% Expected Shortfall from GPD.
        hill_estimate: Hill tail index estimate (alternative to GPD xi).
    """

    threshold: float
    xi: float
    sigma: float = Field(..., gt=0.0)
    n_total: int
    n_exceedances: int
    var_99: float
    var_999: float
    es_99: float
    es_999: float
    hill_estimate: float | None = None


# ---------------------------------------------------------------------------
# EVT risk model
# ---------------------------------------------------------------------------


class EVTRiskModel:
    """Peaks-Over-Threshold EVT model using the Generalised Pareto Distribution.

    Workflow:
    1. Convert returns to losses (negate).
    2. Select threshold at the 95th percentile of losses.
    3. Fit GPD to exceedances (loss − threshold).
    4. Extrapolate extreme quantiles (VaR₉₉₉%, ES₉₉₉%).

    Also computes the Hill estimator as a non-parametric alternative.

    Args:
        threshold_quantile: Quantile for threshold selection (default 0.95).
    """

    def __init__(self, threshold_quantile: float = _DEFAULT_THRESHOLD_QUANTILE) -> None:
        self._q = threshold_quantile
        self._logger = get_logger(__name__)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, returns: pd.Series) -> EVTResult:
        """Fit the GPD to loss exceedances.

        Args:
            returns: Daily log-returns (sign convention: positive = gain).

        Returns:
            :class:`EVTResult` with tail fit parameters and extreme quantiles.

        Raises:
            ValueError: If fewer than ``_MIN_EXCEEDANCES`` losses exceed threshold.
        """
        losses = -returns.dropna().values.astype(np.float64)
        n = len(losses)

        threshold = self._select_threshold(losses)
        exceedances = losses[losses > threshold] - threshold

        if len(exceedances) < _MIN_EXCEEDANCES:
            raise ValueError(
                f"Only {len(exceedances)} exceedances above threshold "
                f"{threshold:.4f}; need at least {_MIN_EXCEEDANCES}."
            )

        xi, sigma = self._fit_gpd(exceedances)
        n_u = len(exceedances)

        # Extreme quantiles via GPD inversion
        var_99 = self._gpd_quantile(threshold, xi, sigma, n, n_u, p=0.99)
        var_999 = self._gpd_quantile(threshold, xi, sigma, n, n_u, p=0.999)
        es_99 = self._gpd_es(threshold, xi, sigma, n, n_u, p=0.99)
        es_999 = self._gpd_es(threshold, xi, sigma, n, n_u, p=0.999)

        # Hill estimator (top 10% of losses)
        k_hill = max(10, n // 10)
        hill = self._hill_estimator(losses, k=k_hill)

        self._logger.info(
            "EVT fitted",
            threshold=round(threshold, 6),
            xi=round(xi, 4),
            sigma=round(sigma, 6),
            n_exceedances=n_u,
            var_999=round(var_999, 6),
        )

        return EVTResult(
            threshold=round(float(threshold), 8),
            xi=round(float(xi), 6),
            sigma=round(float(sigma), 8),
            n_total=n,
            n_exceedances=n_u,
            var_99=round(float(var_99), 8),
            var_999=round(float(var_999), 8),
            es_99=round(float(es_99), 8),
            es_999=round(float(es_999), 8),
            hill_estimate=round(float(hill), 6) if hill is not None else None,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _select_threshold(self, losses: np.ndarray) -> float:
        """Select the GPD threshold as a quantile of the loss distribution.

        Args:
            losses: Loss values (positive = loss).

        Returns:
            Threshold float.
        """
        return float(np.percentile(losses, self._q * 100))

    def _fit_gpd(self, exceedances: np.ndarray) -> tuple[float, float]:
        """Fit Generalised Pareto Distribution to exceedances via MLE.

        Args:
            exceedances: Loss values above the threshold (already threshold-subtracted).

        Returns:
            Tuple (xi, sigma) — shape and scale parameters.
        """
        try:
            # scipy genpareto: c = xi (shape), loc=0 fixed, scale = sigma
            xi, _loc, sigma = stats.genpareto.fit(exceedances, floc=0.0)
            # Clip xi to avoid numerical issues with very heavy / very light tails
            xi = float(np.clip(xi, -0.5, 0.9))
            sigma = max(float(sigma), 1e-8)
            return xi, sigma
        except Exception:
            # MOM fallback
            mean_e = float(exceedances.mean())
            var_e = float(exceedances.var())
            sigma_mom = 0.5 * mean_e * (mean_e**2 / var_e + 1) if var_e > 0 else mean_e
            xi_mom = 0.5 * (mean_e**2 / var_e - 1) if var_e > 0 else 0.0
            return float(np.clip(xi_mom, -0.5, 0.9)), max(sigma_mom, 1e-8)

    @staticmethod
    def _gpd_quantile(
        u: float,
        xi: float,
        sigma: float,
        n: int,
        n_u: int,
        p: float,
    ) -> float:
        """Compute GPD-extrapolated VaR at probability level p.

        Formula: u + (sigma/xi) * [((n/n_u) * (1-p))^{-xi} - 1]
        For xi = 0 (exponential): u - sigma * log((n/n_u)*(1-p))

        Args:
            u: Threshold.
            xi: GPD shape.
            sigma: GPD scale.
            n: Total observations.
            n_u: Number of threshold exceedances.
            p: Probability level (e.g. 0.999 for 99.9% VaR).

        Returns:
            VaR as a loss value (positive = loss).
        """
        prob_factor = (n / n_u) * (1.0 - p)
        prob_factor = max(prob_factor, 1e-10)
        if abs(xi) < 1e-6:
            return u - sigma * np.log(prob_factor)
        return u + (sigma / xi) * (prob_factor ** (-xi) - 1.0)

    @staticmethod
    def _gpd_es(
        u: float,
        xi: float,
        sigma: float,
        n: int,
        n_u: int,
        p: float,
    ) -> float:
        """Compute GPD-extrapolated Expected Shortfall at level p.

        Formula: ES = VaR / (1 - xi) + (sigma - xi*u) / (1 - xi)

        Args:
            u: Threshold.
            xi: GPD shape.
            sigma: GPD scale.
            n: Total observations.
            n_u: Exceedances count.
            p: Probability level.

        Returns:
            ES as a loss value (positive = loss).
        """
        var_p = EVTRiskModel._gpd_quantile(u, xi, sigma, n, n_u, p)
        if xi >= 1.0:
            return float("inf")
        es = (var_p + sigma - xi * u) / (1.0 - xi)
        return max(es, var_p)  # ES must be >= VaR

    @staticmethod
    def _hill_estimator(losses: np.ndarray, k: int) -> float | None:
        """Hill (1975) non-parametric tail index estimator.

        Uses the top-k order statistics of the loss distribution.

        Args:
            losses: All loss values.
            k: Number of top order statistics to use.

        Returns:
            Hill tail index estimate (1/xi_Hill), or None if insufficient data.
        """
        sorted_losses = np.sort(losses)[::-1]  # Descending
        if k >= len(sorted_losses) or k < 2:
            return None
        top_k = sorted_losses[:k]
        threshold_k = sorted_losses[k]  # (k+1)-th order statistic
        if threshold_k <= 0:
            return None
        log_ratios = np.log(top_k / (threshold_k + 1e-10))
        log_ratios = log_ratios[log_ratios > 0]
        if len(log_ratios) == 0:
            return None
        return float(1.0 / log_ratios.mean())
