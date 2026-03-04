"""Value-at-Risk and Expected Shortfall calculator.

Three methodologies:
1. Historical Simulation — non-parametric rolling-window approach.
2. Parametric (Variance-Covariance) — Normal or Student-t MLE fit.
3. Monte Carlo — 10,000 paths from fitted distribution.

Backtesting via Kupiec (unconditional coverage) and Christoffersen
(conditional coverage / independence of violations) tests.

All VaR / ES values are expressed as *return* values (negative = loss).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import scipy.stats as stats
from pydantic import BaseModel, Field

from quantflow.config.constants import TRADING_DAYS_PER_YEAR
from quantflow.config.logging import get_logger

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)

_MIN_OBSERVATIONS = 30  # Minimum for meaningful statistics
_DEFAULT_N_SIMULATIONS = 10_000
_FAT_TAIL_KURTOSIS = 3.5  # Switch to Student-t above this kurtosis


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


class RiskReport(BaseModel):
    """Output of a VaR / ES calculation.

    Attributes:
        method: Calculation method used.
        confidence: VaR confidence level (e.g. 0.99).
        horizon: Holding period in days.
        var: Value-at-Risk (negative = loss, e.g. -0.025 = 2.5% loss).
        es: Expected Shortfall (always <= var; negative = loss).
        distribution: Distribution fitted (``"normal"`` or ``"student_t"``).
        df_t: Degrees of freedom if Student-t was used.
        kupiec_pvalue: Kupiec unconditional coverage test p-value.
        christoffersen_pvalue: Christoffersen independence test p-value.
        n_observations: Number of return observations used.
        n_violations: Number of VaR breaches in the history.
    """

    method: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    horizon: int = Field(..., ge=1)
    var: float  # Negative value representing a loss
    es: float
    distribution: str = "normal"
    df_t: float | None = None
    kupiec_pvalue: float | None = None
    christoffersen_pvalue: float | None = None
    n_observations: int = 0
    n_violations: int = 0


# ---------------------------------------------------------------------------
# Main calculator
# ---------------------------------------------------------------------------


class RiskCalculator:
    """Compute VaR and ES using three methodologies with backtesting.

    Args:
        window: Rolling lookback window in trading days for historical method.
        n_simulations: Number of Monte Carlo paths.
    """

    def __init__(
        self,
        window: int = 252,
        n_simulations: int = _DEFAULT_N_SIMULATIONS,
    ) -> None:
        self._window = window
        self._n_sims = n_simulations
        self._logger = get_logger(__name__)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute_var_es(
        self,
        returns: pd.Series,
        confidence: float = 0.99,
        horizon: int = 1,
        method: Literal["historical", "parametric", "monte_carlo"] = "historical",
        run_backtest: bool = True,
    ) -> RiskReport:
        """Compute VaR and ES for a single return series.

        Args:
            returns: Daily log-returns (float Series, DatetimeIndex).
            confidence: VaR confidence level in ``(0, 1)`` (e.g. 0.99 for 99%).
            horizon: Holding period in days (square-root-of-time scaling applied).
            method: VaR methodology to use.
            run_backtest: Whether to compute Kupiec / Christoffersen tests.

        Returns:
            :class:`RiskReport` with VaR, ES, and backtesting results.

        Raises:
            ValueError: If fewer than ``_MIN_OBSERVATIONS`` valid returns.
        """
        clean = returns.dropna().astype(np.float64)
        if len(clean) < _MIN_OBSERVATIONS:
            raise ValueError(f"Insufficient observations: {len(clean)} < {_MIN_OBSERVATIONS}")

        window_returns = clean.iloc[-self._window :]

        if method == "historical":
            var, es = self._historical(window_returns, confidence, horizon)
            dist = "empirical"
            df_t = None
        elif method == "parametric":
            var, es, dist, df_t = self._parametric(window_returns, confidence, horizon)
        else:  # monte_carlo
            var, es, dist, df_t = self._monte_carlo(window_returns, confidence, horizon)

        kpv: float | None = None
        cpv: float | None = None
        n_obs = len(clean)
        n_viol = 0

        if run_backtest and len(clean) >= 60:
            violations = (clean < var).astype(int).values
            n_viol = int(violations.sum())
            kpv = self._kupiec_test(violations, confidence)
            cpv = self._christoffersen_test(violations)

        self._logger.info(
            "VaR/ES computed",
            method=method,
            confidence=confidence,
            horizon=horizon,
            var=round(var, 6),
            es=round(es, 6),
        )

        return RiskReport(
            method=method,
            confidence=confidence,
            horizon=horizon,
            var=round(float(var), 8),
            es=round(float(es), 8),
            distribution=dist,
            df_t=df_t,
            kupiec_pvalue=kpv,
            christoffersen_pvalue=cpv,
            n_observations=n_obs,
            n_violations=n_viol,
        )

    def compute_all_methods(
        self,
        returns: pd.Series,
        confidence: float = 0.99,
        horizon: int = 1,
    ) -> dict[str, RiskReport]:
        """Compute VaR/ES with all three methods and return all reports.

        Args:
            returns: Daily log-returns.
            confidence: VaR confidence level.
            horizon: Holding period in days.

        Returns:
            Dict mapping method name → :class:`RiskReport`.
        """
        results: dict[str, RiskReport] = {}
        for method in ("historical", "parametric", "monte_carlo"):
            try:
                results[method] = self.compute_var_es(
                    returns,
                    confidence=confidence,
                    horizon=horizon,
                    method=method,
                )
            except Exception as exc:
                self._logger.warning(
                    "VaR method failed",
                    method=method,
                    error=str(exc),
                )
        return results

    def compute_portfolio_var(
        self,
        weights: np.ndarray,
        covariance_matrix: np.ndarray,
        mean_returns: np.ndarray | None = None,
        confidence: float = 0.99,
        horizon: int = 1,
        n_simulations: int = _DEFAULT_N_SIMULATIONS,
    ) -> RiskReport:
        """Compute parametric portfolio VaR from weights and covariance matrix.

        Args:
            weights: Portfolio weight vector (must sum to 1, non-negative).
            covariance_matrix: Asset covariance matrix (annualised).
            mean_returns: Expected daily returns vector (defaults to zeros).
            confidence: VaR confidence level.
            horizon: Holding period in days.
            n_simulations: Monte Carlo paths.

        Returns:
            :class:`RiskReport` using parametric method.
        """
        w = np.asarray(weights, dtype=np.float64)
        cov = np.asarray(covariance_matrix, dtype=np.float64)
        mu = (
            np.asarray(mean_returns, dtype=np.float64)
            if mean_returns is not None
            else np.zeros(len(w))
        )

        # Daily covariance
        daily_cov = cov / TRADING_DAYS_PER_YEAR

        port_mu = float(w @ mu)
        port_var_scalar = float(w @ daily_cov @ w)
        port_sigma = float(np.sqrt(port_var_scalar))

        # Square-root-of-time scaling for horizon > 1
        scaled_mu = port_mu * horizon
        scaled_sigma = port_sigma * np.sqrt(horizon)

        alpha = 1.0 - confidence
        var = float(scaled_mu + stats.norm.ppf(alpha) * scaled_sigma)
        es = float(scaled_mu - scaled_sigma * stats.norm.pdf(stats.norm.ppf(alpha)) / alpha)

        return RiskReport(
            method="portfolio_parametric",
            confidence=confidence,
            horizon=horizon,
            var=round(var, 8),
            es=round(es, 8),
            distribution="normal",
        )

    # ------------------------------------------------------------------
    # VaR/ES methods
    # ------------------------------------------------------------------

    def _historical(
        self,
        returns: pd.Series,
        confidence: float,
        horizon: int,
    ) -> tuple[float, float]:
        """Historical simulation VaR / ES.

        Args:
            returns: Window of historical returns.
            confidence: Confidence level.
            horizon: Holding period (square-root-of-time scaling).

        Returns:
            Tuple of (var, es).
        """
        # Aggregate to horizon-day returns
        if horizon > 1:
            # Non-overlapping windows
            n_periods = len(returns) // horizon
            agg = np.array(
                [returns.iloc[i * horizon : (i + 1) * horizon].sum() for i in range(n_periods)]
            )
        else:
            agg = returns.values.astype(np.float64)

        alpha = 1.0 - confidence
        var = float(np.percentile(agg, alpha * 100))
        tail = agg[agg <= var]
        es = float(tail.mean()) if len(tail) > 0 else var

        return var, min(es, var)

    def _parametric(
        self,
        returns: pd.Series,
        confidence: float,
        horizon: int,
    ) -> tuple[float, float, str, float | None]:
        """Parametric (variance-covariance) VaR / ES.

        Fits Normal; switches to Student-t if excess kurtosis > threshold.

        Args:
            returns: Window of historical returns.
            confidence: Confidence level.
            horizon: Holding period.

        Returns:
            Tuple of (var, es, distribution_name, df_t_or_None).
        """
        r = returns.values.astype(np.float64)
        mu = float(r.mean())
        sigma = float(r.std(ddof=1))
        kurt = float(stats.kurtosis(r, fisher=True))

        alpha = 1.0 - confidence
        sqrt_h = float(np.sqrt(horizon))

        if kurt > _FAT_TAIL_KURTOSIS:
            # Fit Student-t via MLE
            df, loc, scale = stats.t.fit(r, floc=mu)
            df = max(df, 2.1)  # Ensure finite variance
            scaled_loc = loc * horizon
            scaled_scale = scale * sqrt_h
            var = float(scaled_loc + stats.t.ppf(alpha, df) * scaled_scale)
            # ES for Student-t
            t_alpha = stats.t.ppf(alpha, df)
            es = float(
                scaled_loc
                - scaled_scale * (stats.t.pdf(t_alpha, df) / alpha) * ((df + t_alpha**2) / (df - 1))
            )
            return var, min(es, var), "student_t", float(df)
        else:
            scaled_mu = mu * horizon
            scaled_sigma = sigma * sqrt_h
            var = float(scaled_mu + stats.norm.ppf(alpha) * scaled_sigma)
            es = float(scaled_mu - scaled_sigma * stats.norm.pdf(stats.norm.ppf(alpha)) / alpha)
            return var, min(es, var), "normal", None

    def _monte_carlo(
        self,
        returns: pd.Series,
        confidence: float,
        horizon: int,
    ) -> tuple[float, float, str, float | None]:
        """Monte Carlo VaR / ES from fitted distribution.

        Args:
            returns: Window of historical returns.
            confidence: Confidence level.
            horizon: Holding period.

        Returns:
            Tuple of (var, es, distribution_name, df_t_or_None).
        """
        r = returns.values.astype(np.float64)
        kurt = float(stats.kurtosis(r, fisher=True))

        rng = np.random.default_rng(seed=42)

        if kurt > _FAT_TAIL_KURTOSIS:
            df, loc, scale = stats.t.fit(r, floc=r.mean())
            df = max(df, 2.1)
            # Simulate horizon-day returns as sum of daily draws
            daily = stats.t.rvs(
                df, loc=loc, scale=scale, size=(self._n_sims, horizon), random_state=rng
            )
            dist = "student_t"
            df_out: float | None = float(df)
        else:
            mu, sigma = float(r.mean()), float(r.std(ddof=1))
            daily = rng.normal(mu, sigma, size=(self._n_sims, horizon))
            dist = "normal"
            df_out = None

        sim_returns = daily.sum(axis=1)

        alpha = 1.0 - confidence
        var = float(np.percentile(sim_returns, alpha * 100))
        tail = sim_returns[sim_returns <= var]
        es = float(tail.mean()) if len(tail) > 0 else var

        return var, min(es, var), dist, df_out

    # ------------------------------------------------------------------
    # Backtesting
    # ------------------------------------------------------------------

    @staticmethod
    def _kupiec_test(violations: np.ndarray, confidence: float) -> float:
        """Kupiec (1995) Proportion of Failures (POF) test.

        H₀: violation rate = (1 − confidence).

        Args:
            violations: Binary array; 1 if return < VaR on that day.
            confidence: VaR confidence level used.

        Returns:
            p-value (> 0.05 ⟹ model not rejected).
        """
        T = len(violations)
        x = int(violations.sum())
        p = 1.0 - confidence  # Expected violation rate

        if x == 0 or x == T:
            return 1.0  # Degenerate case

        # Log-likelihood ratio statistic
        p_hat = x / T
        try:
            lr = 2.0 * (x * np.log(p_hat / p) + (T - x) * np.log((1.0 - p_hat) / (1.0 - p)))
        except (ValueError, ZeroDivisionError):
            return 1.0

        pvalue = float(1.0 - stats.chi2.cdf(lr, df=1))
        return round(pvalue, 6)

    @staticmethod
    def _christoffersen_test(violations: np.ndarray) -> float:
        """Christoffersen (1998) independence test for VaR violations.

        Tests whether violations cluster in time.

        Args:
            violations: Binary array of VaR breaches.

        Returns:
            p-value (> 0.05 ⟹ violations appear independent).
        """
        n = len(violations)
        if n < 4:
            return 1.0

        # Transition counts
        n00 = n01 = n10 = n11 = 0
        for i in range(n - 1):
            a, b = int(violations[i]), int(violations[i + 1])
            if a == 0 and b == 0:
                n00 += 1
            elif a == 0 and b == 1:
                n01 += 1
            elif a == 1 and b == 0:
                n10 += 1
            else:
                n11 += 1

        n0 = n00 + n01
        n1 = n10 + n11

        if n0 == 0 or n1 == 0 or n01 == 0 or n10 == 0:
            return 1.0

        pi01 = n01 / n0
        pi11 = n11 / n1 if n1 > 0 else 0.0
        pi = (n01 + n11) / (n0 + n1)

        try:
            lr = -2.0 * (
                (n00 + n10) * np.log(1.0 - pi)
                + (n01 + n11) * np.log(pi)
                - n00 * np.log(1.0 - pi01 + 1e-12)
                - n01 * np.log(pi01 + 1e-12)
                - n10 * np.log(1.0 - pi11 + 1e-12)
                - n11 * np.log(pi11 + 1e-12)
            )
        except (ValueError, ZeroDivisionError):
            return 1.0

        pvalue = float(1.0 - stats.chi2.cdf(lr, df=1))
        return round(pvalue, 6)
