"""Stress testing and scenario analysis.

Three scenario types:
1. Historical: replay actual crisis periods on the current portfolio.
2. Factor shock: apply user-defined factor shocks via beta exposures.
3. Monte Carlo stress: simulate extreme scenarios from fat-tailed distribution.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats
from pydantic import BaseModel, Field

from quantflow.config.constants import STRESS_PERIODS, TRADING_DAYS_PER_YEAR
from quantflow.config.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_N_STRESS_SCENARIOS = 50_000
_STRESS_TAIL_DF = 4.0          # t-distribution df for stress MC
_STRESS_VOL_MULTIPLIER = 3.0   # Inflate vol for stress scenarios


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------


class ScenarioResult(BaseModel):
    """Result of a single stress scenario.

    Attributes:
        scenario_name: Identifier for the scenario.
        portfolio_return: Simulated portfolio return over the scenario.
        var_stressed: 1st-percentile return within the scenario window.
        es_stressed: Expected return below var_stressed.
        worst_asset_return: Return of the worst-performing asset.
        best_asset_return: Return of the best-performing asset.
        duration_days: Length of the historical scenario in calendar days.
        metadata: Additional diagnostics.
    """

    scenario_name: str
    portfolio_return: float
    var_stressed: float
    es_stressed: float
    worst_asset_return: float
    best_asset_return: float
    duration_days: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class StressDistribution(BaseModel):
    """Distribution of Monte Carlo stress scenario outcomes.

    Attributes:
        p50: Median portfolio return.
        p10: 10th-percentile return.
        p5: 5th-percentile return.
        p1: 1st-percentile return.
        p01: 0.1th-percentile return.
        expected_shortfall_999: Expected return below 0.1th percentile.
        mean: Mean of simulated returns.
        std: Standard deviation of simulated returns.
        n_scenarios: Number of paths simulated.
    """

    p50: float
    p10: float
    p5: float
    p1: float
    p01: float
    expected_shortfall_999: float
    mean: float
    std: float
    n_scenarios: int


# ---------------------------------------------------------------------------
# Stress tester
# ---------------------------------------------------------------------------


class StressTester:
    """Stress test a portfolio under historical and hypothetical scenarios.

    Args:
        n_stress_scenarios: Number of Monte Carlo paths for stress testing.
    """

    def __init__(self, n_stress_scenarios: int = _DEFAULT_N_STRESS_SCENARIOS) -> None:
        self._n = n_stress_scenarios
        self._logger = get_logger(__name__)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run_historical_scenario(
        self,
        weights: dict[str, float],
        returns_df: pd.DataFrame,
        scenario_name: str,
    ) -> ScenarioResult:
        """Replay a historical crisis period on the current portfolio.

        Args:
            weights: Dict mapping asset ticker → portfolio weight (sum to 1).
            returns_df: DataFrame of daily log-returns (columns = tickers,
                DatetimeIndex).
            scenario_name: Key from ``STRESS_PERIODS`` (e.g. "gfc_2008") or
                a custom string.  Custom scenarios require ``returns_df`` to
                already be filtered to the desired window.

        Returns:
            :class:`ScenarioResult` for this scenario.
        """
        period = STRESS_PERIODS.get(scenario_name)

        if period is not None:
            start, end = period
            mask = (returns_df.index >= start) & (returns_df.index <= end)
            window = returns_df.loc[mask]
        else:
            window = returns_df

        if window.empty:
            self._logger.warning(
                "No data for scenario",
                scenario=scenario_name,
            )
            return ScenarioResult(
                scenario_name=scenario_name,
                portfolio_return=0.0,
                var_stressed=0.0,
                es_stressed=0.0,
                worst_asset_return=0.0,
                best_asset_return=0.0,
            )

        # Align assets — only include assets present in both weights and returns
        common = [a for a in weights if a in window.columns]
        if not common:
            raise ValueError(
                f"No common assets between weights and returns_df for scenario "
                f"'{scenario_name}'. Weights: {list(weights)}, "
                f"Returns columns: {list(window.columns)}"
            )

        w = np.array([weights[a] for a in common], dtype=np.float64)
        w /= w.sum()  # Renormalise after subsetting

        asset_returns = window[common].fillna(0.0)
        port_returns = asset_returns.values @ w  # Daily portfolio returns

        cumulative = float(port_returns.sum())
        daily_percentile_1 = float(np.percentile(port_returns, 1))
        tail = port_returns[port_returns <= daily_percentile_1]
        es_stressed = float(tail.mean()) if len(tail) > 0 else daily_percentile_1

        total_period = asset_returns.sum()
        worst = float(total_period.min())
        best = float(total_period.max())

        duration = len(window)
        self._logger.info(
            "Historical scenario complete",
            scenario=scenario_name,
            portfolio_return=round(cumulative, 4),
            duration_days=duration,
        )

        return ScenarioResult(
            scenario_name=scenario_name,
            portfolio_return=round(cumulative, 6),
            var_stressed=round(daily_percentile_1, 6),
            es_stressed=round(es_stressed, 6),
            worst_asset_return=round(worst, 6),
            best_asset_return=round(best, 6),
            duration_days=duration,
            metadata={"n_assets": len(common), "n_days": duration},
        )

    def run_factor_shock(
        self,
        weights: dict[str, float],
        betas: dict[str, dict[str, float]],
        shock_dict: dict[str, float],
    ) -> ScenarioResult:
        """Apply instantaneous factor shocks to a portfolio.

        Computes portfolio return as the weighted sum of asset-level returns
        implied by each asset's factor beta exposures and the factor shocks.

        Args:
            weights: Dict mapping asset → portfolio weight.
            betas: Dict mapping asset → {factor_name → beta}.
                   Example: ``{"AAPL": {"equity_market": 1.2, "tech": 0.8}}``.
            shock_dict: Factor shock magnitudes.
                   Example: ``{"equity_market": -0.20, "tech": -0.10}``.

        Returns:
            :class:`ScenarioResult` describing the shocked portfolio return.
        """
        assets = list(weights)
        w = np.array([weights[a] for a in assets], dtype=np.float64)
        w /= w.sum()

        asset_returns = np.zeros(len(assets))
        for i, asset in enumerate(assets):
            asset_betas = betas.get(asset, {})
            asset_ret = sum(
                asset_betas.get(factor, 0.0) * shock
                for factor, shock in shock_dict.items()
            )
            asset_returns[i] = asset_ret

        port_return = float(w @ asset_returns)
        worst = float(asset_returns.min())
        best = float(asset_returns.max())

        shock_label = "|".join(f"{k}={v:+.1%}" for k, v in shock_dict.items())
        self._logger.info(
            "Factor shock applied",
            shocks=shock_label,
            portfolio_return=round(port_return, 4),
        )

        return ScenarioResult(
            scenario_name=f"factor_shock:{shock_label}",
            portfolio_return=round(port_return, 6),
            var_stressed=round(port_return, 6),
            es_stressed=round(port_return, 6),
            worst_asset_return=round(worst, 6),
            best_asset_return=round(best, 6),
            metadata={"shocks": shock_dict, "n_assets": len(assets)},
        )

    def run_monte_carlo_stress(
        self,
        returns: pd.DataFrame,
        weights: dict[str, float],
        horizon: int = 21,
    ) -> StressDistribution:
        """Monte Carlo stress test using a fat-tailed (Student-t) distribution.

        Models joint asset returns with Student-t innovations to capture
        tail dependence and correlation breakdown under stress.

        Args:
            returns: Historical daily log-returns DataFrame (columns = tickers).
            weights: Dict mapping ticker → portfolio weight.
            horizon: Simulation horizon in days.

        Returns:
            :class:`StressDistribution` summarising the simulated portfolio P&L.
        """
        common = [a for a in weights if a in returns.columns]
        if not common:
            raise ValueError("No common assets between weights and returns columns.")

        w = np.array([weights[a] for a in common], dtype=np.float64)
        w /= w.sum()

        data = returns[common].dropna().values.astype(np.float64)
        n_assets = len(common)

        # Estimate mean and covariance
        mu = data.mean(axis=0)
        cov = np.cov(data.T)

        # Inflate covariance for stress (vol * sqrt(3) ≈ extreme regime)
        stressed_cov = cov * (_STRESS_VOL_MULTIPLIER**2)

        rng = np.random.default_rng(seed=0)
        # Multivariate Student-t: simulate via chi-squared scaling
        # X = mu + L @ Z / sqrt(chi2/df) where L = cholesky(stressed_cov)
        try:
            L = np.linalg.cholesky(stressed_cov + np.eye(n_assets) * 1e-8)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(np.diag(np.diag(stressed_cov)) + np.eye(n_assets) * 1e-8)

        Z = rng.standard_normal(size=(self._n, n_assets, horizon))
        chi2 = rng.chisquare(df=_STRESS_TAIL_DF, size=(self._n, 1, horizon))
        t_factor = np.sqrt(chi2 / _STRESS_TAIL_DF)

        # Daily stressed returns: (n_sims, n_assets, horizon)
        stressed_daily = mu[np.newaxis, :, np.newaxis] + (L @ Z.transpose(0, 2, 1)).transpose(0, 2, 1) / t_factor

        # Portfolio return over horizon
        port_daily = (stressed_daily * w[np.newaxis, :, np.newaxis]).sum(axis=1)  # (n_sims, horizon)
        port_total = port_daily.sum(axis=1)  # (n_sims,)

        p50 = float(np.percentile(port_total, 50))
        p10 = float(np.percentile(port_total, 10))
        p5 = float(np.percentile(port_total, 5))
        p1 = float(np.percentile(port_total, 1))
        p01 = float(np.percentile(port_total, 0.1))
        tail = port_total[port_total <= p01]
        es_999 = float(tail.mean()) if len(tail) > 0 else p01

        self._logger.info(
            "Monte Carlo stress complete",
            n_scenarios=self._n,
            horizon=horizon,
            p1=round(p1, 4),
            p01=round(p01, 4),
        )

        return StressDistribution(
            p50=round(p50, 6),
            p10=round(p10, 6),
            p5=round(p5, 6),
            p1=round(p1, 6),
            p01=round(p01, 6),
            expected_shortfall_999=round(es_999, 6),
            mean=round(float(port_total.mean()), 6),
            std=round(float(port_total.std()), 6),
            n_scenarios=self._n,
        )

    def run_all_historical(
        self,
        weights: dict[str, float],
        returns_df: pd.DataFrame,
    ) -> dict[str, ScenarioResult]:
        """Run all built-in historical stress scenarios.

        Args:
            weights: Dict mapping asset → portfolio weight.
            returns_df: Full history of daily log-returns.

        Returns:
            Dict mapping scenario_name → :class:`ScenarioResult`.
        """
        results: dict[str, ScenarioResult] = {}
        for name in STRESS_PERIODS:
            try:
                results[name] = self.run_historical_scenario(weights, returns_df, name)
            except Exception as exc:
                self._logger.warning(
                    "Scenario failed",
                    scenario=name,
                    error=str(exc),
                )
        return results
