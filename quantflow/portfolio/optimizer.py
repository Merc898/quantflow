"""Mean-Variance Optimization (MVO) with multiple covariance estimators.

Implements:
- Covariance estimation: sample, Ledoit-Wolf, OAS, Random Matrix Theory (RMT).
- Expected returns: sample mean, CAPM-implied, signal-implied.
- Optimization via cvxpy (CLARABEL solver).
- Efficient frontier computation (50 points).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from quantflow.config.constants import (
    MAX_VARIANCE_WEIGHT,
    MIN_VARIANCE_WEIGHT,
    RISK_AVERSION_LAMBDA,
    TRADING_DAYS_PER_YEAR,
)
from quantflow.config.logging import get_logger

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)

_MIN_OBSERVATIONS = 60  # Minimum rows for reliable covariance
_N_FRONTIER_POINTS = 50


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------


class EfficientFrontierPoint(BaseModel):
    """A single point on the efficient frontier.

    Attributes:
        expected_return: Annualised expected portfolio return.
        expected_vol: Annualised expected portfolio volatility.
        sharpe_ratio: Sharpe ratio (assuming zero risk-free rate).
        weights: Dict mapping asset ticker → weight.
    """

    expected_return: float
    expected_vol: float
    sharpe_ratio: float
    weights: dict[str, float] = Field(default_factory=dict)


class OptimizationResult(BaseModel):
    """Result of a portfolio optimization.

    Attributes:
        weights: Dict mapping asset ticker → optimal weight.
        expected_return: Annualised expected return.
        expected_vol: Annualised expected volatility.
        sharpe_ratio: Sharpe ratio (zero risk-free rate).
        covariance_method: Covariance estimator used.
        return_method: Expected return estimator used.
        efficient_frontier: List of frontier points (if computed).
        metadata: Diagnostics (condition number, n_assets, etc.).
    """

    weights: dict[str, float]
    expected_return: float
    expected_vol: float
    sharpe_ratio: float
    covariance_method: str
    return_method: str
    efficient_frontier: list[EfficientFrontierPoint] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# MVO Optimizer
# ---------------------------------------------------------------------------


class MVOOptimizer:
    """Markowitz Mean-Variance Optimizer with modern enhancements.

    Supports multiple covariance estimators (sample, Ledoit-Wolf, OAS, RMT)
    and expected return methods (sample mean, CAPM, signal-implied).

    Args:
        risk_aversion: Lambda (λ) in the mean-variance objective
            ``max w'μ − (λ/2) w'Σw``.
        max_weight: Maximum weight per asset.
        min_weight: Minimum weight per asset (long-only by default).
    """

    def __init__(
        self,
        risk_aversion: float = RISK_AVERSION_LAMBDA,
        max_weight: float = MAX_VARIANCE_WEIGHT,
        min_weight: float = MIN_VARIANCE_WEIGHT,
    ) -> None:
        self._lam = risk_aversion
        self._max_w = max_weight
        self._min_w = min_weight
        self._logger = get_logger(__name__)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def optimize(
        self,
        returns: pd.DataFrame,
        covariance_method: Literal["sample", "ledoit_wolf", "oas", "rmt"] = "ledoit_wolf",
        return_method: Literal["sample", "capm", "signal"] = "sample",
        signal_returns: pd.Series | None = None,
        market_returns: pd.Series | None = None,
        risk_free_rate: float = 0.0,
        compute_frontier: bool = False,
    ) -> OptimizationResult:
        """Run MVO and return the optimal portfolio.

        Args:
            returns: DataFrame of daily log-returns (columns = tickers).
            covariance_method: Covariance estimation method.
            return_method: Expected return estimation method.
            signal_returns: Signal-implied annualised returns (for
                ``return_method="signal"``).
            market_returns: Market benchmark returns for CAPM estimation.
            risk_free_rate: Annualised risk-free rate.
            compute_frontier: Whether to compute the efficient frontier.

        Returns:
            :class:`OptimizationResult` with optimal weights and diagnostics.

        Raises:
            ValueError: If fewer than ``_MIN_OBSERVATIONS`` rows.
        """
        if len(returns) < _MIN_OBSERVATIONS:
            raise ValueError(f"Need at least {_MIN_OBSERVATIONS} observations, got {len(returns)}")

        assets = list(returns.columns)
        n = len(assets)
        data = returns.dropna().values.astype(np.float64)

        # --- Covariance estimation ---
        cov_daily = self._estimate_covariance(data, method=covariance_method)
        cov_annual = cov_daily * TRADING_DAYS_PER_YEAR

        # --- Expected returns ---
        mu = self._estimate_returns(
            data,
            returns,
            cov_annual,
            assets,
            method=return_method,
            signal_returns=signal_returns,
            market_returns=market_returns,
            risk_free_rate=risk_free_rate,
        )

        # --- Optimize ---
        w_opt = self._solve_mvo(mu, cov_annual)

        # Annualised portfolio stats
        port_ret = float(w_opt @ mu)
        port_vol = float(np.sqrt(w_opt @ cov_annual @ w_opt))
        sharpe = (port_ret - risk_free_rate) / port_vol if port_vol > 0 else 0.0

        weights_dict = {assets[i]: round(float(w_opt[i]), 6) for i in range(n)}

        # Efficient frontier
        frontier: list[EfficientFrontierPoint] = []
        if compute_frontier:
            frontier = self._compute_efficient_frontier(mu, cov_annual, assets, risk_free_rate)

        cond_num = float(np.linalg.cond(cov_annual))
        self._logger.info(
            "MVO optimized",
            n_assets=n,
            covariance_method=covariance_method,
            return_method=return_method,
            expected_return=round(port_ret, 4),
            expected_vol=round(port_vol, 4),
            sharpe=round(sharpe, 4),
        )

        return OptimizationResult(
            weights=weights_dict,
            expected_return=round(port_ret, 6),
            expected_vol=round(port_vol, 6),
            sharpe_ratio=round(sharpe, 6),
            covariance_method=covariance_method,
            return_method=return_method,
            efficient_frontier=frontier,
            metadata={
                "n_assets": n,
                "n_observations": len(data),
                "covariance_condition_number": round(cond_num, 2),
                "risk_aversion": self._lam,
            },
        )

    # ------------------------------------------------------------------
    # Covariance estimators
    # ------------------------------------------------------------------

    def _estimate_covariance(
        self,
        data: np.ndarray,
        method: str,
    ) -> np.ndarray:
        """Estimate the daily covariance matrix using the specified method.

        Args:
            data: Returns array of shape (T, N).
            method: One of "sample", "ledoit_wolf", "oas", "rmt".

        Returns:
            Daily covariance matrix (N×N).
        """
        if method == "sample":
            return np.cov(data.T)

        if method == "ledoit_wolf":
            from sklearn.covariance import LedoitWolf

            lw = LedoitWolf().fit(data)
            return lw.covariance_

        if method == "oas":
            from sklearn.covariance import OAS

            oas = OAS().fit(data)
            return oas.covariance_

        if method == "rmt":
            return self._rmt_clean_covariance(np.cov(data.T), T=data.shape[0], N=data.shape[1])

        raise ValueError(f"Unknown covariance method: {method!r}")

    def _rmt_clean_covariance(
        self,
        cov: np.ndarray,
        T: int,
        N: int,
    ) -> np.ndarray:
        """Clean sample covariance via Random Matrix Theory (Marchenko-Pastur).

        Eigenvalues below λ_max of the Marchenko-Pastur distribution are
        replaced by the average noise eigenvalue.

        Args:
            cov: Sample covariance matrix (N×N).
            T: Number of observations.
            N: Number of assets.

        Returns:
            RMT-cleaned covariance matrix.
        """
        q = T / N  # Ratio T/N (must be >= 1 for Marchenko-Pastur)
        sigma2 = float(np.trace(cov)) / N  # Average variance

        # Marchenko-Pastur upper bound
        lambda_max = sigma2 * (1.0 + np.sqrt(1.0 / q)) ** 2

        # Eigendecompose
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = eigvals.real
        eigvecs = eigvecs.real

        # Separate signal vs noise eigenvalues
        signal_mask = eigvals > lambda_max
        noise_eigvals = eigvals[~signal_mask]
        avg_noise = float(noise_eigvals.mean()) if len(noise_eigvals) > 0 else sigma2 / N

        # Replace noise eigenvalues with average noise level
        cleaned_vals = np.where(signal_mask, eigvals, avg_noise)
        cleaned_vals = np.maximum(cleaned_vals, 1e-8)  # Ensure PSD

        # Reconstruct
        cov_clean = eigvecs @ np.diag(cleaned_vals) @ eigvecs.T

        # Rescale to match original diagonal (preserve individual variances)
        original_std = np.sqrt(np.diag(cov))
        cleaned_std = np.sqrt(np.diag(cov_clean).clip(1e-12))
        scale = original_std / cleaned_std
        cov_clean = cov_clean * np.outer(scale, scale)

        return cov_clean

    # ------------------------------------------------------------------
    # Expected return estimators
    # ------------------------------------------------------------------

    def _estimate_returns(
        self,
        data: np.ndarray,
        returns_df: pd.DataFrame,
        cov_annual: np.ndarray,
        assets: list[str],
        method: str,
        signal_returns: pd.Series | None,
        market_returns: pd.Series | None,
        risk_free_rate: float,
    ) -> np.ndarray:
        """Estimate annualised expected returns.

        Args:
            data: Returns array (T, N).
            returns_df: Original returns DataFrame for CAPM betas.
            cov_annual: Annualised covariance matrix.
            assets: Asset ticker list (same order as columns).
            method: "sample", "capm", or "signal".
            signal_returns: Signal-implied returns for "signal" method.
            market_returns: Market returns for "capm" method.
            risk_free_rate: Annualised risk-free rate.

        Returns:
            Annualised expected return vector (N,).
        """
        N = data.shape[1]

        if method == "sample":
            return data.mean(axis=0) * TRADING_DAYS_PER_YEAR

        if method == "capm":
            if market_returns is None:
                self._logger.warning("CAPM method needs market_returns; falling back to sample")
                return data.mean(axis=0) * TRADING_DAYS_PER_YEAR

            aligned = returns_df.join(market_returns.rename("_mkt"), how="inner").dropna()
            mkt = aligned["_mkt"].values
            mkt_var = float(np.var(mkt, ddof=1))
            if mkt_var < 1e-10:
                return data.mean(axis=0) * TRADING_DAYS_PER_YEAR

            mkt_excess_return = float(mkt.mean()) * TRADING_DAYS_PER_YEAR - risk_free_rate
            mu = np.zeros(N)
            for i, asset in enumerate(assets):
                if asset in aligned.columns:
                    cov_im = float(np.cov(aligned[asset].values, mkt)[0, 1])
                    beta = cov_im / mkt_var
                    mu[i] = risk_free_rate + beta * mkt_excess_return
                else:
                    mu[i] = data[:, i].mean() * TRADING_DAYS_PER_YEAR
            return mu

        if method == "signal":
            if signal_returns is None:
                self._logger.warning("Signal method needs signal_returns; falling back to sample")
                return data.mean(axis=0) * TRADING_DAYS_PER_YEAR

            mu = np.zeros(N)
            for i, asset in enumerate(assets):
                mu[i] = float(signal_returns.get(asset, data[:, i].mean() * TRADING_DAYS_PER_YEAR))
            return mu

        raise ValueError(f"Unknown return method: {method!r}")

    # ------------------------------------------------------------------
    # Optimizer (cvxpy)
    # ------------------------------------------------------------------

    def _solve_mvo(self, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Solve the MVO problem via cvxpy.

        Objective: maximize w'μ − (λ/2) w'Σw
        Subject to: Σw = 1, min_w ≤ w_i ≤ max_w

        Falls back to equal weights if solver fails.

        Args:
            mu: Annualised expected return vector (N,).
            cov: Annualised covariance matrix (N×N).

        Returns:
            Optimal weight vector (N,).
        """
        import cvxpy as cp

        n = len(mu)
        w = cp.Variable(n)

        objective = cp.Maximize(mu @ w - (self._lam / 2) * cp.quad_form(w, cov))  # type: ignore[attr-defined]
        constraints = [
            cp.sum(w) == 1.0,  # type: ignore[attr-defined]
            w >= self._min_w,
            w <= self._max_w,
        ]
        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=cp.CLARABEL, warm_start=True)
            if w.value is not None and problem.status in ("optimal", "optimal_inaccurate"):
                raw = np.array(w.value, dtype=np.float64)
                # Clip and renormalise for numerical cleanliness
                raw = np.clip(raw, self._min_w, self._max_w)
                return raw / raw.sum()
        except Exception as exc:
            self._logger.warning("cvxpy solve failed, using equal weights", error=str(exc))

        # Equal weight fallback
        return np.full(n, 1.0 / n)

    # ------------------------------------------------------------------
    # Efficient frontier
    # ------------------------------------------------------------------

    def _compute_efficient_frontier(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        assets: list[str],
        risk_free_rate: float = 0.0,
        n_points: int = _N_FRONTIER_POINTS,
    ) -> list[EfficientFrontierPoint]:
        """Compute the mean-variance efficient frontier.

        Args:
            mu: Annualised expected returns.
            cov: Annualised covariance matrix.
            assets: Asset tickers (for labelling weights).
            risk_free_rate: Annualised risk-free rate.
            n_points: Number of frontier points.

        Returns:
            List of :class:`EfficientFrontierPoint` objects.
        """
        import cvxpy as cp

        n = len(mu)
        mu_min = float(mu.min())
        mu_max = float(mu.max())
        target_returns = np.linspace(mu_min, mu_max, n_points)

        frontier: list[EfficientFrontierPoint] = []
        for target in target_returns:
            w = cp.Variable(n)
            obj = cp.Minimize(cp.quad_form(w, cov))  # type: ignore[attr-defined]
            constraints = [
                cp.sum(w) == 1.0,  # type: ignore[attr-defined]
                w >= self._min_w,
                w <= self._max_w,
                mu @ w >= target,
            ]
            try:
                cp.Problem(obj, constraints).solve(solver=cp.CLARABEL)
                if w.value is None:
                    continue
                wv = np.clip(np.array(w.value), 0.0, 1.0)
                wv /= wv.sum()
                pret = float(wv @ mu)
                pvol = float(np.sqrt(wv @ cov @ wv))
                sharpe = (pret - risk_free_rate) / pvol if pvol > 0 else 0.0
                frontier.append(
                    EfficientFrontierPoint(
                        expected_return=round(pret, 6),
                        expected_vol=round(pvol, 6),
                        sharpe_ratio=round(sharpe, 6),
                        weights={assets[i]: round(float(wv[i]), 4) for i in range(n)},
                    )
                )
            except Exception:
                continue

        return frontier
