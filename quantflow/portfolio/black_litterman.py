"""Black-Litterman portfolio optimization.

Combines equilibrium market returns (reverse-optimized from market-cap weights)
with investor views derived from the quantitative model ensemble to produce
a posterior expected return vector and MVO-optimal weights.

Reference: Black & Litterman (1992), He & Litterman (1999).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantflow.config.constants import BL_DELTA, BL_TAU, TRADING_DAYS_PER_YEAR
from quantflow.config.logging import get_logger
from quantflow.portfolio.optimizer import MVOOptimizer, OptimizationResult

logger = get_logger(__name__)

_MIN_OBSERVATIONS = 60


class BlackLittermanOptimizer:
    """Black-Litterman optimizer that incorporates quantitative views.

    Steps:
    1. Compute equilibrium returns Π = δ · Σ · w_market.
    2. Build views matrix P and view returns Q from model signals.
    3. Compute posterior: μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹[(τΣ)⁻¹Π + P'Ω⁻¹Q].
    4. Optimize with μ_BL as expected returns via cvxpy.

    Args:
        tau: Uncertainty scaling factor (typically 1/T).
        delta: Market risk aversion coefficient.
    """

    def __init__(
        self,
        tau: float = BL_TAU,
        delta: float = BL_DELTA,
    ) -> None:
        self._tau = tau
        self._delta = delta
        self._logger = get_logger(__name__)
        self._mvo = MVOOptimizer()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def optimize(
        self,
        returns: pd.DataFrame,
        market_weights: pd.Series,
        signal_views: dict[str, float],
        view_confidences: dict[str, float] | None = None,
    ) -> OptimizationResult:
        """Run Black-Litterman optimization.

        Args:
            returns: Daily log-returns DataFrame (columns = tickers).
            market_weights: Market-cap weights (pd.Series, same tickers as returns).
            signal_views: Dict mapping asset → view on annualised excess return.
                Example: ``{"AAPL": 0.10, "MSFT": -0.05}``.
                Positive = bullish view (expect 10% excess return).
            view_confidences: Dict mapping asset → confidence in view ``[0, 1]``.
                Higher confidence → smaller view uncertainty Ω.
                Defaults to 0.5 for all views.

        Returns:
            :class:`OptimizationResult` with BL-optimal weights and posterior returns.

        Raises:
            ValueError: If fewer than ``_MIN_OBSERVATIONS`` rows.
        """
        if len(returns) < _MIN_OBSERVATIONS:
            raise ValueError(f"Need at least {_MIN_OBSERVATIONS} observations, got {len(returns)}")

        # Align assets
        common = [a for a in returns.columns if a in market_weights.index]
        if not common:
            raise ValueError("No common assets between returns and market_weights.")

        assets = common
        N = len(assets)
        ret_data = returns[assets].dropna().values.astype(np.float64)

        # Daily and annual covariance
        from sklearn.covariance import LedoitWolf

        lw = LedoitWolf().fit(ret_data)
        cov_daily = lw.covariance_
        cov_annual = cov_daily * TRADING_DAYS_PER_YEAR

        # Market weights (aligned and normalised)
        w_mkt = np.array([float(market_weights.get(a, 1.0 / N)) for a in assets], dtype=np.float64)
        w_mkt = w_mkt / w_mkt.sum()

        # --- Step 1: Equilibrium returns ---
        pi = self._equilibrium_returns(w_mkt, cov_annual)

        # --- Step 2: Build views ---
        if not signal_views:
            # No views: use equilibrium returns
            mu_bl = pi
        else:
            if view_confidences is None:
                view_confidences = dict.fromkeys(signal_views, 0.5)

            P, Q, Omega = self._build_views(assets, cov_annual, signal_views, view_confidences)
            # --- Step 3: Posterior ---
            mu_bl, _ = self._posterior(pi, cov_annual, P, Q, Omega)

        # --- Step 4: Optimize with BL returns ---
        mu_series = pd.Series(mu_bl, index=assets)
        result = self._mvo.optimize(
            returns[assets],
            covariance_method="ledoit_wolf",
            return_method="signal",
            signal_returns=mu_series,
        )

        # Annotate with BL metadata
        result.metadata["bl_tau"] = self._tau
        result.metadata["bl_delta"] = self._delta
        result.metadata["n_views"] = len(signal_views)
        result.metadata["equilibrium_returns"] = {
            assets[i]: round(float(pi[i]), 6) for i in range(N)
        }
        result.metadata["posterior_returns"] = {
            assets[i]: round(float(mu_bl[i]), 6) for i in range(N)
        }
        result.return_method = "black_litterman"

        self._logger.info(
            "Black-Litterman optimized",
            n_assets=N,
            n_views=len(signal_views),
            expected_return=result.expected_return,
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _equilibrium_returns(
        self,
        market_weights: np.ndarray,
        cov_annual: np.ndarray,
    ) -> np.ndarray:
        """Compute equilibrium (implied) returns via reverse optimization.

        Π = δ · Σ · w_market

        Args:
            market_weights: Market-cap weight vector.
            cov_annual: Annualised covariance matrix.

        Returns:
            Equilibrium return vector (annualised).
        """
        return self._delta * cov_annual @ market_weights

    def _build_views(
        self,
        assets: list[str],
        cov_annual: np.ndarray,
        signal_views: dict[str, float],
        view_confidences: dict[str, float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build the views pick matrix P, view returns Q, and uncertainty Omega.

        Each view is an absolute view on a single asset (P has one 1 per row).
        Uncertainty Ω_kk = (1 − conf_k) * P_k Σ P_k' * τ.

        Args:
            assets: Ordered list of asset tickers.
            cov_annual: Annualised covariance matrix.
            signal_views: Asset → expected excess return.
            view_confidences: Asset → confidence in [0, 1].

        Returns:
            Tuple of (P, Q, Omega):
                P: Pick matrix (K × N).
                Q: View returns (K,).
                Omega: Diagonal view uncertainty matrix (K × K).
        """
        asset_idx = {a: i for i, a in enumerate(assets)}
        N = len(assets)

        view_assets = [a for a in signal_views if a in asset_idx]
        K = len(view_assets)

        P = np.zeros((K, N))
        Q = np.zeros(K)
        omega_diag = np.zeros(K)

        for k, asset in enumerate(view_assets):
            idx = asset_idx[asset]
            P[k, idx] = 1.0
            Q[k] = signal_views[asset]
            conf = float(view_confidences.get(asset, 0.5))
            conf = np.clip(conf, 0.01, 0.99)
            # Omega_kk = (1 - conf) * tau * P_k Sigma P_k'
            p_sigma_pt = float(P[k] @ cov_annual @ P[k])
            omega_diag[k] = (1.0 - conf) * self._tau * p_sigma_pt + 1e-8

        Omega = np.diag(omega_diag)
        return P, Q, Omega

    def _posterior(
        self,
        pi: np.ndarray,
        cov_annual: np.ndarray,
        P: np.ndarray,
        Q: np.ndarray,
        Omega: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the Black-Litterman posterior mean and covariance.

        μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ · [(τΣ)⁻¹Π + P'Ω⁻¹Q]
        Σ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ + Σ

        Args:
            pi: Equilibrium return vector.
            cov_annual: Annual covariance matrix.
            P: Pick matrix (K × N).
            Q: View returns (K,).
            Omega: View uncertainty (K × K diagonal).

        Returns:
            Tuple (posterior_mu, posterior_cov).
        """
        tau_sigma = self._tau * cov_annual
        tau_sigma_inv = np.linalg.inv(tau_sigma + np.eye(len(pi)) * 1e-8)
        omega_inv = np.linalg.inv(Omega + np.eye(len(Q)) * 1e-8)

        M_inv = tau_sigma_inv + P.T @ omega_inv @ P
        M = np.linalg.inv(M_inv + np.eye(len(pi)) * 1e-8)

        rhs = tau_sigma_inv @ pi + P.T @ omega_inv @ Q
        mu_bl = M @ rhs

        # Posterior covariance: M + Sigma
        sigma_bl = M + cov_annual
        return mu_bl, sigma_bl
