"""Hierarchical Risk Parity (HRP) portfolio optimizer.

Implements Lopez de Prado (2016). HRP avoids matrix inversion entirely,
making it numerically stable and well-conditioned for large universes.

Steps:
1. Compute correlation-based distance matrix.
2. Hierarchical clustering (Ward linkage).
3. Quasi-diagonalise covariance via dendrogram leaf ordering.
4. Recursive bisection to allocate weights.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.cluster.hierarchy as sch

from quantflow.config.constants import HRP_LINKAGE_METHOD, TRADING_DAYS_PER_YEAR
from quantflow.config.logging import get_logger
from quantflow.portfolio.optimizer import OptimizationResult

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)

_MIN_OBSERVATIONS = 30


class HRPOptimizer:
    """Hierarchical Risk Parity optimizer (Lopez de Prado, 2016).

    No matrix inversion — robust and stable for any number of assets.

    Args:
        linkage_method: Hierarchical clustering linkage method.
    """

    def __init__(self, linkage_method: str = HRP_LINKAGE_METHOD) -> None:
        self._linkage = linkage_method
        self._logger = get_logger(__name__)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def optimize(self, returns: pd.DataFrame) -> OptimizationResult:
        """Compute HRP weights for the given return history.

        Args:
            returns: Daily log-returns DataFrame (columns = tickers,
                DatetimeIndex).  At least ``_MIN_OBSERVATIONS`` rows required.

        Returns:
            :class:`OptimizationResult` with HRP-optimal weights.

        Raises:
            ValueError: If fewer than ``_MIN_OBSERVATIONS`` valid rows.
        """
        data = returns.dropna(how="any").values.astype(np.float64)
        if len(data) < _MIN_OBSERVATIONS:
            raise ValueError(f"Need at least {_MIN_OBSERVATIONS} observations, got {len(data)}")

        assets = list(returns.columns)
        N = len(assets)
        cov = np.cov(data.T)
        corr = np.corrcoef(data.T)

        # 1. Distance matrix
        dist = self._correlation_distance(corr)

        # 2. Hierarchical clustering
        link = sch.linkage(dist[np.triu_indices(N, k=1)], method=self._linkage)

        # 3. Quasi-diagonalisation (leaf ordering)
        sorted_idx = self._quasi_diagonalize(link, N)

        # 4. Recursive bisection
        weights_arr = self._recursive_bisection(cov, sorted_idx)

        # Sanity: clip negatives from numerical noise and renormalise
        weights_arr = np.maximum(weights_arr, 0.0)
        total = weights_arr.sum()
        if total < 1e-10:
            weights_arr = np.full(N, 1.0 / N)
        else:
            weights_arr /= total

        weights_dict = {assets[i]: round(float(weights_arr[i]), 6) for i in range(N)}

        # Portfolio stats
        cov_annual = cov * TRADING_DAYS_PER_YEAR
        w = weights_arr
        port_vol = float(np.sqrt(w @ cov_annual @ w))
        port_ret = float((data.mean(axis=0) * TRADING_DAYS_PER_YEAR) @ w)
        sharpe = port_ret / port_vol if port_vol > 0 else 0.0

        self._logger.info(
            "HRP optimized",
            n_assets=N,
            expected_vol=round(port_vol, 4),
        )

        return OptimizationResult(
            weights=weights_dict,
            expected_return=round(port_ret, 6),
            expected_vol=round(port_vol, 6),
            sharpe_ratio=round(sharpe, 6),
            covariance_method="sample",
            return_method="sample",
            metadata={
                "n_assets": N,
                "n_observations": len(data),
                "linkage_method": self._linkage,
                "algorithm": "HRP",
            },
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _correlation_distance(corr: np.ndarray) -> np.ndarray:
        """Convert correlation matrix to distance matrix.

        d_ij = sqrt((1 − ρ_ij) / 2) ∈ [0, 1].

        Args:
            corr: Correlation matrix (N×N).

        Returns:
            Distance matrix (N×N), symmetric with zeros on diagonal.
        """
        dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))
        np.fill_diagonal(dist, 0.0)
        return dist

    @staticmethod
    def _quasi_diagonalize(link: np.ndarray, n: int) -> list[int]:
        """Extract the leaf order from the dendrogram for quasi-diagonalisation.

        Uses scipy's ``leaves_list`` to get the optimal leaf order.

        Args:
            link: Linkage matrix from hierarchical clustering.
            n: Number of original observations (assets).

        Returns:
            Ordered list of original asset indices.
        """
        return list(sch.leaves_list(link))

    def _recursive_bisection(
        self,
        cov: np.ndarray,
        sorted_items: list[int],
    ) -> np.ndarray:
        """Allocate weights by recursive bisection.

        Splits the sorted asset list into two halves at each step.
        Weights are allocated inversely proportional to the minimum-variance
        portfolio variance of each cluster.

        Args:
            cov: Daily covariance matrix (N×N).
            sorted_items: Quasi-diagonalised asset index ordering.

        Returns:
            Weight array (same length as sorted_items), sums to 1.
        """
        # Map from original index to position in sorted_items
        pos_of = {orig: pos for pos, orig in enumerate(sorted_items)}
        n = len(sorted_items)
        weights = np.ones(n)

        def bisect(items: list[int]) -> None:
            if len(items) <= 1:
                return
            mid = len(items) // 2
            left = items[:mid]
            right = items[mid:]

            var_l = self._cluster_variance(cov, left)
            var_r = self._cluster_variance(cov, right)
            total = var_l + var_r + 1e-12
            alpha = 1.0 - var_l / total  # Right cluster gets alpha

            for idx in left:
                weights[pos_of[idx]] *= 1.0 - alpha
            for idx in right:
                weights[pos_of[idx]] *= alpha

            bisect(left)
            bisect(right)

        bisect(sorted_items)
        return weights

    @staticmethod
    def _cluster_variance(cov: np.ndarray, cluster: list[int]) -> float:
        """Compute the minimum-variance portfolio variance for a cluster.

        Uses inverse-variance weights (diagonal approximation) for robustness.

        Args:
            cov: Full covariance matrix.
            cluster: List of asset indices forming this cluster.

        Returns:
            Cluster variance float.
        """
        sub_cov = cov[np.ix_(cluster, cluster)]
        inv_diag = 1.0 / np.diag(sub_cov).clip(1e-8)
        w = inv_diag / inv_diag.sum()
        return float(w @ sub_cov @ w)
