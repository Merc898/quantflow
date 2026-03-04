"""Dynamic model weight calibration based on recent IC performance.

Weights are proportional to the recent (trailing 63-day) Information
Coefficient of each model, subject to floor and cap constraints.

Supports regime-conditional weights: a separate weight vector is maintained
per market regime, since different models dominate in different environments.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy.stats as stats

from quantflow.config.constants import LOOKBACK_LONG
from quantflow.config.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_FLOOR = 0.01   # Minimum weight — keep all models alive
_DEFAULT_CAP = 0.30     # Maximum weight — no single model dominates
_IC_HORIZON = 21        # Forward return horizon for IC computation (days)


class DynamicWeightCalibrator:
    """IC-based dynamic model weight calibrator.

    Computes Information Coefficients for each model over a rolling window
    and allocates weights proportional to positive ICs, with regime
    conditioning so that model blends adapt to market environments.

    Args:
        ic_lookback: Trailing window (in days) for IC computation.
        ic_horizon: Forward return horizon for IC labels (in days).
        floor: Minimum weight per model.
        cap: Maximum weight per model.
    """

    def __init__(
        self,
        ic_lookback: int = LOOKBACK_LONG,
        ic_horizon: int = _IC_HORIZON,
        floor: float = _DEFAULT_FLOOR,
        cap: float = _DEFAULT_CAP,
    ) -> None:
        self._lookback = ic_lookback
        self._horizon = ic_horizon
        self._floor = floor
        self._cap = cap
        self._logger = get_logger(__name__)

        # Regime-conditional weight cache: regime_name → {model_name → weight}
        self._regime_weights: dict[str, dict[str, float]] = {}
        self._last_calibrated: datetime | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute_weights(
        self,
        model_signals: dict[str, pd.Series],
        realized_returns: pd.Series,
        regime: str | None = None,
    ) -> dict[str, float]:
        """Compute dynamic weights from trailing IC performance.

        Args:
            model_signals: Dict mapping model_name → signal Series
                (DatetimeIndex, values in [-1, +1]).
            realized_returns: Realised daily log-return Series (same index).
            regime: Current market regime label (for regime-conditional
                weights).  If provided and previously calibrated, returns
                cached regime weights.

        Returns:
            Dict mapping model_name → normalised weight, summing to 1.
        """
        if not model_signals:
            return {}

        # Use cached regime weights if available
        if regime is not None and regime in self._regime_weights:
            # Filter to models present in current signal set
            cached = {
                m: w for m, w in self._regime_weights[regime].items()
                if m in model_signals
            }
            if cached:
                return self._apply_floor_cap(cached)

        # Compute forward returns (shifted back to align with signals)
        fwd_returns = realized_returns.shift(-self._horizon)

        ic_scores: dict[str, float] = {}
        for name, signal in model_signals.items():
            ic = self._compute_ic(signal, fwd_returns)
            ic_scores[name] = ic
            self._logger.debug(
                "Model IC computed",
                model=name,
                ic=round(ic, 4),
                regime=regime or "none",
            )

        weights = self._ic_to_weights(ic_scores)

        # Cache for this regime
        if regime is not None:
            self._regime_weights[regime] = dict(weights)

        self._last_calibrated = datetime.now(tz=timezone.utc)
        self._logger.info(
            "Weights calibrated",
            n_models=len(weights),
            regime=regime or "none",
            weights=weights,
        )
        return weights

    def get_regime_weights(self, regime: str) -> dict[str, float] | None:
        """Retrieve cached regime-conditional weights.

        Args:
            regime: Regime label.

        Returns:
            Cached weight dict or ``None`` if not yet calibrated for this regime.
        """
        return self._regime_weights.get(regime)

    def icir(
        self,
        model_signals: dict[str, pd.Series],
        realized_returns: pd.Series,
        rolling_window: int = 63,
    ) -> dict[str, float]:
        """Compute IC Information Ratio (ICIR) = mean(IC) / std(IC) per model.

        Args:
            model_signals: Dict mapping model_name → signal Series.
            realized_returns: Realised daily return Series.
            rolling_window: Lookback for ICIR estimation.

        Returns:
            Dict mapping model_name → ICIR value.
        """
        fwd = realized_returns.shift(-self._horizon)
        icir_dict: dict[str, float] = {}

        for name, signal in model_signals.items():
            aligned = pd.concat(
                [signal.rename("sig"), fwd.rename("ret")], axis=1
            ).dropna().iloc[-rolling_window:]

            if len(aligned) < 10:
                icir_dict[name] = 0.0
                continue

            # Rolling 21-day ICs
            ic_series = []
            step = max(1, self._horizon // 2)
            for i in range(0, len(aligned) - self._horizon, step):
                chunk = aligned.iloc[i : i + self._horizon]
                if len(chunk) < 5:
                    continue
                ic, _ = stats.spearmanr(chunk["sig"], chunk["ret"])
                if np.isfinite(ic):
                    ic_series.append(float(ic))

            if len(ic_series) >= 3:
                icir_dict[name] = float(np.mean(ic_series) / (np.std(ic_series) + 1e-8))
            else:
                icir_dict[name] = 0.0

        return icir_dict

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_ic(self, signal: pd.Series, fwd_returns: pd.Series) -> float:
        """Compute Spearman IC over the trailing lookback window.

        Args:
            signal: Model signal Series.
            fwd_returns: Forward return Series (already shifted).

        Returns:
            Spearman IC float in [-1, +1].  Returns 0.0 if insufficient data.
        """
        aligned = pd.concat(
            [signal.rename("sig"), fwd_returns.rename("ret")], axis=1
        ).dropna().iloc[-self._lookback:]

        if len(aligned) < 10:
            return 0.0

        ic, _ = stats.spearmanr(aligned["sig"], aligned["ret"])
        return float(ic) if np.isfinite(ic) else 0.0

    def _ic_to_weights(self, ic_scores: dict[str, float]) -> dict[str, float]:
        """Convert IC scores to normalised weights.

        Formula: w_i = max(IC_i, 0) / Σ max(IC_j, 0)
        Special case: if all ICs ≤ 0, use equal weights.

        Args:
            ic_scores: Dict mapping model_name → IC value.

        Returns:
            Normalised weight dict.
        """
        positive_ic = {m: max(ic, 0.0) for m, ic in ic_scores.items()}
        total = sum(positive_ic.values())

        if total < 1e-8:
            # All ICs non-positive → equal weights
            n = len(ic_scores)
            raw = {m: 1.0 / n for m in ic_scores}
        else:
            raw = {m: v / total for m, v in positive_ic.items()}

        return self._apply_floor_cap(raw)

    def _apply_floor_cap(self, raw: dict[str, float]) -> dict[str, float]:
        """Apply floor and cap constraints and renormalise using iterative projection.

        A single clip-then-normalise pass can violate constraints after
        renormalisation (e.g. if one weight is clipped to the cap, dividing
        by a total < 1 can push it back above the cap).  This method
        iterates until all weights genuinely satisfy [floor, cap] and sum = 1.

        Args:
            raw: Raw weight dict (may not sum to 1 after floor/cap).

        Returns:
            Constrained and renormalised weight dict.
        """
        n = len(raw)
        if n == 0:
            return {}

        models = list(raw.keys())
        total = sum(raw.values())
        if total < 1e-8:
            w = np.full(n, 1.0 / n)
        else:
            w = np.array([raw[m] / total for m in models], dtype=np.float64)

        # Iterative projection: clip, renormalise, repeat until stable
        for _ in range(200):
            w = np.clip(w, self._floor, self._cap)
            t = w.sum()
            if t < 1e-8:
                w = np.full(n, 1.0 / n)
                break
            w = w / t
            # Check convergence: all weights within [floor, cap]
            if np.all(w >= self._floor - 1e-9) and np.all(w <= self._cap + 1e-9):
                break

        return {m: round(float(w[i]), 6) for i, m in enumerate(models)}
