"""Markov-Switching (Hidden Markov Model) regime detection.

Uses statsmodels MarkovAutoregression for regime classification.
Outputs smoothed state probabilities and regime-conditional forecasts.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from quantflow.config.logging import get_logger
from quantflow.models.base import BaseQuantModel, ModelOutput

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)


_REGIME_LABELS_2 = {0: "BULL", 1: "BEAR"}
_REGIME_LABELS_3 = {0: "BULL", 1: "NEUTRAL", 2: "BEAR"}


class MarkovSwitchingModel(BaseQuantModel):
    """Regime detection via Markov-Switching model.

    Fits a Markov-Switching autoregression on log returns.
    Number of regimes chosen by BIC over 2–3 regime models.

    Regimes are labelled by mean return:
    - Highest mean return ⟹ BULL
    - Lowest mean return ⟹ BEAR
    - Middle (if 3 regimes) ⟹ NEUTRAL

    Args:
        symbol: Ticker symbol.
        n_regimes: Number of regimes (2 or 3).  If ``None``, auto-select
            by BIC.
        order: Autoregressive order within each regime.
    """

    def __init__(
        self,
        symbol: str,
        n_regimes: int | None = None,
        order: int = 0,
    ) -> None:
        """Initialise the Markov-Switching model.

        Args:
            symbol: Ticker symbol.
            n_regimes: Fixed number of regimes (2 or 3) or None for auto.
            order: AR order within each regime.
        """
        super().__init__("MarkovSwitching", symbol)
        self.n_regimes = n_regimes
        self.order = order

        self._result: Any = None
        self._regime_stats: dict[int, dict[str, float]] = {}
        self._smoothed_probs: np.ndarray | None = None
        self._regime_labels: dict[int, str] = {}
        self._chosen_k: int = 2

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame) -> MarkovSwitchingModel:
        """Fit Markov-Switching AR model on log returns.

        Args:
            data: DataFrame with ``close`` column.

        Returns:
            Self (fitted model).
        """

        close = data["close"].astype(np.float64)
        log_ret = np.log(close / close.shift(1)).dropna().values * 100  # scale to %

        if self.n_regimes is not None:
            best_k = self.n_regimes
            best_result = self._fit_k(log_ret, best_k)
        else:
            best_result = None
            best_bic = np.inf
            best_k = 2
            for k in (2, 3):
                try:
                    res = self._fit_k(log_ret, k)
                    if res is not None and res.bic < best_bic:
                        best_bic = res.bic
                        best_result = res
                        best_k = k
                except Exception:
                    pass

        if best_result is None:
            self._logger.warning(
                "Markov-Switching fit failed, using neutral defaults",
                symbol=self.symbol,
            )
            self._is_fitted = True
            self._chosen_k = 2
            return self

        self._result = best_result
        self._chosen_k = best_k
        self._smoothed_probs = np.array(best_result.smoothed_marginal_probabilities)

        # Characterise each regime by its conditional mean and std
        for k in range(best_k):
            probs = self._smoothed_probs[:, k]
            weighted_ret = float(np.average(log_ret[-len(probs) :], weights=probs))
            weighted_std = float(
                np.sqrt(np.average((log_ret[-len(probs) :] - weighted_ret) ** 2, weights=probs))
            )
            self._regime_stats[k] = {"mean": weighted_ret, "std": weighted_std}

        # Label regimes by mean return (descending)
        sorted_regimes = sorted(
            self._regime_stats.items(), key=lambda x: x[1]["mean"], reverse=True
        )
        labels = list(_REGIME_LABELS_3.values()) if best_k == 3 else list(_REGIME_LABELS_2.values())
        for i, (regime_idx, _) in enumerate(sorted_regimes):
            self._regime_labels[regime_idx] = labels[i] if i < len(labels) else "UNKNOWN"

        self._is_fitted = True
        self._log_fit_complete(
            n_regimes=best_k,
            bic=round(best_result.bic, 2),
            regime_labels=self._regime_labels,
        )
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, data: pd.DataFrame | None = None) -> ModelOutput:
        """Generate a regime-probability-weighted return forecast.

        Returns:
            :class:`ModelOutput` with:
            - ``signal``: Probability-weighted expected return (normalised).
            - ``regime``: Current most-likely regime label.
            - ``metadata.regime_probs``: Dict mapping regime labels to probs.
        """
        self._require_fitted()

        if self._smoothed_probs is None or self._result is None:
            return self._neutral_output()

        # Last smoothed probabilities
        last_probs = self._smoothed_probs[-1]

        # Weighted forecast return (sum over regimes)
        forecast_return = 0.0
        forecast_var = 0.0
        for k in range(self._chosen_k):
            stats = self._regime_stats.get(k, {"mean": 0.0, "std": 1.0})
            p = float(last_probs[k])
            forecast_return += p * stats["mean"]
            forecast_var += p * (stats["std"] ** 2 + stats["mean"] ** 2)
        forecast_var -= forecast_return**2
        forecast_std = max(np.sqrt(forecast_var), 1e-6)

        # Current regime
        current_regime_idx = int(np.argmax(last_probs))
        current_regime = self._regime_labels.get(current_regime_idx, "UNKNOWN")

        # Signal from normalised weighted return
        raw_signal = forecast_return / (forecast_std + 1e-8)
        signal = self.normalise_signal(raw_signal)

        # Confidence: high when regime probability is concentrated
        confidence = float(last_probs.max())

        regime_probs = {
            self._regime_labels.get(k, f"regime_{k}"): round(float(last_probs[k]), 3)
            for k in range(self._chosen_k)
        }

        metadata: dict[str, Any] = {
            "n_regimes": self._chosen_k,
            "regime_probs": regime_probs,
            "bic": round(self._result.bic, 2),
        }

        return ModelOutput(
            model_name=self.model_name,
            symbol=self.symbol,
            timestamp=datetime.now(tz=UTC),
            signal=signal,
            confidence=confidence,
            forecast_return=round(forecast_return / 100.0, 6),  # convert % back
            forecast_std=round(forecast_std / 100.0, 6),
            regime=current_regime,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fit_k(self, log_ret: np.ndarray, k: int) -> Any:
        """Fit a k-regime Markov-Switching AR model."""
        from statsmodels.tsa.regime_switching.markov_autoregression import (
            MarkovAutoregression,
        )

        model = MarkovAutoregression(
            log_ret,
            k_regimes=k,
            order=self.order,
            switching_ar=False,
            switching_variance=True,
        )
        result = model.fit(disp=False)
        return result

    def _neutral_output(self) -> ModelOutput:
        """Return neutral output when the model failed to fit."""
        return ModelOutput(
            model_name=self.model_name,
            symbol=self.symbol,
            timestamp=datetime.now(tz=UTC),
            signal=0.0,
            confidence=0.0,
            forecast_return=0.0,
            forecast_std=0.02,
            regime="UNKNOWN",
            metadata={"error": "model_not_fitted"},
        )
