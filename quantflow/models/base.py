"""Base class and output schema for all QuantFlow quantitative models.

Every model in quantflow.models inherits from BaseQuantModel and returns
a ModelOutput Pydantic object that the signal fusion engine consumes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from quantflow.config.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


class ModelOutput(BaseModel):
    """Standardised output produced by every quantitative model.

    Attributes:
        model_name: Identifier of the producing model.
        symbol: Ticker symbol this output is for.
        timestamp: UTC timestamp of when the output was produced.
        signal: Normalised signal in ``[-1.0, +1.0]``.
            Positive ⟹ bullish, negative ⟹ bearish.
        confidence: Estimate of signal reliability in ``[0.0, 1.0]``.
        forecast_return: Point estimate of expected return (e.g. 21-day).
        forecast_std: Standard deviation / uncertainty of the forecast.
        regime: Optional regime label (e.g. "HIGH_VOL", "BEAR").
        metadata: Model-specific diagnostics (AIC, BIC, p-values, etc.).
    """

    model_name: str
    symbol: str
    timestamp: datetime
    signal: float = Field(..., ge=-1.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    forecast_return: float
    forecast_std: float = Field(..., ge=0.0)
    regime: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("forecast_std")
    @classmethod
    def std_must_be_finite(cls, v: float) -> float:
        """Ensure forecast standard deviation is finite and non-negative."""
        if not np.isfinite(v) or v < 0:
            raise ValueError(f"forecast_std must be finite and ≥ 0, got {v}")
        return v

    @field_validator("signal")
    @classmethod
    def signal_must_be_finite(cls, v: float) -> float:
        """Ensure signal is finite."""
        if not np.isfinite(v):
            raise ValueError(f"signal must be finite, got {v}")
        return v


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class BaseQuantModel(ABC):
    """Abstract base class for all QuantFlow quantitative models.

    Subclasses must implement :meth:`fit` and :meth:`predict`.
    The base class provides shared utilities for:
    - Computing information coefficients (IC).
    - Normalising signals to ``[-1, +1]``.
    - Checking for NaN / Inf in inputs.
    - Logging structured diagnostics.

    Args:
        model_name: Human-readable name for this model instance.
        symbol: Ticker symbol (used in ``ModelOutput``).
    """

    def __init__(self, model_name: str, symbol: str) -> None:
        """Initialise the base model.

        Args:
            model_name: Identifier for this model.
            symbol: Ticker symbol.
        """
        self.model_name = model_name
        self.symbol = symbol
        self._logger = get_logger(self.__class__.__name__)
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> "BaseQuantModel":
        """Fit the model on historical data.

        Args:
            data: DataFrame containing at minimum a ``close`` column
                  with a UTC DatetimeIndex.  Additional columns (e.g.
                  ``volume``, macro features) may be used by specific models.

        Returns:
            Self (for method chaining).
        """
        ...

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> ModelOutput:
        """Generate a :class:`ModelOutput` from (possibly new) data.

        Args:
            data: The same format as used in :meth:`fit`.  Must not contain
                  data points that were not yet available at prediction time.

        Returns:
            :class:`ModelOutput` with signal, confidence, and diagnostics.
        """
        ...

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    @staticmethod
    def normalise_signal(raw_signal: float, clip: float = 3.0) -> float:
        """Map an arbitrary raw signal to ``[-1, +1]`` via tanh.

        Args:
            raw_signal: Raw signal value (e.g. z-score, IC-weighted sum).
            clip: Pre-tanh clip to prevent numerical issues.

        Returns:
            Signal in ``[-1.0, +1.0]``.
        """
        if not np.isfinite(raw_signal):
            return 0.0
        clipped = np.clip(raw_signal, -clip, clip)
        return float(np.tanh(clipped))

    @staticmethod
    def compute_ic(
        signals: pd.Series,
        future_returns: pd.Series,
        method: str = "spearman",
    ) -> float:
        """Compute Information Coefficient between signals and realised returns.

        Args:
            signals: Predicted signal values.
            future_returns: Forward returns aligned to signal timestamps.
            method: Correlation method — "spearman" (rank) or "pearson".

        Returns:
            IC as a float in ``[-1, +1]``.  Returns 0.0 if insufficient data.
        """
        aligned = pd.concat([signals, future_returns], axis=1).dropna()
        if len(aligned) < 10:
            return 0.0
        if method == "spearman":
            ic = aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman")
        else:
            ic = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
        return float(ic) if np.isfinite(ic) else 0.0

    @staticmethod
    def check_numeric(
        arr: np.ndarray | pd.Series,
        name: str = "input",
    ) -> None:
        """Raise ``ValueError`` if *arr* contains NaN or Inf.

        Args:
            arr: Numeric array to check.
            name: Name used in the error message.

        Raises:
            ValueError: If any NaN or Inf values are found.
        """
        arr_np = np.asarray(arr, dtype=np.float64)
        n_nan = int(np.isnan(arr_np).sum())
        n_inf = int(np.isinf(arr_np).sum())
        if n_nan > 0 or n_inf > 0:
            raise ValueError(
                f"{name}: {n_nan} NaN and {n_inf} Inf values detected."
            )

    def _require_fitted(self) -> None:
        """Raise ``RuntimeError`` if the model has not been fitted.

        Raises:
            RuntimeError: If :meth:`fit` has not been called yet.
        """
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.model_name} must be fitted before calling predict(). "
                "Call fit(data) first."
            )

    def _log_fit_complete(self, **diagnostics: Any) -> None:
        """Log structured diagnostics after fitting.

        Args:
            **diagnostics: Key-value pairs to include in the log event.
        """
        self._logger.info(
            "Model fitted",
            model=self.model_name,
            symbol=self.symbol,
            **diagnostics,
        )
