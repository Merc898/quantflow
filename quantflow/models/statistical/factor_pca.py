"""PCA / Dynamic PCA Factor Model for idiosyncratic alpha extraction.

Decomposes a cross-section of returns into common factors and idiosyncratic
returns.  The idiosyncratic return momentum is the primary signal.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from quantflow.config.constants import LOOKBACK_1Y, MOMENTUM_LOOKBACKS
from quantflow.config.logging import get_logger
from quantflow.models.base import BaseQuantModel, ModelOutput

logger = get_logger(__name__)

_MIN_VARIANCE_EXPLAINED = 0.80  # Extract factors explaining >80% variance


class PCARiskFactorModel(BaseQuantModel):
    """PCA factor model for idiosyncratic return extraction.

    Steps:
    1. Fit rolling 252-day PCA on a universe return matrix.
    2. Extract the top K factors that explain >80% of variance.
    3. Project target returns onto factor space → factor exposures.
    4. Compute idiosyncratic return (residual after removing factor returns).
    5. Signal: idiosyncratic momentum (alpha vs factors).

    Args:
        symbol: Target ticker symbol.
        universe_col: Column in input data containing returns of the universe.
            If the DataFrame only has a ``close`` column, we use that alone.
        window: Rolling window for PCA re-estimation (default 252 days).
        min_variance: Minimum cumulative variance for factor count selection.
    """

    def __init__(
        self,
        symbol: str,
        universe_col: str | None = None,
        window: int = LOOKBACK_1Y,
        min_variance: float = _MIN_VARIANCE_EXPLAINED,
    ) -> None:
        """Initialise the PCA factor model.

        Args:
            symbol: Target ticker symbol.
            universe_col: Optional name for universe return matrix column.
            window: PCA rolling window in trading days.
            min_variance: Minimum cumulative variance explained threshold.
        """
        super().__init__("PCA_FactorModel", symbol)
        self.universe_col = universe_col
        self.window = window
        self.min_variance = min_variance

        self._pca: PCA | None = None
        self._scaler: StandardScaler | None = None
        self._n_factors: int = 1
        self._target_loadings: np.ndarray | None = None
        self._idiosyncratic_returns: pd.Series | None = None
        self._explained_variance: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame) -> "PCARiskFactorModel":
        """Fit the PCA factor model on the last ``window`` rows of data.

        Args:
            data: DataFrame with a ``close`` column (or multiple asset columns
                  representing the universe).  UTC DatetimeIndex.

        Returns:
            Self (fitted model).
        """
        # Build the return matrix
        ret_matrix = self._build_return_matrix(data)

        self._scaler = StandardScaler()
        scaled = self._scaler.fit_transform(ret_matrix)

        # Fit PCA on the full windowed return matrix
        n_max = min(ret_matrix.shape[1], ret_matrix.shape[0] // 2)
        pca = PCA(n_components=n_max)
        pca.fit(scaled)

        # Select number of factors for >min_variance explained
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        n_factors = int(np.searchsorted(cum_var, self.min_variance) + 1)
        n_factors = max(1, min(n_factors, n_max))

        # Re-fit with the optimal number of factors
        self._pca = PCA(n_components=n_factors)
        self._pca.fit(scaled)
        self._n_factors = n_factors
        self._explained_variance = self._pca.explained_variance_ratio_

        # Project target asset returns onto factor space
        # Target is always the first column (or a named column)
        target_ret = ret_matrix[:, 0]
        factor_scores = self._pca.transform(scaled)

        # OLS regression: target_ret = factor_scores @ beta + epsilon
        X = np.column_stack([factor_scores, np.ones(len(factor_scores))])
        betas, _, _, _ = np.linalg.lstsq(X, target_ret, rcond=None)
        self._target_loadings = betas

        # Compute idiosyncratic returns
        fitted = X @ betas
        idio_ret = target_ret - fitted

        # Store as a Series aligned to the data index
        idx = data.index[-len(idio_ret):]
        self._idiosyncratic_returns = pd.Series(idio_ret, index=idx)

        self._is_fitted = True
        self._log_fit_complete(
            n_factors=n_factors,
            explained_variance=round(float(cum_var[n_factors - 1]), 3),
        )
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, data: pd.DataFrame | None = None) -> ModelOutput:
        """Generate a signal from idiosyncratic momentum.

        Args:
            data: Optional new data.

        Returns:
            :class:`ModelOutput` with idiosyncratic momentum signal.
        """
        self._require_fitted()
        assert self._idiosyncratic_returns is not None
        assert self._explained_variance is not None

        idio = self._idiosyncratic_returns.dropna()
        if len(idio) < 21:
            return self._neutral_output()

        # Idiosyncratic momentum (63-day cumulative idio return)
        idio_mom_63 = float(idio.iloc[-63:].sum()) if len(idio) >= 63 else float(idio.sum())
        idio_std = float(idio.std())

        raw_signal = idio_mom_63 / (idio_std * np.sqrt(63) + 1e-8)
        signal = self.normalise_signal(raw_signal)

        # Forecast: next-day idio return (simple last value)
        forecast_return = float(idio.iloc[-1])
        forecast_std = idio_std

        # Confidence: higher when more variance is explained by factors
        total_explained = float(self._explained_variance.sum())
        confidence = min(0.75, total_explained * 0.8)

        # Scree data for frontend visualisation
        scree_data = {
            f"PC{i+1}": round(float(v), 4)
            for i, v in enumerate(self._explained_variance)
        }

        metadata: dict[str, Any] = {
            "n_factors": self._n_factors,
            "total_variance_explained": round(total_explained, 3),
            "idio_mom_63d": round(idio_mom_63, 4),
            "idio_std": round(idio_std, 4),
            "scree": scree_data,
        }

        return ModelOutput(
            model_name=self.model_name,
            symbol=self.symbol,
            timestamp=datetime.now(tz=timezone.utc),
            signal=signal,
            confidence=confidence,
            forecast_return=round(forecast_return, 6),
            forecast_std=round(forecast_std, 6),
            regime=None,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_return_matrix(self, data: pd.DataFrame) -> np.ndarray:
        """Build a 2-D return matrix with the target as the first column."""
        close = data["close"].astype(np.float64)
        target_ret = np.log(close / close.shift(1)).dropna().values

        # Use additional numeric columns as the universe (if available)
        numeric_cols = [c for c in data.columns if c != "close" and data[c].dtype.kind in "fc"]
        if numeric_cols:
            extras = []
            for col in numeric_cols:
                col_ret = np.log(
                    data[col].astype(np.float64) / data[col].astype(np.float64).shift(1)
                ).dropna().values
                # Align to target_ret length
                min_len = min(len(target_ret), len(col_ret))
                extras.append(col_ret[-min_len:])
            min_len = min(len(target_ret), *[len(e) for e in extras])
            ret_matrix = np.column_stack(
                [target_ret[-min_len:]] + [e[-min_len:] for e in extras]
            )
        else:
            ret_matrix = target_ret.reshape(-1, 1)

        # Use last `window` rows
        ret_matrix = ret_matrix[-self.window:]
        return ret_matrix.astype(np.float64)

    def _neutral_output(self) -> ModelOutput:
        """Return neutral output with zero signal."""
        return ModelOutput(
            model_name=self.model_name,
            symbol=self.symbol,
            timestamp=datetime.now(tz=timezone.utc),
            signal=0.0,
            confidence=0.0,
            forecast_return=0.0,
            forecast_std=0.02,
            regime=None,
            metadata={"error": "insufficient_data"},
        )
