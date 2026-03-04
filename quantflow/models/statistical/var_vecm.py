"""VAR / VECM model for multivariate return forecasting.

Uses statsmodels for Vector Autoregression with Johansen cointegration
testing to switch to VECM when series are cointegrated.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen

from quantflow.config.logging import get_logger
from quantflow.models.base import BaseQuantModel, ModelOutput

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)

_MAX_LAGS = 10


class VARModel(BaseQuantModel):
    """Vector Autoregression model with automatic VECM switching.

    Fits a VAR on a multivariate return series (target asset + auxiliary
    variables).  If Johansen cointegration is detected (trace test, 5%),
    switches to VECM to exploit the long-run equilibrium relationship.

    Args:
        symbol: Ticker symbol for the target asset.
        target_col: Column name of the target return series.
        max_lags: Maximum VAR lag order to consider.
        horizon: Forecast horizon in steps (trading days).
    """

    def __init__(
        self,
        symbol: str,
        target_col: str = "target_return",
        max_lags: int = _MAX_LAGS,
        horizon: int = 1,
    ) -> None:
        """Initialise the VAR/VECM model.

        Args:
            symbol: Ticker symbol.
            target_col: Name of the target return column.
            max_lags: Max lag search range.
            horizon: Steps-ahead forecast horizon.
        """
        super().__init__("VAR_VECM", symbol)
        self.target_col = target_col
        self.max_lags = max_lags
        self.horizon = horizon

        self._fitted_model: Any = None
        self._is_vecm: bool = False
        self._target_idx: int = 0
        self._vol_estimate: float = 0.02
        self._col_names: list[str] = []
        self._lag_order: int = 1

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame) -> VARModel:
        """Fit VAR or VECM on a multivariate log-return DataFrame.

        The input DataFrame should contain multiple return series.
        The ``target_col`` (or ``close``) is used for signal generation.

        Args:
            data: DataFrame with multiple columns of returns (or prices
                  that will be converted to log-returns).  UTC DatetimeIndex.

        Returns:
            Self (fitted model).
        """
        df = self._prepare_returns(data)
        self._col_names = list(df.columns)

        if self.target_col in df.columns:
            self._target_idx = df.columns.tolist().index(self.target_col)
        else:
            self._target_idx = 0

        self._vol_estimate = float(df.iloc[:, self._target_idx].std())

        # Johansen cointegration test on levels (for VECM switch)
        is_cointegrated = self._johansen_test(data)

        if is_cointegrated:
            self._fit_vecm(df)
        else:
            self._fit_var(df)

        self._is_fitted = True
        self._log_fit_complete(
            is_vecm=self._is_vecm,
            lag_order=self._lag_order,
            n_series=len(self._col_names),
        )
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, data: pd.DataFrame | None = None) -> ModelOutput:
        """Forecast the target return and generate a signal.

        Args:
            data: Optional new data for updating the model.

        Returns:
            :class:`ModelOutput` with h-step-ahead return forecast.
        """
        self._require_fitted()

        try:
            if self._is_vecm:
                fc = self._fitted_model.predict(steps=self.horizon)
                forecast_return = float(np.sum(fc[:, self._target_idx]))
            else:
                result = self._fitted_model
                fc = result.forecast(
                    result.endog[-self._lag_order :],
                    steps=self.horizon,
                )
                forecast_return = float(np.sum(fc[:, self._target_idx]))
        except Exception as exc:
            self._logger.warning("VAR prediction failed", error=str(exc))
            return self._fallback_output()

        forecast_std = self._vol_estimate * np.sqrt(self.horizon)
        raw_signal = forecast_return / (forecast_std + 1e-8)
        signal = self.normalise_signal(raw_signal)

        confidence = min(0.65, max(0.25, 0.5 - abs(raw_signal) * 0.03))

        metadata: dict[str, Any] = {
            "is_vecm": self._is_vecm,
            "lag_order": self._lag_order,
            "horizon": self.horizon,
            "n_series": len(self._col_names),
        }

        return ModelOutput(
            model_name=self.model_name,
            symbol=self.symbol,
            timestamp=datetime.now(tz=UTC),
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

    def _prepare_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert price columns to log-returns."""
        # If data already looks like returns (small magnitudes), use directly
        if data.select_dtypes(include=[np.number]).abs().max().max() < 1.0:
            return data.select_dtypes(include=[np.number]).dropna()

        # Otherwise compute log-returns from price columns
        numeric = data.select_dtypes(include=[np.number])
        log_ret = np.log(numeric / numeric.shift(1)).dropna()
        return log_ret

    def _johansen_test(self, data: pd.DataFrame) -> bool:
        """Run Johansen trace test for cointegration.

        Returns:
            True if at least one cointegrating relationship is found at 5%.
        """
        try:
            price_cols = data.select_dtypes(include=[np.number]).dropna()
            if price_cols.shape[1] < 2:
                return False
            result = coint_johansen(price_cols.values, det_order=0, k_ar_diff=1)
            # Trace statistic vs 5% critical value
            trace_stat = result.lr1[0]
            crit_5pct = result.cvt[0, 1]
            return bool(trace_stat > crit_5pct)
        except Exception:
            return False

    def _fit_var(self, df: pd.DataFrame) -> None:
        """Fit a VAR model with AIC-selected lag order."""
        model = VAR(df)
        try:
            lag_results = model.select_order(maxlags=min(self.max_lags, len(df) // 10))
            self._lag_order = int(lag_results.aic)
        except Exception:
            self._lag_order = 1
        self._lag_order = max(1, min(self._lag_order, self.max_lags))
        self._fitted_model = model.fit(maxlags=self._lag_order)
        self._is_vecm = False

    def _fit_vecm(self, df: pd.DataFrame) -> None:
        """Fit a VECM model."""
        try:
            self._lag_order = 1
            vecm = VECM(df, k_ar_diff=self._lag_order, coint_rank=1)
            self._fitted_model = vecm.fit()
            self._is_vecm = True
        except Exception:
            self._fit_var(df)

    def _fallback_output(self) -> ModelOutput:
        """Return neutral output when prediction fails."""
        return ModelOutput(
            model_name=self.model_name,
            symbol=self.symbol,
            timestamp=datetime.now(tz=UTC),
            signal=0.0,
            confidence=0.0,
            forecast_return=0.0,
            forecast_std=self._vol_estimate,
            metadata={"error": "prediction_failed"},
        )
