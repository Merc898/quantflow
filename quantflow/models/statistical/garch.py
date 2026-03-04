"""GARCH family models for conditional volatility forecasting.

Implements GARCH(1,1), EGARCH(1,1), TGARCH(1,1) via the ``arch`` library.
Outputs a volatility forecast and regime label used by the signal engine.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy import stats

from quantflow.config.constants import (
    GARCH_P,
    GARCH_PERSISTENCE_WARNING,
    GARCH_Q,
    LOOKBACK_1Y,
    TRADING_DAYS_PER_YEAR,
    VOL_REGIME_EXTREME_THRESHOLD,
    VOL_REGIME_HIGH_THRESHOLD,
    VOL_REGIME_LOW_THRESHOLD,
)
from quantflow.config.logging import get_logger
from quantflow.models.base import BaseQuantModel, ModelOutput

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)

GarchVariant = Literal["GARCH", "EGARCH", "TGARCH"]


class GARCHModel(BaseQuantModel):
    """Conditional volatility model using the GARCH family.

    Fits one of GARCH(1,1), EGARCH(1,1), or TGARCH(1,1) on log returns,
    forecasts conditional variance one step ahead, and derives a
    volatility-regime-aware signal.

    The signal contribution is based on inverse-volatility weighting:
    high conditional vol → lower (more cautious) signal.

    Args:
        symbol: Ticker symbol.
        model_type: One of "GARCH", "EGARCH", "TGARCH".
        p: ARCH lag order (default 1).
        q: GARCH lag order (default 1).
        dist: Error distribution — "normal", "t", or "ged".
    """

    def __init__(
        self,
        symbol: str,
        model_type: GarchVariant = "GARCH",
        p: int = GARCH_P,
        q: int = GARCH_Q,
        dist: str = "t",
    ) -> None:
        """Initialise the GARCH model.

        Args:
            symbol: Ticker symbol.
            model_type: GARCH variant to fit.
            p: ARCH order.
            q: GARCH order.
            dist: Residual distribution ("normal", "t", "ged").
        """
        super().__init__(f"GARCH_{model_type}", symbol)
        self.model_type: GarchVariant = model_type
        self.p = p
        self.q = q
        self.dist = dist

        self._result: Any = None
        self._log_returns: pd.Series | None = None
        self._historical_vol: pd.Series | None = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame) -> GARCHModel:
        """Fit the GARCH model on log returns.

        Automatically tests for ARCH effects before fitting.
        Chooses Student-t distribution if residuals are non-normal.

        Args:
            data: DataFrame with a ``close`` column (UTC DatetimeIndex).

        Returns:
            Self (fitted model).
        """
        from arch import arch_model

        close = data["close"].astype(np.float64)
        log_ret = np.log(close / close.shift(1)).dropna() * 100  # scale to %

        # Jarque-Bera test: use Student-t distribution for heavy tails
        use_t = True  # default to t-dist
        _jb_stat, jb_pval = stats.jarque_bera(log_ret)
        if jb_pval > 0.05:
            use_t = False  # near-normal residuals → Gaussian sufficient

        dist = "t" if use_t else "normal"

        vol_spec: dict[str, Any] = {
            "GARCH": {"vol": "Garch", "p": self.p, "q": self.q},
            "EGARCH": {"vol": "EGARCH", "p": self.p, "q": self.q},
            "TGARCH": {"vol": "GARCH", "p": self.p, "o": 1, "q": self.q},
        }[self.model_type]

        am = arch_model(
            log_ret,
            mean="Zero",
            dist=dist,  # type: ignore[arg-type]
            **vol_spec,
        )

        self._result = am.fit(disp="off", show_warning=False)
        self._log_returns = log_ret

        # Store rolling 252-day historical vol for regime classification
        self._historical_vol = (
            log_ret.rolling(LOOKBACK_1Y).std() * np.sqrt(TRADING_DAYS_PER_YEAR) / 100.0
        )

        # Check persistence
        params = self._result.params
        if self.model_type == "GARCH":
            alpha = params.get("alpha[1]", 0.0)
            beta = params.get("beta[1]", 0.0)
            persistence = alpha + beta
            if persistence > GARCH_PERSISTENCE_WARNING:
                self._logger.warning(
                    "Near-unit-root variance persistence",
                    persistence=round(persistence, 4),
                    model=self.model_name,
                    symbol=self.symbol,
                )

        self._is_fitted = True
        self._log_fit_complete(
            aic=round(self._result.aic, 2),
            bic=round(self._result.bic, 2),
            dist=dist,
            jb_pval=round(float(jb_pval), 4),
        )
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, data: pd.DataFrame | None = None) -> ModelOutput:
        """Forecast one-step-ahead conditional volatility.

        Args:
            data: Not used (GARCH forecasts from the fitted model's last state).
                  Accepted for interface compatibility.

        Returns:
            :class:`ModelOutput` with:
            - ``signal``: Inverse-vol-normalised signal in ``[-1, +1]``.
              Negative (bearish) when vol is high relative to history.
            - ``forecast_std``: One-step-ahead annualised conditional vol.
            - ``regime``: "LOW_VOL" / "MEDIUM_VOL" / "HIGH_VOL" / "EXTREME_VOL".
        """
        self._require_fitted()
        assert self._result is not None
        assert self._historical_vol is not None

        # One-step-ahead forecast (h=1)
        forecast = self._result.forecast(horizon=1, reindex=False)
        cond_var_pct_sq = float(forecast.variance.values[-1, 0])
        # Convert: (% daily vol)² → annualised decimal vol
        cond_vol_annual = np.sqrt(cond_var_pct_sq) / 100.0 * np.sqrt(TRADING_DAYS_PER_YEAR)

        # Regime classification (percentile of historical vol distribution)
        hist_vol = self._historical_vol.dropna()
        if len(hist_vol) > 0:
            pct = float(stats.percentileofscore(hist_vol, cond_vol_annual))
        else:
            pct = 50.0

        if pct >= VOL_REGIME_EXTREME_THRESHOLD:
            regime = "EXTREME_VOL"
        elif pct >= VOL_REGIME_HIGH_THRESHOLD:
            regime = "HIGH_VOL"
        elif pct <= VOL_REGIME_LOW_THRESHOLD:
            regime = "LOW_VOL"
        else:
            regime = "MEDIUM_VOL"

        # Signal: invert vol → high vol = bearish signal
        target_vol = 0.15  # 15% target annual vol
        raw_signal = -(cond_vol_annual - target_vol) / target_vol
        signal = self.normalise_signal(raw_signal)

        # Confidence inversely proportional to vol regime
        confidence_map = {
            "LOW_VOL": 0.75,
            "MEDIUM_VOL": 0.60,
            "HIGH_VOL": 0.45,
            "EXTREME_VOL": 0.30,
        }
        confidence = confidence_map[regime]

        metadata: dict[str, Any] = {
            "cond_vol_annual": round(cond_vol_annual, 4),
            "vol_percentile": round(pct, 1),
            "aic": round(self._result.aic, 2),
            "bic": round(self._result.bic, 2),
            "model_type": self.model_type,
            "dist": self._result.model.distribution.name,
        }

        # Add persistence for GARCH variants
        if self.model_type == "GARCH":
            params = self._result.params
            alpha = float(params.get("alpha[1]", np.nan))
            beta = float(params.get("beta[1]", np.nan))
            metadata["persistence"] = round(alpha + beta, 4)

        return ModelOutput(
            model_name=self.model_name,
            symbol=self.symbol,
            timestamp=datetime.now(tz=UTC),
            signal=signal,
            confidence=confidence,
            forecast_return=0.0,  # GARCH forecasts vol, not return
            forecast_std=round(cond_vol_annual, 6),
            regime=regime,
            metadata=metadata,
        )
