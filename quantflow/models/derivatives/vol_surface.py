"""Implied volatility surface model (SVI parameterisation).

Fits a Stochastic Volatility Inspired (SVI) smile to market data
when available; otherwise constructs a synthetic surface from
historical realized volatility, skew, and term-structure.

SVI total-variance smile (Gatheral 2004):
  w(k) = a + b * (rho*(k - m) + sqrt((k - m)^2 + sigma^2))

where k = log-moneyness, w = total implied variance = sigma_impl^2 * T.

Signal decomposition:
  1. ATM vol level   (high vol → bearish)
  2. Term-structure slope (inverted short/long → crisis → bearish)
  3. Return skewness as put-skew proxy (more negative → bearish)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, Field
from scipy.optimize import minimize

from quantflow.config.constants import TRADING_DAYS_PER_YEAR
from quantflow.config.logging import get_logger
from quantflow.models.base import BaseQuantModel, ModelOutput

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)

# SVI parameter bounds
_SVI_BOUNDS = [
    (1e-6, 2.0),  # a  ≥ 0
    (1e-6, 5.0),  # b  > 0
    (-0.999, 0.999),  # rho ∈ (-1, 1)
    (-1.0, 1.0),  # m
    (1e-4, 2.0),  # sigma > 0
]


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class SVIParameters(BaseModel):
    """SVI smile parameters for a single expiry.

    Attributes:
        expiry_days: Days to expiry.
        a: Vertical shift (ATM level contribution).
        b: Slope / wing steepness.
        rho: Put-call asymmetry (skew).
        m: ATM offset.
        sigma: Smile curvature.
        fit_rmse: Root-mean-square error of the fit.
    """

    expiry_days: int
    a: float
    b: float
    rho: float
    m: float
    sigma: float
    fit_rmse: float = 0.0


class VolSurfaceResult(BaseModel):
    """Fitted volatility surface summary.

    Attributes:
        atm_vol_short: ATM implied vol for short expiry (~21d).
        atm_vol_long: ATM implied vol for long expiry (~252d).
        term_structure_slope: atm_vol_short / atm_vol_long (>1 = inverted).
        return_skewness: Historical return skewness (proxy for put skew).
        vol_percentile: Current short-term vol percentile vs 252d history.
        svi_params: Fitted SVI parameters per expiry.
    """

    atm_vol_short: float
    atm_vol_long: float
    term_structure_slope: float
    return_skewness: float
    vol_percentile: float
    svi_params: list[SVIParameters] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class VolSurfaceModel(BaseQuantModel):
    """Implied volatility surface model.

    Generates a trading signal from:
    - ATM vol level relative to its history.
    - Term structure slope (inverted = stressed → bearish).
    - Return distribution skewness (proxy for put-call skew).

    When ``implied_vol_data`` is passed to :meth:`fit`, fits actual SVI
    parameters to market quotes.  Otherwise, derives a synthetic surface
    from historical realized volatility.

    Args:
        symbol: Ticker symbol.
        expiry_days: Expiry tenors (days) to model.
        vol_lookback: Lookback window (days) for historical vol.
    """

    def __init__(
        self,
        symbol: str,
        expiry_days: tuple[int, ...] = (21, 63, 126, 252),
        vol_lookback: int = 252,
    ) -> None:
        super().__init__("VolSurfaceModel", symbol)
        self._expiry_days = expiry_days
        self._vol_lookback = vol_lookback
        self._surface: VolSurfaceResult | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        data: pd.DataFrame,
        implied_vol_data: dict[int, dict[float, float]] | None = None,
    ) -> VolSurfaceModel:
        """Fit the volatility surface.

        Args:
            data: OHLCV DataFrame with UTC DatetimeIndex.
            implied_vol_data: Optional nested dict
                ``{expiry_days → {log_moneyness → implied_vol}}``.
                When provided, SVI is fit to market quotes.

        Returns:
            Self.
        """
        log_returns = np.log(data["close"] / data["close"].shift(1)).dropna()
        if len(log_returns) < 42:
            raise ValueError(f"VolSurfaceModel requires ≥42 observations; got {len(log_returns)}")

        svi_fits: list[SVIParameters] = []
        if implied_vol_data:
            for exp_days, smile_data in implied_vol_data.items():
                svi = self._fit_svi_smile(exp_days, smile_data)
                svi_fits.append(svi)

        self._surface = self._build_surface(log_returns, svi_fits)
        self._is_fitted = True
        self._log_fit_complete(
            atm_vol_short=round(self._surface.atm_vol_short, 4),
            atm_vol_long=round(self._surface.atm_vol_long, 4),
            term_slope=round(self._surface.term_structure_slope, 4),
            vol_pct=round(self._surface.vol_percentile, 2),
        )
        return self

    def predict(self, data: pd.DataFrame) -> ModelOutput:
        """Generate a vol-surface-based trading signal.

        Args:
            data: OHLCV DataFrame.

        Returns:
            :class:`ModelOutput`.
        """
        self._require_fitted()
        assert self._surface is not None

        surf = self._surface
        signal, confidence = self._signal_from_surface(surf)

        # Regime from vol percentile
        pct = surf.vol_percentile
        regime = "HIGH_VOL" if pct > 75 else ("LOW_VOL" if pct < 25 else "MEDIUM_VOL")

        return ModelOutput(
            model_name=self.model_name,
            symbol=self.symbol,
            timestamp=datetime.now(tz=UTC),
            signal=round(signal, 6),
            confidence=round(confidence, 6),
            forecast_return=0.0,
            forecast_std=round(surf.atm_vol_short, 6),
            regime=regime,
            metadata={
                "atm_vol_short": surf.atm_vol_short,
                "atm_vol_long": surf.atm_vol_long,
                "term_structure_slope": surf.term_structure_slope,
                "return_skewness": surf.return_skewness,
                "vol_percentile": surf.vol_percentile,
                "n_svi_fits": len(surf.svi_params),
            },
        )

    # ------------------------------------------------------------------
    # SVI fitting
    # ------------------------------------------------------------------

    @staticmethod
    def _svi_total_variance(params: np.ndarray, k: np.ndarray) -> np.ndarray:
        """Compute SVI total implied variance for log-moneyness k.

        Args:
            params: [a, b, rho, m, sigma].
            k: Log-moneyness array.

        Returns:
            Total variance w(k).
        """
        a, b, rho, m, sigma = params
        return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))

    def _fit_svi_smile(
        self,
        expiry_days: int,
        smile_data: dict[float, float],
    ) -> SVIParameters:
        """Fit SVI parameters to market implied vols.

        Args:
            expiry_days: Days to expiry.
            smile_data: Dict of {log_moneyness → implied_vol}.

        Returns:
            :class:`SVIParameters`.
        """
        k = np.array(list(smile_data.keys()))
        iv = np.array(list(smile_data.values()))
        T = expiry_days / TRADING_DAYS_PER_YEAR
        w_market = iv**2 * T  # total implied variance

        def objective(p: np.ndarray) -> float:
            w_model = self._svi_total_variance(p, k)
            return float(np.mean((w_model - w_market) ** 2))

        # Initial guess: ATM vol level
        atm_vol = float(np.median(iv))
        x0 = np.array([atm_vol**2 * T * 0.5, 0.1, -0.3, 0.0, 0.3])

        try:
            res = minimize(
                objective,
                x0,
                method="L-BFGS-B",
                bounds=_SVI_BOUNDS,
                options={"maxiter": 1000, "ftol": 1e-10},
            )
            params = res.x
            rmse = float(np.sqrt(res.fun))
        except Exception:
            params = x0
            rmse = float("inf")

        a, b, rho, m, sigma = params
        return SVIParameters(
            expiry_days=expiry_days,
            a=float(a),
            b=float(b),
            rho=float(rho),
            m=float(m),
            sigma=float(sigma),
            fit_rmse=rmse,
        )

    # ------------------------------------------------------------------
    # Surface construction from historical data
    # ------------------------------------------------------------------

    def _build_surface(
        self,
        returns: pd.Series,
        svi_fits: list[SVIParameters],
    ) -> VolSurfaceResult:
        """Build a vol surface summary from historical return data.

        Args:
            returns: Daily log-return series.
            svi_fits: Pre-fitted SVI parameters (may be empty).

        Returns:
            :class:`VolSurfaceResult`.
        """
        ann = TRADING_DAYS_PER_YEAR

        # Short-term (21d) and long-term (252d) realized vol
        if len(returns) >= 21:
            vol_short = float(returns.iloc[-21:].std() * np.sqrt(ann))
        else:
            vol_short = float(returns.std() * np.sqrt(ann))

        if len(returns) >= 252:
            vol_long = float(returns.iloc[-252:].std() * np.sqrt(ann))
        else:
            vol_long = vol_short

        vol_short = max(vol_short, 1e-4)
        vol_long = max(vol_long, 1e-4)
        term_slope = vol_short / vol_long

        # Return skewness (proxy for put-call skew; negative = fat left tail)
        skew_window = min(len(returns), self._vol_lookback)
        skewness = float(returns.iloc[-skew_window:].skew())
        if not np.isfinite(skewness):
            skewness = 0.0

        # Vol percentile vs rolling 252d
        if len(returns) >= 252:
            rolling_vol = returns.rolling(21).std().dropna() * np.sqrt(ann)
            vol_pct = float((rolling_vol < vol_short).mean() * 100)
        else:
            vol_pct = 50.0

        return VolSurfaceResult(
            atm_vol_short=round(vol_short, 6),
            atm_vol_long=round(vol_long, 6),
            term_structure_slope=round(term_slope, 6),
            return_skewness=round(skewness, 6),
            vol_percentile=round(vol_pct, 2),
            svi_params=svi_fits,
        )

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    @staticmethod
    def _signal_from_surface(surf: VolSurfaceResult) -> tuple[float, float]:
        """Derive trading signal from surface characteristics.

        Components:
        - ATM vol level: high vol → bearish (negative).
        - Term structure inversion: slope > 1.2 → bearish.
        - Skewness: more negative → more bearish.

        Args:
            surf: Fitted vol surface result.

        Returns:
            Tuple of (signal, confidence) both in appropriate ranges.
        """
        # 1. Vol-level component: scale vol_percentile to [-1, 1]
        # 0th pct → +1 (bullish low vol), 100th pct → -1 (bearish high vol)
        vol_signal = 1.0 - 2.0 * (surf.vol_percentile / 100.0)

        # 2. Term-structure component
        # Inverted (short > long by >20%) → crisis → bearish
        ts_signal = -float(np.tanh((surf.term_structure_slope - 1.0) * 3.0))

        # 3. Skewness component (more negative = fat left tail = bearish)
        # Typical range: -2 to +0.5 for equity returns
        skew_signal = float(np.tanh(surf.return_skewness / 2.0))

        # Weighted combination
        combined = 0.5 * vol_signal + 0.3 * ts_signal + 0.2 * skew_signal
        combined = float(np.clip(combined, -1.0, 1.0))

        # Confidence: high when vol percentile is extreme (clear regime signal)
        # Low in the middle of the vol range
        distance_from_center = abs(surf.vol_percentile - 50.0) / 50.0
        confidence = float(np.clip(0.40 + 0.40 * distance_from_center, 0.40, 0.80))

        return combined, confidence
