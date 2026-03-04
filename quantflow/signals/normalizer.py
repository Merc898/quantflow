"""Signal normalization pipeline.

Before fusion, ALL signals are normalized to a common scale:
1. Winsorize at [1%, 99%] to clip outliers.
2. Cross-sectional z-score (if multiple assets provided).
3. Time-series rolling z-score (252-day window).
4. Clip to [-3, +3].
5. Optional: isotonic regression for monotonicity enforcement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantflow.config.constants import (
    FEATURE_WINSORIZE_LOWER,
    FEATURE_WINSORIZE_UPPER,
    FEATURE_ZSCORE_CLIP,
    LOOKBACK_1Y,
)
from quantflow.config.logging import get_logger

logger = get_logger(__name__)


class SignalNormalizer:
    """Normalize raw model signals to a common z-score scale.

    Handles both single-series normalization and cross-sectional
    (multi-asset) normalization for the signal aggregation pipeline.

    Args:
        winsorize_lower: Lower winsorization quantile (default 0.01).
        winsorize_upper: Upper winsorization quantile (default 0.99).
        zscore_clip: Clip z-scores to ``[-clip, clip]`` (default 3.0).
        window: Rolling window for time-series z-score (default 252 days).
    """

    def __init__(
        self,
        winsorize_lower: float = FEATURE_WINSORIZE_LOWER,
        winsorize_upper: float = FEATURE_WINSORIZE_UPPER,
        zscore_clip: float = FEATURE_ZSCORE_CLIP,
        window: int = LOOKBACK_1Y,
    ) -> None:
        self._low = winsorize_lower
        self._high = winsorize_upper
        self._clip = zscore_clip
        self._window = window
        self._logger = get_logger(__name__)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def normalize(
        self,
        signal: pd.Series,
        cross_section: pd.DataFrame | None = None,
        use_isotonic: bool = False,
    ) -> pd.Series:
        """Normalize a single signal series.

        Applies: winsorize → (cross-sectional z-score) → time-series z-score → clip.

        Args:
            signal: Raw signal Series (DatetimeIndex or integer index).
            cross_section: Optional DataFrame of same-index signals for other
                assets.  If provided, cross-sectional z-scoring is applied.
            use_isotonic: Whether to apply isotonic regression post-normalization
                to enforce monotonicity (useful for rank-based signals).

        Returns:
            Normalized signal Series with the same index, clipped to
            ``[-clip, clip]``.
        """
        s = signal.copy().astype(np.float64)

        # Step 1: Winsorize
        s = self._winsorize(s)

        # Step 2: Cross-sectional z-score (optional)
        if cross_section is not None:
            s = self._cross_sectional_zscore(s, cross_section)

        # Step 3: Time-series rolling z-score
        s = self._rolling_zscore(s)

        # Step 4: Clip
        s = s.clip(-self._clip, self._clip).fillna(0.0)

        # Step 5: Isotonic regression (optional)
        if use_isotonic:
            s = self._isotonic_transform(s)

        return s

    def normalize_batch(
        self,
        signals: dict[str, pd.Series],
        use_isotonic: bool = False,
    ) -> dict[str, pd.Series]:
        """Normalize a batch of signals with cross-sectional z-scoring.

        Aligns all signals on a common index before cross-sectional
        standardization.

        Args:
            signals: Dict mapping model_name → raw signal Series.
            use_isotonic: Apply isotonic regression to each signal.

        Returns:
            Dict mapping model_name → normalized signal Series.
        """
        if not signals:
            return {}

        # Align on common index
        df = pd.DataFrame(signals).sort_index()
        normalized: dict[str, pd.Series] = {}

        for name in df.columns:
            try:
                normalized[name] = self.normalize(
                    df[name].dropna(),
                    cross_section=df.drop(columns=[name]),
                    use_isotonic=use_isotonic,
                )
            except Exception as exc:
                self._logger.warning(
                    "Signal normalization failed",
                    model=name,
                    error=str(exc),
                )
                normalized[name] = pd.Series(dtype=float)

        return normalized

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _winsorize(self, s: pd.Series) -> pd.Series:
        """Clip signal values at empirical quantile bounds.

        Args:
            s: Raw signal Series.

        Returns:
            Winsorized Series.
        """
        lo = float(s.quantile(self._low))
        hi = float(s.quantile(self._high))
        return s.clip(lo, hi)

    def _cross_sectional_zscore(
        self, s: pd.Series, cross_section: pd.DataFrame
    ) -> pd.Series:
        """Apply cross-sectional (universe-relative) standardization.

        At each time step, subtract the cross-sectional mean and divide
        by the cross-sectional standard deviation.

        Args:
            s: Signal for the target asset.
            cross_section: Signals for peer assets (same index).

        Returns:
            Cross-sectionally standardized signal.
        """
        # Align
        aligned = cross_section.reindex(s.index)
        combined = pd.concat([s.rename("__target__"), aligned], axis=1)

        cs_mean = combined.mean(axis=1)
        cs_std = combined.std(axis=1).replace(0, np.nan)

        normalized = (s - cs_mean) / cs_std
        return normalized.fillna(s)  # Fall back to original if std is zero

    def _rolling_zscore(self, s: pd.Series) -> pd.Series:
        """Compute rolling time-series z-score.

        Args:
            s: Winsorized signal Series.

        Returns:
            Time-series z-score Series.
        """
        roll_mean = s.rolling(self._window, min_periods=max(5, self._window // 10)).mean()
        roll_std = s.rolling(self._window, min_periods=max(5, self._window // 10)).std()
        roll_std = roll_std.replace(0, np.nan)
        return ((s - roll_mean) / roll_std).fillna(0.0)

    @staticmethod
    def _isotonic_transform(s: pd.Series) -> pd.Series:
        """Apply isotonic (monotone) regression to the signal.

        Useful to enforce that higher raw signal values always map to
        higher normalized values (removes noise-driven inversions).

        Args:
            s: Normalized signal Series.

        Returns:
            Isotonic-transformed Series with the same index.
        """
        try:
            from sklearn.isotonic import IsotonicRegression

            valid = s.dropna()
            if len(valid) < 10:
                return s
            iso = IsotonicRegression(out_of_bounds="clip")
            X = np.arange(len(valid)).reshape(-1, 1)
            fitted = iso.fit_transform(X.ravel(), valid.values)
            result = s.copy()
            result.loc[valid.index] = fitted
            return result
        except ImportError:
            return s
