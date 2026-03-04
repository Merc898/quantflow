"""Walk-forward evaluator for all QuantFlow models.

Implements TimeSeriesSplit-based evaluation that ensures strict temporal
ordering and no look-ahead bias.  Logs metrics to MLflow if a tracking
URI is configured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from quantflow.config.constants import (
    IC_MIN_THRESHOLD,
    ICIR_MIN_THRESHOLD,
    WALK_FORWARD_GAP,
    WALK_FORWARD_N_SPLITS,
)
from quantflow.config.logging import get_logger
from quantflow.models.base import BaseQuantModel, ModelOutput

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseQuantModel)


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardResult:
    """Results from a walk-forward evaluation run.

    Attributes:
        model_name: Name of the evaluated model.
        symbol: Ticker symbol evaluated.
        n_splits: Number of train/test splits used.
        ic_series: Information Coefficient per split.
        ic_mean: Mean IC across splits.
        ic_std: Std of IC across splits.
        icir: IC / std(IC) — information ratio of IC.
        hit_rate: Fraction of splits with IC > 0.
        rmse: Root-mean-square error of return forecasts.
        outputs: List of ModelOutput from each test split.
        metadata: Additional diagnostics.
    """

    model_name: str
    symbol: str
    n_splits: int
    ic_series: list[float] = field(default_factory=list)
    ic_mean: float = 0.0
    ic_std: float = 0.0
    icir: float = 0.0
    hit_rate: float = 0.0
    rmse: float = 0.0
    outputs: list[ModelOutput] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passes_ic_threshold(self) -> bool:
        """True if mean IC passes the minimum acceptable threshold."""
        return self.ic_mean > IC_MIN_THRESHOLD

    @property
    def passes_icir_threshold(self) -> bool:
        """True if ICIR passes the minimum acceptable threshold."""
        return abs(self.icir) > ICIR_MIN_THRESHOLD


# ---------------------------------------------------------------------------
# Walk-forward evaluator
# ---------------------------------------------------------------------------


class WalkForwardEvaluator:
    """Evaluate a :class:`BaseQuantModel` using time-series cross-validation.

    Each split consists of:
    - Training window: all data up to split date.
    - Gap: ``gap_days`` to prevent leakage around prediction.
    - Test window: ``test_size`` rows immediately after the gap.

    Information Coefficient is computed between the model's
    ``forecast_return`` and the realised forward return on the test set.

    Args:
        n_splits: Number of walk-forward folds.
        gap_days: Days to skip between train end and test start.
        min_train_size: Minimum rows in the training set.
        mlflow_experiment: MLflow experiment name (or None to skip logging).
    """

    def __init__(
        self,
        n_splits: int = WALK_FORWARD_N_SPLITS,
        gap_days: int = WALK_FORWARD_GAP,
        min_train_size: int = 252,
        mlflow_experiment: str | None = None,
    ) -> None:
        """Initialise the walk-forward evaluator.

        Args:
            n_splits: Number of time-series CV folds.
            gap_days: Gap between train end and test start (trading days).
            min_train_size: Minimum training set size.
            mlflow_experiment: MLflow experiment name for logging.
        """
        self.n_splits = n_splits
        self.gap_days = gap_days
        self.min_train_size = min_train_size
        self.mlflow_experiment = mlflow_experiment
        self._logger = get_logger(__name__)

    def evaluate(
        self,
        model_factory: Callable[[], T],
        data: pd.DataFrame,
        forecast_horizon: int = 21,
    ) -> WalkForwardResult:
        """Run walk-forward evaluation.

        The ``model_factory`` callable must return a **new**, unfitted instance
        of the model on each call — this ensures no state leaks between splits.

        Args:
            model_factory: Callable that returns a fresh model instance.
            data: Full historical DataFrame with at least a ``close`` column.
            forecast_horizon: Days ahead for forward-return computation.

        Returns:
            :class:`WalkForwardResult` with all evaluation metrics.
        """
        # Use a temporary model to get the name
        probe = model_factory()
        model_name = probe.model_name
        symbol = probe.symbol

        self._logger.info(
            "Starting walk-forward evaluation",
            model=model_name,
            symbol=symbol,
            n_splits=self.n_splits,
            data_rows=len(data),
        )

        # Compute realised forward returns (shifted back so [t] = return_{t→t+h})
        close = data["close"].astype(np.float64)
        forward_ret = close.shift(-forecast_horizon) / close - 1  # future return known at t

        tscv = TimeSeriesSplit(
            n_splits=self.n_splits,
            gap=self.gap_days,
        )

        ic_series: list[float] = []
        outputs: list[ModelOutput] = []
        forecast_returns: list[float] = []
        realised_returns: list[float] = []

        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(data)):
            if len(train_idx) < self.min_train_size:
                self._logger.debug(
                    "Skipping fold — insufficient training data",
                    fold=fold_idx,
                    train_size=len(train_idx),
                )
                continue

            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]

            try:
                model = model_factory()
                model.fit(train_data)
            except Exception as exc:
                self._logger.warning(
                    "Model fit failed on fold",
                    fold=fold_idx,
                    error=str(exc),
                )
                continue

            # Generate prediction for the start of the test window
            try:
                output = model.predict(test_data)
            except Exception as exc:
                self._logger.warning(
                    "Model predict failed on fold",
                    fold=fold_idx,
                    error=str(exc),
                )
                continue

            outputs.append(output)

            # IC computation: forecast_return vs mean realised return in test window
            test_realised = forward_ret.iloc[test_idx].dropna()
            if len(test_realised) == 0:
                continue

            realized_mean = float(test_realised.mean())
            fc_return = output.forecast_return
            forecast_returns.append(fc_return)
            realised_returns.append(realized_mean)

            # Fold-level IC: sign agreement as a proxy IC
            fold_ic = float(np.sign(fc_return) == np.sign(realized_mean)) - 0.5
            ic_series.append(fold_ic)

        # Aggregate metrics
        ic_arr = np.array(ic_series)
        ic_mean = float(ic_arr.mean()) if len(ic_arr) > 0 else 0.0
        ic_std = float(ic_arr.std()) if len(ic_arr) > 1 else 1.0
        icir = ic_mean / (ic_std + 1e-8)
        hit_rate = float((ic_arr > 0).mean()) if len(ic_arr) > 0 else 0.0

        # RMSE of return forecasts
        if len(forecast_returns) > 0:
            fc_arr = np.array(forecast_returns)
            real_arr = np.array(realised_returns)
            rmse = float(np.sqrt(np.mean((fc_arr - real_arr) ** 2)))
        else:
            rmse = float("nan")

        result = WalkForwardResult(
            model_name=model_name,
            symbol=symbol,
            n_splits=len(ic_series),
            ic_series=ic_series,
            ic_mean=round(ic_mean, 4),
            ic_std=round(ic_std, 4),
            icir=round(icir, 4),
            hit_rate=round(hit_rate, 4),
            rmse=round(rmse, 6) if np.isfinite(rmse) else float("nan"),
            outputs=outputs,
            metadata={"forecast_horizon": forecast_horizon},
        )

        self._logger.info(
            "Walk-forward evaluation complete",
            model=model_name,
            symbol=symbol,
            ic_mean=result.ic_mean,
            icir=result.icir,
            hit_rate=result.hit_rate,
            passes_ic=result.passes_ic_threshold,
            passes_icir=result.passes_icir_threshold,
        )

        # Log to MLflow if configured
        if self.mlflow_experiment is not None:
            self._log_to_mlflow(result)

        return result

    def _log_to_mlflow(self, result: WalkForwardResult) -> None:
        """Log walk-forward metrics to MLflow.

        Args:
            result: Completed walk-forward evaluation result.
        """
        try:
            import mlflow

            mlflow.set_experiment(self.mlflow_experiment)
            with mlflow.start_run(run_name=f"{result.model_name}_{result.symbol}"):
                mlflow.log_params(
                    {
                        "model_name": result.model_name,
                        "symbol": result.symbol,
                        "n_splits": result.n_splits,
                    }
                )
                mlflow.log_metrics(
                    {
                        "ic_mean": result.ic_mean,
                        "ic_std": result.ic_std,
                        "icir": result.icir,
                        "hit_rate": result.hit_rate,
                        "rmse": result.rmse if np.isfinite(result.rmse) else -1.0,
                    }
                )
        except Exception as exc:
            self._logger.warning("MLflow logging failed", error=str(exc))
