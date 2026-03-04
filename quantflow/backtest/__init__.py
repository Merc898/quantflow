"""Backtesting framework with walk-forward validation.

Key exports:
- :class:`BacktestConfig` — engine configuration
- :class:`BacktestEngine` — main walk-forward engine
- :class:`BacktestResult` — result container
- :class:`PerformanceMetrics` — full metric suite (Spec 09)
- :class:`BaseStrategy` — abstract strategy interface
- :class:`QuantFlowStrategy` — built-in momentum + HRP strategy
- :class:`Trade` — individual trade record
- :func:`generate_pdf_report` — PDF report generation
- :func:`compute_performance_metrics` — standalone metric computation
"""

from quantflow.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    BaseStrategy,
    PerformanceMetrics,
    QuantFlowStrategy,
    Trade,
    compute_performance_metrics,
)
from quantflow.backtest.report import generate_pdf_report

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResult",
    "BaseStrategy",
    "PerformanceMetrics",
    "QuantFlowStrategy",
    "Trade",
    "compute_performance_metrics",
    "generate_pdf_report",
]
