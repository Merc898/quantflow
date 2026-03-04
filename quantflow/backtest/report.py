"""PDF report generation for backtest results.

Uses ``matplotlib.backends.backend_pdf.PdfPages`` — no extra dependencies
beyond the core scientific stack.  Generates a multi-page PDF with:

1. Cover page — key statistics summary table
2. Equity curve — strategy vs benchmark (rebased to 100)
3. Underwater chart — rolling drawdown
4. Rolling Sharpe — 252-day trailing Sharpe ratio
5. IC series — information coefficient over time with rolling mean
6. Stress period bar chart — Sharpe and max-drawdown comparison
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter

from quantflow.config.logging import get_logger

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from quantflow.backtest.engine import BacktestResult, PerformanceMetrics

logger = get_logger(__name__)

# ── Style constants ──────────────────────────────────────────────────────────
_FIG_W = 11.0
_FIG_H = 8.5
_STRATEGY_COLOR = "#2196F3"
_BENCHMARK_COLOR = "#FF5722"
_POSITIVE_COLOR = "#4CAF50"
_NEGATIVE_COLOR = "#F44336"
_GRID_ALPHA = 0.25
_TITLE_SIZE = 14
_LABEL_SIZE = 11
_TICK_SIZE = 9


def generate_pdf_report(
    result: BacktestResult,
    output_path: str | Path,
    strategy_name: str = "QuantFlow Strategy",
) -> Path:
    """Generate a six-page PDF backtest report.

    Args:
        result: Completed :class:`BacktestResult` from
                :class:`~quantflow.backtest.engine.BacktestEngine`.
        output_path: File path for the PDF.  Parent directories must exist.
        strategy_name: Display name for the strategy (used in titles).

    Returns:
        Resolved :class:`pathlib.Path` to the generated PDF.
    """
    output_path = Path(output_path)
    logger.info("Generating PDF report", path=str(output_path))

    with PdfPages(output_path) as pdf:
        _page_summary(pdf, result, strategy_name)
        _page_equity_curve(pdf, result, strategy_name)
        _page_drawdown(pdf, result, strategy_name)
        _page_rolling_sharpe(pdf, result, strategy_name)
        _page_ic_series(pdf, result)
        _page_stress_periods(pdf, result)

        # PDF metadata
        d = pdf.infodict()
        d["Title"] = f"{strategy_name} — Backtest Report"
        d["Author"] = "QuantFlow"
        d["Subject"] = "Walk-Forward Backtest Analysis"
        d["CreationDate"] = datetime.now()

    logger.info("PDF report written", path=str(output_path))
    return output_path.resolve()


# ---------------------------------------------------------------------------
# Page helpers
# ---------------------------------------------------------------------------


def _page_summary(pdf: PdfPages, result: BacktestResult, name: str) -> None:
    """Render the cover / summary statistics page."""
    m = result.metrics
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    ax.axis("off")

    title = f"{name}\nBacktest Report"
    ax.text(
        0.5,
        0.92,
        title,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.85,
        f"{result.config.start_date}  →  {result.config.end_date}  "
        f"(${result.config.initial_capital:,.0f} initial capital)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        color="#555",
    )

    rows = _metric_rows(m)
    n = len(rows)
    mid = (n + 1) // 2
    left_rows, right_rows = rows[:mid], rows[mid:]

    for col_offset, group in enumerate([left_rows, right_rows]):
        x_label = 0.02 + col_offset * 0.50
        x_value = 0.25 + col_offset * 0.50
        y_start = 0.78
        step = 0.045

        ax.text(
            x_label,
            y_start + step,
            "Metric",
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            color="#333",
        )
        ax.text(
            x_value,
            y_start + step,
            "Value",
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            color="#333",
        )
        ax.plot(
            [col_offset * 0.50, 0.50 + col_offset * 0.50],
            [y_start + step * 0.5, y_start + step * 0.5],
            color="#999",
            linewidth=0.8,
            transform=ax.transAxes,
        )

        for j, (label, value, color) in enumerate(group):
            y = y_start - j * step
            "#F5F5F5" if j % 2 == 0 else "white"
            ax.text(x_label, y, label, transform=ax.transAxes, fontsize=9, va="top", color="#222")
            ax.text(
                x_value,
                y,
                value,
                transform=ax.transAxes,
                fontsize=9,
                va="top",
                fontweight="bold",
                color=color,
            )

    _add_footer(fig, result)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_equity_curve(pdf: PdfPages, result: BacktestResult, name: str) -> None:
    """Render equity curve vs benchmark (rebased to 100)."""
    eq = result.equity_series()
    bench = result.benchmark_series()
    if eq.empty:
        return

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))

    eq_rebased = eq / eq.iloc[0] * 100
    ax.plot(
        eq_rebased.index,
        eq_rebased.values,
        color=_STRATEGY_COLOR,
        linewidth=1.5,
        label=f"{name} (net)",
    )

    if not bench.empty:
        bench_rebased = bench / bench.iloc[0] * 100
        ax.plot(
            bench_rebased.index,
            bench_rebased.values,
            color=_BENCHMARK_COLOR,
            linewidth=1.2,
            linestyle="--",
            label=result.config.benchmark_symbol,
            alpha=0.85,
        )

    ax.axhline(100, color="#999", linewidth=0.8, linestyle=":")
    ax.set_title(f"{name} — Equity Curve (rebased to 100)", fontsize=_TITLE_SIZE, pad=12)
    ax.set_ylabel("Portfolio Value (rebased)", fontsize=_LABEL_SIZE)
    ax.legend(fontsize=_LABEL_SIZE)
    ax.grid(alpha=_GRID_ALPHA)
    ax.tick_params(labelsize=_TICK_SIZE)

    # Annotate final values
    ax.annotate(
        f"Final: {eq_rebased.iloc[-1]:.1f}",
        xy=(eq_rebased.index[-1], eq_rebased.iloc[-1]),
        xytext=(-60, 10),
        textcoords="offset points",
        fontsize=9,
        color=_STRATEGY_COLOR,
        arrowprops={"arrowstyle": "->", "color": _STRATEGY_COLOR, "lw": 0.8},
    )

    _add_footer(fig, result)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_drawdown(pdf: PdfPages, result: BacktestResult, name: str) -> None:
    """Render underwater (drawdown) chart."""
    eq = result.equity_series()
    if eq.empty:
        return

    cum = eq / eq.iloc[0]
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max.clip(lower=1e-8) * 100

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    ax.fill_between(dd.index, dd.values, 0, color=_NEGATIVE_COLOR, alpha=0.55, label="Drawdown")
    ax.plot(dd.index, dd.values, color=_NEGATIVE_COLOR, linewidth=0.8)

    ax.axhline(
        result.metrics.max_drawdown * 100,
        color="#B71C1C",
        linewidth=0.8,
        linestyle="--",
        label=f"Max DD: {result.metrics.max_drawdown:.1%}",
    )

    ax.set_title(f"{name} — Underwater Chart", fontsize=_TITLE_SIZE, pad=12)
    ax.set_ylabel("Drawdown (%)", fontsize=_LABEL_SIZE)
    ax.legend(fontsize=_LABEL_SIZE)
    ax.grid(alpha=_GRID_ALPHA)
    ax.tick_params(labelsize=_TICK_SIZE)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))

    _add_footer(fig, result)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_rolling_sharpe(pdf: PdfPages, result: BacktestResult, name: str) -> None:
    """Render 252-day trailing Sharpe ratio."""
    eq = result.equity_series()
    if eq.empty or len(eq) < 252:
        return

    ret = eq.pct_change().dropna()
    roll_sharpe = (
        ret.rolling(252)
        .apply(lambda r: (r.mean() / r.std() * np.sqrt(252)) if r.std() > 1e-8 else 0.0)
        .dropna()
    )

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    ax.plot(roll_sharpe.index, roll_sharpe.values, color=_STRATEGY_COLOR, linewidth=1.2)
    ax.fill_between(
        roll_sharpe.index,
        roll_sharpe.values,
        0,
        where=roll_sharpe.values >= 0,
        color=_POSITIVE_COLOR,
        alpha=0.30,
    )
    ax.fill_between(
        roll_sharpe.index,
        roll_sharpe.values,
        0,
        where=roll_sharpe.values < 0,
        color=_NEGATIVE_COLOR,
        alpha=0.30,
    )
    ax.axhline(0, color="#555", linewidth=0.8, linestyle=":")
    ax.axhline(1.0, color=_POSITIVE_COLOR, linewidth=0.8, linestyle="--", label="Sharpe = 1.0")

    ax.set_title(f"{name} — Rolling 252-Day Sharpe Ratio", fontsize=_TITLE_SIZE, pad=12)
    ax.set_ylabel("Sharpe Ratio", fontsize=_LABEL_SIZE)
    ax.legend(fontsize=_LABEL_SIZE)
    ax.grid(alpha=_GRID_ALPHA)
    ax.tick_params(labelsize=_TICK_SIZE)

    _add_footer(fig, result)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_ic_series(pdf: PdfPages, result: BacktestResult) -> None:
    """Render IC per rebalancing period with 12-period rolling mean."""
    if not result.ic_series:
        return

    ic_dates = pd.to_datetime(result.ic_dates)
    ic_vals = np.array(result.ic_series, dtype=np.float64)
    ic_s = pd.Series(ic_vals, index=ic_dates)
    roll_mean = ic_s.rolling(window=min(12, len(ic_s))).mean()

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    colors = [_POSITIVE_COLOR if v >= 0 else _NEGATIVE_COLOR for v in ic_vals]
    ax.bar(ic_dates, ic_vals, color=colors, width=15, alpha=0.70, label="IC per period")
    ax.plot(
        roll_mean.index,
        roll_mean.values,
        color=_STRATEGY_COLOR,
        linewidth=1.8,
        label="12-period rolling mean",
    )
    ax.axhline(0, color="#555", linewidth=0.8, linestyle=":")
    ax.axhline(0.03, color="#888", linewidth=0.8, linestyle="--", label="IC = 0.03 target")

    ic_mean = result.metrics.ic_mean
    icir = result.metrics.icir
    ax.set_title(
        f"Information Coefficient Series  |  Mean IC={ic_mean:.3f}  ICIR={icir:.2f}",
        fontsize=_TITLE_SIZE,
        pad=12,
    )
    ax.set_ylabel("Spearman IC", fontsize=_LABEL_SIZE)
    ax.legend(fontsize=_LABEL_SIZE)
    ax.grid(alpha=_GRID_ALPHA)
    ax.tick_params(labelsize=_TICK_SIZE)

    _add_footer(fig, result)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_stress_periods(pdf: PdfPages, result: BacktestResult) -> None:
    """Render stress-period bar chart (Sharpe + max drawdown)."""
    sp = result.stress_periods
    if not sp:
        return

    periods = list(sp.keys())
    sharpes = [sp[p].sharpe_ratio for p in periods]
    drawdowns = [sp[p].max_drawdown * 100 for p in periods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(_FIG_W, _FIG_H))

    # Sharpe
    colors_s = [_POSITIVE_COLOR if s >= 0 else _NEGATIVE_COLOR for s in sharpes]
    ax1.barh(periods, sharpes, color=colors_s, alpha=0.80, edgecolor="white")
    ax1.axvline(0, color="#555", linewidth=0.8)
    ax1.set_title("Sharpe Ratio by Stress Period", fontsize=_TITLE_SIZE - 1)
    ax1.set_xlabel("Annualised Sharpe", fontsize=_LABEL_SIZE)
    ax1.tick_params(labelsize=_TICK_SIZE)
    ax1.grid(alpha=_GRID_ALPHA, axis="x")
    for i, v in enumerate(sharpes):
        ax1.text(
            v + 0.02,
            i,
            f"{v:.2f}",
            va="center",
            fontsize=9,
            color=_POSITIVE_COLOR if v >= 0 else _NEGATIVE_COLOR,
        )

    # Max drawdown
    colors_d = [_NEGATIVE_COLOR] * len(drawdowns)
    ax2.barh(periods, drawdowns, color=colors_d, alpha=0.80, edgecolor="white")
    ax2.axvline(0, color="#555", linewidth=0.8)
    ax2.set_title("Max Drawdown by Stress Period", fontsize=_TITLE_SIZE - 1)
    ax2.set_xlabel("Max Drawdown (%)", fontsize=_LABEL_SIZE)
    ax2.tick_params(labelsize=_TICK_SIZE)
    ax2.grid(alpha=_GRID_ALPHA, axis="x")
    for i, v in enumerate(drawdowns):
        ax2.text(v - 0.5, i, f"{v:.1f}%", va="center", ha="right", fontsize=9, color="white")

    fig.suptitle("Performance in Stress Periods", fontsize=_TITLE_SIZE + 1, fontweight="bold")
    _add_footer(fig, result)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _metric_rows(m: PerformanceMetrics) -> list[tuple[str, str, str]]:
    """Build (label, formatted_value, colour) rows for the summary table."""

    def pct(v: float) -> str:
        return f"{v:.2%}"

    def x2(v: float) -> str:
        return f"{v:.2f}×"

    def f4(v: float) -> str:
        return f"{v:.4f}"

    def f2(v: float) -> str:
        return f"{v:.2f}"

    pos = _POSITIVE_COLOR
    neg = _NEGATIVE_COLOR
    neu = "#222"

    def colour_sharpe(v: float) -> str:
        return pos if v >= 1.0 else (neg if v < 0 else neu)

    def colour_pct(v: float, good_positive: bool = True) -> str:
        if good_positive:
            return pos if v > 0 else neg
        return neg if v > 0 else pos

    return [
        ("Total Return", pct(m.total_return), colour_pct(m.total_return)),
        ("CAGR", pct(m.cagr), colour_pct(m.cagr)),
        ("Annualised Vol", pct(m.annualized_vol), neu),
        ("Sharpe Ratio", f2(m.sharpe_ratio), colour_sharpe(m.sharpe_ratio)),
        ("Sortino Ratio", f2(m.sortino_ratio), colour_sharpe(m.sortino_ratio)),
        ("Calmar Ratio", f2(m.calmar_ratio), colour_sharpe(m.calmar_ratio)),
        ("Max Drawdown", pct(m.max_drawdown), neg),
        ("Max DD Duration", f"{m.max_drawdown_duration_days}d", neu),
        ("VaR 95% (1d)", pct(m.var_95_1d), neg),
        ("CVaR 95% (1d)", pct(m.cvar_95_1d), neg),
        ("Benchmark Return", pct(m.benchmark_return), neu),
        ("Alpha (ann.)", pct(m.alpha), colour_pct(m.alpha)),
        ("Beta", f2(m.beta), neu),
        ("Tracking Error", pct(m.tracking_error), neu),
        ("Information Ratio", f2(m.information_ratio), colour_sharpe(m.information_ratio)),
        ("IC Mean", f4(m.ic_mean), colour_pct(m.ic_mean)),
        ("ICIR", f2(m.icir), colour_pct(m.icir)),
        ("Hit Rate", pct(m.hit_rate), colour_pct(m.hit_rate, True)),
        ("Annual Turnover", pct(m.annual_turnover), neu),
        ("Total Costs (bps)", f"{m.total_costs_bps:.1f}", neg),
        ("Gross vs Net Sharpe", f2(m.gross_vs_net_sharpe), colour_pct(-m.gross_vs_net_sharpe)),
        ("Sharpe t-stat", f2(m.sharpe_t_stat), colour_pct(m.sharpe_t_stat - 2.0)),
        ("IC t-stat", f2(m.ic_t_stat), colour_pct(m.ic_t_stat - 2.0)),
    ]


def _add_footer(fig: Figure, result: BacktestResult) -> None:
    """Add a footer line with period and generation timestamp."""
    fig.text(
        0.5,
        0.01,
        f"QuantFlow Backtest  |  "
        f"{result.config.start_date} → {result.config.end_date}  |  "
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#888",
    )
