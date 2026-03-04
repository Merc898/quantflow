"""Portfolio optimization and risk endpoints.

All endpoints require Premium subscription.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from quantflow.api.auth.dependencies import CurrentUser, require_tier
from quantflow.config.constants import TIER_PREMIUM
from quantflow.config.logging import get_logger
from quantflow.portfolio.hrp import HRPOptimizer
from quantflow.portfolio.optimizer import MVOOptimizer
from quantflow.risk.stress_tester import StressTester
from quantflow.risk.var_es import RiskCalculator

logger = get_logger(__name__)

router = APIRouter(tags=["portfolio"], dependencies=[require_tier(TIER_PREMIUM)])

_MIN_SYMBOLS = 2
_MAX_SYMBOLS = 50
_HISTORY_DAYS = "2y"


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class OptimizeRequest(BaseModel):
    """Portfolio optimisation request.

    Attributes:
        symbols: List of ticker symbols to include.
        method: Optimisation method.
        covariance_method: Covariance estimator for MVO.
        signal_views: Optional dict mapping symbol → signal ([-1,1]) for
            Black-Litterman.
    """

    symbols: list[str] = Field(..., min_length=_MIN_SYMBOLS, max_length=_MAX_SYMBOLS)
    method: str = Field(default="mvo", pattern="^(mvo|hrp|black_litterman)$")
    covariance_method: str = Field(
        default="ledoit_wolf",
        pattern="^(sample|ledoit_wolf|oas|rmt)$",
    )
    signal_views: dict[str, float] | None = None


class WeightEntry(BaseModel):
    """Single asset weight in an optimised portfolio."""

    symbol: str
    weight: float
    expected_return: float | None = None


class OptimizeResponse(BaseModel):
    """Optimised portfolio response."""

    method: str
    weights: list[WeightEntry]
    expected_portfolio_return: float
    expected_portfolio_vol: float
    sharpe_ratio: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class StressTestRequest(BaseModel):
    """Stress test request."""

    symbols: list[str] = Field(..., min_length=1, max_length=_MAX_SYMBOLS)
    weights: dict[str, float]
    scenarios: list[str] = Field(
        default=["covid_crash_2020", "gfc_2008"],
        max_length=6,
    )


class StressTestResponse(BaseModel):
    """Stress test response."""

    scenarios: dict[str, Any]
    monte_carlo: dict[str, Any]


class RiskReportResponse(BaseModel):
    """Risk report response."""

    symbol: str
    var_95_1d: float
    var_99_1d: float
    es_95_1d: float
    es_99_1d: float
    method: str
    n_observations: int
    kupiec_pvalue: float | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fetch_returns(symbols: list[str]) -> pd.DataFrame:
    """Download price data and compute log returns.

    Args:
        symbols: Ticker symbols.

    Returns:
        DataFrame of daily log returns (columns = symbols).

    Raises:
        HTTPException 404: No price data available.
        HTTPException 400: Fewer than 2 valid symbols.
    """
    tickers = " ".join(symbols)
    data = yf.download(tickers, period=_HISTORY_DAYS, progress=False, auto_adjust=True)
    if data.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Could not fetch price data for the requested symbols.",
        )

    # Handle single vs multi-symbol download
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data[["Close"]] if len(symbols) == 1 else data

    close.columns = [str(c).upper() for c in close.columns]
    close = close.dropna(how="all")

    # Keep only symbols with sufficient data
    valid = close.columns[close.notna().sum() >= 63].tolist()
    if len(valid) < _MIN_SYMBOLS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Need at least {_MIN_SYMBOLS} symbols with ≥63 days of data.",
        )

    returns = np.log(close[valid] / close[valid].shift(1)).dropna()
    return returns


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/optimize",
    response_model=OptimizeResponse,
    summary="Run portfolio optimisation (MVO / HRP / Black-Litterman)",
)
async def optimize_portfolio(
    body: OptimizeRequest,
    user: CurrentUser,
) -> OptimizeResponse:
    """Compute optimal portfolio weights.

    Fetches 2 years of price data, computes log returns, then runs the
    selected optimisation method.
    """
    symbols = [s.upper() for s in body.symbols]
    returns = _fetch_returns(symbols)
    method = body.method.lower()

    if method == "hrp":
        opt = HRPOptimizer()
        result = opt.optimize(returns)
    elif method == "black_litterman":
        from quantflow.portfolio.black_litterman import BlackLittermanOptimizer

        mkt_weights = {s: 1.0 / len(returns.columns) for s in returns.columns}
        views = body.signal_views or {}
        opt_bl = BlackLittermanOptimizer()
        result = opt_bl.optimize(returns, mkt_weights, views)
    else:
        # Default: MVO
        opt_mvo = MVOOptimizer()
        result = opt_mvo.optimize(
            returns,
            covariance_method=body.covariance_method,  # type: ignore[arg-type]
        )

    weights_list = [WeightEntry(symbol=sym, weight=w) for sym, w in result.weights.items()]

    return OptimizeResponse(
        method=method,
        weights=weights_list,
        expected_portfolio_return=result.expected_return,
        expected_portfolio_vol=result.expected_vol,
        sharpe_ratio=result.sharpe_ratio,
        metadata={"covariance_method": result.covariance_method},
    )


@router.get(
    "/efficient-frontier",
    summary="Compute efficient frontier",
)
async def efficient_frontier(
    symbols: str = Query(..., description="Comma-separated ticker symbols"),
    n_points: int = Query(default=30, ge=10, le=100),
) -> dict[str, Any]:
    """Compute the mean-variance efficient frontier.

    Returns a list of (return, volatility, weights) triples spanning
    the efficient frontier.
    """
    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    if len(symbol_list) < _MIN_SYMBOLS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"At least {_MIN_SYMBOLS} symbols required.",
        )

    returns = _fetch_returns(symbol_list)
    opt = MVOOptimizer()
    result = opt.optimize(returns, compute_frontier=True)

    frontier_points = [
        {
            "expected_return": p.expected_return,
            "expected_vol": p.expected_vol,
            "sharpe": p.sharpe_ratio,
        }
        for p in result.efficient_frontier[:n_points]
    ]

    return {
        "n_points": len(frontier_points),
        "frontier": frontier_points,
        "optimal": {
            "expected_return": result.expected_return,
            "expected_vol": result.expected_vol,
            "sharpe": result.sharpe_ratio,
        },
    }


@router.post(
    "/stress-test",
    response_model=StressTestResponse,
    summary="Run portfolio stress tests",
)
async def stress_test(
    body: StressTestRequest,
    user: CurrentUser,
) -> StressTestResponse:
    """Run historical and Monte Carlo stress tests on a portfolio.

    Fetches price data for all symbols, then runs:
    - Historical scenario replay for each requested scenario.
    - Monte Carlo simulation under stress conditions.
    """
    symbols = [s.upper() for s in body.symbols]
    returns = _fetch_returns(symbols)

    # Align provided weights to available symbols
    available = set(returns.columns)
    weights = {s: w for s, w in body.weights.items() if s in available}
    if not weights:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No weight symbols overlap with available price data.",
        )

    # Normalise weights to sum to 1
    total_w = sum(weights.values())
    weights = {s: w / total_w for s, w in weights.items()}

    tester = StressTester()

    scenario_results: dict[str, Any] = {}
    for scenario in body.scenarios:
        try:
            res = tester.run_historical_scenario(weights, returns, scenario)
            scenario_results[scenario] = {
                "portfolio_return": res.portfolio_return,
                "worst_asset_return": res.worst_asset_return,
                "best_asset_return": res.best_asset_return,
                "scenario": res.scenario_name,
            }
        except Exception as exc:
            scenario_results[scenario] = {"error": str(exc)}

    # Monte Carlo
    mc = tester.run_monte_carlo_stress(returns, weights)
    mc_result = {
        "p5": mc.p5,
        "p1": mc.p1,
        "expected_shortfall_999": mc.expected_shortfall_999,
        "mean": mc.mean,
        "n_scenarios": mc.n_scenarios,
    }

    return StressTestResponse(scenarios=scenario_results, monte_carlo=mc_result)


@router.get(
    "/risk-report",
    summary="Compute VaR / ES risk report for a symbol",
)
async def risk_report(
    symbol: str = Query(..., description="Ticker symbol"),
    confidence: float = Query(default=0.99, ge=0.90, le=0.999),
    method: str = Query(default="historical", pattern="^(historical|parametric|monte_carlo)$"),
) -> RiskReportResponse:
    """Compute 1-day VaR and ES using three methods.

    Returns VaR/ES at both 95% and 99% confidence along with Kupiec backtest.
    """
    data = yf.download(symbol.upper(), period="2y", progress=False, auto_adjust=True)
    if data.empty or len(data) < 63:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Insufficient data for '{symbol}'.",
        )

    close = data["Close"].squeeze() if isinstance(data.columns, pd.MultiIndex) else data["Close"]

    returns = np.log(close / close.shift(1)).dropna()

    calc = RiskCalculator()
    report_99 = calc.compute_var_es(returns, confidence=0.99, method=method)  # type: ignore[arg-type]
    report_95 = calc.compute_var_es(returns, confidence=0.95, method=method)  # type: ignore[arg-type]

    return RiskReportResponse(
        symbol=symbol.upper(),
        var_95_1d=report_95.var,
        var_99_1d=report_99.var,
        es_95_1d=report_95.es,
        es_99_1d=report_99.es,
        method=method,
        n_observations=report_99.n_observations,
        kupiec_pvalue=report_99.kupiec_pvalue,
    )
