"""Signal endpoints: get, history, explain, batch, screener.

All endpoints require authentication.  History > 30 days and SHAP
explainability require Premium tier.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import desc, select

from quantflow.api.auth.dependencies import (
    CurrentUser,
    DbDep,
    check_symbol_limit,
    require_tier,
)
from quantflow.config.constants import TIER_FREE, TIER_PREMIUM
from quantflow.config.logging import get_logger
from quantflow.db.models import Recommendation

if TYPE_CHECKING:
    from quantflow.signals.recommendation import FinalRecommendation

logger = get_logger(__name__)

router = APIRouter(tags=["signals"])


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class SignalResponse(BaseModel):
    """Lightweight signal response for screener/list endpoints."""

    symbol: str
    recommendation: str
    signal_strength: float
    confidence: float
    suggested_position_size: float
    regime: str | None = None
    timestamp: datetime
    risk_warnings: list[str] = Field(default_factory=list)


class HistoryResponse(BaseModel):
    """Paginated signal history response."""

    symbol: str
    count: int
    signals: list[SignalResponse]


class ScreenerResponse(BaseModel):
    """Screener result: ranked list of signals."""

    count: int
    ranked: list[SignalResponse]
    generated_at: datetime


class BatchRequest(BaseModel):
    """Batch signal request body."""

    symbols: list[str] = Field(..., min_length=1, max_length=50)


class BatchResponse(BaseModel):
    """Batch signal response."""

    requested: int
    signals: list[SignalResponse]
    errors: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rec_to_db(rec: FinalRecommendation) -> Recommendation:
    """Convert a FinalRecommendation to an ORM Recommendation row."""
    return Recommendation(
        time=rec.timestamp,
        symbol=rec.symbol,
        recommendation=rec.recommendation,
        signal_strength=rec.signal_strength,
        confidence=rec.confidence,
        position_size=rec.suggested_position_size,
        expected_return_21d=rec.expected_return_21d,
        expected_vol_21d=rec.expected_vol_21d,
        var_95_1d=rec.var_95_1d,
        max_drawdown_estimate=rec.max_drawdown_estimate,
        model_contributions=rec.model_contributions,
        top_bullish_factors=rec.top_bullish_factors,
        top_bearish_factors=rec.top_bearish_factors,
        regime=rec.regime.model_dump() if rec.regime else {},
        rationale=rec.rationale,
        risk_warnings=rec.risk_warnings,
        data_quality_score=rec.data_quality_score,
        models_used=rec.models_used,
        models_available=rec.models_available,
    )


def _row_to_response(row: Recommendation) -> SignalResponse:
    """Convert ORM row to SignalResponse."""
    regime_label: str | None = None
    if isinstance(row.regime, dict) and row.regime:
        regime_label = row.regime.get("overall_regime")
    return SignalResponse(
        symbol=row.symbol,
        recommendation=row.recommendation,
        signal_strength=row.signal_strength,
        confidence=row.confidence,
        suggested_position_size=row.position_size,
        regime=regime_label,
        timestamp=row.time,
        risk_warnings=row.risk_warnings or [],
    )


async def _generate_signal(symbol: str) -> FinalRecommendation:
    """Generate a fresh signal for a symbol using minimal models.

    This is a lightweight synchronous approximation used when no cached
    recommendation is available.  Full async signal generation with all
    50+ models runs via Celery workers.

    Args:
        symbol: Ticker symbol (e.g. "AAPL").

    Returns:
        :class:`FinalRecommendation`.

    Raises:
        HTTPException 404: Symbol not found or insufficient data.
    """
    from quantflow.signals.aggregator import EnsembleAggregator
    from quantflow.signals.recommendation import RecommendationEngine
    from quantflow.signals.regime_detector import RegimeDetector

    try:
        data = yf.download(symbol, period="2y", progress=False, auto_adjust=True)
        if data.empty or len(data) < 63:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Insufficient market data for symbol '{symbol}'.",
            )

        # Flatten MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0].lower() for col in data.columns]
        else:
            data.columns = [c.lower() for c in data.columns]

        returns = np.log(data["close"] / data["close"].shift(1)).dropna()

        # Regime
        detector = RegimeDetector()
        regime = detector.detect(returns, prices=data["close"])

        # Minimal aggregation (no model outputs — data-quality signal only)
        aggregator = EnsembleAggregator()
        agg = aggregator.aggregate(
            model_outputs=[], realized_vol=float(returns.iloc[-21:].std() * np.sqrt(252))
        )

        # Recommendation
        engine = RecommendationEngine(models_available=1)
        rec = engine.generate(
            symbol=symbol,
            composite_signal=agg.risk_scaled_signal,
            confidence=0.40,
            regime=regime,
            returns=returns,
        )
        return rec

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Signal generation failed", symbol=symbol, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Signal generation failed for '{symbol}': {exc}",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/{symbol}",
    response_model=SignalResponse,
    summary="Get latest recommendation for a symbol",
)
async def get_signal(
    symbol: str,
    user: CurrentUser,
    db: DbDep,
) -> SignalResponse:
    """Return the most recent recommendation for *symbol*.

    If no cached recommendation exists, generates one on-the-fly
    using lightweight models (full ensemble runs via background workers).
    """
    symbol = symbol.upper()
    result = await db.execute(
        select(Recommendation)
        .where(Recommendation.symbol == symbol)
        .order_by(desc(Recommendation.time))
        .limit(1)
    )
    row = result.scalar_one_or_none()
    if row is not None:
        return _row_to_response(row)

    # Generate fresh signal
    rec = await _generate_signal(symbol)
    db_row = _rec_to_db(rec)
    db.add(db_row)
    return SignalResponse(
        symbol=rec.symbol,
        recommendation=rec.recommendation,
        signal_strength=rec.signal_strength,
        confidence=rec.confidence,
        suggested_position_size=rec.suggested_position_size,
        regime=rec.regime.overall_regime if rec.regime else None,
        timestamp=rec.timestamp,
        risk_warnings=rec.risk_warnings,
    )


@router.get(
    "/{symbol}/history",
    response_model=HistoryResponse,
    summary="Get historical signals for a symbol",
)
async def get_signal_history(
    symbol: str,
    user: CurrentUser,
    db: DbDep,
    days: int = Query(default=30, ge=1, le=1095),
    limit: int = Query(default=90, ge=1, le=500),
) -> HistoryResponse:
    """Return historical recommendation time series.

    Free tier: max 30 days history.
    Premium+: up to 3 years.
    """
    symbol = symbol.upper()
    if user.subscription_tier == TIER_FREE and days > 30:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Free tier is limited to 30 days of history. Upgrade to Premium for full history.",
        )

    since = datetime.now(tz=UTC) - timedelta(days=days)
    result = await db.execute(
        select(Recommendation)
        .where(Recommendation.symbol == symbol, Recommendation.time >= since)
        .order_by(desc(Recommendation.time))
        .limit(limit)
    )
    rows = result.scalars().all()
    return HistoryResponse(
        symbol=symbol,
        count=len(rows),
        signals=[_row_to_response(r) for r in rows],
    )


@router.get(
    "/{symbol}/explain",
    summary="Get explainability report for latest signal",
    dependencies=[require_tier(TIER_PREMIUM)],
)
async def get_explanation(
    symbol: str,
    user: CurrentUser,
    db: DbDep,
) -> dict[str, Any]:
    """Return the full explainability report for the latest signal.

    Includes: model contributions, confidence decomposition, top features,
    and a narrative rationale (LLM-generated or template).

    Requires Premium subscription.
    """
    symbol = symbol.upper()
    result = await db.execute(
        select(Recommendation)
        .where(Recommendation.symbol == symbol)
        .order_by(desc(Recommendation.time))
        .limit(1)
    )
    row = result.scalar_one_or_none()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No recommendation found for '{symbol}'.",
        )

    return {
        "symbol": row.symbol,
        "timestamp": row.time.isoformat(),
        "recommendation": row.recommendation,
        "signal_strength": row.signal_strength,
        "confidence": row.confidence,
        "rationale": row.rationale or "No narrative available.",
        "model_contributions": row.model_contributions,
        "top_bullish_factors": row.top_bullish_factors,
        "top_bearish_factors": row.top_bearish_factors,
        "regime": row.regime,
        "risk_warnings": row.risk_warnings,
        "data_quality_score": row.data_quality_score,
        "models_used": row.models_used,
        "models_available": row.models_available,
    }


@router.post(
    "/batch",
    response_model=BatchResponse,
    summary="Batch signal generation",
)
async def batch_signals(
    body: BatchRequest,
    user: CurrentUser,
    db: DbDep,
) -> BatchResponse:
    """Generate signals for multiple symbols in one call.

    Free tier: max 5 symbols per batch.
    Premium+: up to 50 symbols.
    """
    symbols = [s.upper() for s in body.symbols]
    check_symbol_limit(symbols, user.subscription_tier)

    signals: list[SignalResponse] = []
    errors: dict[str, str] = {}

    for symbol in symbols:
        try:
            result = await db.execute(
                select(Recommendation)
                .where(Recommendation.symbol == symbol)
                .order_by(desc(Recommendation.time))
                .limit(1)
            )
            row = result.scalar_one_or_none()
            if row is not None:
                signals.append(_row_to_response(row))
            else:
                rec = await _generate_signal(symbol)
                signals.append(
                    SignalResponse(
                        symbol=rec.symbol,
                        recommendation=rec.recommendation,
                        signal_strength=rec.signal_strength,
                        confidence=rec.confidence,
                        suggested_position_size=rec.suggested_position_size,
                        regime=rec.regime.overall_regime if rec.regime else None,
                        timestamp=rec.timestamp,
                        risk_warnings=rec.risk_warnings,
                    )
                )
        except HTTPException as exc:
            errors[symbol] = exc.detail
        except Exception as exc:
            errors[symbol] = str(exc)

    return BatchResponse(requested=len(symbols), signals=signals, errors=errors)


@router.get(
    "/universe/screener",
    response_model=ScreenerResponse,
    summary="Screen universe by signal strength",
    dependencies=[require_tier(TIER_PREMIUM)],
)
async def screener(
    user: CurrentUser,
    db: DbDep,
    min_confidence: float = Query(default=0.50, ge=0.0, le=1.0),
    limit: int = Query(default=50, ge=1, le=200),
) -> ScreenerResponse:
    """Return symbols ranked by signal strength from the database.

    Filters to the most recent recommendation per symbol,
    with confidence ≥ ``min_confidence``.
    Requires Premium subscription.
    """
    from sqlalchemy import func

    # Latest recommendation per symbol via subquery
    subq = (
        select(
            Recommendation.symbol,
            func.max(Recommendation.time).label("max_time"),
        )
        .group_by(Recommendation.symbol)
        .subquery()
    )

    result = await db.execute(
        select(Recommendation)
        .join(
            subq,
            (Recommendation.symbol == subq.c.symbol) & (Recommendation.time == subq.c.max_time),
        )
        .where(Recommendation.confidence >= min_confidence)
        .order_by(desc(Recommendation.signal_strength))
        .limit(limit)
    )
    rows = result.scalars().all()
    return ScreenerResponse(
        count=len(rows),
        ranked=[_row_to_response(r) for r in rows],
        generated_at=datetime.now(tz=UTC),
    )
