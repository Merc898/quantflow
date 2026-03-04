"""WebSocket endpoints for real-time signal and market streaming.

Endpoints:
  WS /ws/signals/{symbol}   — stream FinalRecommendation updates
  WS /ws/market             — stream market regime + macro summary
  WS /ws/portfolio/{id}     — stream portfolio P&L (placeholder)

Authentication: Bearer token passed as a ``token`` query parameter
(browsers cannot set custom headers for WS upgrades).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status
from jose import JWTError

from quantflow.api.auth.jwt import decode_token
from quantflow.api.websockets.manager import manager
from quantflow.config.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["websockets"])

# How often to push updates (seconds)
_SIGNAL_PUSH_INTERVAL = 60
_MARKET_PUSH_INTERVAL = 300


# ---------------------------------------------------------------------------
# Authentication helper
# ---------------------------------------------------------------------------


async def _authenticate_ws(websocket: WebSocket, token: str | None) -> tuple[str, str] | None:
    """Validate JWT from query param and return (user_id, tier).

    Closes the WebSocket with code 1008 (Policy Violation) if invalid.

    Args:
        websocket: Incoming WebSocket.
        token: Bearer token from query parameter.

    Returns:
        Tuple of (user_id, tier) on success, or ``None`` on failure.
    """
    if not token:
        await websocket.close(code=1008, reason="Authentication required.")
        return None
    try:
        payload = decode_token(token)
        return payload.sub, payload.tier
    except (JWTError, Exception):
        await websocket.close(code=1008, reason="Invalid or expired token.")
        return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.websocket("/signals/{symbol}")
async def ws_signals(
    websocket: WebSocket,
    symbol: str,
    token: str | None = Query(default=None, description="Bearer JWT token"),
) -> None:
    """Stream real-time signal updates for a symbol.

    Sends a JSON message every ``_SIGNAL_PUSH_INTERVAL`` seconds containing
    the latest recommendation for the symbol.  Clients can also send
    ``{"action": "ping"}`` to receive a ``{"action": "pong"}`` keepalive.

    Message format::

        {
            "type": "signal",
            "symbol": "AAPL",
            "recommendation": "BUY",
            "signal_strength": 0.42,
            "confidence": 0.71,
            "timestamp": "2026-01-01T12:00:00Z"
        }
    """
    auth = await _authenticate_ws(websocket, token)
    if auth is None:
        return

    user_id, tier = auth
    symbol = symbol.upper()
    channel = f"signals:{symbol}"

    await manager.connect(websocket, channel)
    logger.info("Signal WS connected", symbol=symbol, user_id=user_id)

    try:
        # Send initial welcome message
        await manager.send_personal(websocket, {
            "type": "connected",
            "channel": channel,
            "symbol": symbol,
            "message": f"Subscribed to {symbol} signal stream.",
        })

        # Keep alive loop: push updates and handle incoming messages
        push_task = asyncio.create_task(_push_signal_loop(websocket, symbol, channel))

        try:
            while True:
                # Wait for client messages (ping/unsubscribe) with timeout
                data = await asyncio.wait_for(websocket.receive_json(), timeout=_SIGNAL_PUSH_INTERVAL)
                action = data.get("action", "")
                if action == "ping":
                    await manager.send_personal(websocket, {"action": "pong"})
        except asyncio.TimeoutError:
            # No client message received — loop continues (push_task is still running)
            pass
        except WebSocketDisconnect:
            pass
        finally:
            push_task.cancel()

    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(websocket, channel)
        logger.info("Signal WS disconnected", symbol=symbol, user_id=user_id)


async def _push_signal_loop(
    websocket: WebSocket,
    symbol: str,
    channel: str,
) -> None:
    """Background task: push signal updates every interval.

    Args:
        websocket: Client WebSocket connection.
        symbol: Symbol to stream.
        channel: Channel name.
    """
    while True:
        await asyncio.sleep(_SIGNAL_PUSH_INTERVAL)
        try:
            payload = _build_signal_payload(symbol)
            await manager.send_personal(websocket, payload)
        except Exception as exc:
            logger.warning("Signal push failed", symbol=symbol, error=str(exc))
            break


def _build_signal_payload(symbol: str) -> dict[str, Any]:
    """Build a lightweight signal payload without DB access.

    In production this would query the latest cached recommendation.
    Here we return a heartbeat-style update with metadata.

    Args:
        symbol: Ticker symbol.

    Returns:
        JSON-serialisable dict.
    """
    return {
        "type": "signal_update",
        "symbol": symbol,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "message": f"Signal refresh for {symbol}. Use REST endpoint for full data.",
    }


@router.websocket("/market")
async def ws_market(
    websocket: WebSocket,
    token: str | None = Query(default=None, description="Bearer JWT token"),
) -> None:
    """Stream macro market regime updates.

    Sends a regime snapshot every ``_MARKET_PUSH_INTERVAL`` seconds.

    Message format::

        {
            "type": "market_update",
            "regime": "BULL_LOW_VOL",
            "macro_sentiment": 0.25,
            "timestamp": "2026-01-01T12:00:00Z"
        }
    """
    auth = await _authenticate_ws(websocket, token)
    if auth is None:
        return

    user_id, tier = auth
    channel = "market"
    await manager.connect(websocket, channel)
    logger.info("Market WS connected", user_id=user_id)

    try:
        await manager.send_personal(websocket, {
            "type": "connected",
            "channel": channel,
            "message": "Subscribed to market regime stream.",
        })

        while True:
            await asyncio.sleep(_MARKET_PUSH_INTERVAL)
            payload = {
                "type": "market_update",
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "message": "Market regime refresh. See /api/v1/intelligence/macro for full report.",
            }
            await manager.send_personal(websocket, payload)

    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(websocket, channel)
        logger.info("Market WS disconnected", user_id=user_id)


@router.websocket("/portfolio/{portfolio_id}")
async def ws_portfolio(
    websocket: WebSocket,
    portfolio_id: str,
    token: str | None = Query(default=None, description="Bearer JWT token"),
) -> None:
    """Stream portfolio P&L and risk updates (placeholder).

    Full implementation requires a live positions store (future phase).
    """
    auth = await _authenticate_ws(websocket, token)
    if auth is None:
        return

    channel = f"portfolio:{portfolio_id}"
    await manager.connect(websocket, channel)

    try:
        await manager.send_personal(websocket, {
            "type": "connected",
            "channel": channel,
            "message": "Portfolio streaming will be available in a future release.",
        })
        # Keep connection alive
        while True:
            await asyncio.sleep(30)
            await manager.send_personal(websocket, {"type": "heartbeat"})
    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(websocket, channel)
