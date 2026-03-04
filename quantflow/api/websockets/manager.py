"""WebSocket connection manager.

Maintains a mapping of channel → set of active WebSocket connections.
Provides broadcast, unicast, and graceful disconnect helpers.

Channels follow the pattern ``signals:{SYMBOL}`` or ``market``.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import defaultdict
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from quantflow.config.logging import get_logger

logger = get_logger(__name__)


class ConnectionManager:
    """Manage active WebSocket connections grouped by channel.

    Thread-safe for use within a single asyncio event loop (FastAPI).

    Attributes:
        _connections: Map of channel → set of WebSocket connections.
        _lock: Async lock protecting the connections dict.
    """

    def __init__(self) -> None:
        self._connections: dict[str, set[WebSocket]] = defaultdict(set)
        self._lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self, websocket: WebSocket, channel: str) -> None:
        """Accept and register a new connection on a channel.

        Args:
            websocket: Incoming WebSocket connection.
            channel: Channel identifier (e.g. ``"signals:AAPL"``).
        """
        await websocket.accept()
        async with self._lock:
            self._connections[channel].add(websocket)
        logger.info("WebSocket connected", channel=channel, n=self._channel_size(channel))

    async def disconnect(self, websocket: WebSocket, channel: str) -> None:
        """Remove a connection from a channel (silent if not found).

        Args:
            websocket: WebSocket to remove.
            channel: Channel identifier.
        """
        async with self._lock:
            self._connections[channel].discard(websocket)
            if not self._connections[channel]:
                del self._connections[channel]
        logger.info("WebSocket disconnected", channel=channel)

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------

    async def broadcast(self, channel: str, message: Any) -> None:
        """Broadcast a JSON-serialisable message to all channel subscribers.

        Stale connections are silently removed.

        Args:
            channel: Channel identifier.
            message: JSON-serialisable payload.
        """
        async with self._lock:
            subscribers = set(self._connections.get(channel, set()))

        dead: list[WebSocket] = []
        for ws in subscribers:
            try:
                await ws.send_json(message)
            except (WebSocketDisconnect, RuntimeError):
                dead.append(ws)
            except Exception as exc:
                logger.warning("WebSocket send failed", channel=channel, error=str(exc))
                dead.append(ws)

        if dead:
            async with self._lock:
                for ws in dead:
                    self._connections[channel].discard(ws)

    async def send_personal(self, websocket: WebSocket, message: Any) -> None:
        """Send a message to a single WebSocket connection.

        Args:
            websocket: Target connection.
            message: JSON-serialisable payload.
        """
        with contextlib.suppress(WebSocketDisconnect, RuntimeError):
            await websocket.send_json(message)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def channel_count(self) -> dict[str, int]:
        """Return the subscriber count per channel.

        Returns:
            Dict mapping channel → subscriber count.
        """
        return {ch: len(subs) for ch, subs in self._connections.items()}

    def total_connections(self) -> int:
        """Return the total number of active WebSocket connections."""
        return sum(len(subs) for subs in self._connections.values())

    def _channel_size(self, channel: str) -> int:
        return len(self._connections.get(channel, set()))


# Global singleton used across all WebSocket endpoints
manager = ConnectionManager()
