"""Prometheus metrics middleware for QuantFlow FastAPI application.

Exposes:
- ``http_requests_total`` — request counter labelled by method, endpoint, status
- ``http_request_duration_seconds`` — request latency histogram
- ``http_requests_in_progress`` — active requests gauge
- ``quantflow_signals_generated_total`` — business metric: signals computed
- ``quantflow_active_websocket_connections`` — live WS connections

The ``/metrics`` endpoint (handled in main.py) returns a text exposition
format compatible with Prometheus scraping.
"""

from __future__ import annotations

import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.routing import Match

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

if _PROMETHEUS_AVAILABLE:
    REQUEST_COUNT = Counter(
        "http_requests_total",
        "Total HTTP request count",
        ["method", "endpoint", "http_status"],
    )

    REQUEST_LATENCY = Histogram(
        "http_request_duration_seconds",
        "HTTP request latency in seconds",
        ["method", "endpoint"],
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )

    REQUESTS_IN_PROGRESS = Gauge(
        "http_requests_in_progress",
        "Number of HTTP requests currently being processed",
        ["method", "endpoint"],
    )

    SIGNALS_GENERATED = Counter(
        "quantflow_signals_generated_total",
        "Total number of trading signals generated",
        ["symbol", "recommendation"],
    )

    ACTIVE_WS_CONNECTIONS = Gauge(
        "quantflow_active_websocket_connections",
        "Number of active WebSocket connections",
    )

    CELERY_TASK_COUNT = Counter(
        "quantflow_celery_tasks_total",
        "Total Celery tasks dispatched",
        ["task_name", "status"],
    )


def _get_route_path(request: Request) -> str:
    """Extract the matched route path template (e.g. ``/api/v1/signals/{symbol}``).

    Falls back to the raw URL path if no route matches.

    Args:
        request: Incoming HTTP request.

    Returns:
        Route path string for use as a Prometheus label.
    """
    for route in request.app.routes:
        match, _ = route.matches(request.scope)
        if match == Match.FULL:
            return route.path  # type: ignore[attr-defined]
    # Truncate long unknown paths to avoid label cardinality explosion
    raw = request.url.path
    return raw if len(raw) <= 80 else raw[:80]


class PrometheusMetricsMiddleware(BaseHTTPMiddleware):
    """Record per-request Prometheus metrics.

    Tracks request count, latency, and in-progress count.  Skips the
    ``/metrics`` endpoint itself to avoid self-referential noise.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process a request, recording timing and status metrics.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware or route handler.

        Returns:
            HTTP response (unchanged).
        """
        if not _PROMETHEUS_AVAILABLE:
            return await call_next(request)

        # Skip the metrics endpoint itself
        if request.url.path == "/metrics":
            return await call_next(request)

        method = request.method
        endpoint = _get_route_path(request)
        start = time.perf_counter()

        REQUESTS_IN_PROGRESS.labels(method=method, endpoint=endpoint).inc()
        try:
            response: Response = await call_next(request)
            status = str(response.status_code)
        except Exception:
            status = "500"
            raise
        finally:
            duration = time.perf_counter() - start
            REQUESTS_IN_PROGRESS.labels(method=method, endpoint=endpoint).dec()
            REQUEST_COUNT.labels(method=method, endpoint=endpoint, http_status=status).inc()
            REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)

        return response


def metrics_response() -> Response:
    """Generate a Prometheus text-format metrics response.

    Returns:
        ``text/plain`` response suitable for ``GET /metrics``.
    """
    if not _PROMETHEUS_AVAILABLE:
        return Response(
            content="# prometheus_client not installed\n",
            media_type="text/plain",
        )
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


def record_signal_generated(symbol: str, recommendation: str) -> None:
    """Increment the signal-generated counter.

    Args:
        symbol: Ticker symbol for which the signal was generated.
        recommendation: Recommendation label (e.g. ``"BUY"``).
    """
    if _PROMETHEUS_AVAILABLE:
        SIGNALS_GENERATED.labels(symbol=symbol, recommendation=recommendation).inc()


def set_active_ws_connections(count: int) -> None:
    """Update the active WebSocket connections gauge.

    Args:
        count: Current number of active connections.
    """
    if _PROMETHEUS_AVAILABLE:
        ACTIVE_WS_CONNECTIONS.set(count)
