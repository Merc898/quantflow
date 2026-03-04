"""Request/response logging middleware with structured JSON output.

Logs each request with:
- method, path, status_code, duration_ms
- user_id (from JWT if present)
- request_id (UUID generated per request)
"""

from __future__ import annotations

import time
import uuid
from typing import Awaitable, Callable

import structlog
from fastapi import Request, Response
from jose import JWTError
from starlette.middleware.base import BaseHTTPMiddleware

from quantflow.api.auth.jwt import decode_token
from quantflow.config.logging import get_logger

logger = get_logger(__name__)

# Paths to skip verbose logging
_QUIET_PATHS = {"/health", "/api/docs", "/api/redoc", "/openapi.json"}


def _extract_user_id(request: Request) -> str | None:
    """Extract user ID from JWT without raising exceptions.

    Args:
        request: Incoming request.

    Returns:
        User ID string or ``None``.
    """
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    try:
        payload = decode_token(auth[7:])
        return payload.sub
    except (JWTError, Exception):
        return None


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all HTTP requests with timing and user context.

    Args:
        app: ASGI application.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = str(uuid.uuid4())[:8]
        user_id = _extract_user_id(request)

        # Bind context for the duration of this request
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            user_id=user_id,
        )

        start = time.perf_counter()
        try:
            response = await call_next(request)
            duration_ms = round((time.perf_counter() - start) * 1000, 2)

            if request.url.path not in _QUIET_PATHS:
                log_fn = logger.warning if response.status_code >= 400 else logger.info
                log_fn(
                    "HTTP request",
                    method=request.method,
                    path=request.url.path,
                    status=response.status_code,
                    duration_ms=duration_ms,
                )

            response.headers["X-Request-ID"] = request_id
            return response

        except Exception as exc:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.error(
                "Unhandled exception",
                method=request.method,
                path=request.url.path,
                error=str(exc),
                duration_ms=duration_ms,
            )
            raise
        finally:
            structlog.contextvars.clear_contextvars()
