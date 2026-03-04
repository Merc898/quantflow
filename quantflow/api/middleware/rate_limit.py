"""Per-tier rate limiting middleware using Redis sliding-window counters.

Each authenticated request is counted against the user's daily quota.
Unauthenticated requests use the IP address as the key (free-tier limits).

Redis key:  ``rate_limit:{user_id_or_ip}:{YYYY-MM-DD}``
TTL:        86 400 seconds (auto-expires next day)

When Redis is unavailable the middleware fails open (passes requests
through) and logs a warning — never block the API due to a Redis outage.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Awaitable, Callable

import redis.asyncio as aioredis
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from jose import JWTError
from starlette.middleware.base import BaseHTTPMiddleware

from quantflow.api.auth.jwt import decode_token
from quantflow.config.constants import (
    RATE_LIMIT_FREE,
    RATE_LIMIT_INSTITUTIONAL,
    RATE_LIMIT_PREMIUM,
)
from quantflow.config.logging import get_logger
from quantflow.config.settings import settings

logger = get_logger(__name__)

_TIER_LIMITS: dict[str, int] = {
    "free": RATE_LIMIT_FREE,
    "premium": RATE_LIMIT_PREMIUM,
    "institutional": RATE_LIMIT_INSTITUTIONAL,
}

# Paths excluded from rate limiting
_EXCLUDED_PATHS = {"/health", "/api/docs", "/api/redoc", "/openapi.json"}


def _get_redis_key(identifier: str) -> str:
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    return f"rate_limit:{identifier}:{today}"


def _extract_user_identity(request: Request) -> tuple[str, str]:
    """Extract (identifier, tier) from the request for rate-limit bucketing.

    Tries to decode the JWT without raising, falls back to IP address.

    Args:
        request: Incoming FastAPI request.

    Returns:
        Tuple of (identifier_string, tier_string).
    """
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        try:
            payload = decode_token(token)
            return f"user:{payload.sub}", payload.tier
        except (JWTError, Exception):
            pass

    api_key = request.headers.get("X-API-Key", "")
    if api_key:
        return f"apikey:{api_key[:16]}", "premium"

    # Fall back to IP
    client_ip = request.client.host if request.client else "unknown"
    return f"ip:{client_ip}", "free"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter backed by Redis.

    Args:
        app: ASGI application.
        redis_url: Redis connection URL.
    """

    def __init__(self, app: object, redis_url: str | None = None) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._redis_url = redis_url or settings.redis_url
        self._redis: aioredis.Redis | None = None

    async def _get_redis(self) -> aioredis.Redis | None:
        """Lazily create Redis connection."""
        if self._redis is None:
            try:
                self._redis = aioredis.from_url(
                    self._redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_connect_timeout=1,
                    socket_timeout=1,
                )
                await self._redis.ping()
            except Exception as exc:
                logger.warning("Redis unavailable for rate limiting", error=str(exc))
                self._redis = None
        return self._redis

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        # Skip excluded paths
        if request.url.path in _EXCLUDED_PATHS:
            return await call_next(request)

        identifier, tier = _extract_user_identity(request)
        limit = _TIER_LIMITS.get(tier, RATE_LIMIT_FREE)

        redis = await self._get_redis()
        if redis is not None:
            key = _get_redis_key(identifier)
            try:
                current = await redis.incr(key)
                if current == 1:
                    await redis.expire(key, 86_400)

                remaining = max(0, limit - current)
                reset_ts = int(time.time()) + 86_400

                if current > limit:
                    logger.warning(
                        "Rate limit exceeded",
                        identifier=identifier,
                        tier=tier,
                        count=current,
                        limit=limit,
                    )
                    return JSONResponse(
                        status_code=429,
                        content={
                            "detail": "Rate limit exceeded. Please slow down or upgrade your plan.",
                            "limit": limit,
                            "reset_at": reset_ts,
                        },
                        headers={
                            "X-RateLimit-Limit": str(limit),
                            "X-RateLimit-Remaining": "0",
                            "X-RateLimit-Reset": str(reset_ts),
                            "Retry-After": str(reset_ts - int(time.time())),
                        },
                    )

                response = await call_next(request)
                response.headers["X-RateLimit-Limit"] = str(limit)
                response.headers["X-RateLimit-Remaining"] = str(remaining)
                response.headers["X-RateLimit-Reset"] = str(reset_ts)
                return response

            except Exception as exc:
                logger.warning("Rate limit check failed", error=str(exc))
                # Fail open
                return await call_next(request)
        else:
            return await call_next(request)
