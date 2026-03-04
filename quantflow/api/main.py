"""QuantFlow FastAPI application entry point.

Configures:
- Sentry error tracking (via SENTRY_DSN env var)
- CORS middleware (allow all origins in development)
- Request logging middleware (structlog)
- Prometheus metrics middleware
- Per-tier rate limiting middleware (Redis)
- API routers for auth, signals, portfolio, intelligence, subscription
- WebSocket router for real-time streaming
- ``/metrics`` endpoint for Prometheus scraping
- Lifespan context manager for startup/shutdown

Usage::

    uvicorn quantflow.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from quantflow.api.auth.router import router as auth_router
from quantflow.api.middleware.logging import RequestLoggingMiddleware
from quantflow.api.middleware.metrics import PrometheusMetricsMiddleware, metrics_response
from quantflow.api.middleware.rate_limit import RateLimitMiddleware
from quantflow.api.routers.admin import router as admin_router
from quantflow.api.routers.intelligence import router as intelligence_router
from quantflow.api.routers.portfolio import router as portfolio_router
from quantflow.api.routers.signals import router as signals_router
from quantflow.api.routers.subscription import router as subscription_router
from quantflow.api.websockets.router import router as ws_router
from quantflow.config.logging import get_logger, setup_logging
from quantflow.config.settings import settings

setup_logging()
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Sentry initialisation (no-op when SENTRY_DSN is unset)
# ---------------------------------------------------------------------------

_sentry_dsn = os.environ.get("SENTRY_DSN", "")
if _sentry_dsn:
    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

        sentry_sdk.init(
            dsn=_sentry_dsn,
            integrations=[FastApiIntegration(), SqlalchemyIntegration()],
            traces_sample_rate=float(os.environ.get("SENTRY_TRACES_SAMPLE_RATE", "0.1")),
            environment=settings.environment,
            release=os.environ.get("APP_VERSION", "1.0.0"),
            # PII scrubbing
            send_default_pii=False,
        )
        logger.info("Sentry initialised", environment=settings.environment)
    except ImportError:
        logger.warning("sentry-sdk not installed — error tracking disabled")


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: startup → yield → shutdown.

    Startup:
        - Log configuration summary.
        - (Future: warm up model cache, validate DB connection.)

    Shutdown:
        - Graceful cleanup.
    """
    logger.info(
        "QuantFlow API starting",
        environment=settings.environment,
        log_level=settings.log_level,
    )
    yield
    logger.info("QuantFlow API shutting down")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured :class:`FastAPI` instance.
    """
    app = FastAPI(
        title="QuantFlow API",
        description=(
            "Institutional-grade quantitative research and trading signal platform. "
            "Combines 50+ quantitative models, agentic AI intelligence, "
            "and portfolio optimization into a single SaaS API."
        ),
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------
    # CORS
    # ------------------------------------------------------------------
    allowed_origins = (
        ["*"]
        if settings.is_development
        else [
            "https://app.quantflow.io",
            "https://www.quantflow.io",
        ]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Custom middleware (added in reverse order — last added = first run)
    # ------------------------------------------------------------------
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(PrometheusMetricsMiddleware)
    app.add_middleware(RequestLoggingMiddleware)

    # ------------------------------------------------------------------
    # Exception handlers
    # ------------------------------------------------------------------

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            "Unhandled exception",
            path=request.url.path,
            method=request.method,
            error=str(exc),
            exc_info=True,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An internal server error occurred.", "code": "internal_error"},
        )

    # ------------------------------------------------------------------
    # Routers
    # ------------------------------------------------------------------
    app.include_router(auth_router, prefix="/api/v1/auth")
    app.include_router(admin_router, prefix="/api/v1/admin")
    app.include_router(signals_router, prefix="/api/v1/signals")
    app.include_router(portfolio_router, prefix="/api/v1/portfolio")
    app.include_router(intelligence_router, prefix="/api/v1/intelligence")
    app.include_router(subscription_router, prefix="/api/v1/subscription")
    app.include_router(ws_router, prefix="/ws")

    # ------------------------------------------------------------------
    # Utility endpoints
    # ------------------------------------------------------------------

    @app.get("/health", tags=["system"], summary="Health check")
    async def health_check() -> dict[str, str]:
        """Return API health status.

        Always returns ``{"status": "ok"}`` if the API is running.
        Monitoring systems can poll this endpoint.
        """
        return {"status": "ok", "version": "1.0.0", "environment": settings.environment}

    @app.get("/api/v1/system/stats", tags=["system"], summary="WebSocket connection stats")
    async def system_stats() -> dict[str, object]:
        """Return current WebSocket and system statistics."""
        from quantflow.api.websockets.manager import manager

        return {
            "ws_total_connections": manager.total_connections(),
            "ws_channels": manager.channel_count(),
        }

    @app.get(
        "/metrics",
        tags=["system"],
        summary="Prometheus metrics",
        include_in_schema=False,
    )
    async def prometheus_metrics() -> Response:
        """Expose Prometheus-format metrics for scraping."""
        return metrics_response()

    return app


# ---------------------------------------------------------------------------
# Application instance
# ---------------------------------------------------------------------------

app = create_app()
