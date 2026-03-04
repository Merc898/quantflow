"""Structured JSON logging configuration using structlog.

Provides consistent, machine-readable logging throughout QuantFlow.
All logs are output as JSON for easy parsing and aggregation.
"""

import logging
import sys
from typing import TYPE_CHECKING, Any

import structlog

from quantflow.config.settings import settings

if TYPE_CHECKING:
    from structlog.types import Processor


def setup_logging() -> None:
    """Configure structured logging for the application.

    Sets up structlog with JSON formatting for production and
    human-readable console output for development.
    """
    # Common processors for all environments
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.UnicodeDecoder(),
    ]

    if settings.is_production:
        # Production: JSON output for log aggregation
        processors: list[Processor] = [
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Development: Human-readable console output
        processors = [*shared_processors, structlog.dev.ConsoleRenderer(colors=True)]

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.log_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging to use structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.getLevelName(settings.log_level),
    )

    # Silence noisy third-party loggers in production
    if settings.is_production:
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        A structlog BoundLogger instance.
    """
    return structlog.get_logger(name)


class LogContext:
    """Context manager for adding temporary context to log messages.

    Example:
        with LogContext(symbol="AAPL", model="GARCH"):
            logger.info("Fitting model")
            # Logs will include symbol="AAPL" and model="GARCH"
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize context with key-value pairs.

        Args:
            **kwargs: Key-value pairs to add to logging context.
        """
        self.context = kwargs
        self._token: Any = None

    def __enter__(self) -> "LogContext":
        """Enter context and bind variables."""
        self._token = structlog.contextvars.bind_contextvars(**self.context)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context and unbind variables."""
        if self._token is not None:
            structlog.contextvars.unbind_contextvars(*self.context.keys())


def log_execution_time(func_name: str) -> structlog.stdlib.BoundLogger:
    """Create a logger with execution timing context.

    Args:
        func_name: Name of the function being executed.

    Returns:
        Logger with function_name in context.
    """
    return get_logger().bind(function=func_name)


# Initialize logging on module import
setup_logging()

# Default logger instance
logger = get_logger(__name__)
