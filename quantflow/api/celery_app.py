"""Celery application instance for QuantFlow background tasks.

Usage::

    # Start worker
    celery -A quantflow.api.celery_app worker --loglevel=info -c 4

    # Start beat scheduler
    celery -A quantflow.api.celery_app beat --loglevel=info

    # Monitor tasks
    celery -A quantflow.api.celery_app flower
"""

from __future__ import annotations

import os

from celery import Celery
from celery.signals import task_failure, task_postrun, task_prerun

from quantflow.config.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Application instance
# ---------------------------------------------------------------------------

app = Celery(
    "quantflow",
    broker=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
    backend=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
    include=[
        "quantflow.api.tasks.agent_tasks",
    ],
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

app.conf.update(
    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # Timezone
    timezone="UTC",
    enable_utc=True,
    # Result expiry
    result_expires=3600 * 24,  # 24 hours
    # Task time limits
    task_soft_time_limit=300,  # 5 minutes soft limit
    task_time_limit=600,  # 10 minutes hard limit
    # Retries
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # Concurrency
    worker_prefetch_multiplier=1,  # fair scheduling for long tasks
    # Monitoring
    task_send_sent_event=True,
    worker_send_task_events=True,
)

# ---------------------------------------------------------------------------
# Sentry integration for Celery tasks
# ---------------------------------------------------------------------------

try:
    import sentry_sdk

    _sentry_dsn = os.environ.get("SENTRY_DSN", "")
    if _sentry_dsn:
        from sentry_sdk.integrations.celery import CeleryIntegration

        sentry_sdk.init(
            dsn=_sentry_dsn,
            integrations=[CeleryIntegration()],
            traces_sample_rate=float(os.environ.get("SENTRY_TRACES_SAMPLE_RATE", "0.1")),
            environment=os.environ.get("ENVIRONMENT", "development"),
            release=os.environ.get("APP_VERSION", "1.0.0"),
        )
        logger.info("Sentry initialised for Celery worker")
except ImportError:
    pass  # sentry-sdk not installed

# ---------------------------------------------------------------------------
# Task lifecycle logging
# ---------------------------------------------------------------------------


@task_prerun.connect
def task_prerun_handler(
    task_id: str, task: object, args: tuple, kwargs: dict, **kw: object
) -> None:
    """Log task start."""
    logger.info("Task started", task_id=task_id, task_name=str(task))


@task_postrun.connect
def task_postrun_handler(
    task_id: str,
    task: object,
    args: tuple,
    kwargs: dict,
    retval: object,
    state: str,
    **kw: object,
) -> None:
    """Log task completion."""
    logger.info("Task completed", task_id=task_id, task_name=str(task), state=state)


@task_failure.connect
def task_failure_handler(
    task_id: str,
    exception: Exception,
    traceback: object,
    sender: object,
    **kw: object,
) -> None:
    """Log task failures."""
    logger.error(
        "Task failed",
        task_id=task_id,
        task_name=str(sender),
        error=str(exception),
        exc_info=True,
    )
