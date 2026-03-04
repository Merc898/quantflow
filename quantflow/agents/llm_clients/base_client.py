"""Shared base class for all LLM API clients.

Provides:
- Async httpx session management
- Exponential backoff with jitter on 429 / 503
- Circuit breaker (3 consecutive failures → skip for 1 hour)
- Token usage tracking
- Structured JSON response parsing
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from abc import ABC, abstractmethod
from typing import Any

import httpx

from quantflow.config.logging import get_logger

logger = get_logger(__name__)

# Backoff / retry constants
_MAX_RETRIES = 3
_BASE_DELAY_S = 1.0
_MAX_DELAY_S = 60.0
_CIRCUIT_BREAKER_THRESHOLD = 3
_CIRCUIT_BREAKER_COOLDOWN_S = 3600.0  # 1 hour

# Retryable HTTP status codes
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class LLMClientError(Exception):
    """Raised when an LLM API call fails after all retries."""


class CircuitBreakerOpenError(LLMClientError):
    """Raised when the circuit breaker is open (agent temporarily disabled)."""


class BaseLLMClient(ABC):
    """Abstract base for all LLM API clients.

    Subclasses override :meth:`_call_api` with provider-specific logic.
    Retry / circuit-breaker / logging are handled here.

    Args:
        api_key: Provider API key.
        timeout_s: Per-request timeout in seconds.
        max_retries: Maximum retry attempts.
    """

    def __init__(
        self,
        api_key: str,
        timeout_s: float = 30.0,
        max_retries: int = _MAX_RETRIES,
    ) -> None:
        self._api_key = api_key
        self._timeout = timeout_s
        self._max_retries = max_retries
        self._logger = get_logger(self.__class__.__name__)

        # Circuit breaker state
        self._consecutive_failures: int = 0
        self._circuit_open_until: float = 0.0

        # Token usage tracking
        self._total_tokens_used: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def call_with_retry(
        self,
        payload: dict[str, Any],
        endpoint: str,
    ) -> dict[str, Any]:
        """Call the LLM API with automatic retry and circuit-breaker logic.

        Args:
            payload: Request body dict (will be JSON-serialised).
            endpoint: Full API endpoint URL.

        Returns:
            Parsed JSON response dict.

        Raises:
            CircuitBreakerOpenError: If circuit breaker is currently open.
            LLMClientError: If all retries are exhausted.
        """
        # Circuit breaker check
        if time.monotonic() < self._circuit_open_until:
            raise CircuitBreakerOpenError(
                f"{self.__class__.__name__} circuit breaker is open "
                f"(cooldown until {self._circuit_open_until:.0f})"
            )

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._call_api(payload, endpoint)
                # Reset on success
                self._consecutive_failures = 0
                return response

            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status not in _RETRYABLE_STATUS or attempt == self._max_retries:
                    self._record_failure()
                    raise LLMClientError(
                        f"HTTP {status} from {endpoint}: {exc.response.text[:200]}"
                    ) from exc
                delay = self._backoff_delay(attempt)
                self._logger.warning(
                    "Retryable HTTP error",
                    status=status,
                    attempt=attempt + 1,
                    delay_s=round(delay, 2),
                    endpoint=endpoint,
                )
                await asyncio.sleep(delay)
                last_exc = exc

            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                if attempt == self._max_retries:
                    self._record_failure()
                    raise LLMClientError(f"Network error calling {endpoint}: {exc}") from exc
                delay = self._backoff_delay(attempt)
                self._logger.warning(
                    "Network error, retrying",
                    attempt=attempt + 1,
                    delay_s=round(delay, 2),
                )
                await asyncio.sleep(delay)
                last_exc = exc

        self._record_failure()
        raise LLMClientError(
            f"All {self._max_retries} retries exhausted for {endpoint}"
        ) from last_exc

    def parse_json_from_response(self, text: str) -> dict[str, Any]:
        """Extract and parse JSON from an LLM response string.

        Handles models that wrap JSON in markdown code fences.

        Args:
            text: Raw LLM response text.

        Returns:
            Parsed dict.

        Raises:
            LLMClientError: If no valid JSON can be extracted.
        """
        # Strip markdown code fences
        stripped = text.strip()
        for fence in ("```json", "```"):
            if stripped.startswith(fence):
                stripped = stripped[len(fence) :]
                if stripped.endswith("```"):
                    stripped = stripped[:-3]
                stripped = stripped.strip()
                break

        try:
            return json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise LLMClientError(
                f"Could not parse JSON from LLM response: {exc}\nText: {text[:300]}"
            ) from exc

    @property
    def total_tokens_used(self) -> int:
        """Total tokens consumed by this client instance."""
        return self._total_tokens_used

    # ------------------------------------------------------------------
    # Abstract method
    # ------------------------------------------------------------------

    @abstractmethod
    async def _call_api(
        self,
        payload: dict[str, Any],
        endpoint: str,
    ) -> dict[str, Any]:
        """Make a single (non-retried) API call.

        Args:
            payload: Request body.
            endpoint: Full endpoint URL.

        Returns:
            Parsed JSON response.
        """
        ...

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _backoff_delay(self, attempt: int) -> float:
        """Compute exponential backoff delay with full jitter.

        Args:
            attempt: Zero-indexed attempt number.

        Returns:
            Delay in seconds.
        """
        base = min(_BASE_DELAY_S * (2**attempt), _MAX_DELAY_S)
        return random.uniform(0.0, base)

    def _record_failure(self) -> None:
        """Record a failure; open circuit breaker if threshold reached."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= _CIRCUIT_BREAKER_THRESHOLD:
            self._circuit_open_until = time.monotonic() + _CIRCUIT_BREAKER_COOLDOWN_S
            self._logger.error(
                "Circuit breaker opened",
                agent=self.__class__.__name__,
                cooldown_s=_CIRCUIT_BREAKER_COOLDOWN_S,
            )
