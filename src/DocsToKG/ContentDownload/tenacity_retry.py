"""Tenacity retry strategies and classification for ContentDownload.

Provides:
- Retryability classification (RFC-compliant)
- Retry-After header aware wait strategies
- Tenacity controller builder
- Structured logging and telemetry integration
"""

from __future__ import annotations

import email.utils
import logging
import time
from datetime import datetime
from typing import Callable, Optional

import httpx
import tenacity
from tenacity import RetryCallState, retry_if_exception, retry_if_result

from DocsToKG.ContentDownload.breakers import BreakerOpenError
from DocsToKG.ContentDownload.retries_config import RetriesConfig

LOGGER = logging.getLogger(__name__)


def is_retryable(
    *,
    method: str,
    status: Optional[int] = None,
    exception: Optional[BaseException] = None,
    offline: bool = False,
    breaker_open: bool = False,
    cfg: RetriesConfig,
) -> bool:
    """Determine if a request should be retried.

    Args:
        method: HTTP method (GET, POST, etc.)
        status: HTTP status code (if from response)
        exception: Exception that occurred (if from exception)
        offline: Whether in offline mode (only-if-cached)
        breaker_open: Whether circuit breaker is open
        cfg: Retries configuration

    Returns:
        True if the request should be retried, False otherwise
    """
    # Never retry if breaker is open or offline
    if breaker_open or offline:
        return False

    # Don't retry BreakerOpenError
    if isinstance(exception, BreakerOpenError):
        return False

    # Handle HTTP status codes
    if status is not None:
        # Check if status is in retryable list
        if status in cfg.statuses:
            return True
        # Handle 408 optionally
        if status == 408 and cfg.retry_408:
            return True
        # Never retry other 4xx/5xx
        return False

    # Handle exceptions
    if exception is not None:
        # Retryable network exceptions
        retryable_exceptions = (
            httpx.ConnectError,
            httpx.ReadError,
            httpx.WriteError,
        )

        if isinstance(exception, retryable_exceptions):
            return True

        # Timeout exception (config-gated)
        if isinstance(exception, httpx.TimeoutException) and cfg.retry_on_timeout:
            return True

        # RemoteProtocolError (config-gated)
        if isinstance(exception, httpx.RemoteProtocolError) and cfg.retry_on_remote_protocol:
            return True

        # Never retry client protocol errors
        if isinstance(exception, httpx.LocalProtocolError):
            return False

        return False

    return False


class _WaitRetryAfter(tenacity.wait.wait_base):
    """Wait strategy that prefers Retry-After header over exponential backoff.

    This strategy respects RFC 7231 Retry-After headers when present,
    falling back to exponential jitter backoff when not.
    """

    def __init__(
        self,
        fallback: tenacity.wait.wait_base,
        cap_s: float,
    ) -> None:
        """Initialize wait strategy.

        Args:
            fallback: Fallback wait strategy (e.g., wait_random_exponential)
            cap_s: Maximum Retry-After seconds to honor
        """
        self.fallback = fallback
        self.cap_s = cap_s

    def __call__(self, retry_state: RetryCallState) -> float:
        """Compute wait time.

        Args:
            retry_state: Current Tenacity retry state

        Returns:
            Seconds to wait before next attempt
        """
        # Try to extract Retry-After from last outcome
        outcome = retry_state.outcome
        if outcome is None:
            return self.fallback(retry_state)

        retry_after_s = None

        # Check if it's a response with Retry-After header
        if hasattr(outcome, "value") and hasattr(outcome.value, "headers"):
            response = outcome.value
            retry_after_header = response.headers.get("Retry-After")

            if retry_after_header:
                try:
                    # Try parsing as seconds
                    retry_after_s = int(retry_after_header)
                except ValueError:
                    try:
                        # Try parsing as HTTP-date
                        dt = email.utils.parsedate_to_datetime(retry_after_header)
                        retry_after_s = int(max(0, (dt - datetime.now(dt.tzinfo)).total_seconds()))
                    except (TypeError, ValueError):
                        pass

        # Return Retry-After value if available and valid, capped
        if retry_after_s is not None and retry_after_s > 0:
            wait_s = min(retry_after_s, self.cap_s)
            LOGGER.debug(f"Using Retry-After header: {wait_s}s (capped at {self.cap_s}s)")
            return wait_s

        # Fall back to exponential jitter
        return self.fallback(retry_state)


def _make_retry_predicate(
    cfg: RetriesConfig,
    method: str,
    offline: bool = False,
    breaker_open: bool = False,
) -> Callable:
    """Create a retry predicate for Tenacity.

    Args:
        cfg: Retries configuration
        method: HTTP method
        offline: Whether in offline mode
        breaker_open: Whether circuit breaker is open

    Returns:
        Predicate function for Tenacity
    """
    # Check if method is retryable
    if method not in cfg.methods:
        if cfg.allow_post_if_idempotent and method == "POST":
            # Will check idempotent flag at request time
            pass
        else:
            # Method not retryable, never retry
            def never_retry(outcome):
                return False

            return never_retry

    def predicate(outcome):
        """Check if outcome should be retried."""
        exception = outcome.exception()
        if exception is not None:
            return is_retryable(
                method=method,
                exception=exception,
                offline=offline,
                breaker_open=breaker_open,
                cfg=cfg,
            )

        # Check response status
        value = outcome.value()
        if hasattr(value, "status_code"):
            return is_retryable(
                method=method,
                status=value.status_code,
                offline=offline,
                breaker_open=breaker_open,
                cfg=cfg,
            )

        return False

    return predicate


def build_tenacity_retrying(
    cfg: RetriesConfig,
    method: str = "GET",
    offline: bool = False,
    breaker_open: bool = False,
    before_sleep_hook: Optional[Callable[[RetryCallState], None]] = None,
) -> tenacity.Retrying:
    """Build a Tenacity Retrying controller.

    Args:
        cfg: Retries configuration
        method: HTTP method
        offline: Whether in offline mode
        breaker_open: Whether circuit breaker is open
        before_sleep_hook: Optional hook to run before each sleep

    Returns:
        Configured Tenacity Retrying controller
    """
    # Build retry predicate
    retry_predicate = _make_retry_predicate(
        cfg, method=method, offline=offline, breaker_open=breaker_open
    )

    # Build stop policy (attempt limit OR wall-clock limit)
    stop_policy = tenacity.stop_after_attempt(cfg.max_attempts)
    if cfg.max_total_s > 0:
        stop_policy = stop_policy | tenacity.stop_after_delay(cfg.max_total_s)

    # Build wait strategy with Retry-After support
    fallback_wait = tenacity.wait_random_exponential(
        multiplier=cfg.backoff.multiplier,
        max=cfg.backoff.max_s,
    )
    wait_strategy = _WaitRetryAfter(fallback=fallback_wait, cap_s=cfg.retry_after_cap_s)

    # Build hook
    if before_sleep_hook is None:
        before_sleep_hook = _default_before_sleep_hook

    # Create Retrying controller
    return tenacity.Retrying(
        retry=retry_if_exception(retry_predicate) | retry_if_result(retry_predicate),
        stop=stop_policy,
        wait=wait_strategy,
        sleep=time.sleep,
        before_sleep=before_sleep_hook,
        reraise=True,
    )


def _default_before_sleep_hook(retry_state: RetryCallState) -> None:
    """Default hook to log before sleep.

    Args:
        retry_state: Current retry state
    """
    attempt_num = retry_state.attempt_number
    seconds_since_start = retry_state.seconds_since_start
    next_action = retry_state.next_action

    if next_action is not None and hasattr(next_action, "sleep"):
        wait_s = next_action.sleep
        wait_ms = int(wait_s * 1000)
        LOGGER.warning(
            f"retry attempt={attempt_num} wait_ms={wait_ms} elapsed_s={seconds_since_start:.1f}"
        )
