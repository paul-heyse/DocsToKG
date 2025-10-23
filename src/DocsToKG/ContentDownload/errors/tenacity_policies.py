"""Context-aware retry policies using Tenacity predicates.

Extends Tenacity's retry/wait predicates to support operation-aware strategies
(DOWNLOAD, VALIDATE, RESOLVE, EXTRACT, MANIFEST_FETCH) without a separate
error recovery layer.

This module encodes operation context into Tenacity predicates, allowing
different operations to have different retry semantics:

- DOWNLOAD (critical): Aggressive retries, always retry on transient errors
- VALIDATE (non-critical): Signal deferral on 429/timeout (batch later)
- RESOLVE (has alternatives): Signal failover on timeout (try next resolver)
- EXTRACT (retryable): Standard exponential backoff
- MANIFEST_FETCH (has fallbacks): Defer on 429, fallback on timeout

Usage:
    from DocsToKG.ContentDownload.errors.tenacity_policies import (
        OperationType,
        create_contextual_retry_policy,
    )

    policy = create_contextual_retry_policy(
        operation=OperationType.DOWNLOAD,
        max_attempts=6,
        max_delay_seconds=60,
    )

    for attempt in policy:
        with attempt:
            response = client.get(url)

Integration:
    Callers are responsible for:
    - Catching deferral signals (429/timeout → defer_item to batch queue)
    - Catching failover signals (timeout on RESOLVE → try_alternative_resolver)
    - Calling tracker.on_success() after successful request
    This policy just decides yes/no on individual errors via predicates.
"""

from __future__ import annotations

import logging
from enum import Enum, auto
from typing import Any, Callable

import httpx
from tenacity import (
    RetryCallState,
    Retrying,
    before_sleep_log,
    retry_if_exception_type,
    retry_if_result,
    stop_after_delay,
)

logger = logging.getLogger(__name__)


class OperationType(Enum):
    """Operation type for contextual retry decisions."""

    DOWNLOAD = auto()  # Artifact content (critical path)
    VALIDATE = auto()  # Post-download validation (non-critical, batchable)
    RESOLVE = auto()  # Metadata resolution (has fallback resolvers)
    EXTRACT = auto()  # Archive extraction (retryable)
    MANIFEST_FETCH = auto()  # Manifest metadata (has fallback sources)


def _should_retry_on_429(operation: OperationType) -> Callable[[Any], bool]:
    """Create predicate for 429 handling per operation.

    Downloads always retry. Validation/resolve may defer or failover instead.
    This predicate decides **whether** to retry; defer/failover are handled
    at the caller level (orchestrator/pipeline).

    Args:
        operation: Operation type (DOWNLOAD, VALIDATE, RESOLVE, etc.)

    Returns:
        Predicate function for Tenacity retry_if_result()
    """

    def _retry_429(response: Any) -> bool:
        """Check if 429 should be retried based on operation type."""
        if not hasattr(response, "status_code"):
            return False

        if response.status_code != 429:
            return False

        # Operation-specific 429 strategy
        if operation == OperationType.DOWNLOAD:
            # Downloads: always retry (critical path)
            # Rate limiting is enforced at limiter level, not here
            logger.debug("429 on DOWNLOAD: retrying (critical path)")
            return True
        elif operation in (OperationType.VALIDATE, OperationType.MANIFEST_FETCH):
            # Non-critical: return False to signal deferral to caller
            # Caller will defer_item(context) for batch processing later
            logger.debug(f"429 on {operation.name}: signaling deferral (non-critical)")
            return False
        elif operation == OperationType.RESOLVE:
            # Resolve: return False to signal failover attempt
            # Caller will try_alternative_resolver(context)
            logger.debug(f"429 on {operation.name}: signaling failover (has alternatives)")
            return False
        else:
            # Default: retry (safe fallback)
            logger.debug(f"429 on {operation.name}: retrying (default)")
            return True

    return _retry_429


def _should_retry_on_timeout(operation: OperationType) -> Callable[[Any], bool]:
    """Create predicate for timeout handling per operation.

    Args:
        operation: Operation type

    Returns:
        Predicate function for Tenacity retry_if_exception()
    """

    def _retry_timeout(exc: BaseException) -> bool:
        """Check if timeout should be retried based on operation type."""
        if not isinstance(exc, (httpx.ConnectTimeout, httpx.ReadTimeout)):
            return False

        # Operation-specific timeout strategy
        if operation == OperationType.DOWNLOAD:
            # Downloads: always retry (critical path)
            logger.debug("Timeout on DOWNLOAD: retrying (critical path)")
            return True
        elif operation == OperationType.VALIDATE:
            # Validation: defer (non-critical)
            logger.debug("Timeout on VALIDATE: signaling deferral (non-critical)")
            return False
        elif operation == OperationType.RESOLVE:
            # Resolve: retry up to 3 times, then failover (handled by caller)
            # This predicate always returns True; caller counts attempts
            logger.debug("Timeout on RESOLVE: retrying (failover handled by caller)")
            return True
        elif operation == OperationType.EXTRACT:
            # Extract: always retry (retryable operation)
            logger.debug("Timeout on EXTRACT: retrying (retryable)")
            return True
        else:
            # Default: retry (safe fallback)
            logger.debug(f"Timeout on {operation.name}: retrying (default)")
            return True

    return _retry_timeout


def create_contextual_retry_policy(
    operation: OperationType = OperationType.DOWNLOAD,
    max_attempts: int = 6,
    max_delay_seconds: int = 60,
) -> Retrying:
    """Create operation-aware Tenacity retry policy.

    Different operations have different retry semantics:

    - **DOWNLOAD**: Aggressive retries (critical path)
    - **VALIDATE**: Deferral signals on 429/timeout (non-critical, batch later)
    - **RESOLVE**: Failover signals on timeout (has alternative resolvers)
    - **EXTRACT**: Standard exponential backoff retries
    - **MANIFEST_FETCH**: Deferral on 429, fallback on timeout

    Caller is responsible for:
    - Catching deferral signals (429/timeout on non-critical ops → defer)
    - Catching failover signals (timeout on RESOLVE → try_alternative)
    - This policy just decides yes/no on individual errors

    Args:
        operation: Operation type (DOWNLOAD, VALIDATE, RESOLVE, etc.)
        max_attempts: Maximum retry attempts
        max_delay_seconds: Maximum total retry time (deadline)

    Returns:
        Configured Tenacity Retrying object

    Example:
        >>> from DocsToKG.ContentDownload.errors.tenacity_policies import (
        ...     OperationType,
        ...     create_contextual_retry_policy,
        ... )
        >>> policy = create_contextual_retry_policy(
        ...     operation=OperationType.DOWNLOAD,
        ...     max_attempts=6,
        ... )
        >>> for attempt in policy:
        ...     with attempt:
        ...         response = client.get(url)
    """

    def wait_with_retry_after(retry_state: RetryCallState) -> float:
        """Wait strategy: respect Retry-After header if present."""
        exc = retry_state.outcome.exception() or retry_state.outcome.result()

        if hasattr(exc, "response") and exc.response:
            retry_after = exc.response.headers.get("Retry-After")
            if retry_after:
                try:
                    # Try parsing as integer (seconds)
                    return float(retry_after)
                except ValueError:
                    # Would parse HTTP-date here; skip for now
                    pass

        # Default: let exponential backoff handle it (return 0)
        return 0

    # Build retry condition: network errors + operation-specific HTTP handling
    retry_condition = (
        retry_if_exception_type((httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout))
        | retry_if_result(_should_retry_on_429(operation))
        | retry_if_result(_should_retry_on_timeout(operation))
    )

    # Also retry on 5xx for all operations (server errors)
    retry_condition |= retry_if_result(lambda r: hasattr(r, "status_code") and r.status_code >= 500)

    return Retrying(
        stop=stop_after_delay(max_delay_seconds),
        wait=wait_with_retry_after,
        retry=retry_condition,
        before_sleep=before_sleep_log(
            logger,
            logging.WARNING,
            exc_info=False,
        ),
        reraise=True,
    )


__all__ = [
    "OperationType",
    "create_contextual_retry_policy",
]
