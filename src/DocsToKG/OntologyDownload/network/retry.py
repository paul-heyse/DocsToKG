"""Network retry policies: Tenacity-based backoff for resilient HTTP.

Provides retry policies for resilient HTTP requests with exponential backoff.
Designed to be wrapped around HTTPX calls for handling transient failures:
- Connection errors (retried at transport layer)
- Timeouts (retried with backoff)
- Rate-limiting (429, with Retry-After support)
- Server errors (5xx, with exponential backoff)

Design:
- **Full-jitter exponential backoff**: Reduces synchronized retry storms
- **Retry-After support**: Respects server guidance
- **Conservative defaults**: 6 attempts, 60-second deadline
- **Configurable per-call**: Override defaults for specific endpoints
- **Idempotence enforcement**: Only retries GET/HEAD/DELETE + 4xx/5xx

Example:
    >>> from DocsToKG.OntologyDownload.network.retry import create_http_retry_policy
    >>> import httpx
    >>> policy = create_http_retry_policy(max_attempts=6, max_delay_seconds=30)
    >>> client = httpx.Client()
    >>> for attempt in policy:
    ...     with attempt:
    ...         response = client.get("https://api.example.com/data")
"""

import email.utils
import logging
from datetime import datetime, timezone
from typing import Optional

import httpx
from tenacity import (
    RetryError,
    Retrying,
    before_sleep_log,
    retry_if_exception_type,
    retry_if_result,
    stop_after_delay,
    wait_random_exponential,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Retry Policies
# ============================================================================


def create_http_retry_policy(
    max_attempts: int = 6,
    max_delay_seconds: int = 60,
) -> Retrying:
    """Create Tenacity retry policy for HTTP requests.

    Retry strategy:
    - **Retryable exceptions**: ConnectError, ConnectTimeout, ReadTimeout
    - **Retryable responses**: 429 (rate-limit), 5xx (server error)
    - **Backoff strategy**: Full-jitter exponential (reduces thundering herd)
    - **Max delay**: Overall deadline from first attempt
    - **Retry-After**: Respects server guidance if provided

    Args:
        max_attempts: Maximum number of attempts (default 6)
        max_delay_seconds: Maximum time to retry (seconds, default 60)

    Returns:
        Configured Tenacity Retrying object for use in retry loops

    Example:
        >>> policy = create_http_retry_policy(max_attempts=6, max_delay_seconds=30)
        >>> for attempt in policy:
        ...     with attempt:
        ...         response = client.get(url)

    Note:
        - Use in retry loops: `for attempt in policy: with attempt: ...`
        - Don't use directly with decorators (use `@retry(...)` for that)
        - This is for explicit retry control over complex flows
    """

    def wait_with_retry_after(retry_state):
        """Wait strategy: respect Retry-After header if present."""
        exc = retry_state.outcome.exception()

        # Check for Retry-After header in the response
        if exc and hasattr(exc, "response") and exc.response:
            retry_after_header = exc.response.headers.get("Retry-After")
            if retry_after_header:
                try:
                    # Try parsing as integer (seconds)
                    return int(retry_after_header)
                except ValueError:
                    # Parse as HTTP-date
                    try:
                        dt = email.utils.parsedate_to_datetime(retry_after_header)
                        now = datetime.now(timezone.utc)
                        delay_sec = (dt - now).total_seconds()
                        return max(0, delay_sec)
                    except (TypeError, ValueError):
                        pass

        # Fall back to exponential backoff
        return 0  # Let exponential strategy compute the wait

    def retry_on_status(response):
        """Retry on 429 (rate-limit) or 5xx (server error)."""
        if hasattr(response, "status_code"):
            return response.status_code in {429, 500, 502, 503, 504}
        return False

    return Retrying(
        # Stop after deadline (not just attempts)
        stop=stop_after_delay(max_delay_seconds),
        # Full-jitter exponential backoff: spreads retries, reduces contention
        wait=wait_random_exponential(
            multiplier=0.5,  # Initial: 0.5s * 2^attempt
            max=min(60, max_delay_seconds),  # Cap at deadline
        ),
        # Retry on transient network errors
        retry=(
            retry_if_exception_type(
                (
                    httpx.ConnectError,
                    httpx.ConnectTimeout,
                    httpx.ReadTimeout,
                )
            )
            | retry_if_result(retry_on_status)  # Or 429/5xx responses
        ),
        # Log before sleeping (once per actual retry sleep)
        before_sleep=before_sleep_log(logger, logging.WARNING, exc_info=True),
        # Re-raise original exception on final failure (don't wrap in RetryError)
        reraise=True,
    )


def create_idempotent_retry_policy(
    max_attempts: int = 6,
    max_delay_seconds: int = 60,
    methods: Optional[list] = None,
) -> Retrying:
    """Create retry policy with idempotency enforcement.

    Only retries requests that are safe to retry (GET, HEAD, DELETE)
    or that are explicitly marked as idempotent.

    Args:
        max_attempts: Maximum attempts
        max_delay_seconds: Maximum time to retry
        methods: List of HTTP methods to allow retry (default: GET, HEAD, DELETE)

    Returns:
        Configured Tenacity Retrying object

    Note:
        This is stricter than create_http_retry_policy. Use when you're
        unsure about endpoint idempotence.
    """
    if methods is None:
        methods = {"GET", "HEAD", "DELETE"}
    else:
        methods = set(methods)

    def retry_on_status_idempotent(response):
        """Retry on 429/5xx but only for idempotent methods."""
        if hasattr(response, "request"):
            method = response.request.method
            if method not in methods:
                # Non-idempotent method: don't retry
                return False

        if hasattr(response, "status_code"):
            return response.status_code in {429, 500, 502, 503, 504}
        return False

    return Retrying(
        stop=stop_after_delay(max_delay_seconds),
        wait=wait_random_exponential(
            multiplier=0.5,
            max=min(60, max_delay_seconds),
        ),
        retry=(
            retry_if_exception_type(
                (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout)
            )
            | retry_if_result(retry_on_status_idempotent)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING, exc_info=True),
        reraise=True,
    )


def create_aggressive_retry_policy(
    max_attempts: int = 12,
    max_delay_seconds: int = 300,
) -> Retrying:
    """Create aggressive retry policy for critical operations.

    Use for operations that must succeed (e.g., metadata downloads).
    Higher attempt count and longer deadline, but still respects backoff.

    Args:
        max_attempts: Maximum attempts (default 12)
        max_delay_seconds: Maximum time to retry (default 5 minutes)

    Returns:
        Configured Tenacity Retrying object
    """
    return Retrying(
        stop=stop_after_delay(max_delay_seconds),
        wait=wait_random_exponential(
            multiplier=0.5,
            max=min(300, max_delay_seconds),  # Allow up to 5-minute wait
        ),
        retry=(
            retry_if_exception_type(
                (
                    httpx.ConnectError,
                    httpx.ConnectTimeout,
                    httpx.ReadTimeout,
                )
            )
            | retry_if_result(lambda r: hasattr(r, "status_code") and r.status_code >= 500)
        ),
        before_sleep=before_sleep_log(logger, logging.INFO),  # Log each retry
        reraise=True,
    )


def create_rate_limit_retry_policy(
    max_delay_seconds: int = 120,
) -> Retrying:
    """Create retry policy specialized for rate-limiting (429).

    Always respects Retry-After header. Useful when you expect rate-limiting
    and want to be cooperative with server guidance.

    Args:
        max_delay_seconds: Maximum time to retry (default 2 minutes)

    Returns:
        Configured Tenacity Retrying object
    """

    def wait_rate_limit(retry_state):
        """Wait strategy: mandatory Retry-After respect."""
        exc = retry_state.outcome.exception() or retry_state.outcome.result()

        # Check for explicit Retry-After
        if hasattr(exc, "response") and exc.response:
            retry_after = exc.response.headers.get("Retry-After")
            if retry_after:
                try:
                    return int(retry_after)
                except ValueError:
                    try:
                        dt = email.utils.parsedate_to_datetime(retry_after)
                        delay = (dt - datetime.now(timezone.utc)).total_seconds()
                        return max(1, delay)  # At least 1 second
                    except Exception:
                        pass

        # Fall back to exponential backoff
        return 0

    return Retrying(
        stop=stop_after_delay(max_delay_seconds),
        wait=wait_rate_limit,
        # Only retry on 429
        retry=retry_if_result(
            lambda r: hasattr(r, "status_code") and r.status_code == 429
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


# ============================================================================
# Retry Decorators (for functions/methods)
# ============================================================================


def retry_http_request(
    max_attempts: int = 6,
    max_delay_seconds: int = 60,
):
    """Decorator for HTTP request functions.

    Use on functions that make HTTP calls:
        @retry_http_request(max_attempts=6, max_delay_seconds=30)
        def fetch_data(url):
            return client.get(url)

    Args:
        max_attempts: Maximum attempts
        max_delay_seconds: Maximum time to retry

    Returns:
        Decorator function
    """
    from tenacity import retry

    def retry_on_http_error(exc):
        """Predicate for retryable HTTP errors."""
        if isinstance(exc, httpx.RequestError):
            return isinstance(
                exc,
                (
                    httpx.ConnectError,
                    httpx.ConnectTimeout,
                    httpx.ReadTimeout,
                ),
            )
        if isinstance(exc, httpx.HTTPStatusError):
            # Retry on 5xx
            return exc.response.status_code >= 500
        return False

    return retry(
        stop=stop_after_delay(max_delay_seconds),
        wait=wait_random_exponential(multiplier=0.5, max=min(60, max_delay_seconds)),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


__all__ = [
    "create_http_retry_policy",
    "create_idempotent_retry_policy",
    "create_aggressive_retry_policy",
    "create_rate_limit_retry_policy",
    "retry_http_request",
]
