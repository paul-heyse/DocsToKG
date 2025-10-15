"""Unified HTTP request utilities with retry and backoff support."""

from __future__ import annotations

import random
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Optional, Set

import requests


def parse_retry_after_header(response: requests.Response) -> Optional[float]:
    """Parse ``Retry-After`` header and return wait time in seconds.

    Args:
        response: HTTP response potentially containing ``Retry-After`` header.

    Returns:
        Float seconds to wait, or ``None`` if header missing/invalid.

    Raises:
        None.

    Examples:
        >>> # Integer format
        >>> response = requests.Response()
        >>> response.headers = {"Retry-After": "5"}
        >>> parse_retry_after_header(response)
        5.0

        >>> # HTTP-date format
        >>> response.headers = {"Retry-After": "Wed, 21 Oct 2025 07:28:00 GMT"}
        >>> isinstance(parse_retry_after_header(response), float)
        True
    """

    retry_after = response.headers.get("Retry-After")
    if not retry_after:
        return None

    try:
        return float(retry_after)
    except ValueError:
        pass

    try:
        target_time = parsedate_to_datetime(retry_after)
        if target_time is None:
            return None
        if target_time.tzinfo is None:
            target_time = target_time.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        delta = (target_time - now).total_seconds()
        return max(0.0, delta)
    except (ValueError, TypeError, OverflowError):
        return None


def request_with_retries(
    session: requests.Session,
    method: str,
    url: str,
    *,
    max_retries: int = 3,
    retry_statuses: Optional[Set[int]] = None,
    backoff_factor: float = 0.75,
    respect_retry_after: bool = True,
    **kwargs: Any,
) -> requests.Response:
    """Execute an HTTP request with exponential backoff and retry handling.

    Args:
        session: :class:`requests.Session` used to execute the request.
        method: HTTP method such as ``"GET"`` or ``"HEAD"``.
        url: Fully-qualified URL for the request.
        max_retries: Maximum number of retry attempts before returning the final
            response or raising an exception. Defaults to ``3``.
        retry_statuses: Optional set of HTTP status codes that should trigger a
            retry. Defaults to ``{429, 500, 502, 503, 504}``.
        backoff_factor: Base multiplier for exponential backoff delays.
            Defaults to ``0.75`` seconds.
        respect_retry_after: Whether to parse and obey ``Retry-After`` headers
            when provided by the server. Defaults to ``True``.
        **kwargs: Additional keyword arguments forwarded directly to
            :meth:`requests.Session.request`.

    Returns:
        A :class:`requests.Response` instance on success. The caller is
        responsible for closing the response when streaming content.

    Raises:
        requests.RequestException: If all retry attempts fail due to network
            errors or the session raises an exception.

    Example:
        >>> session = requests.Session()
        >>> response = request_with_retries(session, "HEAD", "https://example.org")
        >>> response.status_code  # doctest: +SKIP
        200
        >>> with request_with_retries(
        ...     session,
        ...     "GET",
        ...     "https://example.org/resource",
        ...     stream=True,
        ... ) as resp:  # doctest: +SKIP
        ...     data = resp.content

    Thread Safety:
        The helper is thread-safe provided the supplied ``session`` can be used
        safely across threads. Standard :class:`requests.Session` instances are
        generally safe for concurrent reads when configured with connection
        pooling adapters.
    """

    if retry_statuses is None:
        retry_statuses = {429, 500, 502, 503, 504}

    request_func = getattr(session, "request", None)
    if not callable(request_func):
        fallback = getattr(session, method.lower(), None)
        if not callable(fallback):
            raise AttributeError(
                f"Session object of type {type(session)!r} lacks 'request' and '{method.lower()}' callables"
            )

        def request_func(*, method: str, url: str, **call_kwargs: Any) -> requests.Response:
            return fallback(url, **call_kwargs)

    last_exception: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            response = request_func(method=method, url=url, **kwargs)

            if response.status_code not in retry_statuses:
                return response

            if attempt >= max_retries:
                return response

            base_delay = backoff_factor * (2**attempt)
            jitter = random.random() * 0.1
            delay = base_delay + jitter

            if respect_retry_after:
                retry_after_delay = parse_retry_after_header(response)
                if retry_after_delay is not None and retry_after_delay > delay:
                    delay = retry_after_delay

            time.sleep(delay)

        except requests.RequestException as exc:
            last_exception = exc
            if attempt >= max_retries:
                raise

            delay = backoff_factor * (2**attempt) + random.random() * 0.1
            time.sleep(delay)

    if last_exception is not None:
        raise last_exception

    raise requests.RequestException(f"Exhausted {max_retries} retries for {method} {url}")


__all__ = ["parse_retry_after_header", "request_with_retries"]
