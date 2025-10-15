"""
Unified Network Utilities

This module consolidates HTTP retry helpers, conditional request utilities, and
session construction logic into a single import surface. Historically these
helpers lived in separate ``http`` and ``conditional`` modules; co-locating
them reduces cross-module bootstrapping and ensures shared defaults remain
aligned.

Key Features:

- ``create_session``: Configure ``requests.Session`` instances with pooled
  adapters and optional header injection.
- ``request_with_retries``: Execute resilient HTTP calls with jittered,
  exponential backoff while honouring ``Retry-After`` directives.
- ``ConditionalRequestHelper``: Build and validate conditional request headers
  for polite revalidation workflows.

Usage:

    from DocsToKG.ContentDownload.network import (
        ConditionalRequestHelper,
        create_session,
        request_with_retries,
    )

    session = create_session({"User-Agent": "DocsToKG/1.0"})
    response = request_with_retries(session, "GET", "https://example.org/resource")
    helper = ConditionalRequestHelper(prior_etag="abc123")
    headers = helper.build_headers()
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Mapping, MutableMapping, Optional, Set, Union

import requests
from requests.adapters import HTTPAdapter

LOGGER = logging.getLogger("DocsToKG.ContentDownload.network")


def create_session(
    headers: Optional[Mapping[str, str]] = None,
    *,
    adapter_max_retries: int = 0,
) -> requests.Session:
    """Return a ``requests.Session`` configured for DocsToKG network requests.

    Args:
        headers: Optional mapping of HTTP headers applied to the session. The
            mapping is copied into the session's header store to avoid mutating
            caller-owned dictionaries.
        adapter_max_retries: Number of retries configured on mounted HTTP
            adapters. Defaults to ``0`` so that :func:`request_with_retries`
            governs retry behaviour.

    Returns:
        ``requests.Session`` instance with HTTP/HTTPS adapters mounted.

    Notes:
        The returned session uses ``HTTPAdapter`` for connection pooling. It is
        safe to share across threads provided callers avoid mutating shared
        mutable state (for example, using ``session.headers.update``) once the
        session is distributed to workers.
    """

    session = requests.Session()

    if headers and hasattr(session, "headers"):
        session_headers = session.headers
        if isinstance(session_headers, MutableMapping):
            session_headers.update(dict(headers))
        else:  # pragma: no cover - defensive guard for non-mapping implementations
            LOGGER.debug(
                "Session.headers is not a mutable mapping; skipping header injection on %r",
                session_headers,
            )

    adapter = HTTPAdapter(max_retries=adapter_max_retries)
    if hasattr(session, "mount"):
        session.mount("http://", adapter)
        session.mount("https://", adapter)

    return session


def parse_retry_after_header(response: requests.Response) -> Optional[float]:
    """Parse ``Retry-After`` header and return wait time in seconds.

    Args:
        response: HTTP response potentially containing ``Retry-After`` header.

    Returns:
        Float seconds to wait, or ``None`` if header missing/invalid.

    Raises:
        None: Invalid headers are tolerated and yield ``None`` without raising.

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
        ValueError: If ``max_retries`` or ``backoff_factor`` are invalid or
            ``url``/``method`` are empty.
        requests.RequestException: If all retry attempts fail due to network
            errors or the session raises an exception.
    """

    if max_retries < 0:
        raise ValueError(f"max_retries must be non-negative, got {max_retries}")
    if backoff_factor < 0:
        raise ValueError(f"backoff_factor must be non-negative, got {backoff_factor}")
    if not method:
        raise ValueError("method must be a non-empty string")
    if not isinstance(url, str) or not url:
        raise ValueError("url must be a non-empty string")

    if retry_statuses is None:
        retry_statuses = {429, 500, 502, 503, 504}
    else:
        retry_statuses = set(retry_statuses)

    request_func = getattr(session, "request", None)
    if not callable(request_func):
        fallback = getattr(session, method.lower(), None)
        if not callable(fallback):
            raise AttributeError(
                f"Session object of type {type(session)!r} lacks 'request' and '{method.lower()}' callables"
            )

        def request_func(
            *,
            method: str,
            url: str,
            **call_kwargs: Any,
        ) -> requests.Response:
            """Invoke the fallback HTTP method when ``Session.request`` is unavailable."""

            return fallback(url, **call_kwargs)

    last_exception: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            response = request_func(method=method, url=url, **kwargs)

            if response.status_code not in retry_statuses:
                return response

            if attempt >= max_retries:
                LOGGER.warning(
                    "Received status %s for %s %s after %s attempts; returning response",
                    response.status_code,
                    method,
                    url,
                    attempt + 1,
                )
                return response

            base_delay = backoff_factor * (2**attempt)
            jitter = random.random() * 0.1
            delay = base_delay + jitter

            if respect_retry_after:
                retry_after_delay = parse_retry_after_header(response)
                if retry_after_delay is not None and retry_after_delay > delay:
                    delay = retry_after_delay

            LOGGER.debug(
                "Retrying %s %s after HTTP %s (attempt %s/%s, delay %.2fs)",
                method,
                url,
                response.status_code,
                attempt + 1,
                max_retries + 1,
                delay,
            )
            time.sleep(delay)

        except requests.Timeout as exc:
            last_exception = exc
            LOGGER.debug(
                "Request %s %s timed out (attempt %s/%s): %s",
                method,
                url,
                attempt + 1,
                max_retries + 1,
                exc,
            )
            if attempt >= max_retries:
                LOGGER.warning(
                    "Exhausted %s retries for %s %s due to timeouts", max_retries, method, url
                )
                raise

            delay = backoff_factor * (2**attempt) + random.random() * 0.1
            time.sleep(delay)

        except requests.ConnectionError as exc:
            last_exception = exc
            LOGGER.debug(
                "Request %s %s encountered connection error (attempt %s/%s): %s",
                method,
                url,
                attempt + 1,
                max_retries + 1,
                exc,
            )
            if attempt >= max_retries:
                LOGGER.warning(
                    "Exhausted %s retries for %s %s due to connection errors",
                    max_retries,
                    method,
                    url,
                )
                raise

            delay = backoff_factor * (2**attempt) + random.random() * 0.1
            time.sleep(delay)

        except requests.RequestException as exc:
            last_exception = exc
            LOGGER.debug(
                "Request %s %s failed (attempt %s/%s): %s",
                method,
                url,
                attempt + 1,
                max_retries + 1,
                exc,
            )
            if attempt >= max_retries:
                LOGGER.warning("Exhausted %s retries for %s %s: %s", max_retries, method, url, exc)
                raise

            delay = backoff_factor * (2**attempt) + random.random() * 0.1
            time.sleep(delay)

    if last_exception is not None:  # pragma: no cover - defensive safety net
        raise last_exception

    raise requests.RequestException(  # pragma: no cover - defensive safety net
        f"Exhausted {max_retries} retries for {method} {url}"
    )


@dataclass
class CachedResult:
    """Represents HTTP 304 Not Modified response with prior metadata."""

    path: str
    sha256: str
    content_length: int
    etag: Optional[str]
    last_modified: Optional[str]


@dataclass
class ModifiedResult:
    """Represents HTTP 200 response requiring fresh download."""

    etag: Optional[str]
    last_modified: Optional[str]


class ConditionalRequestHelper:
    """Utility for constructing conditional requests and interpreting responses."""

    def __init__(
        self,
        prior_etag: Optional[str] = None,
        prior_last_modified: Optional[str] = None,
        prior_sha256: Optional[str] = None,
        prior_content_length: Optional[int] = None,
        prior_path: Optional[str] = None,
    ) -> None:
        if prior_content_length is not None and prior_content_length < 0:
            raise ValueError(
                f"prior_content_length must be non-negative, got {prior_content_length}"
            )
        self.prior_etag = prior_etag
        self.prior_last_modified = prior_last_modified
        self.prior_sha256 = prior_sha256
        self.prior_content_length = prior_content_length
        self.prior_path = prior_path

    def build_headers(self) -> Mapping[str, str]:
        """Generate conditional request headers from cached metadata."""

        headers: dict[str, str] = {}
        if self.prior_etag:
            headers["If-None-Match"] = self.prior_etag
        if self.prior_last_modified:
            headers["If-Modified-Since"] = self.prior_last_modified
        return headers

    def interpret_response(
        self,
        response: requests.Response,
    ) -> Union[CachedResult, ModifiedResult]:
        """Interpret response status and headers as cached or modified result."""

        if not hasattr(response, "status_code") or not hasattr(response, "headers"):
            raise TypeError("response must expose 'status_code' and 'headers' attributes")

        if response.status_code == 304:
            missing_fields = []
            if not self.prior_path:
                missing_fields.append("path")
            if not self.prior_sha256:
                missing_fields.append("sha256")
            if self.prior_content_length is None:
                missing_fields.append("content_length")

            if missing_fields:
                raise ValueError(
                    "HTTP 304 requires complete prior metadata. Missing: "
                    + ", ".join(missing_fields)
                    + ". This indicates a bug in manifest loading or caching logic."
                )
            assert self.prior_path is not None
            assert self.prior_sha256 is not None
            assert self.prior_content_length is not None
            return CachedResult(
                path=self.prior_path,
                sha256=self.prior_sha256,
                content_length=self.prior_content_length,
                etag=self.prior_etag,
                last_modified=self.prior_last_modified,
            )
        return ModifiedResult(
            etag=response.headers.get("ETag"),
            last_modified=response.headers.get("Last-Modified"),
        )


__all__ = [
    "CachedResult",
    "ConditionalRequestHelper",
    "ModifiedResult",
    "create_session",
    "parse_retry_after_header",
    "request_with_retries",
]
