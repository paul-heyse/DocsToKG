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

Args:
    None.

Returns:
    None.

Raises:
    None.
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
        headers: Optional mapping of HTTP headers applied to the session. The mapping
            is copied into the session's header store so caller-owned dictionaries remain
            unchanged.
        adapter_max_retries: Retry count configured on mounted HTTP adapters. Defaults to
            ``0`` so :func:`request_with_retries` governs retry behaviour.

    Returns:
        requests.Session: Session instance with HTTP/HTTPS adapters mounted and ready for pipeline usage.

    Notes:
        The returned session uses ``HTTPAdapter`` for connection pooling. It is safe to share
        across threads provided callers avoid mutating shared mutable state (for example,
        updating ``session.headers``) once the session is distributed to worker threads.
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
        response (requests.Response): HTTP response potentially containing a
            ``Retry-After`` header.

    Returns:
        float | None: Seconds the caller should wait before retrying, or
        ``None`` when the header is absent or invalid.

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
        session: Session used to execute the outbound request.
        method: HTTP method such as ``"GET"`` or ``"HEAD"``.
        url: Fully qualified URL for the request.
        max_retries: Maximum number of retry attempts before returning the final response or
            raising an exception. Defaults to ``3``.
        retry_statuses: HTTP status codes that should trigger a retry. Defaults to
            ``{429, 500, 502, 503, 504}``.
        backoff_factor: Base multiplier for exponential backoff delays in seconds. Defaults to ``0.75``.
        respect_retry_after: Whether to parse and obey ``Retry-After`` headers. Defaults to ``True``.
        **kwargs: Additional keyword arguments forwarded directly to :meth:`requests.Session.request`.

    Returns:
        requests.Response: Successful response object. Callers are responsible for closing the
        response when streaming content.

    Raises:
        ValueError: If ``max_retries`` or ``backoff_factor`` are invalid or ``url``/``method`` are empty.
        requests.RequestException: If all retry attempts fail due to network errors or the session raises an exception.
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
            """Invoke the fallback HTTP method when ``Session.request`` is unavailable.

            Args:
                method: HTTP method name forwarded for logging parity.
                url: Target URL for the request.
                **call_kwargs: Keyword arguments forwarded to the fallback request callable.

            Returns:
                requests.Response: Response returned by the fallback HTTP method.
            """

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
    """Represents an HTTP 304 response resolved via cached metadata.

    Attributes:
        path (str): Filesystem path that stores the cached artifact.
        sha256 (str): SHA-256 checksum associated with the cached payload.
        content_length (int): Size of the cached payload in bytes.
        etag (str | None): Entity tag reported by the origin server.
        last_modified (str | None): ``Last-Modified`` timestamp provided by the
            origin server.
    """

    path: str
    sha256: str
    content_length: int
    etag: Optional[str]
    last_modified: Optional[str]


@dataclass
class ModifiedResult:
    """Represents a fresh HTTP 200 response requiring download.

    Attributes:
        etag (str | None): Entity tag reported by the origin server.
        last_modified (str | None): ``Last-Modified`` timestamp describing the
            remote resource.
    """

    etag: Optional[str]
    last_modified: Optional[str]


class ConditionalRequestHelper:
    """Construct headers and interpret responses for conditional requests.

    The helper encapsulates cached metadata (ETag, Last-Modified, hashes) so the
    caller can generate polite conditional headers and validate upstream 304
    responses before reusing cached artefacts.
    """

    def __init__(
        self,
        prior_etag: Optional[str] = None,
        prior_last_modified: Optional[str] = None,
        prior_sha256: Optional[str] = None,
        prior_content_length: Optional[int] = None,
        prior_path: Optional[str] = None,
    ) -> None:
        """Initialise cached metadata for conditional requests.

        Args:
            prior_etag: Previously observed entity tag for the
                resource.
            prior_last_modified: Prior ``Last-Modified`` timestamp.
            prior_sha256: SHA-256 checksum of the cached payload.
            prior_content_length: Byte length of the cached payload.
            prior_path: Filesystem path storing the cached artefact.

        Returns:
            None

        Raises:
            ValueError: If ``prior_content_length`` is provided but negative.
        """

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
        """Generate conditional request headers from cached metadata.

        Args:
            None

        Returns:
            Mapping[str, str]: Headers suitable for ``requests`` invocations.
        """

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
        """Classify origin responses as cached or modified results.

        Args:
            response (requests.Response): HTTP response returned from the
                conditional request.

        Returns:
            CachedResult | ModifiedResult: Cached metadata when the origin
            reports HTTP 304, otherwise wrapped metadata from a fresh download.

        Raises:
            ValueError: If a 304 response arrives without complete cached
                metadata.
            TypeError: If ``response`` lacks ``status_code`` or ``headers``.
        """

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
