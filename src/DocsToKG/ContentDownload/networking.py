# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.network",
#   "purpose": "HTTP session, retry, and conditional request helpers",
#   "sections": [
#     {
#       "id": "create-session",
#       "name": "create_session",
#       "anchor": "function-create-session",
#       "kind": "function"
#     },
#     {
#       "id": "parse-retry-after-header",
#       "name": "parse_retry_after_header",
#       "anchor": "function-parse-retry-after-header",
#       "kind": "function"
#     },
#     {
#       "id": "request-with-retries",
#       "name": "request_with_retries",
#       "anchor": "function-request-with-retries",
#       "kind": "function"
#     },
#     {
#       "id": "head-precheck",
#       "name": "head_precheck",
#       "anchor": "function-head-precheck",
#       "kind": "function"
#     },
#     {
#       "id": "looks-like-pdf",
#       "name": "_looks_like_pdf",
#       "anchor": "function-looks-like-pdf",
#       "kind": "function"
#     },
#     {
#       "id": "head-precheck-via-get",
#       "name": "_head_precheck_via_get",
#       "anchor": "function-head-precheck-via-get",
#       "kind": "function"
#     },
#     {
#       "id": "cachedresult",
#       "name": "CachedResult",
#       "anchor": "class-cachedresult",
#       "kind": "class"
#     },
#     {
#       "id": "modifiedresult",
#       "name": "ModifiedResult",
#       "anchor": "class-modifiedresult",
#       "kind": "class"
#     },
#     {
#       "id": "conditionalrequesthelper",
#       "name": "ConditionalRequestHelper",
#       "anchor": "class-conditionalrequesthelper",
#       "kind": "class"
#     },
#     {
#       "id": "circuitbreaker",
#       "name": "CircuitBreaker",
#       "anchor": "class-circuitbreaker",
#       "kind": "class"
#     },
#     {
#       "id": "tokenbucket",
#       "name": "TokenBucket",
#       "anchor": "class-tokenbucket",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

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

import hashlib
import logging
import random
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Optional, Set, Union

import requests
from requests.adapters import HTTPAdapter

# --- Globals ---

__all__ = (
    "CachedResult",
    "ConditionalRequestHelper",
    "ModifiedResult",
    "CircuitBreaker",
    "TokenBucket",
    "create_session",
    "head_precheck",
    "parse_retry_after_header",
    "request_with_retries",
)

LOGGER = logging.getLogger("DocsToKG.ContentDownload.network")


# --- Public Functions ---


def create_session(
    headers: Optional[Mapping[str, str]] = None,
    *,
    adapter_max_retries: int = 0,
    pool_connections: int = 64,
    pool_maxsize: int = 128,
) -> requests.Session:
    """Return a ``requests.Session`` configured for DocsToKG network requests.

    Args:
        headers: Optional mapping of HTTP headers applied to the session. The mapping
            is copied into the session's header store so caller-owned dictionaries remain
            unchanged.
        adapter_max_retries: Retry count configured on mounted HTTP adapters. Defaults to
            ``0`` so :func:`request_with_retries` governs retry behaviour.
        pool_connections: Lower bound of connection pools shared across the session's adapters.
        pool_maxsize: Maximum number of connections kept per host in the adapter pool.

    Returns:
        requests.Session: Session instance with HTTP/HTTPS adapters mounted and ready for pipeline usage.

    Raises:
        None.

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

    adapter = HTTPAdapter(
        max_retries=adapter_max_retries,
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize,
    )
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

    if not hasattr(session, "request") or not callable(getattr(session, "request")):
        raise AttributeError(f"Session object of type {type(session)!r} lacks callable 'request'.")

    def request_func(
        *,
        method: str,
        url: str,
        **call_kwargs: Any,
    ) -> requests.Response:
        """Invoke :meth:`requests.Session.request` on the provided session."""

        return session.request(method=method, url=url, **call_kwargs)

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


def head_precheck(
    session: requests.Session,
    url: str,
    timeout: float,
) -> bool:
    """Issue a lightweight request to determine whether ``url`` returns a PDF.

    The helper primarily relies on a HEAD request capped to a short timeout.
    Some providers respond with ``405`` or ``501`` for HEAD requests; in those
    cases a secondary streaming GET probe is issued that reads at most one
    chunk to infer the payload type without downloading the entire resource.

    Args:
        session: HTTP session used for outbound requests.
        url: Candidate download URL.
        timeout: Maximum time budget in seconds for the probe.

    Returns:
        ``True`` when the response appears to represent a binary payload such as
        a PDF. ``False`` when the response clearly corresponds to HTML or an
        error status. Any network exception results in ``True`` to avoid
        blocking legitimate downloads.
    """

    try:
        response = request_with_retries(
            session,
            "HEAD",
            url,
            max_retries=1,
            timeout=min(timeout, 5.0),
            allow_redirects=True,
        )
    except Exception:
        return True

    try:
        if response.status_code in {200, 302, 304}:
            return _looks_like_pdf(response.headers)
        if response.status_code in {405, 501}:
            return _head_precheck_via_get(session, url, timeout)
        return False
    finally:
        response.close()


# --- Private Helpers ---


def _looks_like_pdf(headers: Mapping[str, str]) -> bool:
    """Return ``True`` when response headers suggest a PDF payload."""

    content_type = (headers.get("Content-Type") or "").lower()
    content_length = (headers.get("Content-Length") or "").strip()

    if any(t in content_type for t in ("text/html", "text/plain")) or "json" in content_type:
        return False
    dispo = (headers.get("Content-Disposition") or "").lower()
    if "filename=" in dispo and ".pdf" in dispo:
        return True
    if content_length == "0":
        return False
    return True


def _head_precheck_via_get(
    session: requests.Session,
    url: str,
    timeout: float,
) -> bool:
    """Fallback GET probe for providers that reject HEAD requests."""

    try:
        with request_with_retries(
            session,
            "GET",
            url,
            stream=True,
            timeout=min(timeout, 5.0),
            allow_redirects=True,
        ) as response:
            # Consume at most one chunk to avoid downloading the entire body.
            try:
                next(response.iter_content(chunk_size=1024))
            except StopIteration:
                pass
            except Exception:
                # If iter_content raises, treat as inconclusive and allow download.
                return True

            if response.status_code not in {200, 302, 304}:
                return False
            return _looks_like_pdf(response.headers)
    except Exception:
        return True


# --- Public Classes ---


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

    Examples:
        >>> CachedResult(path="/tmp/file.pdf", sha256="abc", content_length=10, etag=None, last_modified=None)
        CachedResult(path='/tmp/file.pdf', sha256='abc', content_length=10, etag=None, last_modified=None)
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

    Examples:
        >>> ModifiedResult(etag="abc", last_modified="Tue, 15 Nov 1994 12:45:26 GMT")
        ModifiedResult(etag='abc', last_modified='Tue, 15 Nov 1994 12:45:26 GMT')
    """

    etag: Optional[str]
    last_modified: Optional[str]


class ConditionalRequestHelper:
    """Construct headers and interpret responses for conditional requests.

    The helper encapsulates cached metadata (ETag, Last-Modified, hashes) so the
    caller can generate polite conditional headers and validate upstream 304
    responses before reusing cached artefacts.

    Attributes:
        prior_etag: Cached entity tag from a previous download.
        prior_last_modified: Cached ``Last-Modified`` header value.
        prior_sha256: SHA-256 checksum of the cached content.
        prior_content_length: Cached payload length in bytes.
        prior_path: Filesystem path storing the cached artefact.

    Examples:
        >>> helper = ConditionalRequestHelper(prior_etag="abc123")
        >>> helper.build_headers()
        {'If-None-Match': 'abc123'}
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
        if self.prior_etag or self.prior_last_modified:
            missing: list[str] = []
            if not self.prior_sha256:
                missing.append("sha256")
            if self.prior_content_length is None:
                missing.append("content_length")
            if not self.prior_path:
                missing.append("path")
            if missing:
                LOGGER.warning(
                    "resume-metadata-incomplete: falling back to full fetch (missing %s)",
                    ", ".join(missing),
                    extra={
                        "reason": "resume-metadata-incomplete",
                        "missing_resume_fields": missing,
                    },
                )
                return {}
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

            cached_path = Path(self.prior_path)
            if not cached_path.exists():
                raise FileNotFoundError(
                    f"Cached artifact missing at {cached_path}; cannot reuse prior download."
                )

            try:
                actual_size = cached_path.stat().st_size
            except OSError as exc:  # pragma: no cover - defensive
                raise FileNotFoundError(
                    f"Unable to stat cached artifact at {cached_path}: {exc}"
                ) from exc

            if actual_size != self.prior_content_length:
                raise ValueError(
                    "Cached content length mismatch: "
                    f"expected {self.prior_content_length}, got {actual_size}"
                )

            if self.prior_sha256:
                hasher = hashlib.sha256()
                with cached_path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(8192), b""):
                        hasher.update(chunk)
                digest = hasher.hexdigest()
                if digest != self.prior_sha256:
                    raise ValueError(
                        "Cached SHA256 mismatch: " f"expected {self.prior_sha256}, got {digest}"
                    )

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


class CircuitBreaker:
    """Circuit breaker that opens after repeated failures for a cooldown period."""

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        cooldown_seconds: float = 60.0,
        name: str = "circuit",
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be >= 0")
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = float(cooldown_seconds)
        self.name = name
        self._lock = threading.Lock()
        self._failures = 0
        self._open_until: float = 0.0

    def allow(self, *, now: Optional[float] = None) -> bool:
        """Return ``True`` when the breaker permits a new attempt."""

        ts = now if now is not None else time.monotonic()
        with self._lock:
            return ts >= self._open_until

    def record_success(self) -> None:
        """Reset the breaker following a successful attempt."""

        with self._lock:
            self._failures = 0
            self._open_until = 0.0

    def record_failure(self, *, retry_after: Optional[float] = None) -> None:
        """Record a failure and open the breaker if the threshold is reached."""

        with self._lock:
            self._failures += 1
            if self._failures < self.failure_threshold:
                return
            cooldown = retry_after if retry_after is not None else self.cooldown_seconds
            self._open_until = time.monotonic() + max(cooldown, 0.0)
            self._failures = 0

    def cooldown_remaining(self, *, now: Optional[float] = None) -> float:
        """Return seconds remaining until the breaker closes."""

        ts = now if now is not None else time.monotonic()
        with self._lock:
            remaining = self._open_until - ts
            return remaining if remaining > 0 else 0.0


class TokenBucket:
    """Thread-safe token bucket for per-host rate limiting."""

    def __init__(
        self,
        *,
        rate_per_second: float,
        capacity: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if rate_per_second <= 0:
            raise ValueError("rate_per_second must be > 0")
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.rate_per_second = float(rate_per_second)
        self.capacity = float(capacity)
        self.clock = clock
        self._tokens = capacity
        self._lock = threading.Lock()
        self._last = clock()

    def _refill_locked(self, now: float) -> None:
        elapsed = max(now - self._last, 0.0)
        if elapsed <= 0:
            return
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate_per_second)
        self._last = now

    def acquire(self, tokens: float = 1.0, *, now: Optional[float] = None) -> float:
        """Consume tokens and return wait seconds required before proceeding."""

        if tokens <= 0:
            return 0.0
        ts = now if now is not None else self.clock()
        with self._lock:
            self._refill_locked(ts)
            if self._tokens >= tokens:
                self._tokens -= tokens
                return 0.0
            deficit = tokens - self._tokens
            wait = deficit / self.rate_per_second
            self._tokens = 0.0
            self._last = ts
            return wait

    def offer(self, tokens: float = 1.0, *, now: Optional[float] = None) -> bool:
        """Attempt to consume tokens without blocking; return ``True`` if granted."""

        if tokens <= 0:
            return True
        ts = now if now is not None else self.clock()
        with self._lock:
            self._refill_locked(ts)
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False


__all__ = [
    "CachedResult",
    "ConditionalRequestHelper",
    "ModifiedResult",
    "CircuitBreaker",
    "TokenBucket",
    "head_precheck",
    "create_session",
    "parse_retry_after_header",
    "request_with_retries",
]
