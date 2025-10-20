# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.networking",
#   "purpose": "HTTPX client helpers, retry orchestration, and conditional caching",
#   "sections": [
#     {
#       "id": "parse-retry-after-header",
#       "name": "parse_retry_after_header",
#       "anchor": "function-parse-retry-after-header",
#       "kind": "function"
#     },
#     {
#       "id": "contentpolicyviolation",
#       "name": "ContentPolicyViolation",
#       "anchor": "class-contentpolicyviolation",
#       "kind": "class"
#     },
#     {
#       "id": "retryafterjitterwait",
#       "name": "RetryAfterJitterWait",
#       "anchor": "class-retryafterjitterwait",
#       "kind": "class"
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
#       "id": "conditionalrequesthelper",
#       "name": "ConditionalRequestHelper",
#       "anchor": "class-conditionalrequesthelper",
#       "kind": "class"
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
#       "id": "circuitbreaker",
#       "name": "CircuitBreaker",
#       "anchor": "class-circuitbreaker",
#       "kind": "class"
#     },
#   ]
# }
# === /NAVMAP ===

"""Networking primitives and retry policies for DocsToKG content downloads.

Responsibilities
----------------
- Provide a shared, cached :class:`httpx.Client` (via
  :mod:`DocsToKG.ContentDownload.httpx_transport`) for all ContentDownload
  networking.
- Implement resilient request execution through
  :func:`request_with_retries`, delegating to a Tenacity controller that
  combines exponential backoff with jitter, honours ``Retry-After``
  directives, respects CLI ceilings (``retry_after_cap`` /
  ``max_retry_duration``), and applies content-policy enforcement.
- Provide conditional request tooling (:class:`ConditionalRequestHelper`,
  :class:`CachedResult`, :class:`ModifiedResult`) so resolvers can revalidate
  cached artifacts without redownloading payloads unnecessarily.
- Support streaming workflows by surfacing context managers when ``stream=True``
  so download helpers can iteratively write large payloads without buffering.
- Surface failure-suppression primitives (:class:`CircuitBreaker`) used by the
  pipeline, while deferring centralized rate limiting to
  :mod:`DocsToKG.ContentDownload.ratelimit`.
- Expose diagnostic helpers such as :func:`head_precheck` (with GET fallback)
  and :func:`parse_retry_after_header` to keep request policy decisions
- ``request_with_retries`` – wraps HTTP verbs with Tenacity-backed retry,
  backoff, and logging.
- ``ConditionalRequestHelper`` – produces ``If-None-Match``/``If-Modified-Since`` headers.
- ``CircuitBreaker`` – stateful regulator shared across resolvers and download workers.

Typical Usage
-------------
    from DocsToKG.ContentDownload.networking import (
        ConditionalRequestHelper,
        request_with_retries,
    )

    response = request_with_retries(None, "GET", "https://example.org")
    helper = ConditionalRequestHelper(prior_etag="abc123")
    headers = helper.build_headers()
"""

from __future__ import annotations

import contextlib
import logging
import math
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Set, Union, cast
from urllib.parse import urlparse

import httpx
from tenacity import (
    RetryCallState,
    RetryError,
    Retrying,
    before_sleep_log,
    retry_if_exception_type,
    retry_if_result,
    stop_after_attempt,
    stop_after_delay,
    wait_random_exponential,
)
from tenacity.wait import wait_base

from DocsToKG.ContentDownload.httpx_transport import (
    configure_http_client,
    get_http_client,
    purge_http_cache,
)
from DocsToKG.ContentDownload.ratelimit import DEFAULT_ROLE
from DocsToKG.ContentDownload.urls import canonical_for_index, canonical_for_request

# --- Globals ---

__all__ = (
    "CachedResult",
    "ConditionalRequestHelper",
    "ModifiedResult",
    "ContentPolicyViolation",
    "CircuitBreaker",
    "configure_http_client",
    "get_http_client",
    "purge_http_cache",
    "head_precheck",
    "parse_retry_after_header",
    "request_with_retries",
)

LOGGER = logging.getLogger("DocsToKG.ContentDownload.network")

DEFAULT_RETRYABLE_STATUSES: Set[int] = {429, 500, 502, 503, 504}
_TENACITY_BEFORE_SLEEP_LOG = before_sleep_log(LOGGER, logging.DEBUG)

_DEPRECATED_ATTR_ERRORS: Dict[str, str] = {
    "ThreadLocalSessionFactory": (
        "ThreadLocalSessionFactory has been removed; ContentDownload now shares a singleton "
        "HTTPX client. Patch DocsToKG.ContentDownload.httpx_transport.get_http_client() instead."
    ),
    "create_session": (
        "create_session() has been removed; configure or patch the HTTPX client via "
        "DocsToKG.ContentDownload.httpx_transport instead."
    ),
}


def __getattr__(name: str) -> Any:
    """Provide explicit errors for legacy session factory access."""

    if name in _DEPRECATED_ATTR_ERRORS:
        raise RuntimeError(_DEPRECATED_ATTR_ERRORS[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def parse_retry_after_header(response: httpx.Response) -> Optional[float]:
    """Parse ``Retry-After`` header and return wait time in seconds.

    Args:
        response (httpx.Response): HTTP response potentially containing a
            ``Retry-After`` header.

    Returns:
        float | None: Seconds the caller should wait before retrying, or
        ``None`` when the header is absent or invalid.

    Raises:
        None: Invalid headers are tolerated and yield ``None`` without raising.

    Examples:
        >>> # Integer format
        >>> parse_retry_after_header(httpx.Response(503, headers={"Retry-After": "5"}))
        5.0

        >>> # HTTP-date format
        >>> isinstance(parse_retry_after_header(httpx.Response(503, headers={"Retry-After": "Wed, 21 Oct 2025 07:28:00 GMT"})), float)
        True
    """

    retry_after = response.headers.get("Retry-After")
    if not retry_after:
        return None

    try:
        seconds = float(retry_after)
    except (TypeError, ValueError):
        pass
    else:
        if seconds > 0.0 and math.isfinite(seconds):
            return seconds
        return None

    try:
        target_time = parsedate_to_datetime(retry_after)
    except Exception:
        return None

    if target_time is None:
        return None
    if target_time.tzinfo is None:
        target_time = target_time.replace(tzinfo=timezone.utc)

    delta = (target_time - datetime.now(timezone.utc)).total_seconds()
    if delta > 0.0 and math.isfinite(delta):
        return delta
    return None


class ContentPolicyViolation(httpx.HTTPError):
    """Raised when a response violates configured content policies."""

    def __init__(
        self,
        message: str,
        *,
        violation: str,
        policy: Optional[Mapping[str, Any]] = None,
        detail: Optional[str] = None,
        content_type: Optional[str] = None,
        content_length: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.violation = violation
        self.policy = dict(policy) if policy else None
        self.detail = detail
        self.content_type = content_type
        self.content_length = content_length


def _normalise_content_type(value: str) -> str:
    """Return the canonical MIME type component from a Content-Type header."""

    if not value:
        return ""
    return value.split(";", 1)[0].strip().lower()


def _enforce_content_policy(
    response: httpx.Response,
    content_policy: Optional[Mapping[str, Any]],
    *,
    method: str,
    url: str,
) -> None:
    """Raise ContentPolicyViolation when response headers violate policy."""

    if not content_policy:
        return
    headers = getattr(response, "headers", {}) or {}
    allowed_types = content_policy.get("allowed_types")

    content_type_header = headers.get("Content-Type")
    content_type = _normalise_content_type(content_type_header or "")
    if allowed_types and content_type:
        allowed_set = set(allowed_types)
        if content_type not in allowed_set:
            with contextlib.suppress(Exception):
                response.close()
            detail = f"content-type {content_type!r} not in allow-list"
            raise ContentPolicyViolation(
                f"{method} {url} blocked by content policy",
                violation="content-type",
                policy=content_policy,
                detail=detail,
                content_type=content_type,
            )


class RetryAfterJitterWait(wait_base):
    """Tenacity wait strategy that honours ``Retry-After`` headers."""

    def __init__(
        self,
        *,
        respect_retry_after: bool,
        retry_after_cap: Optional[float],
        backoff_max: Optional[float],
        retry_statuses: Set[int],
        fallback_wait: wait_base,
    ) -> None:
        self._respect_retry_after = respect_retry_after
        self._retry_after_cap = retry_after_cap
        if backoff_max is None:
            self._backoff_max: Optional[float] = None
        else:
            self._backoff_max = float(max(backoff_max, 0.0))
        self._retry_statuses = set(retry_statuses)
        self._fallback_wait = fallback_wait

    def _compute_retry_after(self, response: httpx.Response) -> Optional[float]:
        if not self._respect_retry_after:
            return None

        status = getattr(response, "status_code", None)
        if status not in self._retry_statuses or status not in {429, 503}:
            return None

        retry_after = parse_retry_after_header(response)
        if retry_after is None:
            return None

        if self._retry_after_cap is not None:
            retry_after = min(retry_after, self._retry_after_cap)

        if self._backoff_max is not None and self._backoff_max > 0.0:
            retry_after = min(retry_after, self._backoff_max)

        return retry_after

    def __call__(self, retry_state: RetryCallState) -> float:
        fallback_delay = float(self._fallback_wait(retry_state))

        outcome = retry_state.outcome
        if outcome is None:
            return max(0.0, fallback_delay)

        response: Optional[httpx.Response]
        if outcome.failed:
            exc = outcome.exception()
            response = getattr(exc, "response", None) if exc is not None else None
            if response is None:
                return max(0.0, fallback_delay)
        else:
            result = outcome.result()
            if not isinstance(result, httpx.Response):
                return max(0.0, fallback_delay)
            response = result

        if not isinstance(response, httpx.Response):
            return max(0.0, fallback_delay)

        status = getattr(response, "status_code", None)
        if status not in self._retry_statuses:
            return max(0.0, fallback_delay)

        retry_after_delay = self._compute_retry_after(response)
        if retry_after_delay is None:
            return max(0.0, fallback_delay)

        return max(0.0, retry_after_delay)


def _close_response_safely(response: Optional[httpx.Response]) -> None:
    if response is None:
        return
    with contextlib.suppress(Exception):
        response.close()


def _before_sleep_close_response(retry_state: RetryCallState) -> None:
    """Close prior responses and log Tenacity retry metadata."""

    metadata = getattr(retry_state.retry_object, "_docs_retry_meta", {})
    method = metadata.get("method", "")
    url = metadata.get("url", "")
    max_attempts = metadata.get("max_attempts")

    delay = 0.0
    if retry_state.next_action is not None and retry_state.next_action.sleep is not None:
        delay = float(retry_state.next_action.sleep)

    sleep_accumulator = getattr(retry_state.retry_object, "_docs_retry_sleep", 0.0)
    setattr(retry_state.retry_object, "_docs_retry_sleep", sleep_accumulator + max(delay, 0.0))

    outcome = retry_state.outcome
    if outcome is None:
        LOGGER.debug(
            "Retrying %s %s with unknown outcome (attempt %s, delay %.2fs)",
            method,
            url,
            retry_state.attempt_number,
            delay,
        )
        return

    if outcome.failed:
        exc = outcome.exception()
        LOGGER.debug(
            "Retrying %s %s after exception %s (attempt %s%s, delay %.2fs)",
            method,
            url,
            exc,
            retry_state.attempt_number,
            f"/{max_attempts}" if max_attempts else "",
            delay,
        )
        _close_response_safely(getattr(exc, "response", None))
        return

    response = outcome.result()
    status = getattr(response, "status_code", "?")
    LOGGER.debug(
        "Retrying %s %s after HTTP %s (attempt %s%s, delay %.2fs)",
        method,
        url,
        status,
        retry_state.attempt_number,
        f"/{max_attempts}" if max_attempts else "",
        delay,
    )
    _close_response_safely(response)


def _before_sleep_handler(retry_state: RetryCallState) -> None:
    _before_sleep_close_response(retry_state)
    _TENACITY_BEFORE_SLEEP_LOG(retry_state)


def _is_retryable_response(response: Any, retry_statuses: Set[int]) -> bool:
    if not isinstance(response, httpx.Response):
        return False
    status_code = getattr(response, "status_code", None)
    return bool(status_code in retry_statuses)


def _build_retrying_controller(
    *,
    method: str,
    url: str,
    max_retries: int,
    retry_statuses: Set[int],
    backoff_factor: float,
    backoff_max: Optional[float],
    respect_retry_after: bool,
    retry_after_cap: Optional[float],
    max_retry_duration: Optional[float],
) -> Retrying:
    fallback_wait = wait_random_exponential(multiplier=backoff_factor, max=backoff_max)
    wait_strategy = RetryAfterJitterWait(
        respect_retry_after=respect_retry_after,
        retry_after_cap=retry_after_cap,
        backoff_max=backoff_max,
        retry_statuses=retry_statuses,
        fallback_wait=fallback_wait,
    )

    retry_condition = retry_if_exception_type(
        (httpx.TimeoutException, httpx.TransportError, httpx.ProtocolError)
    ) | retry_if_result(lambda result: _is_retryable_response(result, retry_statuses))

    stop_strategy = stop_after_attempt(max_retries + 1)
    if max_retry_duration is not None and max_retry_duration > 0:
        stop_strategy = stop_strategy | stop_after_delay(max_retry_duration)

    def _retry_error_callback(retry_state: RetryCallState) -> Any:
        outcome = retry_state.outcome
        metadata = getattr(retry_state.retry_object, "_docs_retry_meta", {})
        method_name = metadata.get("method", "")
        target_url = metadata.get("url", "")
        attempts = retry_state.attempt_number
        duration_cap = metadata.get("max_retry_duration")

        if outcome is None:
            raise RetryError(retry_state=retry_state)

        if outcome.failed:
            raise RetryError(retry_state=retry_state)

        response = outcome.result()
        status = getattr(response, "status_code", None)

        if duration_cap:
            if (
                retry_state.seconds_since_start is not None
                and retry_state.seconds_since_start >= duration_cap
            ):
                LOGGER.warning(
                    "Exceeded max retry duration %.1fs for %s %s; returning final response",
                    duration_cap,
                    method_name,
                    target_url,
                )
            else:
                LOGGER.warning(
                    "Retry budget exhausted for %s %s after %s attempts; returning final response (status=%s)",
                    method_name,
                    target_url,
                    attempts,
                    status,
                )
        else:
            LOGGER.warning(
                "Retry budget exhausted for %s %s after %s attempts; returning final response (status=%s)",
                method_name,
                target_url,
                attempts,
                status,
            )
        return response

    retrying = Retrying(
        retry=retry_condition,
        wait=wait_strategy,
        stop=stop_strategy,
        sleep=time.sleep,
        reraise=True,
        before_sleep=_before_sleep_handler,
        retry_error_callback=_retry_error_callback,
    )

    retrying._docs_retry_meta = {
        "method": method,
        "url": url,
        "max_attempts": max_retries + 1,
        "max_retry_duration": max_retry_duration,
        "retry_statuses": retry_statuses,
    }
    retrying._docs_retry_sleep = 0.0

    return retrying


def request_with_retries(
    client: Optional[httpx.Client],
    method: str,
    url: str,
    *,
    role: str = DEFAULT_ROLE,
    origin_host: Optional[str] = None,
    original_url: Optional[str] = None,
    max_retries: int = 3,
    retry_statuses: Optional[Set[int]] = None,
    backoff_factor: float = 0.75,
    respect_retry_after: bool = True,
    retry_after_cap: Optional[float] = None,
    content_policy: Optional[Mapping[str, Any]] = None,
    max_retry_duration: Optional[float] = None,
    backoff_max: Optional[float] = 60.0,
    **kwargs: Any,
) -> httpx.Response:
    """Execute an HTTP request using a Tenacity-backed retry controller."""

    http_client = client or get_http_client()

    if not method:
        raise ValueError("HTTP method must be provided")
    if not url:
        raise ValueError("URL must be provided")
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    if backoff_factor < 0:
        raise ValueError("backoff_factor must be non-negative")
    if backoff_max is not None and backoff_max < 0:
        raise ValueError("backoff_max must be non-negative")
    if retry_after_cap is not None and retry_after_cap < 0:
        raise ValueError("retry_after_cap must be non-negative when provided")
    if max_retry_duration is not None and max_retry_duration <= 0:
        raise ValueError("max_retry_duration must be positive when provided")

    role_token = (role or DEFAULT_ROLE).lower()
    policy_role = role_token if role_token in {"landing", "artifact"} else "metadata"
    url_role = cast(str, policy_role)
    source_url = original_url or url

    host_hint = origin_host
    if host_hint is None:
        try:
            parsed_original = urlparse(source_url)
        except Exception:
            parsed_original = None
        if parsed_original is not None:
            host_hint = (parsed_original.hostname or parsed_original.netloc or "").lower() or None

    try:
        request_url = canonical_for_request(source_url, role=url_role, origin_host=host_hint)
    except Exception:
        request_url = source_url

    try:
        canonical_index = canonical_for_index(source_url)
    except Exception:
        canonical_index = request_url

    if retry_after_cap is not None:
        retry_after_cap = float(retry_after_cap)
    if max_retry_duration is not None:
        max_retry_duration = float(max_retry_duration)

    backoff_factor = float(backoff_factor)
    normalized_backoff_max = float(backoff_max) if backoff_max is not None else None

    retry_statuses = (
        set(retry_statuses) if retry_statuses is not None else set(DEFAULT_RETRYABLE_STATUSES)
    )
    retry_statuses = set(retry_statuses)

    timeout = kwargs.pop("timeout", None)
    if timeout is not None:
        if isinstance(timeout, httpx.Timeout):
            http_timeout = timeout
        elif isinstance(timeout, (int, float)):
            timeout_value = float(timeout)
            if timeout_value <= 0:
                raise ValueError("timeout must be positive when provided as a float")
            http_timeout = httpx.Timeout(timeout_value)
        elif isinstance(timeout, tuple) and len(timeout) == 2:
            connect_timeout, read_timeout = timeout
            if connect_timeout is not None and connect_timeout <= 0:
                raise ValueError("connect timeout must be positive")
            if read_timeout is not None and read_timeout <= 0:
                raise ValueError("read timeout must be positive")
            http_timeout = httpx.Timeout(connect=connect_timeout, read=read_timeout)
        elif isinstance(timeout, Mapping):
            http_timeout = httpx.Timeout(**timeout)
        else:
            raise TypeError(
                "timeout must be a float/int, mapping, httpx.Timeout, or a (connect, read) tuple when provided"
            )
        kwargs["timeout"] = http_timeout

    allow_redirects = kwargs.pop("allow_redirects", None)
    if allow_redirects is not None:
        kwargs["follow_redirects"] = bool(allow_redirects)

    stream_enabled = bool(kwargs.pop("stream", False))

    extensions = kwargs.pop("extensions", None)
    if extensions is None:
        extensions = {}
    else:
        extensions = dict(extensions)
    extensions.setdefault("role", role_token)
    extensions.setdefault("docs_original_url", source_url)
    extensions.setdefault("docs_canonical_url", request_url)
    extensions.setdefault("docs_canonical_index", canonical_index)
    kwargs["extensions"] = extensions

    def request_func(*, method: str, url: str, **call_kwargs: Any) -> httpx.Response:
        if stream_enabled:
            return http_client.stream(method=method, url=url, **call_kwargs)
        return http_client.request(method=method, url=url, **call_kwargs)

    controller = _build_retrying_controller(
        method=method,
        url=request_url,
        max_retries=max_retries,
        retry_statuses=retry_statuses,
        backoff_factor=backoff_factor,
        backoff_max=normalized_backoff_max,
        respect_retry_after=respect_retry_after,
        retry_after_cap=retry_after_cap,
        max_retry_duration=max_retry_duration,
    )

    try:
        response = controller(
            request_func,
            method=method,
            url=request_url,
            **kwargs,
        )
    except RetryError as exc:  # pragma: no cover - defensive safety net
        last_attempt = exc.last_attempt
        if last_attempt.failed:
            raise last_attempt.exception()
        response = last_attempt.result()

    attempts = controller.statistics.get("attempt_number", 1)
    total_sleep = getattr(controller, "_docs_retry_sleep", 0.0)

    LOGGER.debug(
        "Completed %s %s after %s attempt(s) with %.2fs cumulative sleep",
        method,
        request_url,
        attempts,
        total_sleep,
    )

    try:
        _enforce_content_policy(response, content_policy, method=method, url=request_url)
    except AttributeError:  # response lacks headers/status
        LOGGER.debug(
            "Response object %s lacks headers for content policy evaluation.",
            type(response).__name__,
        )

    if not isinstance(response, httpx.Response):
        LOGGER.debug(
            "Response object of type %s lacks status_code; treating as success for %s %s.",
            type(response).__name__,
            method,
            request_url,
        )
        return response

    return response


def head_precheck(
    client: Optional[httpx.Client],
    url: str,
    timeout: float,
    *,
    content_policy: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Issue a lightweight request to determine whether ``url`` returns a PDF.

    The helper primarily relies on a HEAD request capped to a short timeout.
    Some providers respond with ``405`` or ``501`` for HEAD requests; in those
    cases a secondary streaming GET probe is issued that reads at most one
    chunk to infer the payload type without downloading the entire resource.

    Args:
        client: Optional HTTPX client override. When ``None``, the shared client is used.
        url: Candidate download URL.
        timeout: Maximum time budget in seconds for the probe.
        content_policy: Optional domain-specific content policy to enforce.

    Returns:
        ``True`` when the response appears to represent a binary payload such as
        a PDF. ``False`` when the response clearly corresponds to HTML or an
        error status. Any network exception results in ``True`` to avoid
        blocking legitimate downloads.
    """

    try:
        response = request_with_retries(
            client,
            "HEAD",
            url,
            role=DEFAULT_ROLE,
            max_retries=1,
            timeout=min(timeout, 5.0),
            allow_redirects=True,
            content_policy=content_policy,
            max_retry_duration=min(timeout, 5.0),
            backoff_max=min(timeout, 5.0),
            retry_after_cap=min(timeout, 5.0),
        )
    except ContentPolicyViolation:
        return False
    except Exception:
        return True

    try:
        if response.status_code in {200, 302, 304}:
            return _looks_like_pdf(response.headers)
        if response.status_code in {405, 501}:
            return _head_precheck_via_get(client, url, timeout, content_policy=content_policy)
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
    client: Optional[httpx.Client],
    url: str,
    timeout: float,
    *,
    content_policy: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Fallback GET probe for providers that reject HEAD requests."""

    try:
        with request_with_retries(
            client,
            "GET",
            url,
            role=DEFAULT_ROLE,
            max_retries=1,
            stream=True,
            timeout=min(timeout, 5.0),
            allow_redirects=True,
            content_policy=content_policy,
            max_retry_duration=min(timeout, 5.0),
            backoff_max=min(timeout, 5.0),
            retry_after_cap=min(timeout, 5.0),
        ) as response:
            # Consume at most one chunk to avoid downloading the entire body.
            try:
                next(response.iter_bytes(chunk_size=1024))
            except StopIteration:
                pass
            except Exception:
                return True

            if response.status_code not in {200, 302, 304}:
                return False
            return _looks_like_pdf(response.headers)
    except ContentPolicyViolation:
        return False
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
    recorded_mtime_ns: Optional[int] = None


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
        prior_mtime_ns: Optional[int] = None,
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
        self.prior_mtime_ns = prior_mtime_ns

    def build_headers(self) -> Mapping[str, str]:
        """Generate conditional request headers from cached metadata.

        Args:
            None

        Returns:
            Mapping[str, str]: Headers suitable for HTTPX invocations.
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
        response: httpx.Response,
    ) -> Union[CachedResult, ModifiedResult]:
        """Classify origin responses as cached or modified results.

        Args:
            response (httpx.Response): HTTP response returned from the
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

            return CachedResult(
                path=self.prior_path,
                sha256=self.prior_sha256,
                content_length=self.prior_content_length,
                etag=self.prior_etag,
                last_modified=self.prior_last_modified,
                recorded_mtime_ns=self.prior_mtime_ns,
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


__all__ = [
    "CachedResult",
    "ConditionalRequestHelper",
    "ModifiedResult",
    "ContentPolicyViolation",
    "CircuitBreaker",
    "configure_http_client",
    "get_http_client",
    "purge_http_cache",
    "head_precheck",
    "parse_retry_after_header",
    "request_with_retries",
]
