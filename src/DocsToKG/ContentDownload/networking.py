# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.networking",
#   "purpose": "HTTPX client helpers, retry orchestration, and conditional caching",
#   "sections": [
#     {
#       "id": "configure-breaker-registry",
#       "name": "configure_breaker_registry",
#       "anchor": "function-configure-breaker-registry",
#       "kind": "function"
#     },
#     {
#       "id": "set-breaker-registry",
#       "name": "set_breaker_registry",
#       "anchor": "function-set-breaker-registry",
#       "kind": "function"
#     },
#     {
#       "id": "reset-breaker-registry",
#       "name": "reset_breaker_registry",
#       "anchor": "function-reset-breaker-registry",
#       "kind": "function"
#     },
#     {
#       "id": "get-breaker-registry",
#       "name": "get_breaker_registry",
#       "anchor": "function-get-breaker-registry",
#       "kind": "function"
#     },
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
#       "id": "normalise-content-type",
#       "name": "_normalise_content_type",
#       "anchor": "function-normalise-content-type",
#       "kind": "function"
#     },
#     {
#       "id": "enforce-content-policy",
#       "name": "_enforce_content_policy",
#       "anchor": "function-enforce-content-policy",
#       "kind": "function"
#     },
#     {
#       "id": "retryafterjitterwait",
#       "name": "RetryAfterJitterWait",
#       "anchor": "class-retryafterjitterwait",
#       "kind": "class"
#     },
#     {
#       "id": "close-response-safely",
#       "name": "_close_response_safely",
#       "anchor": "function-close-response-safely",
#       "kind": "function"
#     },
#     {
#       "id": "before-sleep-close-response",
#       "name": "_before_sleep_close_response",
#       "anchor": "function-before-sleep-close-response",
#       "kind": "function"
#     },
#     {
#       "id": "before-sleep-handler",
#       "name": "_before_sleep_handler",
#       "anchor": "function-before-sleep-handler",
#       "kind": "function"
#     },
#     {
#       "id": "is-retryable-response",
#       "name": "_is_retryable_response",
#       "anchor": "function-is-retryable-response",
#       "kind": "function"
#     },
#     {
#       "id": "build-retrying-controller",
#       "name": "_build_retrying_controller",
#       "anchor": "function-build-retrying-controller",
#       "kind": "function"
#     },
#     {
#       "id": "compute-url-hash",
#       "name": "_compute_url_hash",
#       "anchor": "function-compute-url-hash",
#       "kind": "function"
#     },
#     {
#       "id": "extract-from-cache",
#       "name": "_extract_from_cache",
#       "anchor": "function-extract-from-cache",
#       "kind": "function"
#     },
#     {
#       "id": "extract-revalidated",
#       "name": "_extract_revalidated",
#       "anchor": "function-extract-revalidated",
#       "kind": "function"
#     },
#     {
#       "id": "extract-stale",
#       "name": "_extract_stale",
#       "anchor": "function-extract-stale",
#       "kind": "function"
#     },
#     {
#       "id": "extract-retry-after",
#       "name": "_extract_retry_after",
#       "anchor": "function-extract-retry-after",
#       "kind": "function"
#     },
#     {
#       "id": "extract-rate-delay",
#       "name": "_extract_rate_delay",
#       "anchor": "function-extract-rate-delay",
#       "kind": "function"
#     },
#     {
#       "id": "extract-breaker-state",
#       "name": "_extract_breaker_state",
#       "anchor": "function-extract-breaker-state",
#       "kind": "function"
#     },
#     {
#       "id": "extract-breaker-recorded",
#       "name": "_extract_breaker_recorded",
#       "anchor": "function-extract-breaker-recorded",
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
#     }
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
- Surface failure-suppression primitives (legacy CircuitBreaker removed) used by the
  pipeline, while deferring centralized rate limiting to
  :mod:`DocsToKG.ContentDownload.ratelimit`.
- Expose diagnostic helpers such as :func:`head_precheck` (with GET fallback)
  and :func:`parse_retry_after_header` to keep request policy decisions
- ``request_with_retries`` – wraps HTTP verbs with Tenacity-backed retry,
  backoff, and logging.
- ``ConditionalRequestHelper`` – produces ``If-None-Match``/``If-Modified-Since`` headers.
- Legacy CircuitBreaker removed – now handled by pybreaker-based BreakerRegistry.

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
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Mapping,
    MutableMapping,
    Optional,
    Set,
    Union,
    cast,
)
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
from DocsToKG.ContentDownload.urls import canonical_for_index, canonical_for_request

# Default role for requests
DEFAULT_ROLE = "metadata"

try:  # pragma: no cover - optional when pybreaker missing in envs
    from DocsToKG.ContentDownload.breakers import (
        BreakerClassification,
        BreakerOpenError,
        BreakerRegistry,
        RequestRole,
    )
except Exception:  # pragma: no cover - pybreaker absent, keep networking importable
    BreakerOpenError = None  # type: ignore[assignment]
    RequestRole = None  # type: ignore[assignment]
    is_failure_for_breaker = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - typing-only imports
    from DocsToKG.ContentDownload.breakers import (
        BreakerConfig,
        BreakerListenerFactory,
        BreakerRegistry,
        CooldownStore,
    )

# --- Globals ---

__all__ = (
    "CachedResult",
    "ConditionalRequestHelper",
    "ModifiedResult",
    "ContentPolicyViolation",
    # "CircuitBreaker",  # Legacy - removed
    "BreakerOpenError",
    "configure_breaker_registry",
    "configure_http_client",
    "get_http_client",
    "get_breaker_registry",
    "purge_http_cache",
    "reset_breaker_registry",
    "head_precheck",
    "parse_retry_after_header",
    "request_with_retries",
    "set_breaker_registry",
)

LOGGER = logging.getLogger("DocsToKG.ContentDownload.network")

DEFAULT_RETRYABLE_STATUSES: Set[int] = {429, 500, 502, 503, 504}
_TENACITY_BEFORE_SLEEP_LOG = before_sleep_log(LOGGER, logging.DEBUG)

# Global breaker registry (singleton managed here)
_BREAKER_LOCK = threading.RLock()
_breaker_registry: Optional["BreakerRegistry"] = None

# Exceptions considered breaker failures by default
# (REMOVED DEFAULT_BREAKER_FAILURE_EXCEPTIONS - use BreakerClassification().failure_statuses instead)


def configure_breaker_registry(
    config: "BreakerConfig",
    *,
    cooldown_store: Optional["CooldownStore"] = None,
    listener_factory: Optional["BreakerListenerFactory"] = None,
) -> "BreakerRegistry":
    """Create and register the process-wide breaker registry."""

    from DocsToKG.ContentDownload.breakers import BreakerRegistry  # Local import to avoid cycles

    classify = config.classify
    if not classify.failure_exceptions:
        classify = replace(classify, failure_exceptions=BreakerClassification().failure_exceptions)
        config = replace(config, classify=classify)

    registry = BreakerRegistry(
        config,
        cooldown_store=cooldown_store,
        listener_factory=listener_factory,
    )

    set_breaker_registry(registry)
    return registry


def set_breaker_registry(registry: Optional["BreakerRegistry"]) -> None:
    """Inject an already-created breaker registry (primarily for tests)."""

    global _breaker_registry
    with _BREAKER_LOCK:
        _breaker_registry = registry


def reset_breaker_registry() -> None:
    """Clear the global breaker registry (used in teardown paths)."""

    set_breaker_registry(None)


def get_breaker_registry() -> Optional["BreakerRegistry"]:
    """Return the currently configured breaker registry, if any."""

    with _BREAKER_LOCK:
        return _breaker_registry


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


# --- Telemetry Helpers (Phase 1: HTTP Layer Instrumentation) ---


def _compute_url_hash(url: str) -> str:
    """Hash URL for privacy (SHA256, first 16 chars)."""
    import hashlib

    try:
        return hashlib.sha256(url.encode()).hexdigest()[:16]
    except Exception:
        return "unknown"


def _extract_from_cache(response: httpx.Response) -> Optional[int]:
    """Extract cache hit status from response extensions."""
    try:
        if not hasattr(response, "extensions"):
            return None
        extensions = response.extensions or {}
        # Hishel sets 'from_cache' flag
        if "from_cache" in extensions:
            return 1 if extensions.get("from_cache") else 0
        cache_status = extensions.get("cache_status")
        if cache_status == "HIT":
            return 1
        elif cache_status in ("MISS", "EXPIRED"):
            return 0
    except Exception:
        pass
    return None


def _extract_revalidated(response: httpx.Response) -> Optional[int]:
    """Return 1 if response was a 304 revalidation, else 0 or None."""
    try:
        return 1 if response.status_code == 304 else 0
    except Exception:
        pass
    return None


def _extract_stale(response: httpx.Response) -> Optional[int]:
    """Extract stale flag (SWrV) from response extensions."""
    try:
        if not hasattr(response, "extensions"):
            return None
        extensions = response.extensions or {}
        return 1 if extensions.get("stale") else 0
    except Exception:
        pass
    return None


def _extract_retry_after(response: httpx.Response) -> Optional[int]:
    """Extract Retry-After header value in seconds."""
    try:
        retry_after_str = response.headers.get("Retry-After")
        if retry_after_str:
            return int(float(retry_after_str))
    except (ValueError, TypeError, AttributeError):
        pass
    return None


def _extract_rate_delay(network_meta: Dict[str, Any]) -> Optional[int]:
    """Extract rate limiter wait time from docs_network_meta."""
    try:
        if isinstance(network_meta, dict):
            rate_info = network_meta.get("rate_limiter") or {}
            if isinstance(rate_info, dict):
                delay = rate_info.get("wait_ms")
                if isinstance(delay, (int, float)):
                    return int(delay)
    except Exception:
        pass
    return None


def _extract_breaker_state(breaker_state_info: Dict[str, Any]) -> Optional[str]:
    """Extract circuit breaker state: closed/half_open/open."""
    try:
        if isinstance(breaker_state_info, dict):
            host_state = breaker_state_info.get("breaker_host_state")
            if host_state:
                state_str = str(host_state).lower()
                if "half" in state_str:
                    return "half_open"
                elif "open" in state_str:
                    return "open"
                else:
                    return "closed"
    except Exception:
        pass
    return None


def _extract_breaker_recorded(breaker_state_info: Dict[str, Any]) -> Optional[str]:
    """Extract breaker recorded outcome: success/failure/none."""
    try:
        if isinstance(breaker_state_info, dict):
            recorded = breaker_state_info.get("breaker_recorded")
            if recorded in ("success", "failure", "none"):
                return recorded
    except Exception:
        pass
    return None


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
    resolver: Optional[str] = None,
    telemetry: Optional[Any] = None,
    run_id: Optional[str] = None,
    **kwargs: Any,
) -> httpx.Response:
    """Execute an HTTP request using a Tenacity-backed retry controller."""

    # Capture start time for telemetry
    request_start_time = time.time()

    http_client = client if isinstance(client, httpx.Client) else None
    if http_client is None:
        http_client = get_http_client()

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

    breaker_registry = get_breaker_registry()
    breaker_meta: Dict[str, Any] = {}
    breaker_enabled = (
        breaker_registry is not None
        and RequestRole is not None
        and BreakerOpenError is not None
        and is_failure_for_breaker is not None
    )

    parsed_request = urlparse(request_url)
    request_host = (parsed_request.hostname or parsed_request.netloc or "").lower()

    role_enum: Optional[RequestRole] = None
    if breaker_enabled:
        assert RequestRole is not None  # for type checkers
        role_enum = RequestRole.METADATA
        if role_token == "landing":
            role_enum = RequestRole.LANDING
        elif role_token == "artifact":
            role_enum = RequestRole.ARTIFACT

    if breaker_enabled:
        if role_enum is not None and hasattr(role_enum, "value"):
            breaker_meta["role"] = role_enum.value
        elif role_token:
            breaker_meta["role"] = role_token
        if resolver:
            breaker_meta["resolver"] = resolver

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

    # URL normalization instrumentation (Phase 3A integration)
    from DocsToKG.ContentDownload.urls_networking import (
        apply_role_headers,
        log_url_change_once,
        record_url_normalization,
    )

    try:
        record_url_normalization(source_url, request_url, url_role)
    except ValueError as e:
        LOGGER.error(f"URL normalization failed in strict mode: {e}")
        raise

    # Apply role-based headers (metadata/landing/artifact)
    if "headers" not in kwargs:
        kwargs["headers"] = {}
    kwargs["headers"] = apply_role_headers(kwargs.get("headers"), url_role)

    # Log URL changes once per host to avoid spam
    if request_url != source_url:
        log_url_change_once(source_url, request_url, host_hint)

    # Track in extensions for downstream access
    extensions.setdefault("docs_url_changed", request_url != source_url)

    network_meta_raw = extensions.get("docs_network_meta")
    if isinstance(network_meta_raw, MutableMapping):
        network_meta = network_meta_raw
    else:
        network_meta = {}
        extensions["docs_network_meta"] = network_meta

    def _apply_breaker_meta() -> None:
        if breaker_meta:
            network_meta["breaker"] = dict(breaker_meta)

    def _snapshot_breaker_state(
        *, recorded: Optional[str] = None, retry_after: Optional[float] = None
    ) -> None:
        if recorded is not None:
            breaker_meta["recorded"] = recorded
        if retry_after is not None:
            breaker_meta["retry_after_s"] = retry_after

        if not breaker_enabled or not request_host or breaker_registry is None:
            _apply_breaker_meta()
            return

        try:
            host_state_snapshot = breaker_registry.current_state(request_host)
        except Exception:  # pragma: no cover - defensive
            host_state_snapshot = None

        host_state_value: Optional[str]
        resolver_state_value: Optional[str]
        if isinstance(host_state_snapshot, Mapping):
            host_state_value = cast(Optional[str], host_state_snapshot.get("host"))
            resolver_state_value = cast(Optional[str], host_state_snapshot.get("resolver"))
        else:
            host_state_value = cast(Optional[str], host_state_snapshot)
            resolver_state_value = None

        if host_state_value:
            breaker_meta["host_state"] = host_state_value

        if resolver:
            if resolver_state_value:
                breaker_meta["resolver_state"] = resolver_state_value
            else:
                try:
                    resolver_snapshot = breaker_registry.current_state(
                        request_host, resolver=resolver
                    )
                except Exception:  # pragma: no cover - defensive
                    resolver_snapshot = None
                if isinstance(resolver_snapshot, Mapping):
                    resolver_value = cast(Optional[str], resolver_snapshot.get("resolver"))
                else:
                    resolver_value = cast(Optional[str], resolver_snapshot)
                if resolver_value:
                    breaker_meta["resolver_state"] = resolver_value

        _apply_breaker_meta()

    def request_func(*, method: str, url: str, **call_kwargs: Any) -> httpx.Response:
        if breaker_enabled and request_host and breaker_registry is not None:
            try:
                breaker_registry.allow(
                    request_host,
                    role=role_enum or RequestRole.METADATA if RequestRole is not None else None,
                    resolver=resolver,
                )
            except Exception as exc:
                if BreakerOpenError is not None and isinstance(exc, BreakerOpenError):
                    LOGGER.debug("Breaker open for %s %s: %s", method, request_url, exc)
                    breaker_meta["error"] = str(exc)
                    _snapshot_breaker_state(recorded="blocked")
                    if not hasattr(exc, "breaker_meta"):
                        setattr(exc, "breaker_meta", dict(breaker_meta))
                raise

        try:
            if stream_enabled:
                response = http_client.stream(method=method, url=url, **call_kwargs)
            else:
                response = http_client.request(method=method, url=url, **call_kwargs)
        except Exception as exc:
            if breaker_enabled and request_host and breaker_registry is not None:
                should_count = bool(
                    is_failure_for_breaker
                    and role_enum is not None
                    and is_failure_for_breaker(
                        breaker_registry.config.classify,
                        status=None,
                        exception=exc,
                    )
                )
                if should_count:
                    breaker_registry.on_failure(
                        request_host,
                        role=role_enum or RequestRole.METADATA,
                        resolver=resolver,
                        exception=exc,
                    )
                    breaker_meta["exception_type"] = type(exc).__name__
                    _snapshot_breaker_state(recorded="failure")
                else:
                    _apply_breaker_meta()
            raise

        if not isinstance(response, httpx.Response):
            _apply_breaker_meta()
            return response

        from_cache = bool(response.extensions.get("from_cache"))
        if breaker_enabled and request_host and breaker_registry is not None and not from_cache:
            status = getattr(response, "status_code", None)
            retry_after_s = None
            if status in (429, 503):
                retry_after_s = parse_retry_after_header(response)

            should_count = bool(
                is_failure_for_breaker
                and role_enum is not None
                and is_failure_for_breaker(
                    breaker_registry.config.classify,
                    status=status,
                    exception=None,
                )
            )

            if should_count:
                breaker_registry.on_failure(
                    request_host,
                    role=role_enum or RequestRole.METADATA,
                    resolver=resolver,
                    status=status,
                    retry_after_s=retry_after_s,
                )
                _snapshot_breaker_state(recorded="failure", retry_after=retry_after_s)
            else:
                breaker_registry.on_success(
                    request_host,
                    role=role_enum or RequestRole.METADATA,
                    resolver=resolver,
                )
                _snapshot_breaker_state(recorded="success")
        elif from_cache:
            _snapshot_breaker_state(recorded="cache-hit")
        else:
            _apply_breaker_meta()

        if breaker_meta and hasattr(response, "extensions"):
            response.extensions.setdefault("breaker", dict(breaker_meta))

        return response

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

    # Update breaker based on response and collect state for telemetry
    breaker_state_info: Dict[str, Any] = {}
    if breaker_registry is not None and request_host:
        from DocsToKG.ContentDownload.breakers import RequestRole, is_failure_for_breaker

        recorded: Optional[str] = None

        if isinstance(response, httpx.Response):
            status = response.status_code
            retry_after_s: Optional[float] = None
            if status in (429, 503):
                retry_after_s = parse_retry_after_header(response)

            is_failure = is_failure_for_breaker(
                breaker_registry.config.classify, status=status, exception=None
            )

            if is_failure:
                breaker_registry.on_failure(
                    request_host,
                    role=role_enum,
                    resolver=resolver,
                    status=status,
                    retry_after_s=retry_after_s,
                )
                recorded = "failure"
            else:
                breaker_registry.on_success(request_host, role=role_enum, resolver=resolver)
                recorded = "success"
        else:
            breaker_registry.on_success(request_host, role=role_enum, resolver=resolver)
            recorded = "success"

        breaker_state_info["breaker_host_state"] = breaker_registry.current_state(request_host)
        if resolver:
            breaker_state_info["breaker_resolver_state"] = breaker_registry.current_state(
                request_host, resolver=resolver
            )
        remaining = breaker_registry.cooldown_remaining_ms(request_host, resolver=resolver)
        if remaining is not None:
            breaker_state_info["breaker_open_remaining_ms"] = remaining
        breaker_state_info["breaker_recorded"] = recorded or "none"

    # Store breaker state info in response extensions for telemetry
    if breaker_state_info and hasattr(response, "extensions"):
        response.extensions.update(breaker_state_info)

    if not isinstance(response, httpx.Response):
        LOGGER.debug(
            "Response object of type %s lacks status_code; treating as success for %s %s.",
            type(response).__name__,
            method,
            request_url,
        )
        return response

    # Emit telemetry event (Phase 1: HTTP Layer Instrumentation)
    try:
        if telemetry is not None:
            from DocsToKG.ContentDownload.telemetry_helpers import emit_http_event

            elapsed_ms = int((time.time() - request_start_time) * 1000)

            emit_http_event(
                telemetry=telemetry,
                run_id=run_id or "unknown",
                host=request_host or "unknown",
                role=policy_role,
                method=method,
                status=response.status_code,
                url_hash=_compute_url_hash(canonical_index),
                from_cache=_extract_from_cache(response),
                revalidated=_extract_revalidated(response),
                stale=_extract_stale(response),
                retry_count=attempts - 1,
                retry_after_s=_extract_retry_after(response),
                rate_delay_ms=_extract_rate_delay(network_meta),
                breaker_state=_extract_breaker_state(breaker_state_info),
                breaker_recorded=_extract_breaker_recorded(breaker_state_info),
                elapsed_ms=elapsed_ms,
                error=None,
            )
    except Exception as exc:
        # Silently log telemetry errors to avoid breaking requests
        LOGGER.debug("Telemetry emission failed for %s %s: %s", method, request_url, exc)

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


# Legacy CircuitBreaker class removed - now handled by pybreaker-based BreakerRegistry


__all__ = [
    "CachedResult",
    "ConditionalRequestHelper",
    "ModifiedResult",
    "ContentPolicyViolation",
    # "CircuitBreaker",  # Legacy - now handled by pybreaker-based BreakerRegistry
    "configure_http_client",
    "get_http_client",
    "purge_http_cache",
    "head_precheck",
    "parse_retry_after_header",
    "request_with_retries",
]
