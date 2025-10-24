# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.DocParsing.core.http",
#   "purpose": "Shared HTTP session management for DocParsing network interactions.",
#   "sections": [
#     {
#       "id": "retryafterwait",
#       "name": "_RetryAfterWait",
#       "anchor": "class-retryafterwait",
#       "kind": "class"
#     },
#     {
#       "id": "tenacityclient",
#       "name": "TenacityClient",
#       "anchor": "class-tenacityclient",
#       "kind": "class"
#     },
#     {
#       "id": "normalize-http-timeout",
#       "name": "normalize_http_timeout",
#       "anchor": "function-normalize-http-timeout",
#       "kind": "function"
#     },
#     {
#       "id": "parse-retry-after-header",
#       "name": "_parse_retry_after_header",
#       "anchor": "function-parse-retry-after-header",
#       "kind": "function"
#     },
#     {
#       "id": "get-http-session",
#       "name": "get_http_session",
#       "anchor": "function-get-http-session",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Shared HTTP session management for DocParsing network interactions.

Certain DocParsing stages—particularly DocTags conversion when downloading
checkpoint models—perform lightweight HTTP calls. This module wraps the
``httpx`` client setup with retry/backoff defaults, timeout normalisation, and
thread-safe caching so callers can use a hardened client without duplicating
connection pooling logic.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
from tenacity import (
    Retrying,
    retry_if_exception_type,
    retry_if_result,
    stop_after_attempt,
    wait_random_exponential,
)
from tenacity.wait import wait_base


@dataclass(frozen=True)
class _RetryConfig:
    """Immutable retry policy configuration."""

    retry_total: int
    retry_backoff: float
    status_forcelist: tuple[int, ...]
    allowed_methods: tuple[str, ...]


@dataclass(frozen=True)
class RetryOverrides:
    """Field-level overrides for retry policy configuration."""

    retry_total: int | None = None
    retry_backoff: float | None = None
    status_forcelist: Sequence[int] | None = None
    allowed_methods: Sequence[str] | None = None


def _normalize_status_codes(status_codes: Sequence[int]) -> tuple[int, ...]:
    normalized: list[int] = []
    seen: set[int] = set()
    for code in status_codes:
        coerced = int(code)
        if coerced not in seen:
            seen.add(coerced)
            normalized.append(coerced)
    return tuple(sorted(normalized))


def _normalize_methods(methods: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for method in methods:
        coerced = str(method).upper()
        if coerced not in seen:
            seen.add(coerced)
            normalized.append(coerced)
    return tuple(normalized)


def _create_retry_config(
    *,
    retry_total: int,
    retry_backoff: float,
    status_forcelist: Sequence[int],
    allowed_methods: Sequence[str],
) -> _RetryConfig:
    return _RetryConfig(
        retry_total=max(0, int(retry_total)),
        retry_backoff=max(float(retry_backoff), 0.0),
        status_forcelist=_normalize_status_codes(status_forcelist),
        allowed_methods=_normalize_methods(allowed_methods),
    )


def _coerce_overrides(
    overrides: RetryOverrides | Mapping[str, object] | None,
) -> RetryOverrides | None:
    if overrides is None:
        return None
    if isinstance(overrides, RetryOverrides):
        return overrides
    if isinstance(overrides, Mapping):
        recognized: dict[str, Any] = {}
        for key in ("retry_total", "retry_backoff", "status_forcelist", "allowed_methods"):
            if key in overrides:
                recognized[key] = overrides[key]
        return RetryOverrides(**recognized)
    raise TypeError("retry overrides must be a RetryOverrides instance or mapping")


def _merge_retry_config(
    base: _RetryConfig,
    overrides: RetryOverrides | Mapping[str, object] | None,
) -> _RetryConfig:
    override_obj = _coerce_overrides(overrides)
    if override_obj is None:
        return base

    retry_total = base.retry_total
    if override_obj.retry_total is not None:
        retry_total = max(0, int(override_obj.retry_total))

    retry_backoff = base.retry_backoff
    if override_obj.retry_backoff is not None:
        retry_backoff = max(float(override_obj.retry_backoff), 0.0)

    status_forcelist = base.status_forcelist
    if override_obj.status_forcelist is not None:
        status_forcelist = _normalize_status_codes(override_obj.status_forcelist)

    allowed_methods = base.allowed_methods
    if override_obj.allowed_methods is not None:
        allowed_methods = _normalize_methods(override_obj.allowed_methods)

    return _RetryConfig(
        retry_total=retry_total,
        retry_backoff=retry_backoff,
        status_forcelist=status_forcelist,
        allowed_methods=allowed_methods,
    )


DEFAULT_HTTP_TIMEOUT: tuple[float, float] = (5.0, 30.0)

DEFAULT_RETRY_TOTAL = 5
DEFAULT_RETRY_BACKOFF = 0.5
DEFAULT_STATUS_FORCELIST: tuple[int, ...] = (429, 500, 502, 503, 504)
DEFAULT_ALLOWED_METHODS: tuple[str, ...] = (
    "GET",
    "HEAD",
    "POST",
    "PUT",
    "DELETE",
    "OPTIONS",
    "PATCH",
)


@dataclass(frozen=True)
class _RetryPolicy:
    retry_total: int
    retry_backoff: float
    status_forcelist: tuple[int, ...]
    allowed_methods: tuple[str, ...]

    def replace(
        self,
        *,
        retry_total: int | None = None,
        retry_backoff: float | None = None,
        status_forcelist: Sequence[int] | None = None,
        allowed_methods: Sequence[str] | None = None,
    ) -> _RetryPolicy:
        return _RetryPolicy(
            retry_total=_coerce_retry_total(
                self.retry_total if retry_total is None else retry_total
            ),
            retry_backoff=_coerce_retry_backoff(
                self.retry_backoff if retry_backoff is None else retry_backoff
            ),
            status_forcelist=_coerce_status_forcelist(
                status_forcelist if status_forcelist is not None else self.status_forcelist
            ),
            allowed_methods=_coerce_allowed_methods(
                allowed_methods if allowed_methods is not None else self.allowed_methods
            ),
        )


def _coerce_retry_total(value: int | float) -> int:
    return max(0, int(value))


def _coerce_retry_backoff(value: float | int) -> float:
    return max(0.0, float(value))


def _coerce_status_forcelist(values: Sequence[int] | tuple[int, ...]) -> tuple[int, ...]:
    seen: set[int] = set()
    coerced: list[int] = []
    for candidate in values:
        code = int(candidate)
        if code not in seen:
            seen.add(code)
            coerced.append(code)
    return tuple(sorted(coerced))


def _coerce_allowed_methods(values: Sequence[str] | tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    coerced: list[str] = []
    for candidate in values:
        method = str(candidate).upper()
        if not method:
            continue
        if method not in seen:
            seen.add(method)
            coerced.append(method)
    return tuple(coerced)


_HTTP_SESSION_LOCK = threading.Lock()
_HTTP_SESSION: TenacityClient | None = None
_HTTP_SESSION_TIMEOUT: tuple[float, float] = DEFAULT_HTTP_TIMEOUT
_DEFAULT_RETRY_POLICY = _RetryPolicy(
    retry_total=DEFAULT_RETRY_TOTAL,
    retry_backoff=DEFAULT_RETRY_BACKOFF,
    status_forcelist=DEFAULT_STATUS_FORCELIST,
    allowed_methods=DEFAULT_ALLOWED_METHODS,
)
_HTTP_SESSION_POLICY: _RetryPolicy = _DEFAULT_RETRY_POLICY

__all__ = [
    "DEFAULT_HTTP_TIMEOUT",
    "get_http_session",
    "normalize_http_timeout",
    "request_with_retries",
]


class _RetryAfterWait(wait_base):
    """Tenacity wait strategy that honours Retry-After headers when present."""

    def __init__(self, *, backoff_factor: float) -> None:
        multiplier = max(float(backoff_factor), 0.0)
        self._fallback_wait = wait_random_exponential(multiplier=multiplier or 0.0)

    def __call__(self, retry_state) -> float:  # type: ignore[override]
        delay = float(self._fallback_wait(retry_state))
        outcome = retry_state.outcome
        if outcome is not None and not outcome.failed:
            response = outcome.result()
            if isinstance(response, httpx.Response):
                retry_after = _parse_retry_after_header(response.headers.get("Retry-After"))
                if retry_after is not None:
                    delay = max(delay, retry_after)
        return max(delay, 0.0)


class TenacityClient(httpx.Client):
    """`httpx.Client` subclass that delegates retries to Tenacity."""

    def __init__(
        self,
        *,
        retry_total: int,
        retry_backoff: float,
        status_forcelist: Sequence[int],
        allowed_methods: Sequence[str],
    ) -> None:
        timeout = self._coerce_timeout(DEFAULT_HTTP_TIMEOUT)
        super().__init__(timeout=timeout, follow_redirects=True)
        policy = _DEFAULT_RETRY_POLICY.replace(
            retry_total=retry_total,
            retry_backoff=retry_backoff,
            status_forcelist=status_forcelist,
            allowed_methods=allowed_methods,
        )
        self._retry_total = policy.retry_total
        self._retry_backoff = policy.retry_backoff
        self._status_forcelist = policy.status_forcelist
        self._status_forcelist_set = set(policy.status_forcelist)
        self._allowed_methods = policy.allowed_methods
        self._allowed_methods_set = {method for method in policy.allowed_methods}
        self._retryable_exceptions = (
            httpx.TimeoutException,
            httpx.RequestError,
        )
        self._default_timeout: tuple[float, float] = DEFAULT_HTTP_TIMEOUT
        self._logger = logging.getLogger(__name__)
        self._wait_strategy = _RetryAfterWait(backoff_factor=self._retry_backoff)

    @property
    def retry_policy(self) -> _RetryPolicy:
        return _RetryPolicy(
            retry_total=self._retry_total,
            retry_backoff=self._retry_backoff,
            status_forcelist=self._status_forcelist,
            allowed_methods=self._allowed_methods,
        )

    def clone(
        self,
        *,
        headers: Mapping[str, str] | None = None,
        retry_total: int | None = None,
        retry_backoff: float | None = None,
        status_forcelist: Sequence[int] | None = None,
        allowed_methods: Sequence[str] | None = None,
    ) -> TenacityClient:
        policy = self.retry_policy.replace(
            retry_total=retry_total,
            retry_backoff=retry_backoff,
            status_forcelist=status_forcelist,
            allowed_methods=allowed_methods,
        )
        clone = TenacityClient(
            retry_total=policy.retry_total,
            retry_backoff=policy.retry_backoff,
            status_forcelist=policy.status_forcelist,
            allowed_methods=policy.allowed_methods,
        )
        clone._default_timeout = self._default_timeout
        clone.headers.update(self.headers)
        if headers:
            for key, value in headers.items():
                if value is not None:
                    clone.headers[str(key)] = str(value)
        with suppress(Exception):
            clone.cookies.update(self.cookies)  # type: ignore[arg-type]
        clone.auth = self.auth
        with suppress(Exception):
            clone.params.update(self.params)  # type: ignore[arg-type]
        clone._set_default_timeout(clone._default_timeout)
        return clone

    def clone_with_headers(self, headers: Mapping[str, str]) -> TenacityClient:
        return self.clone(headers=headers)

    def _coerce_timeout(self, timeout: object | None) -> httpx.Timeout:
        if isinstance(timeout, httpx.Timeout):
            return timeout
        if isinstance(timeout, dict):
            return httpx.Timeout(**timeout)
        connect, read = normalize_http_timeout(timeout)
        return httpx.Timeout(connect=connect, read=read, write=read, pool=connect)

    def _set_default_timeout(self, timeout: tuple[float, float]) -> None:
        self._default_timeout = timeout
        self.timeout = self._coerce_timeout(timeout)

    def request(self, method: str, url: str, **kwargs):
        timeout = kwargs.get("timeout", self._default_timeout)
        kwargs["timeout"] = self._coerce_timeout(timeout)

        if self._retry_total <= 0 or method.upper() not in self._allowed_methods_set:
            return super().request(method, url, **kwargs)

        retrying = self._build_retrying(method)

        def _send():
            response = super().request(method, url, **kwargs)
            return response

        return retrying(_send)

    def _build_retrying(self, method: str) -> Retrying:
        retry_predicate = retry_if_exception_type(self._retryable_exceptions)
        if self._status_forcelist_set:
            retry_predicate = retry_predicate | retry_if_result(
                lambda response: isinstance(response, httpx.Response)
                and response.status_code in self._status_forcelist_set
            )

        wait_strategy = self._wait_strategy
        return Retrying(
            retry=retry_predicate,
            wait=wait_strategy,
            stop=stop_after_attempt(self._retry_total + 1),
            sleep=time.sleep,
            reraise=True,
            before_sleep=self._before_sleep,
        )

    def _before_sleep(self, retry_state) -> None:
        outcome = retry_state.outcome
        if outcome is not None and not outcome.failed:
            response = outcome.result()
            if isinstance(response, httpx.Response):
                with suppress(Exception):
                    response.close()

        if not self._logger.isEnabledFor(logging.DEBUG):
            return

        delay = 0.0
        if retry_state.next_action is not None and retry_state.next_action.sleep is not None:
            with suppress(TypeError, ValueError):
                delay = max(float(retry_state.next_action.sleep), 0.0)

        exc = outcome.exception() if outcome is not None and outcome.failed else None
        extra = {
            "extra_fields": {
                "attempt": retry_state.attempt_number,
                "delay": round(delay, 3),
                "exception": repr(exc) if exc is not None else None,
            }
        }
        self._logger.debug("DocParsing HTTP retry", extra=extra)


def normalize_http_timeout(timeout: object | None) -> tuple[float, float]:
    """Normalize timeout inputs into a ``(connect, read)`` tuple of floats."""

    if timeout is None:
        return DEFAULT_HTTP_TIMEOUT

    if isinstance(timeout, httpx.Timeout):
        connect = timeout.connect if timeout.connect is not None else timeout.read
        read = timeout.read if timeout.read is not None else timeout.connect
        connect = DEFAULT_HTTP_TIMEOUT[0] if connect is None else float(connect)
        read = DEFAULT_HTTP_TIMEOUT[1] if read is None else float(read)
        return connect, read

    if isinstance(timeout, Mapping):
        connect_candidate = timeout.get("connect", timeout.get("connect_timeout"))
        read_candidate = timeout.get("read", timeout.get("read_timeout"))
        fallback = timeout.get("default") or timeout.get("timeout")

        def _coerce(value: object, *, default: float) -> float:
            if value is None:
                return default
            try:
                return float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid timeout value {value!r}") from exc

        fallback_connect = fallback_read = None
        if fallback is not None:
            fallback_value = _coerce(fallback, default=DEFAULT_HTTP_TIMEOUT[0])
            fallback_connect = fallback_value
            fallback_read = fallback_value

        connect_default = (
            fallback_connect if fallback_connect is not None else DEFAULT_HTTP_TIMEOUT[0]
        )
        read_default = fallback_read if fallback_read is not None else DEFAULT_HTTP_TIMEOUT[1]

        connect = _coerce(connect_candidate, default=connect_default)
        read = _coerce(read_candidate, default=read_default)
        return connect, read

    def _coerce_pair(values: Sequence[object]) -> tuple[float, float]:
        """Coerce arbitrary iterables into a two-element timeout tuple."""

        items = list(values)
        if not items:
            return DEFAULT_HTTP_TIMEOUT
        if len(items) == 1:
            single = items[0]
            if single is None:
                return DEFAULT_HTTP_TIMEOUT
            return float(single), float(single)

        def _coerce_element(value: object, *, default: float) -> float:
            if value is None:
                return default
            try:
                return float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid timeout value {value!r}") from exc

        connect = _coerce_element(items[0], default=DEFAULT_HTTP_TIMEOUT[0])
        read = _coerce_element(items[1], default=DEFAULT_HTTP_TIMEOUT[1])
        return connect, read

    if isinstance(timeout, (int, float)):
        coerced = float(timeout)
        return coerced, coerced

    if isinstance(timeout, str):
        parts = [part for part in re.split(r"[;,\s]+", timeout) if part]
        if not parts:
            return DEFAULT_HTTP_TIMEOUT
        return _coerce_pair(parts)

    if isinstance(timeout, (list, tuple, set)):
        return _coerce_pair(list(timeout))

    if hasattr(timeout, "__iter__"):
        try:
            return _coerce_pair(list(timeout))  # type: ignore[arg-type]
        except TypeError as exc:  # pragma: no cover - defensive
            raise ValueError("Unable to interpret HTTP timeout iterable") from exc

    raise TypeError(f"Unsupported timeout type: {type(timeout)!r}")


def _parse_retry_after_header(value: str | None) -> float | None:
    if not value:
        return None
    candidate = value.strip()
    if not candidate:
        return None
    try:
        seconds = float(candidate)
    except ValueError:
        try:
            retry_time = parsedate_to_datetime(candidate)
        except (TypeError, ValueError, IndexError):
            return None
        if retry_time is None:
            return None
        if retry_time.tzinfo is None:
            retry_time = retry_time.replace(tzinfo=UTC)
        now = datetime.now(UTC)
        seconds = (retry_time - now).total_seconds()
    return max(float(seconds), 0.0)


def _sanitize_headers(
    headers: Mapping[str, str] | None,
) -> dict[str, str] | None:
    if not headers:
        return None
    sanitized: dict[str, str] = {}
    for key, value in headers.items():
        if value is None:
            continue
        sanitized[str(key)] = str(value)
    return sanitized or None


def get_http_session(
    *,
    timeout: object | None = None,
    base_headers: Mapping[str, str] | None = None,
    retry_total: int | None = None,
    retry_backoff: float | None = None,
    status_forcelist: Sequence[int] | None = None,
    allowed_methods: Sequence[str] | None = None,
) -> tuple[TenacityClient, tuple[float, float]]:
    """Return a shared :class:`httpx.Client` configured with retries."""

    global _HTTP_SESSION, _HTTP_SESSION_TIMEOUT, _HTTP_SESSION_POLICY

    effective_timeout = normalize_http_timeout(timeout)

    requested_policy = (
        _HTTP_SESSION_POLICY if _HTTP_SESSION is not None else _DEFAULT_RETRY_POLICY
    ).replace(
        retry_total=retry_total,
        retry_backoff=retry_backoff,
        status_forcelist=status_forcelist,
        allowed_methods=allowed_methods,
    )

    header_map = _sanitize_headers(base_headers)

    with _HTTP_SESSION_LOCK:
        if _HTTP_SESSION is None:
            _HTTP_SESSION = TenacityClient(
                retry_total=requested_policy.retry_total,
                retry_backoff=requested_policy.retry_backoff,
                status_forcelist=requested_policy.status_forcelist,
                allowed_methods=requested_policy.allowed_methods,
            )
            _HTTP_SESSION_POLICY = requested_policy
            logging.getLogger(__name__).debug(
                "Created Tenacity-backed DocParsing HTTP session",
                extra={
                    "extra_fields": {
                        "retry_total": requested_policy.retry_total,
                        "retry_backoff": requested_policy.retry_backoff,
                        "status_forcelist": [
                            int(code) for code in requested_policy.status_forcelist
                        ],
                        "allowed_methods": [
                            method.upper() for method in requested_policy.allowed_methods
                        ],
                    }
                },
            )

        session: TenacityClient = _HTTP_SESSION
        session._set_default_timeout(effective_timeout)

        overrides: dict[str, object] = {}
        if requested_policy != _HTTP_SESSION_POLICY:
            overrides = {
                "retry_total": requested_policy.retry_total,
                "retry_backoff": requested_policy.retry_backoff,
                "status_forcelist": requested_policy.status_forcelist,
                "allowed_methods": requested_policy.allowed_methods,
            }

        if header_map or overrides:
            session_to_return = session.clone(headers=header_map, **overrides)
            session_to_return._set_default_timeout(effective_timeout)
        else:
            session_to_return = session

        _HTTP_SESSION_TIMEOUT = effective_timeout
        return session_to_return, _HTTP_SESSION_TIMEOUT


def request_with_retries(
    session: TenacityClient | None,
    method: str,
    url: str,
    *,
    timeout: object | None = None,
    base_headers: Mapping[str, str] | None = None,
    retry_total: int | None = None,
    retry_backoff: float | None = None,
    status_forcelist: Sequence[int] | None = None,
    allowed_methods: Sequence[str] | None = None,
    **kwargs,
) -> httpx.Response:
    """Execute an HTTP request with Tenacity-backed retries."""

    if not method:
        raise ValueError("HTTP method must be provided")
    if not url:
        raise ValueError("URL must be provided")

    header_map = _sanitize_headers(base_headers)
    overrides_requested = any(
        value is not None
        for value in (retry_total, retry_backoff, status_forcelist, allowed_methods)
    )

    if session is None:
        session, request_timeout = get_http_session(
            timeout=timeout,
            base_headers=header_map,
            retry_total=retry_total,
            retry_backoff=retry_backoff,
            status_forcelist=status_forcelist,
            allowed_methods=allowed_methods,
        )
    else:
        session_to_use = session
        if header_map or overrides_requested:
            session_to_use = session.clone(
                headers=header_map,
                retry_total=retry_total,
                retry_backoff=retry_backoff,
                status_forcelist=status_forcelist,
                allowed_methods=allowed_methods,
            )
        session = session_to_use
        if timeout is None:
            request_timeout = session._default_timeout
        else:
            request_timeout = normalize_http_timeout(timeout)

    timeout_value = session._coerce_timeout(request_timeout)
    return session.request(method, url, timeout=timeout_value, **kwargs)
