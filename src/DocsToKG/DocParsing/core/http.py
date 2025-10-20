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
from contextlib import suppress
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Mapping, Optional, Sequence, Tuple

import httpx
from tenacity import (
    Retrying,
    retry_if_exception_type,
    retry_if_result,
    stop_after_attempt,
    wait_random_exponential,
)
from tenacity.wait import wait_base

DEFAULT_HTTP_TIMEOUT: Tuple[float, float] = (5.0, 30.0)

_HTTP_SESSION_LOCK = threading.Lock()
_HTTP_SESSION: Optional["TenacityClient"] = None
_HTTP_SESSION_TIMEOUT: Tuple[float, float] = DEFAULT_HTTP_TIMEOUT

__all__ = [
    "DEFAULT_HTTP_TIMEOUT",
    "get_http_session",
    "normalize_http_timeout",
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
        self._retry_total = max(0, int(retry_total))
        self._retry_backoff = float(retry_backoff)
        self._status_forcelist = {int(code) for code in status_forcelist}
        self._allowed_methods = {method.upper() for method in allowed_methods}
        self._retryable_exceptions = (
            httpx.TimeoutException,
            httpx.RequestError,
        )
        self._default_timeout: Tuple[float, float] = DEFAULT_HTTP_TIMEOUT
        self._logger = logging.getLogger(__name__)
        self._wait_strategy = _RetryAfterWait(backoff_factor=self._retry_backoff)

    def clone_with_headers(self, headers: Mapping[str, str]) -> "TenacityClient":
        clone = TenacityClient(
            retry_total=self._retry_total,
            retry_backoff=self._retry_backoff,
            status_forcelist=tuple(self._status_forcelist),
            allowed_methods=tuple(self._allowed_methods),
        )
        clone._default_timeout = self._default_timeout
        clone.headers.update(self.headers)
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

    def _coerce_timeout(self, timeout: Optional[object]) -> httpx.Timeout:
        if isinstance(timeout, httpx.Timeout):
            return timeout
        if isinstance(timeout, dict):
            return httpx.Timeout(**timeout)
        connect, read = normalize_http_timeout(timeout)
        return httpx.Timeout(connect=connect, read=read, write=read, pool=connect)

    def _set_default_timeout(self, timeout: Tuple[float, float]) -> None:
        self._default_timeout = timeout
        self.timeout = self._coerce_timeout(timeout)

    def request(self, method: str, url: str, **kwargs):
        timeout = kwargs.get("timeout", self._default_timeout)
        kwargs["timeout"] = self._coerce_timeout(timeout)

        if self._retry_total <= 0 or method.upper() not in self._allowed_methods:
            return super().request(method, url, **kwargs)

        retrying = self._build_retrying(method)

        def _send():
            response = super().request(method, url, **kwargs)
            return response

        return retrying(_send)

    def _build_retrying(self, method: str) -> Retrying:
        retry_predicate = retry_if_exception_type(self._retryable_exceptions)
        if self._status_forcelist:
            retry_predicate = retry_predicate | retry_if_result(
                lambda response: isinstance(response, httpx.Response)
                and response.status_code in self._status_forcelist
            )

        return Retrying(
            retry=retry_predicate,
            wait=self._wait_strategy,
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


def normalize_http_timeout(timeout: Optional[object]) -> Tuple[float, float]:
    """Normalize timeout inputs into a ``(connect, read)`` tuple of floats."""

    if timeout is None:
        return DEFAULT_HTTP_TIMEOUT

    def _coerce_pair(values: Sequence[object]) -> Tuple[float, float]:
        """Coerce arbitrary iterables into a two-element timeout tuple."""

        extracted = [v for v in values if v is not None]
        if not extracted:
            return DEFAULT_HTTP_TIMEOUT
        if len(extracted) == 1:
            coerced = float(extracted[0])
            return coerced, coerced
        return float(extracted[0]), float(extracted[1])

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


def _parse_retry_after_header(value: Optional[str]) -> Optional[float]:
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
            retry_time = retry_time.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        seconds = (retry_time - now).total_seconds()
    return max(float(seconds), 0.0)


def get_http_session(
    *,
    timeout: Optional[object] = None,
    base_headers: Optional[Mapping[str, str]] = None,
    retry_total: int = 5,
   retry_backoff: float = 0.5,
   status_forcelist: Sequence[int] = (429, 500, 502, 503, 504),
   allowed_methods: Sequence[str] = ("GET", "HEAD", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"),
) -> Tuple[TenacityClient, Tuple[float, float]]:
    """Return a shared :class:`httpx.Client` configured with retries."""

    effective_timeout = normalize_http_timeout(timeout)

    with _HTTP_SESSION_LOCK:
        global _HTTP_SESSION, _HTTP_SESSION_TIMEOUT
        if _HTTP_SESSION is None:
            _HTTP_SESSION = TenacityClient(
                retry_total=retry_total,
                retry_backoff=retry_backoff,
                status_forcelist=status_forcelist,
                allowed_methods=allowed_methods,
            )
            logging.getLogger(__name__).debug(
                "Created Tenacity-backed DocParsing HTTP session",
                extra={
                    "extra_fields": {
                        "retry_total": retry_total,
                        "retry_backoff": retry_backoff,
                        "status_forcelist": [int(code) for code in status_forcelist],
                        "allowed_methods": [method.upper() for method in allowed_methods],
                    }
                },
            )

        session: TenacityClient = _HTTP_SESSION
        session._set_default_timeout(effective_timeout)

        if base_headers:
            header_map = {str(key): str(value) for key, value in base_headers.items() if value is not None}
            session_to_return: TenacityClient = session.clone_with_headers(header_map)
            session_to_return._set_default_timeout(effective_timeout)
        else:
            session_to_return = session

        _HTTP_SESSION_TIMEOUT = effective_timeout
        return session_to_return, _HTTP_SESSION_TIMEOUT
