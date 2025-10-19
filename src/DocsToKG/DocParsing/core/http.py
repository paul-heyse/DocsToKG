"""Shared HTTP session management for DocParsing network interactions.

Certain DocParsing stages—particularly DocTags conversion when downloading
checkpoint models—perform lightweight HTTP calls. This module wraps the
``requests`` session setup with retry/backoff defaults, timeout normalisation,
and thread-safe caching so callers can use a hardened client without duplicating
connection pooling logic.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Mapping, Optional, Sequence, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_HTTP_TIMEOUT: Tuple[float, float] = (5.0, 30.0)

_HTTP_SESSION_LOCK = threading.Lock()
_HTTP_SESSION: Optional[requests.Session] = None
_HTTP_SESSION_TIMEOUT: Tuple[float, float] = DEFAULT_HTTP_TIMEOUT

__all__ = [
    "DEFAULT_HTTP_TIMEOUT",
    "get_http_session",
    "normalize_http_timeout",
]


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


def get_http_session(
    *,
    timeout: Optional[object] = None,
    base_headers: Optional[Mapping[str, str]] = None,
    retry_total: int = 5,
    retry_backoff: float = 0.5,
    status_forcelist: Sequence[int] = (429, 500, 502, 503, 504),
    allowed_methods: Sequence[str] = ("GET", "HEAD", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"),
) -> Tuple[requests.Session, Tuple[float, float]]:
    """Return a shared :class:`requests.Session` configured with retries."""

    effective_timeout = normalize_http_timeout(timeout)

    with _HTTP_SESSION_LOCK:
        global _HTTP_SESSION, _HTTP_SESSION_TIMEOUT
        if _HTTP_SESSION is None:
            session = requests.Session()
            retry = Retry(
                total=retry_total,
                read=retry_total,
                connect=retry_total,
                backoff_factor=retry_backoff,
                status_forcelist=tuple(int(code) for code in status_forcelist),
                allowed_methods=frozenset(method.upper() for method in allowed_methods),
                raise_on_status=False,
            )
            adapter = HTTPAdapter(max_retries=retry)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            _HTTP_SESSION = session
            logging.getLogger(__name__).debug(
                "Created shared HTTP session",
                extra={
                    "extra_fields": {
                        "retry_total": retry_total,
                        "status_forcelist": list(status_forcelist),
                        "allowed_methods": [method.upper() for method in allowed_methods],
                    }
                },
            )

        session_to_return = _HTTP_SESSION
        if base_headers:
            session_to_return = _clone_http_session(_HTTP_SESSION)
            session_to_return.headers.update(
                {key: value for key, value in base_headers.items() if value is not None}
            )

        _HTTP_SESSION_TIMEOUT = effective_timeout
        return session_to_return, _HTTP_SESSION_TIMEOUT


def _clone_http_session(session: requests.Session) -> requests.Session:
    """Create a shallow copy of a :class:`requests.Session` for transient headers."""

    clone = requests.Session()

    # Preserve base configuration while isolating header mutations and mutable containers.
    clone.headers.clear()
    clone.headers.update(session.headers)
    clone.auth = session.auth
    clone.cookies = session.cookies.copy()
    clone.params = session.params.copy()
    clone.proxies = session.proxies.copy()
    clone.verify = session.verify
    clone.cert = session.cert
    clone.trust_env = session.trust_env
    clone.max_redirects = session.max_redirects
    clone.stream = session.stream

    clone.hooks.clear()
    for event, hooks in session.hooks.items():
        clone.hooks[event] = hooks[:]

    clone.adapters.clear()
    for prefix, adapter in session.adapters.items():
        clone.mount(prefix, adapter)

    return clone
