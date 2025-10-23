# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.network.polite_client",
#   "purpose": "Polite HTTP Client: Combines HTTPX + Rate-Limiting for respectful requests.",
#   "sections": [
#     {
#       "id": "politehttpclient",
#       "name": "PoliteHttpClient",
#       "anchor": "class-politehttpclient",
#       "kind": "class"
#     },
#     {
#       "id": "get-polite-http-client",
#       "name": "get_polite_http_client",
#       "anchor": "function-get-polite-http-client",
#       "kind": "function"
#     },
#     {
#       "id": "close-polite-http-client",
#       "name": "close_polite_http_client",
#       "anchor": "function-close-polite-http-client",
#       "kind": "function"
#     },
#     {
#       "id": "reset-polite-http-client",
#       "name": "reset_polite_http_client",
#       "anchor": "function-reset-polite-http-client",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Polite HTTP Client: Combines HTTPX + Rate-Limiting for respectful requests.

Provides a high-level facade that transparently integrates HTTP client and
rate-limiting, allowing callers to simply make requests without worrying about
rate limits or caching. The client automatically:
- Acquires rate-limit slots before requests
- Respects provider rate limits
- Caches responses (RFC 9111)
- Retries transient failures (429/5xx)
- Tracks detailed telemetry

Design:
- **Lazy initialization**: Created on first request
- **Thread-safe**: Safe for concurrent use
- **Service-aware**: Different rate limits per provider (OLS, BioPortal, etc.)
- **Host-aware**: Optional per-host rate-limit keying
- **Observable**: Emits events for rate-limiting and HTTP operations

Example:
    >>> from DocsToKG.OntologyDownload.network import get_polite_http_client
    >>> client = get_polite_http_client()
    >>> response = client.get("https://ols.example.org/search", service="ols")
    >>> # Transparently rate-limited + cached + retried
"""

import logging
import os
import threading
import time
from typing import Optional
from urllib.parse import urlparse

import httpx

from DocsToKG.OntologyDownload.errors import PolicyError
from DocsToKG.OntologyDownload.network.client import (
    close_http_client,
    get_http_client,
)
from DocsToKG.OntologyDownload.ratelimit import (
    emit_acquire_event,
    emit_blocked_event,
    get_rate_limiter,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Global State
# ============================================================================

_polite_client: Optional["PoliteHttpClient"] = None
_polite_client_lock = threading.Lock()
_polite_client_pid: int | None = None


# ============================================================================
# PoliteHttpClient
# ============================================================================


class PoliteHttpClient:
    """Thread-safe HTTP client with transparent rate-limiting.

    Combines HTTPX (with caching) and rate-limiting to provide a high-level
    HTTP interface that respects provider rate limits automatically.

    Attributes:
        _http_client: HTTPX client for actual HTTP requests
        _rate_limiter: Rate limiter manager
        _service: Default service name for rate-limiting
        _host: Optional default host for rate-limit keying
    """

    def __init__(
        self,
        service: str | None = None,
        host: str | None = None,
    ):
        """Initialize PoliteHttpClient.

        Args:
            service: Default service name (e.g., "ols", "bioportal")
                    Used for rate-limit acquisition if not specified in requests
            host: Optional default host for rate-limit secondary keying
        """
        self._service = service or "default"
        self._host = host
        self._http_client: httpx.Client | None = get_http_client()
        self._rate_limiter = get_rate_limiter()

        logger.debug(
            "PoliteHttpClient initialized",
            extra={
                "service": self._service,
                "host": self._host,
            },
        )

    def get(
        self,
        url: str,
        service: str | None = None,
        host: str | None = None,
        weight: int = 1,
        **kwargs,
    ) -> httpx.Response:
        """Perform a GET request with rate-limiting.

        Args:
            url: URL to request
            service: Service name for rate-limiting (uses default if not provided)
            host: Host for secondary rate-limit keying (optional)
            weight: Number of rate-limit slots to acquire
            **kwargs: Additional arguments passed to httpx.Client.get()

        Returns:
            HTTP response

        Raises:
            httpx.HTTPError: On HTTP errors
            ValueError: If weight is invalid
        """
        return self._polite_request(
            method="GET",
            url=url,
            service=service,
            host=host,
            weight=weight,
            **kwargs,
        )

    def post(
        self,
        url: str,
        service: str | None = None,
        host: str | None = None,
        weight: int = 1,
        **kwargs,
    ) -> httpx.Response:
        """Perform a POST request with rate-limiting.

        Args:
            url: URL to request
            service: Service name for rate-limiting
            host: Host for secondary rate-limit keying
            weight: Number of rate-limit slots to acquire
            **kwargs: Additional arguments passed to httpx.Client.post()

        Returns:
            HTTP response
        """
        return self._polite_request(
            method="POST",
            url=url,
            service=service,
            host=host,
            weight=weight,
            **kwargs,
        )

    def request(
        self,
        method: str,
        url: str,
        service: str | None = None,
        host: str | None = None,
        weight: int = 1,
        **kwargs,
    ) -> httpx.Response:
        """Perform a request with rate-limiting.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: URL to request
            service: Service name for rate-limiting
            host: Host for secondary rate-limit keying
            weight: Number of rate-limit slots to acquire
            **kwargs: Additional arguments passed to httpx.Client.request()

        Returns:
            HTTP response
        """
        return self._polite_request(
            method=method,
            url=url,
            service=service,
            host=host,
            weight=weight,
            **kwargs,
        )

    def _polite_request(
        self,
        method: str,
        url: str,
        service: str | None = None,
        host: str | None = None,
        weight: int = 1,
        **kwargs,
    ) -> httpx.Response:
        """Internal method to perform a polite (rate-limited) request.

        Args:
            method: HTTP method
            url: URL to request
            service: Service name for rate-limiting
            host: Host for secondary rate-limit keying
            weight: Number of rate-limit slots to acquire
            **kwargs: Additional HTTPX arguments

        Returns:
            HTTP response
        """
        # Use defaults if not provided
        service = service or self._service
        host = host or self._host or self._extract_host(url)

        # Refresh the underlying HTTP client if it has been closed by a reset.
        http_client = self._ensure_http_client()

        # Acquire rate limit slot
        ts_acquire_start = time.time()
        try:
            acquired = self._rate_limiter.acquire(
                service=service,
                host=host,
                weight=weight,
            )
            elapsed_ms = int((time.time() - ts_acquire_start) * 1000)

            if acquired:
                emit_acquire_event(
                    service=service,
                    host=host,
                    weight=weight,
                    elapsed_ms=elapsed_ms,
                )
            else:
                emit_blocked_event(
                    service=service,
                    host=host,
                    weight=weight,
                    reason="rate_limit_exceeded",
                )
                raise PolicyError(
                    f"Rate limit denied for service '{service}' (host='{host or 'default'}')"
                )

        except Exception as e:
            logger.error(
                "Error acquiring rate limit",
                extra={
                    "service": service,
                    "host": host,
                    "weight": weight,
                    "error": str(e),
                },
            )
            raise

        # Perform HTTP request
        ts_request_start = time.time()
        try:
            response = http_client.request(method=method, url=url, **kwargs)
            elapsed_ms = int((time.time() - ts_request_start) * 1000)

            logger.debug(
                "Polite request completed",
                extra={
                    "method": method,
                    "url": url,
                    "service": service,
                    "host": host,
                    "status": response.status_code,
                    "elapsed_ms": elapsed_ms,
                    "from_cache": response.extensions.get("from_cache", False),
                },
            )

            return response

        except Exception as e:
            logger.error(
                "HTTP request failed",
                extra={
                    "method": method,
                    "url": url,
                    "service": service,
                    "error": str(e),
                },
            )
            raise

    def _ensure_http_client(self) -> httpx.Client:
        """Ensure the underlying HTTP client is open, refreshing if needed."""

        if self._http_client is None or getattr(self._http_client, "is_closed", False):
            logger.debug("Refreshing polite HTTP client binding after reset")
            self._http_client = get_http_client()
        assert self._http_client is not None
        return self._http_client

    @staticmethod
    def _extract_host(url: str) -> str:
        """Extract hostname from URL.

        Args:
            url: URL string

        Returns:
            Hostname (e.g., "api.example.com")
        """
        try:
            parsed = urlparse(url)
            return parsed.netloc or "unknown"
        except Exception:
            return "unknown"

    def close(self) -> None:
        """Close and cleanup the client.

        Call at application shutdown.
        """
        logger.debug("PoliteHttpClient closing")
        try:
            close_http_client()
            self._http_client = None
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(
                "Failed to close underlying HTTP client",
                extra={"error": str(exc)},
            )


# ============================================================================
# Singleton API
# ============================================================================


def get_polite_http_client(
    service: str | None = None,
    host: str | None = None,
) -> PoliteHttpClient:
    """Get or create the shared PoliteHttpClient singleton.

    Returns:
        The PoliteHttpClient instance

    Behavior:
        - First call: Creates client, binds to current PID
        - Subsequent calls: Returns same instance (thread-safe)
        - Process forked: Child detects PID change, creates new client
    """
    global _polite_client, _polite_client_pid

    # Quick path: already initialized, same PID
    if _polite_client is not None and _polite_client_pid == os.getpid():
        return _polite_client

    # Slow path: need to create or recreate
    with _polite_client_lock:
        # Double-check after lock acquired
        if _polite_client is not None and _polite_client_pid == os.getpid():
            return _polite_client

        # Close old client if PID changed
        if _polite_client is not None and _polite_client_pid != os.getpid():
            logger.debug("Process forked; closing old polite client and creating new one")
            _polite_client.close()
            _polite_client = None

        # Create new client
        _polite_client = PoliteHttpClient(service=service, host=host)
        _polite_client_pid = os.getpid()

        logger.debug(
            "Polite HTTP client created and bound",
            extra={
                "service": service or "default",
                "host": host or "auto-detected",
                "pid": _polite_client_pid,
            },
        )

        return _polite_client


def close_polite_http_client() -> None:
    """Close the polite HTTP client and release resources.

    Safe to call multiple times or when no client has been created.
    """
    global _polite_client

    with _polite_client_lock:
        if _polite_client is not None:
            try:
                _polite_client.close()
                logger.debug("Polite HTTP client closed")
            except Exception as e:
                logger.error(f"Error closing polite HTTP client: {e}")
            finally:
                _polite_client = None


def reset_polite_http_client() -> None:
    """Reset the polite HTTP client (primarily for testing).

    Call this between test cases to force creation of a fresh client.
    """
    close_polite_http_client()
    global _polite_client_pid

    _polite_client_pid = None


__all__ = [
    "PoliteHttpClient",
    "get_polite_http_client",
    "close_polite_http_client",
    "reset_polite_http_client",
]
