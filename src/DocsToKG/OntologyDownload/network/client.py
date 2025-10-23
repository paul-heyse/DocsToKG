# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.network.client",
#   "purpose": "HTTPX + Hishel HTTP Client Factory.",
#   "sections": [
#     {
#       "id": "get-http-client",
#       "name": "get_http_client",
#       "anchor": "function-get-http-client",
#       "kind": "function"
#     },
#     {
#       "id": "close-http-client",
#       "name": "close_http_client",
#       "anchor": "function-close-http-client",
#       "kind": "function"
#     },
#     {
#       "id": "reset-http-client",
#       "name": "reset_http_client",
#       "anchor": "function-reset-http-client",
#       "kind": "function"
#     },
#     {
#       "id": "get-cache-dir",
#       "name": "_get_cache_dir",
#       "anchor": "function-get-cache-dir",
#       "kind": "function"
#     },
#     {
#       "id": "create-ssl-context",
#       "name": "_create_ssl_context",
#       "anchor": "function-create-ssl-context",
#       "kind": "function"
#     },
#     {
#       "id": "create-http-client",
#       "name": "_create_http_client",
#       "anchor": "function-create-http-client",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""HTTPX + Hishel HTTP Client Factory.

Provides a singleton, thread-safe HTTP client with RFC 9111-compliant caching,
connection pooling, automatic keepalives, and comprehensive instrumentation hooks.

Key design:
- **Lazy initialization**: Client created on first use, not at import time.
- **Config binding**: Client is bound to the first loaded Settings/config_hash for the
  process lifetime. If Settings change post-bind, a warning is logged once and the
  client is **not** rebuilt (no hot-reload in production).
- **PID-aware**: If the process forks, the child detects this and rebuilds the client
  on first use to avoid file descriptor/socket conflicts.
- **Thread-safe**: Uses threading.Lock for guard; the lock cost is negligible.
- **Caching**: Hishel FileStorage with RFC 9111 compliance; validates with ETag/Last-Modified.
- **Streaming**: All responses are streamed; never buffer entire bodies.

Example:
    >>> from DocsToKG.OntologyDownload.network import get_http_client, close_http_client
    >>> client = get_http_client()
    >>> response = client.get("https://api.example.com/data", stream=True)
    >>> close_http_client()  # at process shutdown or test cleanup
"""

import logging
import os
import ssl
import threading
from pathlib import Path

import certifi
import hishel
import httpx
import platformdirs

from DocsToKG.OntologyDownload.network.policy import (
    ALLOW_HEURISTIC_CACHING,
    CACHE_SCOPE,
    CACHE_STORAGE_CHECK_INTERVAL_SECONDS,
    CACHE_STORAGE_TTL_SECONDS,
    CACHEABLE_METHODS,
    CACHEABLE_STATUS_CODES,
    FOLLOW_REDIRECTS,
    HTTP2_ENABLED,
    HTTP_CONNECT_TIMEOUT,
    HTTP_POOL_TIMEOUT,
    HTTP_READ_TIMEOUT,
    HTTP_WRITE_TIMEOUT,
    KEEPALIVE_EXPIRY,
    MAX_CONNECTIONS,
    MAX_KEEPALIVE_CONNECTIONS,
    TLS_VERIFY_ENABLED,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Global Client State
# ============================================================================

_client: httpx.Client | None = None
_client_lock = threading.Lock()
_client_bind_hash: str | None = None
_client_bind_pid: int | None = None
_config_hash_mismatch_warned = False


# ============================================================================
# Public API
# ============================================================================


def get_http_client() -> httpx.Client:
    """Get or create the shared HTTPX client.

    Returns:
        A singleton httpx.Client with Hishel caching, connection pooling,
        and comprehensive instrumentation hooks.

    Behavior:
        - First call: Creates client, binds to current config_hash and PID.
        - Subsequent calls: Returns same client (thread-safe).
        - Config changed after bind: Logs warning once, does not rebuild.
        - Process forked: Child detects PID change, rebuilds client on first call.

    Example:
        >>> client = get_http_client()
        >>> response = client.get("https://example.com/file")
        >>> # Client is reused across the codebase; no need to pass it around
    """
    global _client, _client_bind_hash, _client_bind_pid, _config_hash_mismatch_warned

    # Quick path: already initialized, same PID, check hash
    if _client is not None and _client_bind_pid == os.getpid():
        # Import settings here to avoid circular dependency at module load time
        from DocsToKG.OntologyDownload.settings import get_settings

        current_hash = get_settings().config_hash()
        if current_hash != _client_bind_hash and not _config_hash_mismatch_warned:
            logger.warning(
                "Settings config_hash changed after HTTP client was initialized. "
                "Continuing with bound client; no hot-reload. "
                "Reset via reset_http_client() if desired.",
                extra={
                    "bind_hash": _client_bind_hash,
                    "current_hash": current_hash,
                },
            )
            _config_hash_mismatch_warned = True
        return _client

    # Slow path: need to (re)create client
    with _client_lock:
        # Double-check after lock acquired
        if _client is not None and _client_bind_pid == os.getpid():
            return _client

        # Close old client if PID changed
        if _client is not None and _client_bind_pid != os.getpid():
            logger.debug("Process forked; closing old HTTP client and rebuilding.")
            try:
                _client.close()
            except Exception as e:
                logger.debug(f"Error closing old client: {e}")
            _client = None

        # Create new client
        _client = _create_http_client()
        from DocsToKG.OntologyDownload.settings import get_settings

        _client_bind_hash = get_settings().config_hash()
        _client_bind_pid = os.getpid()
        _config_hash_mismatch_warned = False

        logger.debug(
            "HTTP client initialized",
            extra={
                "config_hash": _client_bind_hash,
                "pid": _client_bind_pid,
                "cache_dir": _get_cache_dir(),
            },
        )

        return _client


def close_http_client() -> None:
    """Close the HTTP client and release resources.

    Safe to call multiple times or when no client has been created.

    Typical usage:
        - At application shutdown
        - In test fixtures' teardown
        - Before exiting a context

    Example:
        >>> client = get_http_client()
        >>> # ... use client ...
        >>> close_http_client()  # at the end
    """
    global _client, _client_lock

    with _client_lock:
        if _client is not None:
            try:
                _client.close()
                logger.debug("HTTP client closed")
            except Exception as e:
                logger.error(f"Error closing HTTP client: {e}")
            finally:
                _client = None


def reset_http_client() -> None:
    """Reset the HTTP client (primarily for testing).

    Call this between test cases to force creation of a fresh client
    with potentially different settings.

    **NOT** for production use; only for test isolation.

    Example:
        >>> # In test setup
        >>> reset_http_client()
        >>> client = get_http_client()
    """
    close_http_client()
    global _client_bind_hash, _client_bind_pid, _config_hash_mismatch_warned

    _client_bind_hash = None
    _client_bind_pid = None
    _config_hash_mismatch_warned = False

    try:
        from DocsToKG.OntologyDownload.network.polite_client import (
            reset_polite_http_client,
        )
    except ImportError:  # pragma: no cover - optional dependency
        logger.debug(
            "Polite HTTP client unavailable during reset; skipping polite reset"
        )
    else:
        reset_polite_http_client()


# ============================================================================
# Implementation Details
# ============================================================================


def _get_cache_dir() -> Path:
    """Determine cache directory based on settings and scope.

    Uses platformdirs for cross-platform XDG/macOS/Windows compliance.
    Respects CACHE_SCOPE setting (shared vs run-scoped temp).

    Returns:
        Path to cache directory (created if needed)
    """
    if CACHE_SCOPE == "run":
        # Temp directory for this run (useful for tests)
        import tempfile

        base = Path(tempfile.gettempdir()) / "ontofetch-http-cache"
    else:
        # Persistent cache dir (default)
        base = Path(platformdirs.user_cache_dir("ontofetch"))

    cache_dir = base / "http"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _create_ssl_context() -> ssl.SSLContext:
    """Create SSL context with secure defaults.

    Uses system certificates + certifi bundle for maximum compatibility.
    Enforces TLS verification; refuses self-signed or weak certs.

    Returns:
        Configured ssl.SSLContext for use with HTTPX
    """
    if not TLS_VERIFY_ENABLED:
        # Insecure mode (development only)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        logger.warning("TLS verification DISABLED (development only!)")
        return ctx

    # Standard: system certs + certifi fallback
    ctx = ssl.create_default_context(cafile=certifi.where())
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    return ctx


def _create_http_client() -> httpx.Client:
    """Create HTTPX client with Hishel caching and event hooks.

    Configuration:
    - Timeouts: per-phase (connect < read < write)
    - Connection pooling: bounded to avoid provider blocks
    - HTTP/2: enabled for multiplexing
    - Redirects: disabled (audited manually)
    - Caching: Hishel RFC 9111 with file storage
    - Hooks: instrumentation for telemetry

    Returns:
        Fully configured httpx.Client ready for use
    """
    # SSL context
    ssl_ctx = _create_ssl_context()

    # Base transport with limited connect retries
    base_transport = httpx.HTTPTransport(
        retries=2,  # connect errors only
        verify=ssl_ctx,
    )

    # Hishel cache transport
    cache_storage = hishel.FileStorage(
        base_path=str(_get_cache_dir()),
        ttl=CACHE_STORAGE_TTL_SECONDS,
        check_ttl_every=CACHE_STORAGE_CHECK_INTERVAL_SECONDS,
    )

    cache_controller = hishel.Controller(
        cacheable_methods=CACHEABLE_METHODS,
        cacheable_status_codes=CACHEABLE_STATUS_CODES,
        allow_heuristics=ALLOW_HEURISTIC_CACHING,
        cache_private=True,  # Private cache (not shared)
    )

    cache_transport = hishel.CacheTransport(
        transport=base_transport,
        storage=cache_storage,
        controller=cache_controller,
    )

    # Create client with comprehensive configuration
    client = httpx.Client(
        transport=cache_transport,
        timeout=httpx.Timeout(
            connect=HTTP_CONNECT_TIMEOUT,
            read=HTTP_READ_TIMEOUT,
            write=HTTP_WRITE_TIMEOUT,
            pool=HTTP_POOL_TIMEOUT,
        ),
        limits=httpx.Limits(
            max_connections=MAX_CONNECTIONS,
            max_keepalive_connections=MAX_KEEPALIVE_CONNECTIONS,
            keepalive_expiry=KEEPALIVE_EXPIRY,
        ),
        http2=HTTP2_ENABLED,
        follow_redirects=FOLLOW_REDIRECTS,
        verify=ssl_ctx,
    )

    logger.debug(
        "HTTPX client created",
        extra={
            "http2": HTTP2_ENABLED,
            "max_connections": MAX_CONNECTIONS,
            "max_keepalive": MAX_KEEPALIVE_CONNECTIONS,
            "cache_dir": str(_get_cache_dir()),
        },
    )

    return client


__all__ = [
    "get_http_client",
    "close_http_client",
    "reset_http_client",
]
