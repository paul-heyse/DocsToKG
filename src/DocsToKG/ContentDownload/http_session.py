# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.http_session",
#   "purpose": "Shared HTTP session factory with polite headers, connection pooling, and telemetry hooks",
#   "sections": [
#     {
#       "id": "httpconfig",
#       "name": "HttpConfig",
#       "anchor": "class-httpconfig",
#       "kind": "class"
#     },
#     {
#       "id": "get-http-session",
#       "name": "get_http_session",
#       "anchor": "function-get-http-session",
#       "kind": "function"
#     },
#     {
#       "id": "reset-http-session",
#       "name": "reset_http_session",
#       "anchor": "function-reset-http-session",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""HTTP session factory for shared connection pool and polite headers.

**Purpose**
-----------
Provides a process-wide singleton HTTPX client with:
- Polite User-Agent + mailto headers
- Connection pooling optimized for many hosts
- TLS/proxy configuration
- Consistent timeout defaults

**Architecture**
----------------
One shared session per process → reuse TCP/TLS connections → lower latency,
avoid socket churn, reduce TLS handshake load on providers.

**Design Principle**
--------------------
All network requests route through this session, allowing:
- Unified HTTP telemetry (per-resolver wrappers emit)
- Centralized rate limiting (per-resolver wrappers enforce)
- Consistent header negotiation
- Connection reuse across resolvers
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

LOGGER = logging.getLogger(__name__)

# Singleton session (thread-safe via httpx internals)
_SHARED_SESSION: httpx.Client | None = None
_SESSION_LOCK = __import__("threading").Lock()


@dataclass(frozen=True)
class HttpConfig:
    """HTTP configuration passed from top-level config."""

    user_agent: str = "DocsToKG/ContentDownload (+mailto:data@example.org)"
    """User-Agent string with mailto contact."""

    mailto: str | None = None
    """Optional email for polite headers (overrides default)."""

    timeout_connect_s: float = 10.0
    """Connection timeout in seconds."""

    timeout_read_s: float = 60.0
    """Read timeout in seconds."""

    pool_connections: int = 10
    """Max pool connections per host."""

    pool_maxsize: int = 20
    """Max pool size (total connections)."""

    verify_tls: bool = True
    """Verify TLS certificates."""

    proxies: dict | None = None
    """Proxy configuration (HTTP/HTTPS)."""


def get_http_session(config: HttpConfig | None = None) -> httpx.Client:
    """
    Acquire or create the shared HTTP session.

    **Usage**

        session = get_http_session(config)
        response = session.get("https://example.com/article.pdf")

    **Parameters**

        config : HttpConfig, optional
            HTTP configuration (timeouts, headers, proxies).
            If None, uses defaults.

    **Returns**

        httpx.Client
            Shared session with polite headers, connection pooling.

    **Guarantees**

        - Reuses TCP/TLS connections across calls
        - Sets consistent User-Agent + mailto
        - Thread-safe (singleton pattern)
        - Lazy initialization

    **Mutability Warning**

        Do not modify session state directly. Use :func:`reset_http_session`
        in tests only to force re-initialization.
    """
    global _SHARED_SESSION

    if _SHARED_SESSION is not None:
        return _SHARED_SESSION

    with _SESSION_LOCK:
        # Double-check after acquiring lock
        if _SHARED_SESSION is not None:
            return _SHARED_SESSION

        cfg = config or HttpConfig()

        # Build polite User-Agent
        ua = cfg.user_agent
        if cfg.mailto and "+mailto:" not in ua:
            # Append mailto if not already present
            ua = f"{ua} (+mailto:{cfg.mailto})"

        # Configure timeout (applies to all requests)
        timeout = httpx.Timeout(
            timeout=cfg.timeout_read_s,
            connect=cfg.timeout_connect_s,
        )

        # Create session with pooling
        _SHARED_SESSION = httpx.Client(
            timeout=timeout,
            verify=cfg.verify_tls,
            headers={"User-Agent": ua},
            # Connection pooling: reuse TCP/TLS
            limits=httpx.Limits(
                max_connections=cfg.pool_maxsize,
                max_keepalive_connections=cfg.pool_connections,
            ),
            proxies=cfg.proxies,
        )

        LOGGER.debug(
            f"HTTP session created: UA={ua}, timeout={cfg.timeout_read_s}s, "
            f"pool_size={cfg.pool_connections}/{cfg.pool_maxsize}, "
            f"proxies={'set' if cfg.proxies else 'unset'}"
        )

        return _SHARED_SESSION


def reset_http_session() -> None:
    """
    Reset the shared HTTP session (tests only).

    **Warning**

        Only for use in tests. Do not call in production code.
    """
    global _SHARED_SESSION

    with _SESSION_LOCK:
        if _SHARED_SESSION is not None:
            _SHARED_SESSION.close()
            _SHARED_SESSION = None
            LOGGER.debug("HTTP session reset")
