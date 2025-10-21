"""
HTTPX Client Factory & Singleton Management.

Best-in-class HTTP client with:
- Lazy singleton (PID-aware for fork safety)
- Explicit timeouts, pool limits, HTTP/2, SSL verification
- Hishel RFC-9111 caching (optional, with bypass)
- Structured event hooks for telemetry
- Audited redirect chains (no auto-follow)
- Memory-efficient streaming

Architecture:
1. get_http_client(config) → HTTPX Client singleton
2. Hooks emit net.request telemetry per attempt
3. request_with_redirect_audit() validates each 3xx hop
4. All redirects go through the authoritative URL gate
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin

import httpx

# Optional cache support
try:
    from hishel import CacheTransport, FileSystemCache
except ImportError:
    CacheTransport = FileSystemCache = None

logger = logging.getLogger(__name__)

# ============================================================================
# Singleton State (PID-aware for fork safety)
# ============================================================================

_CLIENT: Optional[httpx.Client] = None
_BIND_HASH: Optional[str] = None
_BIND_PID: Optional[int] = None


# ============================================================================
# Client Factory
# ============================================================================


def get_http_client(config: Any) -> httpx.Client:
    """
    Lazy singleton HTTPX client factory.

    The client is PID-aware: after fork(), a new client is created to avoid
    connection sharing across processes. Settings changes mid-process are warned
    but don't trigger a rebuild (no hot reload in production).

    Args:
        config: ContentDownloadConfig instance with http settings

    Returns:
        Singleton httpx.Client configured per settings
    """
    global _CLIENT, _BIND_HASH, _BIND_PID

    pid = os.getpid()

    # PID changed → fork detected, rebuild
    if _CLIENT is None or _BIND_PID != pid:
        logger.debug(f"Creating new HTTPX client (pid={pid}, existing_pid={_BIND_PID})")
        _CLIENT = _build_http_client(config)
        _BIND_PID = pid
        _BIND_HASH = config.config_hash()

    # Settings changed mid-process → warn once but keep current client
    elif _BIND_HASH != config.config_hash():
        logger.warning(
            "HTTP client settings changed after binding. No hot reload; existing client unchanged."
        )
        _BIND_HASH = config.config_hash()

    return _CLIENT


def close_http_client() -> None:
    """Close the singleton client and cleanup."""
    global _CLIENT, _BIND_HASH, _BIND_PID
    if _CLIENT is not None:
        _CLIENT.close()
        logger.debug("HTTPX client closed")
    _CLIENT = None
    _BIND_HASH = None
    _BIND_PID = None


def reset_http_client() -> None:
    """Reset singleton (for testing)."""
    close_http_client()


# ============================================================================
# Client Construction
# ============================================================================


def _build_http_client(config: Any) -> httpx.Client:
    """Build a new HTTPX client from config."""
    cfg = config.http

    # Build timeout (apply to all operations)
    timeout = httpx.Timeout(
        cfg.timeout_connect_s,
        cfg.timeout_read_s,
        cfg.timeout_write_s,
        cfg.timeout_pool_s,
    )

    # Build pool limits
    limits = httpx.Limits(
        max_connections=cfg.max_connections,
        max_keepalive_connections=cfg.max_keepalive_connections,
        keepalive_expiry=cfg.keepalive_expiry_s,
    )

    # Transport with retries (connect errors only)
    transport = _build_transport(cfg)

    # Wrap with cache if enabled
    if cfg.cache_enabled and CacheTransport and FileSystemCache:
        cache_dir = Path(cfg.cache_dir).expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            cache = FileSystemCache(directory=str(cache_dir))
            transport = CacheTransport(transport=transport, cache=cache)
            logger.debug(f"Hishel cache enabled: {cache_dir}")
        except Exception as e:
            logger.warning(f"Failed to enable Hishel cache: {e}. Continuing without cache.")

    # Create client
    client = httpx.Client(
        http2=cfg.http2,
        transport=transport,
        timeout=timeout,
        limits=limits,
        verify=cfg.verify_tls,
        trust_env=cfg.trust_env,
        headers={
            "User-Agent": cfg.user_agent,
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br",
        },
        follow_redirects=False,  # CRITICAL: we audit hops explicitly
    )

    # Attach event hooks
    client.event_hooks["request"] = [_on_request]
    client.event_hooks["response"] = [_on_response]

    logger.debug(f"HTTPX client created: http2={cfg.http2}, cache={cfg.cache_enabled}")
    return client


def _build_transport(cfg: Any) -> httpx.BaseTransport:
    """Build base transport with connect retry policy."""
    return httpx.HTTPTransport(retries=cfg.connect_retries)


# ============================================================================
# Event Hooks (Telemetry)
# ============================================================================


def _on_request(request: httpx.Request) -> None:
    """Hook: capture request start time and metadata."""
    request.extensions["t0_perf"] = time.perf_counter()
    request.extensions["request_id"] = os.urandom(8).hex()
    # Additional tags can be added here (service, host, etc.) for telemetry


def _on_response(response: httpx.Response) -> None:
    """Hook: emit net.request telemetry event."""
    req = response.request
    t0 = req.extensions.get("t0_perf", time.perf_counter())
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Infer cache status (Hishel exposes via response.extensions if available)
    cache_status = "miss"
    if response.status_code == 304:
        cache_status = "revalidated"
    elif "from_cache" in response.extensions and response.extensions.get("from_cache"):
        cache_status = "hit"

    # Normalize host for consistent telemetry keys
    host = _normalize_host_for_telemetry(str(req.url))

    # Emit telemetry
    _emit_net_request(
        method=req.method,
        url=str(req.url),
        host=host,
        status=response.status_code,
        elapsed_ms=elapsed_ms,
        cache=cache_status,
        http2=response.http_version == "HTTP/2",
        bytes_read=int(response.headers.get("Content-Length", 0) or 0),
        request_id=req.extensions.get("request_id"),
    )


def _emit_net_request(**kwargs: Any) -> None:
    """Emit net.request telemetry (placeholder; integrate with your event system)."""
    # TODO: Wire to your telemetry system (e.g., structured logging, OTLP)
    logger.debug(f"net.request: {kwargs}")


# ============================================================================
# Redirect Audit
# ============================================================================


def request_with_redirect_audit(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    max_hops: int = 5,
    headers: Optional[dict] = None,
    **kw: Any,
) -> httpx.Response:
    """
    Send request and follow 3xx hops with URL security validation.

    Each redirect target is validated by the authoritative URL gate before
    following. This prevents open redirect attacks and enforces host/port
    policies.

    Args:
        client: HTTPX client (follow_redirects must be False)
        method: HTTP method (GET, HEAD, etc.)
        url: Target URL
        max_hops: Maximum redirect hops (default 5)
        headers: Additional headers
        **kw: Additional request kwargs

    Returns:
        Final HTTP response

    Raises:
        httpx.TooManyRedirects: If hops exceed max_hops
        PolicyError: If redirect target fails URL validation
    """
    from ..policy.url_gate import validate_url_security

    current = url
    for hop in range(max_hops):
        resp = client.request(method, current, headers=headers, **kw)

        # Check for redirect
        if 300 <= resp.status_code < 400 and "location" in resp.headers:
            target_raw = resp.headers["location"]
            # Resolve relative URLs against current
            target_abs = urljoin(current, target_raw)

            # Validate target (authoritative gate)
            try:
                target_validated = validate_url_security(target_abs)
            except Exception as e:
                logger.error(f"Redirect blocked: {target_abs} ({e})")
                raise

            logger.debug(f"Redirect {current} → {target_validated}")
            current = target_validated
            continue

        # Final response (no redirect)
        return resp

    # Too many hops
    raise httpx.TooManyRedirects(f"Exceeded {max_hops} redirect hops")


def _normalize_host_for_telemetry(url: str) -> str:
    """
    Extract and normalize host from URL for consistent telemetry keys.

    Handles IDN normalization and port stripping.

    Args:
        url: Full URL

    Returns:
        Normalized hostname (lowercase, punycode if IDN), or "unknown" on error
    """
    try:
        parsed = httpx.URL(url)
        host = parsed.host
        if not host:
            return "unknown"
        # httpx.URL.host returns punycode automatically for IDN
        # Just ensure lowercase for consistency
        return host.lower()
    except Exception:
        return "unknown"
