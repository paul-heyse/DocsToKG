"""
Network layer for ContentDownload.

Provides best-in-class HTTPX client pool management, redirect audit,
caching (Hishel), rate limiting integration, and structured telemetry.

Architecture:
- One lazy singleton HTTPX client per process (PID-aware)
- Explicit timeouts, pool limits, HTTP/2, SSL verification
- Hishel RFC-9111 caching with bypass support
- Audited redirect chains (no auto-follow)
- Per-hop telemetry emission (net.request events)
- Streaming for memory discipline
- Tenacity for status-aware retries (429/5xx)
"""

from .client import (
    close_http_client,
    get_http_client,
    request_with_redirect_audit,
    reset_http_client,
)

__all__ = [
    "get_http_client",
    "close_http_client",
    "reset_http_client",
    "request_with_redirect_audit",
]
