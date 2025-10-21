"""
Structured Telemetry for HTTP Network Layer.

Provides net.request event builders and emission for HTTP operations.
Integrates with ContentDownload telemetry system to provide:

- Per-request timing and metadata
- Cache hit/miss/revalidated tracking
- HTTP protocol version detection
- Error classification
- Request/response correlation via request_id
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# Event Types & Enums
# ============================================================================


class CacheStatus(str, Enum):
    """HTTP cache status."""

    HIT = "hit"
    REVALIDATED = "revalidated"
    MISS = "miss"
    BYPASS = "bypass"


class RequestStatus(str, Enum):
    """Request outcome status."""

    SUCCESS = "success"
    REDIRECT = "redirect"
    ERROR = "error"
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"


# ============================================================================
# Event Data Classes
# ============================================================================


@dataclass(frozen=True)
class NetRequestEvent:
    """
    Structured net.request event.

    Emitted for every HTTP request (including redirects and retries).
    Correlates via request_id for tracing across hops.
    """

    # Event metadata
    ts: str  # ISO 8601 timestamp (UTC)
    event_type: str = "net.request"
    request_id: str  # UUID for correlation

    # Request details
    method: str  # GET, HEAD, POST, etc.
    url: str  # URL (may be redacted)
    host: str  # Hostname only

    # Response
    status_code: int  # HTTP status (0 if error before response)
    status: RequestStatus  # SUCCESS, REDIRECT, ERROR, etc.

    # Timing
    elapsed_ms: float  # Wall-clock duration
    ttfb_ms: Optional[float] = None  # Time to first byte (if tracked)

    # Cache
    cache: CacheStatus = CacheStatus.MISS
    from_cache: bool = False  # True if from Hishel

    # Protocol
    http_version: str = "HTTP/1.1"  # HTTP/1.0, HTTP/1.1, HTTP/2
    http2: bool = False

    # Data
    bytes_read: int = 0
    bytes_written: int = 0

    # Error details (if status != SUCCESS)
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    # Context
    attempt: int = 1  # Attempt number (for retries)
    hop: int = 1  # Redirect hop number
    redirect_target: Optional[str] = None

    # Additional metadata
    service: Optional[str] = None  # Service name (resolver, provider, etc.)
    role: Optional[str] = None  # Role (metadata, landing, artifact, etc.)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "ts": self.ts,
            "event_type": self.event_type,
            "request_id": self.request_id,
            "method": self.method,
            "url": self.url,
            "host": self.host,
            "status_code": self.status_code,
            "status": self.status.value,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "ttfb_ms": round(self.ttfb_ms, 2) if self.ttfb_ms else None,
            "cache": self.cache.value,
            "from_cache": self.from_cache,
            "http_version": self.http_version,
            "http2": self.http2,
            "bytes_read": self.bytes_read,
            "bytes_written": self.bytes_written,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "attempt": self.attempt,
            "hop": self.hop,
            "redirect_target": self.redirect_target,
            "service": self.service,
            "role": self.role,
        }


# ============================================================================
# Event Builder (Fluent Interface)
# ============================================================================


class NetRequestEventBuilder:
    """
    Builder for net.request events.

    Provides a fluent interface for constructing events with required
    and optional fields.
    """

    def __init__(self, request_id: str):
        """Initialize builder with request ID."""
        self.request_id = request_id
        self.method = "GET"
        self.url = ""
        self.host = ""
        self.status_code = 0
        self.status = RequestStatus.SUCCESS
        self.elapsed_ms = 0.0
        self.ttfb_ms: Optional[float] = None
        self.cache = CacheStatus.MISS
        self.from_cache = False
        self.http_version = "HTTP/1.1"
        self.http2 = False
        self.bytes_read = 0
        self.bytes_written = 0
        self.error_code: Optional[str] = None
        self.error_message: Optional[str] = None
        self.attempt = 1
        self.hop = 1
        self.redirect_target: Optional[str] = None
        self.service: Optional[str] = None
        self.role: Optional[str] = None

    def with_request(
        self, method: str, url: str, host: str
    ) -> NetRequestEventBuilder:
        """Set request details."""
        self.method = method
        self.url = url
        self.host = host
        return self

    def with_response(
        self, status_code: int, http_version: str, http2: bool = False
    ) -> NetRequestEventBuilder:
        """Set response details."""
        self.status_code = status_code
        self.http_version = http_version
        self.http2 = http2
        return self

    def with_timing(self, elapsed_ms: float, ttfb_ms: Optional[float] = None) -> NetRequestEventBuilder:
        """Set timing information."""
        self.elapsed_ms = elapsed_ms
        self.ttfb_ms = ttfb_ms
        return self

    def with_cache(
        self, cache: CacheStatus, from_cache: bool = False
    ) -> NetRequestEventBuilder:
        """Set cache status."""
        self.cache = cache
        self.from_cache = from_cache
        return self

    def with_data(self, bytes_read: int = 0, bytes_written: int = 0) -> NetRequestEventBuilder:
        """Set data transfer sizes."""
        self.bytes_read = bytes_read
        self.bytes_written = bytes_written
        return self

    def with_error(
        self, error_code: str, error_message: str = "", status: RequestStatus = RequestStatus.ERROR
    ) -> NetRequestEventBuilder:
        """Set error details."""
        self.error_code = error_code
        self.error_message = error_message
        self.status = status
        return self

    def with_redirect(self, hop: int, target: str) -> NetRequestEventBuilder:
        """Set redirect details."""
        self.hop = hop
        self.redirect_target = target
        self.status = RequestStatus.REDIRECT
        return self

    def with_attempt(self, attempt: int) -> NetRequestEventBuilder:
        """Set attempt number (for retries)."""
        self.attempt = attempt
        return self

    def with_context(self, service: Optional[str] = None, role: Optional[str] = None) -> NetRequestEventBuilder:
        """Set context (service, role)."""
        self.service = service
        self.role = role
        return self

    def build(self) -> NetRequestEvent:
        """Build and return the event."""
        return NetRequestEvent(
            ts=datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            event_type="net.request",
            request_id=self.request_id,
            method=self.method,
            url=self.url,
            host=self.host,
            status_code=self.status_code,
            status=self.status,
            elapsed_ms=self.elapsed_ms,
            ttfb_ms=self.ttfb_ms,
            cache=self.cache,
            from_cache=self.from_cache,
            http_version=self.http_version,
            http2=self.http2,
            bytes_read=self.bytes_read,
            bytes_written=self.bytes_written,
            error_code=self.error_code,
            error_message=self.error_message,
            attempt=self.attempt,
            hop=self.hop,
            redirect_target=self.redirect_target,
            service=self.service,
            role=self.role,
        )


# ============================================================================
# Event Emitter (Pluggable)
# ============================================================================


class NetRequestEmitter:
    """
    Pluggable emitter for net.request events.

    Can be extended to send events to various sinks:
    - Structured logging (default)
    - SQLite telemetry
    - JSONL manifest
    - OTLP/Prometheus
    - Custom handlers
    """

    def __init__(self):
        """Initialize emitter."""
        self.handlers: list[callable] = [self._log_handler]

    def add_handler(self, handler: callable) -> None:
        """Register a custom event handler."""
        self.handlers.append(handler)

    def emit(self, event: NetRequestEvent) -> None:
        """Emit event to all registered handlers."""
        for handler in self.handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Event handler failed: {e}", exc_info=True)

    @staticmethod
    def _log_handler(event: NetRequestEvent) -> None:
        """Default handler: structured logging."""
        msg = (
            f"net.request: {event.method} {event.url} → {event.status_code} "
            f"({event.elapsed_ms:.1f}ms, cache={event.cache.value})"
        )
        if event.error_code:
            msg += f" [ERROR: {event.error_code}]"
        logger.debug(msg)


# ============================================================================
# Global Singleton Emitter
# ============================================================================

_EMITTER: Optional[NetRequestEmitter] = None


def get_net_request_emitter() -> NetRequestEmitter:
    """Get or create the global net.request emitter."""
    global _EMITTER
    if _EMITTER is None:
        _EMITTER = NetRequestEmitter()
    return _EMITTER


def reset_net_request_emitter() -> None:
    """Reset emitter (for testing)."""
    global _EMITTER
    _EMITTER = None
