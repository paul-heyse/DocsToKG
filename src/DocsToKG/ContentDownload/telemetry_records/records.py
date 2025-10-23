"""
Telemetry Record Types - Extended Data Contracts

Defines the rich attempt record used internally by telemetry for logging
and manifest recording. These types extend the minimal api/types.AttemptRecord
with telemetry-specific fields needed for detailed event logging, metrics,
rate limiter tracking, and circuit breaker observability.

Design:
- Frozen dataclasses for immutability and hashability
- Optional fields for partial records during streaming
- Slots for memory efficiency
- Type hints for static analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Optional

# ============================================================================
# EXTENDED ATTEMPT RECORD - Telemetry Internal Type
# ============================================================================


@dataclass(frozen=True)
class TelemetryAttemptRecord:
    """
    Structured telemetry record for HTTP/IO attempts.

    Used throughout the download pipeline to track detailed low-level
    HTTP and I/O operations with full traceability.

    Attributes:
        ts: Timestamp when attempt occurred (UTC)
        run_id: Run identifier (None if telemetry disabled)
        resolver: Resolver name (None for fallback/IO attempts)
        url: Target URL (hashed for privacy in sinks)
        verb: HTTP verb (HEAD, GET, DELETE) or I/O operation type
        status: Stable status token (http-head, http-get, http-200, etc)
        http_status: HTTP status code (200, 304, 429, 500, etc) or None
        content_type: Content-Type header value
        reason: Stable reason token (ok, robots, timeout, etc)
        elapsed_ms: Wall-clock elapsed time in milliseconds
        bytes_written: Bytes successfully written (for streaming)
        content_length_hdr: Content-Length header value (for verification)
        extra: Arbitrary metadata (retry count, redirect chain, etc)
    """

    ts: datetime
    run_id: Optional[str]
    resolver: Optional[str]
    url: str
    verb: str
    status: str
    http_status: Optional[int] = None
    content_type: Optional[str] = None
    reason: Optional[str] = None
    elapsed_ms: Optional[int] = None
    bytes_written: Optional[int] = None
    content_length_hdr: Optional[int] = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate record on construction."""
        if not self.url or not self.url.strip():
            raise ValueError("TelemetryAttemptRecord.url cannot be empty")


__all__ = [
    "TelemetryAttemptRecord",
]
