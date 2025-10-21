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
from typing import Any, Mapping, Optional

from DocsToKG.ContentDownload.api.types import AttemptStatus, ReasonCode

# ============================================================================
# EXTENDED ATTEMPT RECORD - Telemetry Internal Type
# ============================================================================


@dataclass(frozen=True, slots=True)
class TelemetryAttemptRecord:
    """
    Rich attempt record used internally by telemetry for event logging.
    
    Extends the minimal api/types.AttemptRecord with fields needed for:
    - Detailed event logging (reason, work_id, resolver order)
    - Network metrics (rate limiter state, breaker state)
    - Retry and HTTP metadata (retry_after, content_length, sha256)
    - Operational context (dry_run, from_cache)
    
    This record is NOT the external data contract. It's used only for
    internal telemetry operations and manifest writing.
    
    Design:
    - Contains all fields used by telemetry.py for manifest/metrics
    - Backward compatible with existing manifest JSON (via conversion)
    - Supports partial records (e.g., during streaming)
    """

    # Core attempt fields (from api/types.AttemptRecord)
    run_id: str
    resolver_name: str
    url: str
    status: AttemptStatus
    http_status: Optional[int] = None
    elapsed_ms: Optional[int] = None

    # Extended telemetry fields
    work_id: Optional[str] = None
    """Work artifact ID (e.g., OpenAlex W123)"""

    reason: Optional[ReasonCode] = None
    """Normalized reason code (e.g., 'not-modified', 'robots')"""

    resolver_order: Optional[int] = None
    """Resolver priority (0 = first, 1 = second, etc.)"""

    resolver_wall_time_ms: Optional[int] = None
    """Time resolver.resolve() took (milliseconds)"""

    content_type: Optional[str] = None
    """HTTP Content-Type header from response"""

    content_length: Optional[int] = None
    """HTTP Content-Length or actual bytes written"""

    sha256: Optional[str] = None
    """SHA256 digest of downloaded content (after finalize)"""

    dry_run: bool = False
    """Whether this was a dry-run attempt (no write)"""

    retry_after: Optional[int] = None
    """Retry-After header value (seconds) if present"""

    # Rate limiter tracking
    rate_limiter_wait_ms: Optional[int] = None
    """Time spent waiting for rate limiter token"""

    rate_limiter_role: Optional[str] = None
    """Rate limiter role (metadata, landing, artifact)"""

    rate_limiter_mode: Optional[str] = None
    """Rate limiter mode (e.g., 'wait:250', 'fail-fast')"""

    rate_limiter_backend: Optional[str] = None
    """Rate limiter backend (in-memory, sqlite, redis, etc.)"""

    # Circuit breaker tracking
    from_cache: Optional[bool] = None
    """Whether response came from HTTP cache"""

    breaker_host_state: Optional[str] = None
    """Circuit breaker state for host (open, closed, half-open)"""

    breaker_resolver_state: Optional[str] = None
    """Circuit breaker state for resolver context"""

    breaker_open_remaining_ms: Optional[int] = None
    """Time until breaker transitions from open (milliseconds)"""

    breaker_recorded: Optional[bool] = None
    """Whether breaker recorded this event"""

    # Catchall for future/custom metadata
    metadata: Mapping[str, Any] = field(default_factory=dict)
    """Additional structured metadata (extensible)"""

    def __post_init__(self) -> None:
        """Validate record invariants."""
        # Basic validation: ensure at least core fields are set
        if not self.url or not self.url.strip():
            raise ValueError("TelemetryAttemptRecord.url cannot be empty")


@dataclass(frozen=True)
class PipelineResult:
    """
    Legacy pipeline result type for backward compatibility.
    
    ⚠️  DEPRECATED: Used only by old telemetry code paths.
    Do not use in new code.
    """

    success: bool
    resolver_name: Optional[str] = None
    outcome: Optional[Any] = None
    meta: Mapping[str, Any] = field(default_factory=dict)


__all__ = [
    "TelemetryAttemptRecord",
    "PipelineResult",
]
