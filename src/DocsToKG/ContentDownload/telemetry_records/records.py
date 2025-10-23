# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.telemetry_records.records",
#   "purpose": "Telemetry Record Types - Extended Data Contracts.",
#   "sections": [
#     {
#       "id": "telemetryattemptrecord",
#       "name": "TelemetryAttemptRecord",
#       "anchor": "class-telemetryattemptrecord",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

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

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.api import DownloadOutcome
    from DocsToKG.ContentDownload.core import ReasonCode

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
    run_id: str | None
    resolver: str | None
    url: str
    verb: str
    status: str
    http_status: int | None = None
    content_type: str | None = None
    reason: str | None = None
    elapsed_ms: int | None = None
    bytes_written: int | None = None
    content_length_hdr: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate record on construction."""
        if not self.url or not self.url.strip():
            raise ValueError("TelemetryAttemptRecord.url cannot be empty")


@dataclass(frozen=True)
class PipelineResult:
    """Resolver pipeline result captured for manifest emission.

    Attributes
    ----------
    resolver_name:
        Resolver that produced the terminal outcome. Required.
    url:
        Final URL associated with the resolver outcome. ``None`` when
        no network request was issued.
    outcome:
        Finalized :class:`DownloadOutcome` produced by the pipeline.
    success:
        Flag indicating whether the pipeline produced a successful
        outcome.
    reason:
        Canonical reason token emitted by the pipeline when no outcome
        was produced.
    reason_detail:
        Optional human readable detail complementing ``reason``.
    html_paths:
        Iterable of HTML artefact paths captured during resolution.
    canonical_url:
        Canonicalised URL hint propagated into the manifest entry.
    original_url:
        Original URL hint propagated into the manifest entry.
    """

    resolver_name: str
    url: str | None
    outcome: "DownloadOutcome | None" = None
    success: bool = False
    reason: "ReasonCode | str | None" = None
    reason_detail: "ReasonCode | str | None" = None
    html_paths: Iterable[str] = field(default_factory=tuple)
    canonical_url: str | None = None
    original_url: str | None = None

    def __post_init__(self) -> None:
        if not self.resolver_name or not self.resolver_name.strip():
            raise ValueError("PipelineResult.resolver_name cannot be empty")

        raw_paths = self.html_paths
        if raw_paths is None:
            normalized_paths: tuple[str, ...] = ()
        else:
            if isinstance(raw_paths, (str, bytes)):
                raise TypeError("PipelineResult.html_paths must be an iterable of paths")
            normalized_paths = tuple(str(path) for path in raw_paths)
        object.__setattr__(self, "html_paths", normalized_paths)

        for attr_name in ("url", "canonical_url", "original_url"):
            value = getattr(self, attr_name)
            if value is None:
                continue
            text = str(value).strip()
            object.__setattr__(self, attr_name, text or None)


__all__ = [
    "PipelineResult",
    "TelemetryAttemptRecord",
]
