"""
Canonical API Types for ContentDownload Pipeline

Provides frozen, immutable dataclasses as contracts between resolvers,
download execution, and pipeline orchestration. All types are frozen with
slots for memory efficiency and immutability guarantees.

Data Flow:
  Resolver.resolve() → ResolverResult
  ResolverResult.plans → DownloadPlan[] → download_execution
  prepare_candidate_download(plan) → DownloadPlan
  stream_candidate_payload(plan) → DownloadStreamResult
  finalize_candidate_download(plan, stream) → DownloadOutcome
  Pipeline records outcome + manifest

Design Principles:
  - Frozen dataclasses prevent accidental mutation
  - Slots reduce memory footprint
  - Literal types prevent invalid string values
  - All fields documented with semantic meaning
  - No large nested structures (keep meta dict small)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional, Sequence

# ============================================================================
# STABLE TOKEN VOCABULARIES (Public Contract)
# ============================================================================

#: Final outcome classification
OutcomeClass = Literal["success", "skip", "error"]

#: HTTP attempt status tokens (used in telemetry)
AttemptStatus = Literal[
    "http-head",
    "http-get",
    "http-200",
    "http-304",
    "robots-fetch",
    "robots-disallowed",
    "retry",
    "size-mismatch",
    "content-policy-skip",
    "download-error",
]

#: Normalized reason codes (for analytics and recovery logic)
ReasonCode = Literal[
    "ok",
    "not-modified",
    "retry-after",
    "backoff",
    "robots",
    "policy-type",
    "policy-size",
    "timeout",
    "conn-error",
    "tls-error",
    "too-large",
    "unexpected-ct",
    "size-mismatch",
]


# ============================================================================
# CORE API PAYLOADS
# ============================================================================


@dataclass(frozen=True, slots=True)
class DownloadPlan:
    """
    Concrete download plan returned by a resolver.

    Represents a single URL candidate to fetch along with metadata about
    expected content type and optional HTTP hints (etag, last-modified).

    Immutable: intended for passing between resolvers → pipeline → execution.
    """

    url: str
    """The URL to fetch (required, non-empty)."""

    resolver_name: str
    """Which resolver generated this plan (e.g., 'unpaywall', 'arxiv', 'crossref')."""

    referer: Optional[str] = None
    """Optional Referer header value if needed by upstream."""

    expected_mime: Optional[str] = None
    """Expected MIME type (used for content policy validation)."""

    etag: Optional[str] = None
    """Optional ETag for conditional GET (If-None-Match)."""

    last_modified: Optional[str] = None
    """Optional Last-Modified header value for conditional GET (If-Modified-Since)."""

    max_bytes_override: Optional[int] = None
    """Optional per-plan byte cap (overrides global setting if set)."""

    def __post_init__(self) -> None:
        """Validate plan after construction."""
        if not self.url or not self.url.strip():
            raise ValueError("DownloadPlan.url cannot be empty")
        if not self.resolver_name or not self.resolver_name.strip():
            raise ValueError("DownloadPlan.resolver_name cannot be empty")


@dataclass(frozen=True, slots=True)
class DownloadStreamResult:
    """
    Result of streaming a payload to disk.

    Represents the low-level outcome of the GET request, file write,
    and any size/type validation before finalization.

    Path remains temporary (*.part) until finalize_candidate_download
    promotes it to the final destination.
    """

    path_tmp: str
    """Path to the temporary file (pre-rename, pre-final-name)."""

    bytes_written: int
    """Number of bytes actually written to disk."""

    http_status: int
    """HTTP response status code (200, 304, etc.)."""

    content_type: Optional[str] = None
    """HTTP Content-Type header value from the response."""


@dataclass(frozen=True, slots=True)
class DownloadOutcome:
    """
    Final outcome of a download attempt.

    Recorded in the manifest and used for metrics/telemetry.
    Represents the last state after all integrity checks and finalization.
    """

    ok: bool
    """True if download succeeded and file is ready, False otherwise."""

    classification: OutcomeClass
    """Status category: 'success' | 'skip' | 'error'."""

    path: Optional[str] = None
    """Final path to downloaded file (None if not successful)."""

    reason: Optional[ReasonCode] = None
    """Normalized reason token for why skip/error occurred."""

    meta: Mapping[str, Any] = field(default_factory=dict)
    """Small structured metadata (keep lightweight; don't store large blobs here)."""

    def __post_init__(self) -> None:
        """Validate outcome invariants."""
        # Validate classification enum
        valid_classes: set[OutcomeClass] = {"success", "skip", "error"}
        if self.classification not in valid_classes:
            raise ValueError(
                f"DownloadOutcome.classification must be one of {valid_classes}, "
                f"got {self.classification!r}"
            )

        # Enforce invariant: ok=True ⇒ classification='success'
        if self.ok and self.classification != "success":
            raise ValueError(
                f"DownloadOutcome.ok=True requires classification='success', "
                f"got {self.classification!r}"
            )

        # Enforce invariant: ok=False ⇒ path=None
        if not self.ok and self.path is not None:
            raise ValueError(
                f"DownloadOutcome.ok=False implies path must be None, got path={self.path!r}"
            )


@dataclass(frozen=True, slots=True)
class ResolverResult:
    """
    Result of resolver execution.

    A resolver can return zero or more DownloadPlans. The pipeline
    tries them in order until one succeeds or all are exhausted.

    Empty plans=() means this resolver has nothing to contribute.
    """

    plans: Sequence[DownloadPlan]
    """Zero or more plans to try (in order). Immutable sequence."""

    notes: Mapping[str, Any] = field(default_factory=dict)
    """Optional diagnostic/telemetry notes (resolver state, debug info, etc.)."""


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    """
    Record of a single download attempt (for telemetry).

    Lightweight summary of one resolver/plan attempt for logging and metrics.
    """

    run_id: str
    """Unique run identifier stamped on all records in a session."""

    resolver_name: str
    """Resolver that generated the plan."""

    url: str
    """URL that was attempted."""

    status: AttemptStatus
    """Outcome status token (http-head, http-200, robots-disallowed, etc.)."""

    http_status: Optional[int] = None
    """HTTP response status code if applicable."""

    elapsed_ms: Optional[int] = None
    """Time spent on this attempt (in milliseconds)."""

    meta: Mapping[str, Any] = field(default_factory=dict)
    """Optional metadata (cache_hit, bytes, reason codes, etc.)."""
