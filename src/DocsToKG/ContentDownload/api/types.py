"""
Unified API Types for ContentDownload Pipeline

Provides frozen dataclasses as contracts between resolvers, download execution,
and pipeline orchestration. All types are frozen to prevent accidental mutation.

Data flow:
  Resolver → DownloadPlan → Download Execution → DownloadOutcome → Manifest
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class DownloadPlan:
    """
    Concrete download plan returned by a resolver.

    Represents a single URL to fetch along with metadata about the expected
    content and hints for the download executor.
    """

    url: str
    """The URL to fetch."""

    resolver_name: str
    """Which resolver generated this plan (e.g., 'unpaywall', 'arxiv')."""

    referer: Optional[str] = None
    """Referer header value (if needed by the upstream)."""

    expected_mime: Optional[str] = None
    """Expected MIME type (used for content policy checks)."""

    def __post_init__(self) -> None:
        """Validate after construction."""
        if not self.url:
            raise ValueError("url cannot be empty")
        if not self.resolver_name:
            raise ValueError("resolver_name cannot be empty")


@dataclass(frozen=True)
class DownloadStreamResult:
    """
    Result of streaming a payload to disk.

    Represents the low-level outcome of the GET request and file write.
    """

    path_tmp: str
    """Temporary file path (may not be final if integrity checks fail)."""

    bytes_written: int
    """Number of bytes written to disk."""

    http_status: int
    """HTTP response status code."""

    content_type: Optional[str] = None
    """HTTP Content-Type header value."""


@dataclass(frozen=True)
class DownloadOutcome:
    """
    Final outcome of a download attempt.

    Recorded in the manifest and used for metrics/telemetry.
    """

    ok: bool
    """True if download succeeded, False otherwise."""

    path: Optional[str] = None
    """Final path to downloaded file (None if not successful)."""

    classification: str = "error"
    """Status category: 'success' | 'skip' | 'error'."""

    reason: Optional[str] = None
    """Normalized reason token (see taxonomy in ARCHITECTURE.md)."""

    meta: Optional[Dict[str, Any]] = None
    """Additional metadata (e.g., content hash, retry count)."""

    def __post_init__(self) -> None:
        """Validate classification."""
        valid = {"success", "skip", "error"}
        if self.classification not in valid:
            raise ValueError(f"classification must be in {valid}, got {self.classification!r}")

        if self.ok and self.classification != "success":
            raise ValueError("ok=True requires classification='success'")


@dataclass(frozen=True)
class ResolverResult:
    """
    Result of a resolver attempting to find a download plan.

    A resolver may return zero plans (didn't find anything), one plan (typical),
    or multiple plans (rare cases like duplicate mirrors).
    """

    plans: List[DownloadPlan]
    """List of download plans (often 0 or 1, occasionally more)."""

    notes: Optional[Dict[str, Any]] = None
    """Optional resolver-specific notes for debugging."""

    def __post_init__(self) -> None:
        """Validate."""
        if self.plans is None:
            object.__setattr__(self, "plans", [])


@dataclass(frozen=True)
class AttemptRecord:
    """
    Single network attempt during download (for telemetry).

    Every HTTP HEAD, GET, robots.txt fetch, retry, and backoff event
    produces one attempt record in the attempt log.
    """

    ts: str
    """ISO 8601 UTC timestamp."""

    run_id: Optional[str]
    """Run identifier (for correlation)."""

    resolver: Optional[str]
    """Resolver name (e.g., 'unpaywall'), or None for non-resolver attempts."""

    url: str
    """URL being accessed."""

    verb: str
    """HTTP verb: 'HEAD', 'GET', 'ROBOTS', 'RETRY', etc."""

    status: str
    """Attempt category: 'http-head', 'http-get', 'http-200', etc."""

    http_status: Optional[int] = None
    """HTTP status code (if applicable)."""

    content_type: Optional[str] = None
    """HTTP Content-Type header (if applicable)."""

    elapsed_ms: Optional[int] = None
    """Elapsed time in milliseconds."""

    bytes_written: Optional[int] = None
    """Bytes written to disk (if applicable)."""

    content_length_hdr: Optional[int] = None
    """Content-Length header value (if present)."""

    reason: Optional[str] = None
    """Normalized reason token (e.g., 'ok', 'retry', 'timeout')."""

    extra: Dict[str, Any] = field(default_factory=dict)
    """Additional structured data (e.g., retry count, sleep duration)."""
