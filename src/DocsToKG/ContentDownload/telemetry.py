from __future__ import annotations

"""
Shared telemetry contracts for DocsToKG content downloads.

The download CLI and resolver pipeline both emit attempt and manifest telemetry.
This module centralises the shared dataclasses and protocols so callers import a
consistent contract, avoiding subtle drift between components.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.resolvers import AttemptRecord


@dataclass
class ManifestEntry:
    """Structured manifest entry describing a resolved artifact."""

    timestamp: str
    work_id: str
    title: str
    publication_year: Optional[int]
    resolver: Optional[str]
    url: Optional[str]
    path: Optional[str]
    classification: str
    content_type: Optional[str]
    reason: Optional[str]
    html_paths: List[str] = field(default_factory=list)
    sha256: Optional[str] = None
    content_length: Optional[int] = None
    etag: Optional[str] = None
    last_modified: Optional[str] = None
    extracted_text_path: Optional[str] = None
    dry_run: bool = False


class AttemptSink(Protocol):
    """Protocol implemented by telemetry sinks used by the pipeline and CLI."""

    def log_attempt(self, record: "AttemptRecord", *, timestamp: Optional[str] = None) -> None:
        """Record a resolver attempt."""

    def log_manifest(self, entry: ManifestEntry) -> None:
        """Persist a manifest entry."""

    def log_summary(self, summary: Dict[str, Any]) -> None:
        """Store aggregated run metrics."""

    def close(self) -> None:
        """Release any resources held by the sink."""

