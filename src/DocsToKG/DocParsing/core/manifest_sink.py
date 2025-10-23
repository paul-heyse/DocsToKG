# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.DocParsing.core.manifest_sink",
#   "purpose": "Unified manifest sink for DocParsing stages.",
#   "sections": [
#     {
#       "id": "manifestsink",
#       "name": "ManifestSink",
#       "anchor": "class-manifestsink",
#       "kind": "class"
#     },
#     {
#       "id": "manifestentry",
#       "name": "ManifestEntry",
#       "anchor": "class-manifestentry",
#       "kind": "class"
#     },
#     {
#       "id": "jsonlmanifestsink",
#       "name": "JsonlManifestSink",
#       "anchor": "class-jsonlmanifestsink",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""Unified manifest sink for DocParsing stages.

This module provides a protocol and implementation for writing stage manifest
entries (success, skip, failure) with atomic, lock-aware JSONL appending using
the lock-aware JsonlWriter component from io.py.

All stages use this abstraction to ensure consistent base fields and reliable
concurrent writes even when multiple processes report progress simultaneously.
The implementation leverages FileLock and atomic appends to prevent manifest
corruption during concurrent access, making it safe for distributed pipelines.

Key components:
- ManifestSink: Protocol defining the manifest writing interface
- JsonlManifestSink: Implementation using atomic JSONL appends
- ManifestEntry: Dataclass for individual entries

All writes are atomic and process-safe, suitable for distributed pipelines
where multiple workers may write concurrently.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, runtime_checkable

from filelock import FileLock

__all__ = [
    "ManifestSink",
    "JsonlManifestSink",
    "ManifestEntry",
]


@runtime_checkable
class ManifestSink(Protocol):
    """Protocol for writing stage manifest entries atomically."""

    def log_success(
        self,
        stage: str,
        item_id: str,
        input_path: Path | str,
        output_paths: Mapping[str, Path | str],
        duration_s: float,
        schema_version: str,
        extras: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Write a success manifest entry.

        Args:
            stage: Stage name (e.g., "doctags", "chunk", "embed").
            item_id: Stable item identifier.
            input_path: Primary input artifact path.
            output_paths: Map of output artifact names to paths.
            duration_s: Execution duration in seconds.
            schema_version: Schema version for this entry.
            extras: Stage-specific metadata (optional).
        """
        ...

    def log_skip(
        self,
        stage: str,
        item_id: str,
        input_path: Path | str,
        output_path: Path | str,
        duration_s: float,
        schema_version: str,
        reason: str = "resume-satisfied",
        extras: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Write a skip manifest entry.

        Args:
            stage: Stage name.
            item_id: Item identifier.
            input_path: Input artifact path.
            output_path: Output artifact path.
            duration_s: Skip check duration.
            schema_version: Schema version.
            reason: Reason for skip (e.g., "resume-satisfied", "user-skip").
            extras: Stage-specific metadata (optional).
        """
        ...

    def log_failure(
        self,
        stage: str,
        item_id: str,
        input_path: Path | str,
        output_path: Path | str,
        duration_s: float,
        schema_version: str,
        error: str,
        extras: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Write a failure manifest entry.

        Args:
            stage: Stage name.
            item_id: Item identifier.
            input_path: Input artifact path.
            output_path: Output artifact path.
            duration_s: Execution duration.
            schema_version: Schema version.
            error: Error message or code.
            extras: Stage-specific metadata (optional).
        """
        ...


@dataclass(slots=True, frozen=True)
class ManifestEntry:
    """Structured manifest entry with base and optional extra fields."""

    # Base fields (required across all stages)
    stage: str
    doc_id: str
    status: str  # "success", "skip", "failure"
    duration_s: float
    input_path: str
    output_path: str
    schema_version: str

    # Optional fields
    input_hash: Optional[str] = None
    attempts: int = 1
    reason: Optional[str] = None  # for skip
    error: Optional[str] = None  # for failure

    # Extra fields (stage-specific)
    extras: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize entry to JSON line format."""
        payload = asdict(self)
        if not payload["extras"]:
            del payload["extras"]
        return json.dumps(payload, default=str)


class JsonlManifestSink:
    """JSONL manifest sink with atomic writes via FileLock."""

    def __init__(self, manifest_path: Path | str) -> None:
        """Initialize sink pointing to manifest JSONL file.

        Args:
            manifest_path: Path to manifest JSONL file.
        """
        self.manifest_path = Path(manifest_path)
        self.lock_path = self.manifest_path.with_suffix(self.manifest_path.suffix + ".lock")

    def _append_entry(self, entry: ManifestEntry) -> None:
        """Append entry to manifest with FileLock for atomicity."""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

        with FileLock(str(self.lock_path), timeout=30.0):
            with open(self.manifest_path, "a", encoding="utf-8") as f:
                f.write(entry.to_json())
                f.write("\n")
                f.flush()

    def log_success(
        self,
        stage: str,
        item_id: str,
        input_path: Path | str,
        output_paths: Mapping[str, Path | str],
        duration_s: float,
        schema_version: str,
        extras: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Write a success entry."""
        output_path_str = str(list(output_paths.values())[0]) if output_paths else ""
        entry = ManifestEntry(
            stage=stage,
            doc_id=item_id,
            status="success",
            duration_s=float(duration_s),
            input_path=str(input_path),
            output_path=output_path_str,
            schema_version=schema_version,
            extras=dict(extras) if extras else {},
        )
        self._append_entry(entry)

    def log_skip(
        self,
        stage: str,
        item_id: str,
        input_path: Path | str,
        output_path: Path | str,
        duration_s: float,
        schema_version: str,
        reason: str = "resume-satisfied",
        extras: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Write a skip entry."""
        entry = ManifestEntry(
            stage=stage,
            doc_id=item_id,
            status="skip",
            duration_s=float(duration_s),
            input_path=str(input_path),
            output_path=str(output_path),
            schema_version=schema_version,
            reason=reason,
            extras=dict(extras) if extras else {},
        )
        self._append_entry(entry)

    def log_failure(
        self,
        stage: str,
        item_id: str,
        input_path: Path | str,
        output_path: Path | str,
        duration_s: float,
        schema_version: str,
        error: str,
        extras: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Write a failure entry."""
        entry = ManifestEntry(
            stage=stage,
            doc_id=item_id,
            status="failure",
            duration_s=float(duration_s),
            input_path=str(input_path),
            output_path=str(output_path),
            schema_version=schema_version,
            error=error,
            extras=dict(extras) if extras else {},
        )
        self._append_entry(entry)
