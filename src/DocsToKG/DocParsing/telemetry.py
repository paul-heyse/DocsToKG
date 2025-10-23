# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.DocParsing.telemetry",
#   "purpose": "Telemetry data structures and JSONL sinks for DocParsing stages.",
#   "sections": [
#     {
#       "id": "default-writer",
#       "name": "_default_writer",
#       "anchor": "function-default-writer",
#       "kind": "function"
#     },
#     {
#       "id": "attempt",
#       "name": "Attempt",
#       "anchor": "class-attempt",
#       "kind": "class"
#     },
#     {
#       "id": "manifestentry",
#       "name": "ManifestEntry",
#       "anchor": "class-manifestentry",
#       "kind": "class"
#     },
#     {
#       "id": "telemetrysink",
#       "name": "TelemetrySink",
#       "anchor": "class-telemetrysink",
#       "kind": "class"
#     },
#     {
#       "id": "input-bytes",
#       "name": "_input_bytes",
#       "anchor": "function-input-bytes",
#       "kind": "function"
#     },
#     {
#       "id": "stagetelemetry",
#       "name": "StageTelemetry",
#       "anchor": "class-stagetelemetry",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""Telemetry data structures and JSONL sinks for DocParsing stages.

Each stage records structured attempts and manifest entries to support resume
logic and observability dashboards. This module defines dataclasses for representing
those events plus TelemetrySink implementations that append them to JSONL files
using a lock-aware writer (via dependency injection), guaranteeing atomic writes
even when multiple processes report progress concurrently.

Key Components:
- Attempt: Dataclass representing a single processing attempt (status, duration, metadata)
- ManifestEntry: Dataclass representing successful pipeline output (tokens, schema version)
- TelemetrySink: Manages persistent storage of attempts and manifest entries to JSONL
- StageTelemetry: Binds a sink to a specific run ID and stage name for convenient logging
- DEFAULT_JSONL_WRITER: Provides lock-aware appending for concurrent-safe writes

The TelemetrySink and StageTelemetry accept an optional writer dependency (defaulting
to DEFAULT_JSONL_WRITER), enabling both production use and test injection of custom
writers without modifying telemetry logic.

Example:
    from DocsToKG.DocParsing.telemetry import TelemetrySink, StageTelemetry
    from pathlib import Path

    # Create a telemetry sink for a pipeline run
    sink = TelemetrySink(
        attempts_path=Path("attempts.jsonl"),
        manifest_path=Path("manifest.jsonl")
    )

    # Bind to a specific stage and run
    stage_telemetry = StageTelemetry(
        sink=sink,
        run_id="run-2025-10-21",
        stage="embedding"
    )

    # Log attempt completion (uses lock-aware writer internally)
    stage_telemetry.log_attempt_success(
        file_id="doc1",
        duration_s=1.23,
        output_path="vectors.npy"
    )
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from DocsToKG.DocParsing.io import DEFAULT_JSONL_WRITER

if TYPE_CHECKING:
    from DocsToKG.DocParsing.embedding.backends.base import ProviderIdentity

__all__ = [
    "Attempt",
    "ManifestEntry",
    "TelemetrySink",
    "StageTelemetry",
]


def _default_writer(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Append ``rows`` to ``path`` using the lock-aware JsonlWriter."""
    return DEFAULT_JSONL_WRITER(path, rows)


@dataclass(slots=True)
class Attempt:
    """Describe a pipeline attempt for a single document."""

    run_id: str
    file_id: str
    stage: str
    status: str
    reason: str | None
    started_at: float
    finished_at: float
    bytes: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ManifestEntry:
    """Describe a successful pipeline output."""

    run_id: str
    file_id: str
    stage: str
    output_path: str
    tokens: int
    schema_version: str | None
    duration_s: float
    metadata: dict[str, Any] = field(default_factory=dict)


class TelemetrySink:
    """Persistence helper for attempt and manifest telemetry."""

    def __init__(
        self,
        attempts_path: Path,
        manifest_path: Path,
        *,
        writer: Callable[[Path, Iterable[dict[str, Any]]], int | None] | None = None,
    ) -> None:
        """Initialise sink paths and ensure parent directories exist.

        Args:
            attempts_path: Path to the attempts JSONL file.
            manifest_path: Path to the manifest JSONL file.
            writer: Optional custom writer callable (path, rows) -> int | None.
                    Defaults to DEFAULT_JSONL_WRITER (lock-aware appending).
        """

        self._attempts_path = attempts_path
        self._manifest_path = manifest_path
        self._attempts_path.parent.mkdir(parents=True, exist_ok=True)
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = writer or _default_writer

    @property
    def writer(self) -> Callable[[Path, Iterable[dict[str, Any]]], int | None]:
        """Return the default writer for this sink."""

        return self._writer

    def _append_payload(
        self,
        path: Path,
        payload: dict[str, Any],
        *,
        writer: Callable[[Path, Iterable[dict[str, Any]]], int | None] | None = None,
    ) -> None:
        """Append ``payload`` to ``path`` using the provided writer."""

        active_writer = writer or self._writer
        active_writer(path, [payload])

    def write_attempt(
        self,
        attempt: Attempt,
        *,
        writer: Callable[[Path, Iterable[dict[str, Any]]], int | None] | None = None,
    ) -> None:
        """Append ``attempt`` to the attempts log."""

        payload = asdict(attempt)
        metadata = dict(payload.pop("metadata", {}) or {})
        payload.update(metadata)
        self._append_payload(self._attempts_path, payload, writer=writer)

    def write_manifest_entry(
        self,
        entry: ManifestEntry,
        *,
        writer: Callable[[Path, Iterable[dict[str, Any]]], int | None] | None = None,
    ) -> None:
        """Append ``entry`` to the manifest log."""

        payload = asdict(entry)
        metadata = dict(payload.pop("metadata", {}) or {})
        payload.update(metadata)
        payload.setdefault("doc_id", entry.file_id)
        self._append_payload(self._manifest_path, payload, writer=writer)

    def write_provider_event(
        self,
        payload: dict[str, Any],
        *,
        writer: Callable[[Path, Iterable[dict[str, Any]]], int | None] | None = None,
    ) -> None:
        """Append provider telemetry to the attempts log."""

        self._append_payload(self._attempts_path, payload, writer=writer)


def _input_bytes(path: Path | str) -> int:
    """Best-effort size lookup for ``path`` returning zero on failure."""

    try:
        return Path(path).stat().st_size
    except (OSError, ValueError):
        return 0


class StageTelemetry:
    """Lightweight helper binding a sink to a specific stage/run."""

    def __init__(
        self,
        sink: TelemetrySink,
        *,
        run_id: str,
        stage: str,
        writer: Callable[[Path, Iterable[dict[str, Any]]], int | None] | None = None,
    ) -> None:
        """Bind the telemetry sink to a specific run identifier and stage.

        Args:
            sink: The TelemetrySink instance managing attempt/manifest paths.
            run_id: Unique identifier for this pipeline run.
            stage: Name of the stage (e.g., 'doctags', 'chunk', 'embed').
            writer: Optional custom writer callable. Defaults to sink's writer
                    (which defaults to DEFAULT_JSONL_WRITER for lock-aware appending).
        """

        self._sink = sink
        self._run_id = run_id
        self._stage = stage
        self._writer = writer or sink.writer

    def record_attempt(
        self,
        *,
        doc_id: str,
        input_path: Path | str,
        status: str,
        duration_s: float = 0.0,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist an Attempt entry describing the outcome for ``doc_id``."""

        finished = time.time()
        started = finished - max(duration_s, 0.0)
        payload = Attempt(
            run_id=self._run_id,
            file_id=doc_id,
            stage=self._stage,
            status=status,
            reason=reason,
            started_at=started,
            finished_at=finished,
            bytes=_input_bytes(input_path),
            metadata=metadata or {},
        )
        self._sink.write_attempt(payload, writer=self._writer)

    def write_manifest(
        self,
        *,
        doc_id: str,
        output_path: Path | str,
        schema_version: str | None,
        duration_s: float,
        tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append a manifest row for ``doc_id`` and optional metadata."""

        entry = ManifestEntry(
            run_id=self._run_id,
            file_id=doc_id,
            stage=self._stage,
            output_path=str(output_path),
            tokens=tokens or 0,
            schema_version=schema_version,
            duration_s=duration_s,
            metadata=metadata or {},
        )
        self._sink.write_manifest_entry(entry, writer=self._writer)

    def log_success(
        self,
        *,
        doc_id: str,
        input_path: Path | str,
        output_path: Path | str,
        schema_version: str,
        duration_s: float,
        tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a successful attempt and mirror the manifest entry."""

        self.record_attempt(
            doc_id=doc_id,
            input_path=input_path,
            status="success",
            duration_s=duration_s,
            metadata=metadata,
        )
        self.write_manifest(
            doc_id=doc_id,
            output_path=output_path,
            tokens=tokens,
            schema_version=schema_version,
            duration_s=duration_s,
            metadata=metadata,
        )

    def log_failure(
        self,
        *,
        doc_id: str,
        input_path: Path | str,
        duration_s: float,
        reason: str,
        metadata: dict[str, Any] | None = None,
        manifest_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a failure attempt and optionally log manifest metadata."""

        self.record_attempt(
            doc_id=doc_id,
            input_path=input_path,
            status="failure",
            duration_s=duration_s,
            reason=reason,
            metadata=metadata,
        )
        if manifest_metadata is not None:
            self.write_manifest(
                doc_id=doc_id,
                output_path=manifest_metadata.get("output_path", ""),
                tokens=0,
                schema_version=manifest_metadata.get("schema_version", ""),
                duration_s=duration_s,
                metadata=manifest_metadata,
            )

    def log_skip(
        self,
        *,
        doc_id: str,
        input_path: Path | str,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a skipped attempt and optional manifest metadata."""

        self.record_attempt(
            doc_id=doc_id,
            input_path=input_path,
            status="skip",
            duration_s=0.0,
            reason=reason,
            metadata=metadata,
        )
        if metadata is not None:
            self.write_manifest(
                doc_id=doc_id,
                output_path=metadata.get("output_path", ""),
                tokens=0,
                schema_version=metadata.get("schema_version"),
                duration_s=0.0,
                metadata=metadata,
            )

    def log_config(
        self,
        *,
        output_path: Path | str,
        schema_version: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record the configuration manifest emitted at startup."""

        self.write_manifest(
            doc_id="__config__",
            output_path=output_path,
            tokens=0,
            schema_version=schema_version,
            duration_s=0.0,
            metadata=metadata,
        )

    def log_provider_event(
        self,
        *,
        provider: ProviderIdentity,
        phase: str,
        data: dict[str, Any],
    ) -> None:
        """Record telemetry for provider lifecycle and batch metrics."""

        payload: dict[str, Any] = {
            "run_id": self._run_id,
            "file_id": f"__provider__/{provider.name}",
            "stage": self._stage,
            "status": f"provider:{phase}",
            "provider": provider.name,
            "provider_version": provider.version,
            "phase": phase,
            "timestamp": time.time(),
        }
        for key, value in data.items():
            if value is not None:
                payload[key] = value
        payload = {k: v for k, v in payload.items() if v is not None}
        self._sink.write_provider_event(payload, writer=self._writer)
