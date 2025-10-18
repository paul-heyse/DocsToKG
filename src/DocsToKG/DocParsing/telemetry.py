"""Telemetry sink interfaces for DocParsing pipelines."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = [
    "Attempt",
    "ManifestEntry",
    "TelemetrySink",
    "StageTelemetry",
]


@dataclass(slots=True)
class Attempt:
    """Describe a pipeline attempt for a single document."""

    run_id: str
    file_id: str
    stage: str
    status: str
    reason: Optional[str]
    started_at: float
    finished_at: float
    bytes: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ManifestEntry:
    """Describe a successful pipeline output."""

    run_id: str
    file_id: str
    stage: str
    output_path: str
    tokens: int
    schema_version: str
    duration_s: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class TelemetrySink:
    """Persistence helper for attempt and manifest telemetry."""

    def __init__(self, attempts_path: Path, manifest_path: Path) -> None:
        """Initialise sink paths and ensure parent directories exist."""

        self._attempts_path = attempts_path
        self._manifest_path = manifest_path
        self._attempts_path.parent.mkdir(parents=True, exist_ok=True)
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)

    def write_attempt(self, attempt: Attempt) -> None:
        """Append ``attempt`` to the attempts log."""

        payload = asdict(attempt)
        metadata = dict(payload.pop("metadata", {}) or {})
        payload.update(metadata)
        with open(self._attempts_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def write_manifest_entry(self, entry: ManifestEntry) -> None:
        """Append ``entry`` to the manifest log."""

        payload = asdict(entry)
        metadata = dict(payload.pop("metadata", {}) or {})
        payload.update(metadata)
        with open(self._manifest_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _input_bytes(path: Path | str) -> int:
    """Best-effort size lookup for ``path`` returning zero on failure."""

    try:
        return Path(path).stat().st_size
    except (OSError, ValueError):
        return 0


class StageTelemetry:
    """Lightweight helper binding a sink to a specific stage/run."""

    def __init__(self, sink: TelemetrySink, *, run_id: str, stage: str) -> None:
        self._sink = sink
        self._run_id = run_id
        self._stage = stage

    def record_attempt(
        self,
        *,
        doc_id: str,
        input_path: Path | str,
        status: str,
        duration_s: float = 0.0,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
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
        self._sink.write_attempt(payload)

    def write_manifest(
        self,
        *,
        doc_id: str,
        output_path: Path | str,
        tokens: Optional[int] = None,
        schema_version: str,
        duration_s: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
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
        self._sink.write_manifest_entry(entry)

    def log_success(
        self,
        *,
        doc_id: str,
        input_path: Path | str,
        output_path: Path | str,
        tokens: Optional[int] = None,
        schema_version: str,
        duration_s: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
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
        metadata: Optional[Dict[str, Any]] = None,
        manifest_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
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
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.record_attempt(
            doc_id=doc_id,
            input_path=input_path,
            status="skip",
            duration_s=0.0,
            reason=reason,
            metadata=metadata,
        )

    def log_config(
        self,
        *,
        output_path: Path | str,
        schema_version: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.write_manifest(
            doc_id="__config__",
            output_path=output_path,
            tokens=0,
            schema_version=schema_version,
            duration_s=0.0,
            metadata=metadata,
        )
