"""Telemetry contracts shared by DocsToKG content download tooling.

Centralises manifest and attempt payload structures so the CLI, resolver
pipeline, and downstream analytics speak a consistent schema. Keeping these
definitions in one place prevents subtle drift between components while allowing
test doubles to implement the same interfaces.

Args:
    None

Returns:
    None

Raises:
    None
"""

from __future__ import annotations

import csv
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Protocol

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.resolvers import AttemptRecord

from DocsToKG.ContentDownload.classifications import PDF_LIKE, Classification

MANIFEST_SCHEMA_VERSION = 2


@dataclass
class ManifestEntry:
    """Structured manifest entry describing a resolved artifact.

    Attributes:
        schema_version: Integer identifying the manifest schema revision.
        timestamp: ISO-8601 timestamp describing when the artifact was stored.
        work_id: Primary identifier (e.g., OpenAlex work ID) for the artifact.
        title: Human-readable work title.
        publication_year: Optional publication year derived from upstream data.
        resolver: Resolver name that sourced the artifact, when known.
        url: Final URL from which the artifact was fetched.
        path: Local filesystem path where the artifact is stored.
        classification: Classifier label describing the download outcome.
        content_type: MIME type or equivalent classification when available.
        reason: Diagnostic reason for failure or classification notes.
        html_paths: Any extracted HTML content paths stored alongside the PDF.
        sha256: Optional SHA256 digest of the stored artifact.
        content_length: Content size in bytes when reported by the server.
        etag: Server-provided entity tag when available.
        last_modified: HTTP ``Last-Modified`` header value if supplied.
        extracted_text_path: Path to extracted text artefacts when produced.
        dry_run: Flag indicating whether the artifact was processed in dry-run mode.

    Examples:
        >>> entry = ManifestEntry(
        ...     schema_version=2,
        ...     timestamp="2025-01-01T00:00:00Z",
        ...     work_id="W123",
        ...     title="Example",
        ...     publication_year=2024,
        ...     resolver="openalex",
        ...     url="https://example.org/doc.pdf",
        ...     path="/tmp/doc.pdf",
        ...     classification="success",
        ...     content_type="application/pdf",
        ...     reason=None,
        ... )
        >>> entry.work_id
        'W123'
    """

    schema_version: int
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
    """Protocol implemented by telemetry sinks used by the pipeline and CLI.

    Implementations typically write to JSONL/CSV and may maintain external
    storage handles; consumers should always call :meth:`close` when finished.

    Attributes:
        None

    Examples:
        >>> class InMemorySink:
        ...     def __init__(self) -> None:
        ...         self.attempts = []
        ...     def log_attempt(self, record, *, timestamp=None):
        ...         self.attempts.append(record)
        ...     def log_manifest(self, entry):
        ...         pass
        ...     def log_summary(self, summary):
        ...         pass
        ...     def close(self):
        ...         pass
        >>> isinstance(InMemorySink(), AttemptSink)
        True
    """

    def log_attempt(self, record: "AttemptRecord", *, timestamp: Optional[str] = None) -> None:
        """Record a resolver attempt.

        Args:
            record: Resolver attempt metadata.
            timestamp: Optional timestamp override for the event.

        Returns:
            None

        Raises:
            Exception: Implementations may propagate write failures.
        """

    def log_manifest(self, entry: ManifestEntry) -> None:
        """Persist a manifest entry.

        Args:
            entry: Manifest payload describing a successfully processed artifact.

        Returns:
            None

        Raises:
            Exception: Implementations may propagate write failures.
        """

    def log_summary(self, summary: Dict[str, Any]) -> None:
        """Store aggregated run metrics.

        Args:
            summary: Aggregated counters or diagnostic information.

        Returns:
            None

        Raises:
            Exception: Implementations may propagate write failures.
        """

    def close(self) -> None:
        """Release any resources held by the sink.

        Args:
            None

        Returns:
            None

        Raises:
            Exception: Implementations may propagate shutdown failures.
        """


def _utc_timestamp() -> str:
    """Return the current time as an ISO 8601 UTC timestamp."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _ensure_parent_exists(path: Path) -> None:
    """Ensure the parent directory for ``path`` exists."""

    path.parent.mkdir(parents=True, exist_ok=True)


class JsonlSink:
    """Thread-safe sink that streams attempt, manifest, and summary events to JSONL files."""

    def __init__(self, path: Path) -> None:
        self._path = path
        _ensure_parent_exists(path)
        self._file = path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def _write(self, payload: Dict[str, Any]) -> None:
        payload.setdefault("timestamp", _utc_timestamp())
        line = json.dumps(payload, sort_keys=True) + "\n"
        with self._lock:
            self._file.write(line)
            self._file.flush()

    def log_attempt(self, record: "AttemptRecord", *, timestamp: Optional[str] = None) -> None:
        """Append an attempt record to the JSONL log.

        Args:
            record: Attempt metadata captured during resolver execution.
            timestamp: Optional ISO-8601 timestamp overriding the current time.
        """
        ts = timestamp or _utc_timestamp()
        self._write(
            {
                "record_type": "attempt",
                "timestamp": ts,
                "work_id": record.work_id,
                "resolver_name": record.resolver_name,
                "resolver_order": record.resolver_order,
                "url": record.url,
                "status": str(record.status),
                "http_status": record.http_status,
                "content_type": record.content_type,
                "elapsed_ms": record.elapsed_ms,
                "resolver_wall_time_ms": record.resolver_wall_time_ms,
                "reason": record.reason,
                "metadata": record.metadata,
                "sha256": record.sha256,
                "content_length": record.content_length,
                "dry_run": record.dry_run,
            }
        )

    def log_manifest(self, entry: ManifestEntry) -> None:
        """Append a manifest record representing a persisted document.

        Args:
            entry: Manifest entry produced once a resolver finalises output.
        """
        self._write(
            {
                "record_type": "manifest",
                "schema_version": entry.schema_version,
                "timestamp": entry.timestamp,
                "work_id": entry.work_id,
                "title": entry.title,
                "publication_year": entry.publication_year,
                "resolver": entry.resolver,
                "url": entry.url,
                "path": entry.path,
                "classification": str(entry.classification),
                "content_type": entry.content_type,
                "reason": entry.reason,
                "html_paths": entry.html_paths,
                "sha256": entry.sha256,
                "content_length": entry.content_length,
                "etag": entry.etag,
                "last_modified": entry.last_modified,
                "extracted_text_path": entry.extracted_text_path,
                "dry_run": entry.dry_run,
            }
        )

    def log_summary(self, summary: Dict[str, Any]) -> None:
        """Append a run-level summary describing aggregate metrics.

        Args:
            summary: Mapping containing summary counters and metadata.
        """
        payload = {"record_type": "summary", "timestamp": _utc_timestamp(), **summary}
        self._write(payload)

    def close(self) -> None:
        """Flush buffered data and close the underlying JSONL file handle."""

        with self._lock:
            if not self._file.closed:
                self._file.close()

    def __enter__(self) -> "JsonlSink":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class CsvSink:
    """Lightweight sink that mirrors attempt records into a CSV for spreadsheet review."""

    HEADER = [
        "timestamp",
        "work_id",
        "resolver_name",
        "resolver_order",
        "url",
        "status",
        "http_status",
        "content_type",
        "elapsed_ms",
        "resolver_wall_time_ms",
        "reason",
        "sha256",
        "content_length",
        "dry_run",
        "metadata",
    ]

    def __init__(self, path: Path) -> None:
        _ensure_parent_exists(path)
        exists = path.exists()
        self._file = path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.HEADER)
        self._lock = threading.Lock()
        if not exists:
            self._writer.writeheader()

    def log_attempt(self, record: "AttemptRecord", *, timestamp: Optional[str] = None) -> None:
        """Append an attempt record to the CSV log.

        Args:
            record: Attempt metadata captured during resolver execution.
            timestamp: Optional override for the timestamp column.
        """
        ts = timestamp or _utc_timestamp()
        row = {
            "timestamp": ts,
            "work_id": record.work_id,
            "resolver_name": record.resolver_name,
            "resolver_order": record.resolver_order,
            "url": record.url,
            "status": str(record.status),
            "http_status": record.http_status,
            "content_type": record.content_type,
            "elapsed_ms": record.elapsed_ms,
            "resolver_wall_time_ms": record.resolver_wall_time_ms,
            "reason": record.reason,
            "sha256": record.sha256,
            "content_length": record.content_length,
            "dry_run": record.dry_run,
            "metadata": json.dumps(record.metadata, sort_keys=True) if record.metadata else "",
        }
        with self._lock:
            self._writer.writerow(row)
            self._file.flush()

    def log_manifest(self, entry: ManifestEntry) -> None:  # pragma: no cover
        """Ignore manifest writes for CSV sinks (interface compatibility)."""

        return None

    def log_summary(self, summary: Dict[str, Any]) -> None:  # pragma: no cover
        """Ignore summary writes for CSV sinks (interface compatibility)."""

        return None

    def close(self) -> None:
        """Flush buffered data and close the CSV file handle."""

        with self._lock:
            if not self._file.closed:
                self._file.close()

    def __enter__(self) -> "CsvSink":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class MultiSink:
    """Composite sink that fans out logging calls to multiple sinks."""

    def __init__(self, sinks: Iterable[AttemptSink]):
        self._sinks = list(sinks)

    def log_attempt(self, record: "AttemptRecord", *, timestamp: Optional[str] = None) -> None:
        """Fan out an attempt record to each sink in the composite."""

        ts = timestamp or _utc_timestamp()
        for sink in self._sinks:
            sink.log_attempt(record, timestamp=ts)

    def log_manifest(self, entry: ManifestEntry) -> None:
        """Fan out manifest records to every sink in the collection."""

        for sink in self._sinks:
            sink.log_manifest(entry)

    def log_summary(self, summary: Dict[str, Any]) -> None:
        """Fan out summary payloads to every sink in the collection."""

        for sink in self._sinks:
            sink.log_summary(summary)

    def close(self) -> None:
        """Close sinks while capturing the first raised exception."""

        errors: List[BaseException] = []
        for sink in self._sinks:
            try:
                sink.close()
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)
        if errors:
            raise errors[0]

    def __enter__(self) -> "MultiSink":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.close()
        finally:  # pragma: no cover - defensive
            return None


class ManifestIndexSink:
    """Maintain a JSON index mapping work IDs to their latest PDF artefacts."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._index: Dict[str, Dict[str, Optional[str]]] = {}
        self._lock = threading.Lock()
        self._closed = False

    def log_attempt(
        self, record: "AttemptRecord", *, timestamp: Optional[str] = None
    ) -> None:  # pragma: no cover - intentional no-op
        """No-op to satisfy the telemetry sink interface."""

        return None

    def log_summary(self, summary: Dict[str, Any]) -> None:  # pragma: no cover - no-op
        """No-op because the manifest index only reacts to manifests."""

        return None

    def log_manifest(self, entry: ManifestEntry) -> None:
        """Index the latest manifest metadata for ``entry.work_id``."""

        classification = Classification.from_wire(entry.classification)
        path_value = entry.path
        path_str = str(path_value) if path_value else None
        pdf_path: Optional[str] = None
        sha256: Optional[str] = None
        if classification in PDF_LIKE and path_str:
            pdf_path = path_str
            sha256 = entry.sha256
        payload = {
            "classification": classification.value,
            "pdf_path": pdf_path,
            "sha256": sha256,
        }
        with self._lock:
            self._index[entry.work_id] = payload

    def close(self) -> None:
        """Write the collected manifest index to disk once."""

        with self._lock:
            if self._closed:
                return
            ordered = dict(sorted(self._index.items(), key=lambda item: item[0]))
            self._closed = True
        _ensure_parent_exists(self._path)
        self._path.write_text(json.dumps(ordered, indent=2), encoding="utf-8")

    def __enter__(self) -> "ManifestIndexSink":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class LastAttemptCsvSink:
    """Write a CSV snapshot containing the most recent manifest entry per work."""

    HEADER = [
        "work_id",
        "title",
        "publication_year",
        "resolver",
        "url",
        "classification",
        "path",
        "sha256",
        "content_length",
        "etag",
        "last_modified",
    ]

    def __init__(self, path: Path) -> None:
        self._path = path
        self._records: "OrderedDict[str, Dict[str, str]]" = OrderedDict()
        self._lock = threading.Lock()
        self._closed = False

    def log_attempt(
        self, record: "AttemptRecord", *, timestamp: Optional[str] = None
    ) -> None:  # pragma: no cover - intentional no-op
        """No-op to satisfy the telemetry sink interface."""

        return None

    def log_summary(self, summary: Dict[str, Any]) -> None:  # pragma: no cover - no-op
        """No-op because the CSV sink only records manifest events."""

        return None

    def _normalise(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (str, int)):
            return str(value)
        return str(value)

    def log_manifest(self, entry: ManifestEntry) -> None:
        """Store the most recent manifest attributes for ``entry.work_id``."""

        classification = Classification.from_wire(entry.classification)
        data = {
            "work_id": entry.work_id,
            "title": self._normalise(entry.title),
            "publication_year": self._normalise(entry.publication_year),
            "resolver": self._normalise(entry.resolver),
            "url": self._normalise(entry.url),
            "classification": classification.value,
            "path": self._normalise(entry.path),
            "sha256": self._normalise(entry.sha256),
            "content_length": self._normalise(entry.content_length),
            "etag": self._normalise(entry.etag),
            "last_modified": self._normalise(entry.last_modified),
        }
        with self._lock:
            self._records[entry.work_id] = data
            self._records.move_to_end(entry.work_id)

    def close(self) -> None:
        """Flush collected manifest rows to the CSV file."""

        with self._lock:
            if self._closed:
                return
            rows = list(self._records.values())
            self._closed = True
        _ensure_parent_exists(self._path)
        with self._path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.HEADER)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def __enter__(self) -> "LastAttemptCsvSink":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = [
    "AttemptSink",
    "ManifestEntry",
    "MANIFEST_SCHEMA_VERSION",
    "JsonlSink",
    "CsvSink",
    "MultiSink",
    "ManifestIndexSink",
    "LastAttemptCsvSink",
]
