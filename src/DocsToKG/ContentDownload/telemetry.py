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
import io
import json
import sqlite3
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Protocol, Set, Tuple, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.core import WorkArtifact
    from DocsToKG.ContentDownload.pipeline import AttemptRecord, DownloadOutcome

from DocsToKG.ContentDownload.core import (
    PDF_LIKE,
    Classification,
    ReasonCode,
    atomic_write_text,
    normalize_url,
)

MANIFEST_SCHEMA_VERSION = 2
SQLITE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
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
        reason: Diagnostic reason code (see :class:`ReasonCode`).
        reason_detail: Optional human-readable diagnostic detail.
        html_paths: Any extracted HTML content paths stored alongside the PDF.
        sha256: Optional SHA256 digest of the stored artifact.
        content_length: Content size in bytes when reported by the server.
        etag: Server-provided entity tag when available.
        last_modified: HTTP ``Last-Modified`` header value if supplied.
        extracted_text_path: Path to extracted text artefacts when produced.
        dry_run: Flag indicating whether the artifact was processed in dry-run mode.
        run_id: Unique identifier for the downloader run that produced the entry.

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
    reason_detail: Optional[str] = None
    html_paths: List[str] = field(default_factory=list)
    sha256: Optional[str] = None
    content_length: Optional[int] = None
    etag: Optional[str] = None
    last_modified: Optional[str] = None
    extracted_text_path: Optional[str] = None
    dry_run: bool = False
    run_id: Optional[str] = None

    def __post_init__(self) -> None:
        normalized_classification = Classification.from_wire(self.classification)
        object.__setattr__(self, "classification", normalized_classification.value)

        if isinstance(self.reason, ReasonCode):
            object.__setattr__(self, "reason", self.reason.value)

        reason_detail = self.reason_detail
        if isinstance(reason_detail, ReasonCode):
            object.__setattr__(self, "reason_detail", reason_detail.value)

        length = self.content_length
        if length is not None:
            try:
                coerced = int(length)
            except (TypeError, ValueError):
                coerced = None
            else:
                if coerced < 0:
                    coerced = None
            object.__setattr__(self, "content_length", coerced)


@runtime_checkable
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

    def __enter__(self) -> "AttemptSink":  # pragma: no cover - structural typing helper
        """Enter the runtime context for the sink."""

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - structural typing helper
        """Exit the runtime context for the sink."""


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
        status_text = (
            record.status.value if isinstance(record.status, Classification) else str(record.status)
        )
        self._write(
            {
                "record_type": "attempt",
                "timestamp": ts,
                "run_id": record.run_id,
                "work_id": record.work_id,
                "resolver_name": record.resolver_name,
                "resolver_order": record.resolver_order,
                "url": record.url,
                "status": status_text,
                "http_status": record.http_status,
                "content_type": record.content_type,
                "elapsed_ms": record.elapsed_ms,
                "resolver_wall_time_ms": record.resolver_wall_time_ms,
                "reason": (
                    record.reason.value
                    if isinstance(record.reason, ReasonCode)
                    else record.reason if record.reason is not None else None
                ),
                "reason_detail": getattr(record, "reason_detail", None),
                "metadata": record.metadata,
                "sha256": record.sha256,
                "content_length": record.content_length,
                "dry_run": record.dry_run,
                "retry_after": record.retry_after,
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
                "run_id": entry.run_id,
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
                "reason_detail": entry.reason_detail,
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


class RotatingJsonlSink(JsonlSink):
    """JSONL sink that rotates the log file once it exceeds a configured size."""

    def __init__(self, path: Path, *, max_bytes: int) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive for RotatingJsonlSink")
        self._max_bytes = int(max_bytes)
        self._sequence = 0
        super().__init__(path)
        self._sequence = self._initial_sequence()

    def _initial_sequence(self) -> int:
        base_name = self._path.name
        prefix_len = len(base_name) + 1
        highest = 0
        for candidate in self._path.parent.glob(f"{base_name}.*"):
            suffix = candidate.name[prefix_len:]
            if suffix.isdigit():
                highest = max(highest, int(suffix) + 1)
        return highest

    def _rotate(self) -> None:
        self._file.flush()
        self._file.close()
        while True:
            rotated_name = f"{self._path.name}.{self._sequence:04d}"
            rotated_path = self._path.with_name(rotated_name)
            self._sequence += 1
            if rotated_path.exists():
                continue
            self._path.rename(rotated_path)
            break
        self._file = self._path.open("a", encoding="utf-8")

    def _should_rotate(self, pending_bytes: int) -> bool:
        try:
            current_size = self._file.tell()
        except (OSError, ValueError):  # pragma: no cover - defensive
            try:
                current_size = self._path.stat().st_size
            except OSError:
                current_size = 0
        return current_size + pending_bytes > self._max_bytes

    def _write(self, payload: Dict[str, Any]) -> None:
        payload.setdefault("timestamp", _utc_timestamp())
        line = json.dumps(payload, sort_keys=True) + "\n"
        encoded = line.encode("utf-8")
        with self._lock:
            try:
                if self._path.exists() and self._path.stat().st_size >= self._max_bytes:
                    self._rotate()
            except OSError:  # pragma: no cover - defensive
                pass
            if self._should_rotate(len(encoded)):
                self._rotate()
            self._file.write(line)
            self._file.flush()


class CsvSink:
    """Lightweight sink that mirrors attempt records into a CSV for spreadsheet review."""

    HEADER = [
        "timestamp",
        "run_id",
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
        "reason_detail",
        "sha256",
        "content_length",
        "retry_after",
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
            "run_id": record.run_id,
            "work_id": record.work_id,
            "resolver_name": record.resolver_name,
            "resolver_order": record.resolver_order,
            "url": record.url,
            "status": (
                record.status.value
                if isinstance(record.status, Classification)
                else str(record.status)
            ),
            "http_status": record.http_status,
            "content_type": record.content_type,
            "elapsed_ms": record.elapsed_ms,
            "resolver_wall_time_ms": record.resolver_wall_time_ms,
            "reason": (
                record.reason.value
                if isinstance(record.reason, ReasonCode)
                else record.reason if record.reason is not None else None
            ),
            "reason_detail": getattr(record, "reason_detail", None) or "",
            "sha256": record.sha256,
            "content_length": record.content_length,
            "retry_after": record.retry_after,
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
        atomic_write_text(self._path, json.dumps(ordered, indent=2))

    def __enter__(self) -> "ManifestIndexSink":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class LastAttemptCsvSink:
    """Write a CSV snapshot containing the most recent manifest entry per work."""

    HEADER = [
        "work_id",
        "run_id",
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
            "run_id": self._normalise(entry.run_id),
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
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=self.HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        atomic_write_text(self._path, buffer.getvalue())

    def __enter__(self) -> "LastAttemptCsvSink":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class SummarySink:
    """Persist the final metrics summary for a run as JSON."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._summary: Optional[Dict[str, Any]] = None
        self._lock = threading.Lock()
        self._closed = False

    def log_attempt(
        self, record: "AttemptRecord", *, timestamp: Optional[str] = None
    ) -> None:  # pragma: no cover - no-op
        """Accept attempt telemetry for interface parity without persisting it.

        Args:
            record: Resolver attempt metadata emitted by the download pipeline.
            timestamp: Optional ISO-8601 timestamp used when callers override the capture time.

        Returns:
            None
        """
        return None

    def log_manifest(self, entry: ManifestEntry) -> None:  # pragma: no cover - no-op
        """Receive manifest updates while keeping in-memory summary-only semantics.

        Args:
            entry: Manifest entry describing the document artifact that was processed.

        Returns:
            None
        """
        return None

    def log_summary(self, summary: Dict[str, Any]) -> None:
        """Capture the latest run summary prior to shutdown."""

        with self._lock:
            self._summary = dict(summary)

    def close(self) -> None:
        """Write the captured summary to disk exactly once."""

        with self._lock:
            if self._closed:
                return
            payload = dict(self._summary or {})
            self._closed = True
        atomic_write_text(self._path, json.dumps(payload, indent=2, sort_keys=True))

    def __enter__(self) -> "SummarySink":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class SqliteSink:
    """Persist attempts, manifests, and summary records to a SQLite database."""

    def __init__(self, path: Path) -> None:
        _ensure_parent_exists(path)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        try:
            current_version = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
        except Exception:
            current_version = 0
        self._initialise_schema(current_version)
        self._lock = threading.Lock()
        self._closed = False

    def log_attempt(self, record: "AttemptRecord", *, timestamp: Optional[str] = None) -> None:
        """Persist a resolver attempt event for downstream document analytics.

        Args:
            record: Structured attempt data captured during document resolution.
            timestamp: Optional ISO-8601 timestamp overriding the event capture time.

        Returns:
            None
        """
        ts = timestamp or _utc_timestamp()
        metadata_json = json.dumps(record.metadata, sort_keys=True) if record.metadata else None
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO attempts (
                    timestamp,
                    run_id,
                    work_id,
                    resolver_name,
                    resolver_order,
                    url,
                    status,
                    http_status,
                    content_type,
                    elapsed_ms,
                    resolver_wall_time_ms,
                    reason,
                    reason_detail,
                    metadata,
                    sha256,
                    content_length,
                    dry_run,
                    retry_after
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    record.run_id,
                    record.work_id,
                    record.resolver_name,
                    record.resolver_order,
                    record.url,
                    (
                        record.status.value
                        if isinstance(record.status, Classification)
                        else str(record.status)
                    ),
                    record.http_status,
                    record.content_type,
                    record.elapsed_ms,
                    record.resolver_wall_time_ms,
                    (
                        record.reason.value
                        if isinstance(record.reason, ReasonCode)
                        else record.reason if record.reason is not None else None
                    ),
                    getattr(record, "reason_detail", None),
                    metadata_json,
                    record.sha256,
                    record.content_length,
                    1 if record.dry_run else 0,
                    record.retry_after,
                ),
            )
            self._conn.commit()

    def log_manifest(self, entry: ManifestEntry) -> None:
        """Record manifest outcomes for the processed document artifact.

        Args:
            entry: Manifest entry describing the document state after processing.

        Returns:
            None
        """
        html_paths_json = json.dumps(entry.html_paths, sort_keys=True) if entry.html_paths else None
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO manifests (
                    timestamp,
                    run_id,
                    schema_version,
                    work_id,
                    title,
                    publication_year,
                    resolver,
                    url,
                    path,
                    classification,
                    content_type,
                    reason,
                    reason_detail,
                    html_paths,
                    sha256,
                    content_length,
                    etag,
                    last_modified,
                    extracted_text_path,
                    dry_run
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.timestamp,
                    entry.run_id,
                    entry.schema_version,
                    entry.work_id,
                    entry.title,
                    entry.publication_year,
                    entry.resolver,
                    entry.url,
                    entry.path,
                    entry.classification,
                    entry.content_type,
                    entry.reason,
                    entry.reason_detail,
                    html_paths_json,
                    entry.sha256,
                    entry.content_length,
                    entry.etag,
                    entry.last_modified,
                    entry.extracted_text_path,
                    1 if entry.dry_run else 0,
                ),
            )
            self._conn.commit()

    def log_summary(self, summary: Dict[str, Any]) -> None:
        """Store the aggregated run summary in the summaries table.

        Args:
            summary: Aggregated counters and timing metadata for the run.

        Returns:
            None
        """
        run_id = summary.get("run_id")
        if not run_id:
            raise ValueError("summary payload must include run_id")
        payload = json.dumps(summary, indent=2, sort_keys=True)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO summaries (run_id, timestamp, payload)
                VALUES (?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET timestamp=excluded.timestamp, payload=excluded.payload
                """,
                (run_id, _utc_timestamp(), payload),
            )
            self._conn.commit()

    def close(self) -> None:
        """Commit outstanding changes and dispose of the SQLite connection."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._conn.commit()
            self._conn.close()

    def __enter__(self) -> "SqliteSink":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _initialise_schema(self, current_version: int) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                run_id TEXT,
                work_id TEXT,
                resolver_name TEXT,
                resolver_order INTEGER,
                url TEXT,
                status TEXT,
                http_status INTEGER,
                content_type TEXT,
                elapsed_ms REAL,
                resolver_wall_time_ms REAL,
                reason TEXT,
                reason_detail TEXT,
                metadata TEXT,
                sha256 TEXT,
                content_length INTEGER,
                dry_run INTEGER,
                retry_after REAL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS manifests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                run_id TEXT,
                schema_version INTEGER,
                work_id TEXT,
                title TEXT,
                publication_year INTEGER,
                resolver TEXT,
                url TEXT,
                path TEXT,
                classification TEXT,
                content_type TEXT,
                reason TEXT,
                reason_detail TEXT,
                html_paths TEXT,
                sha256 TEXT,
                content_length INTEGER,
                etag TEXT,
                last_modified TEXT,
                extracted_text_path TEXT,
                dry_run INTEGER
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS summaries (
                run_id TEXT PRIMARY KEY,
                timestamp TEXT,
                payload TEXT
            )
            """
        )
        if current_version < SQLITE_SCHEMA_VERSION:
            self._run_migrations(current_version)
        self._conn.execute(f"PRAGMA user_version={SQLITE_SCHEMA_VERSION}")
        self._conn.commit()

    def _run_migrations(self, current_version: int) -> None:
        if current_version < 1:
            current_version = 1
        if current_version < 2:
            self._safe_add_column("attempts", "run_id", "TEXT")
            self._safe_add_column("manifests", "run_id", "TEXT")
            self._safe_add_column("attempts", "retry_after", "REAL")
            self._safe_add_column("attempts", "reason_detail", "TEXT")
            self._safe_add_column("manifests", "reason_detail", "TEXT")
            self._migrate_summary_table()

    def _safe_add_column(self, table: str, column: str, declaration: str) -> None:
        try:
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")
        except sqlite3.OperationalError:
            pass

    def _migrate_summary_table(self) -> None:
        try:
            rows = list(self._conn.execute("SELECT timestamp, payload FROM summaries"))
        except sqlite3.OperationalError:
            rows = []
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS summaries_new (
                run_id TEXT PRIMARY KEY,
                timestamp TEXT,
                payload TEXT
            )
            """
        )
        if rows:
            timestamp, payload = rows[-1]
            self._conn.execute(
                "INSERT OR REPLACE INTO summaries_new (run_id, timestamp, payload) VALUES (?, ?, ?)",
                ("legacy", timestamp, payload),
            )
        self._conn.execute("DROP TABLE IF EXISTS summaries")
        self._conn.execute("ALTER TABLE summaries_new RENAME TO summaries")


def load_previous_manifest(path: Optional[Path]) -> Tuple[Dict[str, Dict[str, Any]], Set[str]]:
    """Load JSONL manifest entries indexed by work ID and normalised URL."""

    per_work: Dict[str, Dict[str, Any]] = {}
    completed: Set[str] = set()
    if not path:
        return per_work, completed

    prefix = f"{path.name}."
    rotated: List[Tuple[int, Path]] = []
    for candidate in path.parent.glob(f"{path.name}.*"):
        if not candidate.is_file():
            continue
        suffix = candidate.name[len(prefix) :]
        if suffix.isdigit():
            rotated.append((int(suffix), candidate))
    rotated.sort()
    ordered_files = [candidate for _, candidate in rotated]
    if path.exists():
        ordered_files.append(path)

    if not ordered_files:
        return per_work, completed

    for file_path in ordered_files:
        with file_path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                data = json.loads(line)
                record_type = data.get("record_type")
                if record_type is None:
                    raise ValueError(
                        "Legacy manifest entries without record_type are no longer supported."
                    )
                if record_type != "manifest":
                    continue

            schema_version_raw = data.get("schema_version")
            if schema_version_raw is None:
                raise ValueError("Manifest entries must include a schema_version field.")
            try:
                schema_version = int(schema_version_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Manifest entry schema_version must be an integer, got {schema_version_raw!r}"
                ) from exc
            if schema_version != MANIFEST_SCHEMA_VERSION:
                qualifier = (
                    "newer"
                    if schema_version > MANIFEST_SCHEMA_VERSION
                    else "older" if schema_version < MANIFEST_SCHEMA_VERSION else "unknown"
                )
                raise ValueError(
                    "Unsupported manifest schema_version {observed} ({qualifier}); expected version {expected}. "
                    "Regenerate the manifest using a compatible DocsToKG downloader release.".format(
                        observed=schema_version,
                        qualifier=qualifier,
                        expected=MANIFEST_SCHEMA_VERSION,
                    )
                )
            data["schema_version"] = schema_version

            work_id = data.get("work_id")
            url = data.get("url")
            if not work_id or not url:
                raise ValueError("Manifest entries must include work_id and url fields.")

            content_length = data.get("content_length")
            if isinstance(content_length, str):
                try:
                    data["content_length"] = int(content_length)
                except ValueError:
                    data["content_length"] = None

            key = normalize_url(url)
            per_work.setdefault(work_id, {})[key] = data

            raw_classification = data.get("classification")
            classification_text = (raw_classification or "").strip()
            if not classification_text:
                raise ValueError("Manifest entries must declare a classification.")

            classification_code = Classification.from_wire(classification_text)
            data["classification"] = classification_code.value
            if classification_code in PDF_LIKE:
                completed.add(work_id)

            raw_reason = data.get("reason")
            if raw_reason is not None:
                data["reason"] = ReasonCode.from_wire(raw_reason).value
            if data.get("reason_detail") is not None:
                detail = data["reason_detail"]
                if detail == "":
                    data["reason_detail"] = None

    return per_work, completed


def load_manifest_url_index(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    """Return a mapping of normalised URLs to manifest metadata from SQLite."""

    if not path or not path.exists():
        return {}
    conn = sqlite3.connect(path)
    try:
        try:
            cursor = conn.execute(
                "SELECT url, path, sha256, classification, etag, last_modified, content_length "
                "FROM manifests ORDER BY timestamp"
            )
        except sqlite3.OperationalError:
            return {}
        mapping: Dict[str, Dict[str, Any]] = {}
        for url, stored_path, sha256, classification, etag, last_modified, content_length in cursor:
            if not url:
                continue
            normalised = normalize_url(url)
            mapping[normalised] = {
                "url": url,
                "path": stored_path,
                "sha256": sha256,
                "classification": classification,
                "etag": etag,
                "last_modified": last_modified,
                "content_length": content_length,
            }
        return mapping
    finally:
        conn.close()


def build_manifest_entry(
    artifact: "WorkArtifact",
    resolver: Optional[str],
    url: Optional[str],
    outcome: Optional["DownloadOutcome"],
    html_paths: List[str],
    *,
    dry_run: bool,
    run_id: Optional[str] = None,
    reason: Optional[ReasonCode | str] = None,
    reason_detail: Optional[str] = None,
) -> ManifestEntry:
    """Create a manifest entry summarising a download attempt."""

    timestamp = _utc_timestamp()
    outcome_reason: Optional[ReasonCode | str] = None
    outcome_detail: Optional[str] = None
    if outcome:
        classification = outcome.classification.value
        path = outcome.path
        content_type = outcome.content_type
        outcome_reason = getattr(outcome, "reason", None)
        outcome_detail = getattr(outcome, "reason_detail", None)
        sha256 = outcome.sha256
        content_length = outcome.content_length
        etag = outcome.etag
        last_modified = outcome.last_modified
        extracted_text_path = outcome.extracted_text_path
    else:
        classification = Classification.MISS.value
        path = None
        content_type = None
        sha256 = None
        content_length = None
        etag = None
        last_modified = None
        extracted_text_path = None

    resolved_reason_code: Optional[ReasonCode]
    if isinstance(reason, ReasonCode):
        resolved_reason_code = reason
    elif reason is not None:
        resolved_reason_code = ReasonCode.from_wire(reason)
    elif outcome:
        resolved_reason_code = (
            outcome_reason
            if isinstance(outcome_reason, ReasonCode)
            else ReasonCode.from_wire(outcome_reason)
        )
    else:
        resolved_reason_code = None

    resolved_detail = reason_detail
    if resolved_detail is None:
        resolved_detail = outcome_detail if outcome_detail else None

    return ManifestEntry(
        schema_version=MANIFEST_SCHEMA_VERSION,
        timestamp=timestamp,
        run_id=run_id,
        work_id=getattr(artifact, "work_id"),
        title=getattr(artifact, "title"),
        publication_year=getattr(artifact, "publication_year"),
        resolver=resolver,
        url=url,
        path=path,
        classification=classification,
        content_type=content_type,
        reason=resolved_reason_code.value if resolved_reason_code else None,
        reason_detail=resolved_detail,
        html_paths=list(html_paths),
        sha256=sha256,
        content_length=content_length,
        etag=etag,
        last_modified=last_modified,
        extracted_text_path=extracted_text_path,
        dry_run=dry_run,
    )


__all__ = [
    "AttemptSink",
    "ManifestEntry",
    "MANIFEST_SCHEMA_VERSION",
    "JsonlSink",
    "RotatingJsonlSink",
    "CsvSink",
    "MultiSink",
    "ManifestIndexSink",
    "LastAttemptCsvSink",
    "SummarySink",
    "SqliteSink",
    "build_manifest_entry",
    "load_previous_manifest",
    "load_manifest_url_index",
]
