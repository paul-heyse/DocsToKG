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

import contextlib
import csv
import io
import json
import logging
import os
import shutil
import sqlite3
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Set,
    Tuple,
    runtime_checkable,
)

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.core import WorkArtifact
    from DocsToKG.ContentDownload.pipeline import AttemptRecord, DownloadOutcome, PipelineResult

from DocsToKG.ContentDownload.core import (
    PDF_LIKE,
    Classification,
    ReasonCode,
    atomic_write_text,
    normalize_classification,
    normalize_reason,
    normalize_url,
)

MANIFEST_SCHEMA_VERSION = 3
SQLITE_SCHEMA_VERSION = 4
CSV_HEADER_TOKENS = {"run_id", "work_id"}


logger = logging.getLogger(__name__)


def normalize_manifest_path(
    path: Optional[str | Path],
    *,
    base: Optional[Path] = None,
) -> Optional[str]:
    """Return an absolute filesystem path suitable for manifest storage.

    Older manifests stored artifact paths relative to the working directory in
    effect when the run executed. To keep those entries usable when resuming a
    run from another working directory, callers may provide ``base`` so that
    relative paths can be resolved against a known directory such as the
    manifest location. For new entries, the helper upgrades any provided path
    (``Path`` objects or strings) to an absolute representation.
    """

    if path is None:
        return None

    if isinstance(path, str):
        candidate_text = path.strip()
        if not candidate_text:
            return None
        candidate = Path(candidate_text)
    else:
        candidate = Path(path)

    if base is not None and not candidate.is_absolute():
        candidate = Path(base) / candidate

    try:
        resolved = candidate.resolve(strict=False)
    except OSError:
        resolved = candidate if candidate.is_absolute() else candidate.absolute()

    return str(resolved)


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
    path_mtime_ns: Optional[int] = None
    run_id: Optional[str] = None

    def __post_init__(self) -> None:
        classification_token = normalize_classification(self.classification)
        if isinstance(classification_token, Classification):
            classification_value = classification_token.value
        else:
            classification_value = Classification.from_wire(classification_token).value
        object.__setattr__(self, "classification", classification_value)

        reason_token = normalize_reason(self.reason)
        if isinstance(reason_token, ReasonCode):
            reason_value = reason_token.value
        else:
            reason_value = reason_token
        object.__setattr__(self, "reason", reason_value)

        detail_token = normalize_reason(self.reason_detail)
        if isinstance(detail_token, ReasonCode):
            detail_value = detail_token.value
        else:
            detail_value = detail_token
        object.__setattr__(self, "reason_detail", detail_value)

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

        mtime_ns = self.path_mtime_ns
        if mtime_ns is not None:
            try:
                coerced_mtime = int(mtime_ns)
            except (TypeError, ValueError):
                coerced_mtime = None
            else:
                if coerced_mtime < 0:
                    coerced_mtime = None
            object.__setattr__(self, "path_mtime_ns", coerced_mtime)


class ManifestUrlIndex:
    """Lazy lookup helper for manifest metadata stored in SQLite."""

    def __init__(self, sqlite_path: Optional[Path], *, eager: bool = False) -> None:
        self._path = sqlite_path
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._loaded_all = False
        if eager:
            self._ensure_loaded()

    def get(self, url: str, default: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Return manifest metadata for ``url`` if present in the SQLite cache.

        Args:
            url: URL whose manifest metadata should be retrieved.
            default: Fallback value returned when the URL has no cached record.

        Returns:
            Manifest metadata dictionary when found; otherwise ``default``.
        """
        normalized = normalize_url(url)
        if normalized in self._cache:
            return self._cache[normalized]
        if self._loaded_all:
            return default
        record = self._fetch_one(normalized)
        if record is None:
            return default
        self._cache[normalized] = record
        return record

    def __contains__(self, url: str) -> bool:  # pragma: no cover - trivial wrapper
        normalized = normalize_url(url)
        if normalized in self._cache:
            return True
        if self._loaded_all or not self._path or not self._path.exists():
            return False
        record = self._fetch_one(normalized)
        if record is None:
            return False
        self._cache[normalized] = record
        return True

    def items(self) -> Iterable[Tuple[str, Dict[str, Any]]]:
        """Iterate over cached manifest entries keyed by normalized URL.

        Returns:
            Iterable of ``(normalized_url, metadata)`` pairs from the cache.
        """
        self._ensure_loaded()
        return self._cache.items()

    def iter_existing_paths(self) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """Stream manifest entries whose artifact paths still exist on disk."""

        if self._loaded_all or not self._path or not self._path.exists():
            for normalized, meta in self._cache.items():
                path_value = meta.get("path")
                if path_value and Path(path_value).exists():
                    yield normalized, meta
            return

        conn = sqlite3.connect(self._path)
        try:
            try:
                cursor = conn.execute(
                    "SELECT url, normalized_url, path, sha256, classification, etag, last_modified, content_length, path_mtime_ns "
                    "FROM manifests ORDER BY timestamp DESC"
                )
            except sqlite3.OperationalError:
                return

            seen: Set[str] = set()
            for (
                url,
                normalized_url,
                stored_path,
                sha256,
                classification,
                etag,
                last_modified,
                content_length,
                path_mtime_ns,
            ) in cursor:
                if not url:
                    continue
                normalized = normalized_url or normalize_url(url)
                if normalized in seen:
                    continue
                if not stored_path:
                    continue
                # Resolve path relative to manifest directory, consistent with _fetch_one
                base_dir = self._path.parent if self._path else None
                resolved_path = normalize_manifest_path(stored_path, base=base_dir)
                if not resolved_path or not Path(resolved_path).exists():
                    continue

                payload = {
                    "url": url,
                    "path": resolved_path,
                    "sha256": sha256,
                    "classification": classification,
                    "etag": etag,
                    "last_modified": last_modified,
                    "content_length": content_length,
                    "mtime_ns": path_mtime_ns,
                }
                self._cache.setdefault(normalized, payload)
                seen.add(normalized)
                yield normalized, payload
        finally:
            conn.close()

    def iter_existing(self) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """Yield manifest entries whose artifact paths still exist on disk."""

        yield from self.iter_existing_paths()

    def as_dict(self) -> Dict[str, Dict[str, Any]]:
        """Return a defensive copy of the manifest cache.

        Returns:
            Dictionary of normalized URLs to manifest metadata.
        """
        self._ensure_loaded()
        return dict(self._cache)

    def _ensure_loaded(self) -> None:
        if self._loaded_all:
            return
        if not self._path or not self._path.exists():
            self._loaded_all = True
            return
        dataset = load_manifest_url_index(self._path)
        self._cache.update(dataset)
        self._loaded_all = True

    def _fetch_one(self, normalized: str) -> Optional[Dict[str, Any]]:
        if not self._path or not self._path.exists():
            return None
        conn = sqlite3.connect(self._path)
        try:
            try:
                cursor = conn.execute(
                    (
                        "SELECT url, path, sha256, classification, etag, last_modified, content_length, path_mtime_ns "
                        "FROM manifests WHERE normalized_url = ? ORDER BY timestamp DESC LIMIT 1"
                    ),
                    (normalized,),
                )
            except sqlite3.OperationalError:
                return None
            row = cursor.fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        (
            url,
            stored_path,
            sha256,
            classification,
            etag,
            last_modified,
            content_length,
            path_mtime_ns,
        ) = row
        base_dir = self._path.parent if self._path else None
        return {
            "url": url,
            "path": normalize_manifest_path(stored_path, base=base_dir),
            "sha256": sha256,
            "classification": classification,
            "etag": etag,
            "last_modified": last_modified,
            "content_length": content_length,
            "mtime_ns": path_mtime_ns,
        }


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


class RunTelemetry(AttemptSink):
    """Telemetry coordinator that centralises manifest aggregation and delegation."""

    def __init__(self, sink: AttemptSink) -> None:
        self._sink = sink

    def log_attempt(self, record: "AttemptRecord", *, timestamp: Optional[str] = None) -> None:
        """Proxy attempt logging to the underlying sink.

        Args:
            record: Resolver attempt payload to record.
            timestamp: Optional ISO-8601 timestamp overriding ``datetime.utcnow``.
        """
        self._sink.log_attempt(record, timestamp=timestamp)

    def log_manifest(self, entry: ManifestEntry) -> None:
        """Forward manifest entries to the configured sink."""
        self._sink.log_manifest(entry)

    def log_summary(self, summary: Dict[str, Any]) -> None:
        """Publish the final run summary to downstream sinks."""
        self._sink.log_summary(summary)

    def __enter__(self) -> "RunTelemetry":
        """Enter the context manager, delegating to the underlying sink if present."""

        enter = getattr(self._sink, "__enter__", None)
        if enter is not None:
            enter()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> Optional[bool]:
        """Exit the context manager, closing the sink when appropriate."""

        exit_method = getattr(self._sink, "__exit__", None)
        if exit_method is not None:
            return exit_method(exc_type, exc, tb)
        self.close()
        return None

    def close(self) -> None:
        """Release sink resources and flush buffered telemetry."""
        self._sink.close()

    def record_manifest(
        self,
        artifact: "WorkArtifact",
        *,
        resolver: Optional[str],
        url: Optional[str],
        outcome: Optional["DownloadOutcome"],
        html_paths: Iterable[str],
        dry_run: bool,
        run_id: Optional[str],
        reason: Optional[ReasonCode | str] = None,
        reason_detail: Optional[str] = None,
    ) -> ManifestEntry:
        """Construct and emit a manifest entry for ``artifact``.

        Args:
            artifact: Work artifact being recorded.
            resolver: Name of the resolver that produced the outcome.
            url: Final URL that yielded the content.
            outcome: Download outcome describing classification and metadata.
            html_paths: HTML artefacts captured alongside the document.
            dry_run: Whether the pipeline ran in dry-run mode.
            run_id: Unique identifier for this pipeline execution.
            reason: Optional diagnostic reason code.
            reason_detail: Optional human-readable reason detail.

        Returns:
            ManifestEntry: Structured manifest entry persisted via the sink.
        """
        entry = build_manifest_entry(
            artifact,
            resolver=resolver,
            url=url,
            outcome=outcome,
            html_paths=list(html_paths),
            dry_run=dry_run,
            run_id=run_id,
            reason=reason,
            reason_detail=reason_detail,
        )
        self.log_manifest(entry)
        return entry

    def record_pipeline_result(
        self,
        artifact: "WorkArtifact",
        result: "PipelineResult",
        *,
        dry_run: bool,
        run_id: Optional[str],
    ) -> ManifestEntry:
        """Record pipeline output, normalising reason metadata on the way out.

        Args:
            artifact: Work artifact that was processed.
            result: Pipeline result encapsulating resolver outcome details.
            dry_run: Whether side effects were suppressed.
            run_id: Unique identifier for the current pipeline execution.

        Returns:
            ManifestEntry: Manifest record produced for downstream sinks.
        """
        outcome = result.outcome
        reason_token = normalize_reason(result.reason) if result.reason else None
        if reason_token is None and outcome is not None:
            reason_token = normalize_reason(getattr(outcome, "reason", None))

        detail_token = normalize_reason(result.reason_detail) if result.reason_detail else None
        if detail_token is None and outcome is not None:
            detail_token = normalize_reason(getattr(outcome, "reason_detail", None))

        return self.record_manifest(
            artifact,
            resolver=result.resolver_name,
            url=result.url,
            outcome=outcome,
            html_paths=list(result.html_paths),
            dry_run=dry_run,
            run_id=run_id,
            reason=reason_token,
            reason_detail=detail_token,
        )


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
                    else record.reason
                    if record.reason is not None
                    else None
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
                else record.reason
                if record.reason is not None
                else None
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
        self._path = path
        alias_candidate = path.with_suffix(".sqlite")
        self._legacy_alias_path = alias_candidate if alias_candidate != path else None
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
                        else record.reason
                        if record.reason is not None
                        else None
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
                    normalized_url,
                    path,
                    path_mtime_ns,
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
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    normalize_url(entry.url) if entry.url else None,
                    entry.path,
                    entry.path_mtime_ns,
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
        self._ensure_legacy_alias()

    def __enter__(self) -> "SqliteSink":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _ensure_legacy_alias(self) -> None:
        """Create a compatibility alias at ``.sqlite`` when using ``.sqlite3`` files."""

        alias = getattr(self, "_legacy_alias_path", None)
        if not alias:
            return
        try:
            target = self._path
            if not target.exists():
                return
            if alias.exists() or alias.is_symlink():
                try:
                    if alias.resolve() == target.resolve():
                        return
                except OSError:
                    pass
                try:
                    alias.unlink()
                except OSError:
                    logger.debug(
                        "Failed to remove stale legacy SQLite alias at %s", alias, exc_info=True
                    )
                    return
            try:
                os.symlink(target, alias)
            except (AttributeError, NotImplementedError, OSError):
                try:
                    shutil.copy2(target, alias)
                except OSError:
                    logger.debug(
                        "Failed to copy legacy SQLite alias from %s to %s",
                        target,
                        alias,
                        exc_info=True,
                    )
        except Exception:
            logger.debug("Unable to create legacy SQLite alias for %s", alias, exc_info=True)

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
                normalized_url TEXT,
                path TEXT,
                path_mtime_ns INTEGER,
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
        if current_version < 3:
            self._safe_add_column("manifests", "path_mtime_ns", "INTEGER")
            self._safe_add_column("manifests", "reason_detail", "TEXT")
            self._migrate_summary_table()
        if current_version < 4:
            self._safe_add_column("manifests", "normalized_url", "TEXT")
            self._populate_normalized_urls()

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

    def _populate_normalized_urls(self) -> None:
        try:
            rows = list(
                self._conn.execute("SELECT id, url FROM manifests WHERE normalized_url IS NULL")
            )
        except sqlite3.OperationalError:
            return
        for row_id, url in rows:
            if not url:
                continue
            normalized = normalize_url(url)
            self._conn.execute(
                "UPDATE manifests SET normalized_url = ? WHERE id = ?",
                (normalized, row_id),
            )
        self._conn.commit()


def _manifest_entry_from_sqlite_row(
    run_id: Any,
    work_id: Any,
    url: Any,
    normalized_url: Any,
    schema_version: Any,
    classification: Any,
    reason: Any,
    reason_detail: Any,
    path_value: Any,
    path_mtime_ns: Any,
    sha256: Any,
    content_length: Any,
    etag: Any,
    last_modified: Any,
) -> Optional[Tuple[str, str, Dict[str, Any], bool]]:
    """Convert a SQLite manifest row into resume metadata."""

    if not work_id or not url:
        return None

    normalized = normalized_url or normalize_url(str(url))
    try:
        schema_version_int = int(schema_version)
    except (TypeError, ValueError):
        schema_version_int = MANIFEST_SCHEMA_VERSION

    classification_enum: Optional[Classification]
    classification_value: Optional[str] = None
    completed = False
    try:
        classification_enum = Classification.from_wire(classification)
    except ValueError:
        classification_enum = None
    else:
        classification_value = classification_enum.value
        if classification_enum in PDF_LIKE:
            completed = True

    reason_value: Optional[str]
    try:
        reason_enum = ReasonCode.from_wire(reason) if reason is not None else None
    except ValueError:
        reason_enum = None
    reason_value = reason_enum.value if reason_enum is not None else reason

    try:
        content_length_value = int(content_length) if content_length is not None else None
    except (TypeError, ValueError):
        content_length_value = None

    path_mtime_value: Optional[int] = None
    if path_mtime_ns is not None:
        try:
            path_mtime_value = int(path_mtime_ns)
        except (TypeError, ValueError):
            path_mtime_value = None

    entry = {
        "record_type": "manifest",
        "schema_version": schema_version_int,
        "run_id": run_id,
        "work_id": work_id,
        "url": url,
        "normalized_url": normalized,
        "classification": classification_value or str(classification or ""),
        "reason": reason_value,
        "reason_detail": reason_detail,
        "path": path_value,
        "path_mtime_ns": path_mtime_value,
        "mtime_ns": path_mtime_value,
        "sha256": sha256,
        "content_length": content_length_value,
        "etag": etag,
        "last_modified": last_modified,
    }

    return str(work_id), normalized, entry, completed


def _iter_resume_rows_from_sqlite(
    sqlite_path: Path,
) -> Iterator[Tuple[str, str, Dict[str, Any], bool]]:
    """Yield manifest resume rows from ``sqlite_path`` lazily."""

    if not sqlite_path.exists():
        return

    try:
        conn = sqlite3.connect(sqlite_path)
    except sqlite3.Error:  # pragma: no cover - defensive guard
        return

    try:
        try:
            cursor = conn.execute(
                (
                    "SELECT run_id, work_id, url, normalized_url, schema_version, "
                    "classification, reason, reason_detail, path, path_mtime_ns, sha256, "
                    "content_length, etag, last_modified "
                    "FROM manifests ORDER BY work_id, normalized_url"
                )
            )
        except sqlite3.OperationalError:
            return

        for row in cursor:
            parsed = _manifest_entry_from_sqlite_row(*row)
            if parsed is None:
                continue
            yield parsed
    finally:
        conn.close()


def _load_resume_from_sqlite(sqlite_path: Path) -> Tuple[Dict[str, Dict[str, Any]], Set[str]]:
    """Return resume metadata reconstructed from the SQLite manifest cache."""

    lookup: Dict[str, Dict[str, Any]] = {}
    completed: Set[str] = set()
    for work_id, normalized, entry, is_pdf_like in _iter_resume_rows_from_sqlite(sqlite_path):
        lookup.setdefault(work_id, {})[normalized] = entry
        if is_pdf_like:
            completed.add(work_id)

    return lookup, completed


def looks_like_csv_resume_target(path: Path) -> bool:
    """Return True when ``path`` likely references a CSV attempts log."""

    lower_name = path.name.lower()
    if lower_name.endswith(".csv") or ".csv." in lower_name:
        return True

    try:
        with path.open("r", encoding="utf-8") as handle:
            for _ in range(5):
                sample = handle.readline()
                if not sample:
                    break
                stripped = sample.strip()
                if not stripped:
                    continue
                if stripped.startswith("{") or stripped.startswith("["):
                    return False
                if "," in stripped:
                    header = {column.strip().strip('"').lower() for column in stripped.split(",")}
                    if CSV_HEADER_TOKENS.issubset(header):
                        return True
                break
    except OSError:
        return lower_name.endswith(".csv") or ".csv." in lower_name

    return False


def looks_like_sqlite_resume_target(path: Path) -> bool:
    """Return True when ``path`` likely references a SQLite manifest cache."""

    lower_name = path.name.lower()
    if lower_name.endswith(".sqlite") or lower_name.endswith(".sqlite3"):
        return True

    try:
        with path.open("rb") as handle:
            header = handle.read(16)
    except OSError:
        return False

    return header.startswith(b"SQLite format 3\x00")


def iter_previous_manifest_entries(
    path: Optional[Path],
    *,
    sqlite_path: Optional[Path] = None,
    allow_sqlite_fallback: bool = False,
    buffer_entries: bool = True,
) -> Iterator[Tuple[str, str, Dict[str, Any], bool]]:
    """Yield resume manifest entries lazily, preferring SQLite when available."""

    if (
        sqlite_path
        and sqlite_path.exists()
        and (not path or not path.exists() or looks_like_sqlite_resume_target(path))
    ):
        yield from _iter_resume_rows_from_sqlite(sqlite_path)
        return

    if not path:
        if allow_sqlite_fallback and sqlite_path and sqlite_path.exists():
            yield from _iter_resume_rows_from_sqlite(sqlite_path)
        return

    if looks_like_sqlite_resume_target(path):
        if not path.exists() or not path.is_file():
            raise ValueError(
                "Resume SQLite cache '{path}' does not exist or is not a file. Provide the matching "
                "manifest.sqlite or manifest.sqlite3 file, or omit --resume-from to start a new run.".format(
                    path=path
                )
            )
        yield from _iter_resume_rows_from_sqlite(path)
        return

    if looks_like_csv_resume_target(path):
        if allow_sqlite_fallback and sqlite_path and sqlite_path.exists():
            yield from _iter_resume_rows_from_sqlite(sqlite_path)
            return
        raise ValueError(
            "Resume manifest '{path}' appears to be a CSV attempts log but no SQLite cache was found. "
            "Provide the matching manifest.sqlite or manifest.sqlite3 file, or resume from a JSONL manifest.".format(
                path=path
            )
        )

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
        if allow_sqlite_fallback and sqlite_path and sqlite_path.exists():
            yield from _iter_resume_rows_from_sqlite(sqlite_path)
            return
        file_error = FileNotFoundError(f"No manifest files found for resume path {path!s}")
        raise ValueError(
            "Resume manifest '{path}' does not exist and no rotated segments were found. "
            "Double-check the --resume-from path or omit it to start a fresh run.".format(path=path)
        ) from file_error

    yielded_any = False

    for file_path in ordered_files:
        line_number: Optional[int] = None
        buffered: Optional[List[Tuple[str, str, Dict[str, Any], bool]]] = [] if buffer_entries else None
        try:
            with file_path.open("r", encoding="utf-8") as handle:
                for line_number, raw in enumerate(handle, start=1):
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Failed to decode JSON: {exc}") from exc
                    if not isinstance(data, dict):
                        raise TypeError(
                            f"Manifest entries must be JSON objects, got {type(data).__name__}"
                        )
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
                            else "older"
                            if schema_version < MANIFEST_SCHEMA_VERSION
                            else "unknown"
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
                    data["normalized_url"] = normalize_url(url)

                    content_length = data.get("content_length")
                    if isinstance(content_length, str):
                        try:
                            data["content_length"] = int(content_length)
                        except ValueError:
                            data["content_length"] = None

                    path_mtime_ns = data.get("path_mtime_ns")
                    if isinstance(path_mtime_ns, str):
                        try:
                            data["path_mtime_ns"] = int(path_mtime_ns)
                        except ValueError:
                            data["path_mtime_ns"] = None
                    data["mtime_ns"] = data.get("path_mtime_ns")

                    raw_classification = data.get("classification")
                    classification_text = (raw_classification or "").strip()
                    if not classification_text:
                        raise ValueError("Manifest entries must declare a classification.")

                    classification_code = Classification.from_wire(classification_text)
                    data["classification"] = classification_code.value

                    completed = classification_code in PDF_LIKE

                    raw_reason = data.get("reason")
                    if raw_reason is not None:
                        data["reason"] = ReasonCode.from_wire(raw_reason).value
                    if data.get("reason_detail") is not None:
                        detail = data["reason_detail"]
                        if detail == "":
                            data["reason_detail"] = None

                    normalized = data["normalized_url"]
                    record = (str(work_id), normalized, data, completed)
                    if buffered is not None:
                        buffered.append(record)
                    else:
                        yielded_any = True
                        yield record
        except Exception as exc:
            location = f"{file_path}:{line_number}" if line_number is not None else str(file_path)
            if allow_sqlite_fallback and sqlite_path and sqlite_path.exists():
                logger.warning("Failed to parse resume manifest at %s", location, exc_info=exc)
                logger.warning(
                    "Falling back to SQLite resume cache '%s' after manifest parse failure at %s.",
                    sqlite_path,
                    location,
                )
                yield from _iter_resume_rows_from_sqlite(sqlite_path)
                return
            raise ValueError(f"Failed to parse resume manifest at {location}: {exc}") from exc

        if buffered is not None:
            for item in buffered:
                yielded_any = True
                yield item

    if not yielded_any and allow_sqlite_fallback and sqlite_path and sqlite_path.exists():
        logger.warning(
            "Resume manifest %s contained no manifest entries; falling back to SQLite %s.",
            path,
            sqlite_path,
        )
        yield from _iter_resume_rows_from_sqlite(sqlite_path)


def load_previous_manifest(
    path: Optional[Path],
    *,
    sqlite_path: Optional[Path] = None,
    allow_sqlite_fallback: bool = False,
) -> Tuple[Dict[str, Dict[str, Any]], Set[str]]:
    """Load manifest entries indexed by work ID and normalised URL."""

    per_work: Dict[str, Dict[str, Any]] = {}
    completed: Set[str] = set()

    for work_id, normalized, entry, is_pdf_like in iter_previous_manifest_entries(
        path,
        sqlite_path=sqlite_path,
        allow_sqlite_fallback=allow_sqlite_fallback,
    ):
        per_work.setdefault(work_id, {})[normalized] = entry
        if is_pdf_like:
            completed.add(work_id)

    return per_work, completed


class JsonlResumeLookup(Mapping[str, Dict[str, Any]]):
    """Lazy resume mapping backed by a temporary SQLite index built from JSONL."""

    def __init__(self, path: Path) -> None:
        if not path:
            raise ValueError("Resume manifest path is required for JsonlResumeLookup")
        self._path = path
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._missing: Set[str] = set()
        self._lock = threading.Lock()
        self._closed = False
        self._tempdir = Path(tempfile.mkdtemp(prefix="docs_resume_jsonl_"))
        self._db_path = self._tempdir / "resume.sqlite3"
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._initialise_schema()
        self._completed_ids: FrozenSet[str] = frozenset()
        self._preload_on_close = False
        try:
            self._populate_from_jsonl()
        except Exception:
            with contextlib.suppress(Exception):
                self.close()
            raise

    def _initialise_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS entries (
                    work_id TEXT NOT NULL,
                    normalized_url TEXT NOT NULL,
                    entry_json TEXT NOT NULL,
                    PRIMARY KEY (work_id, normalized_url)
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS entries_work_id_idx ON entries (work_id)"
            )

    def _populate_from_jsonl(self) -> None:
        completed: Set[str] = set()
        cursor = self._conn.cursor()
        for work_id, normalized, entry, is_pdf_like in iter_previous_manifest_entries(
            self._path,
            allow_sqlite_fallback=False,
            buffer_entries=False,
        ):
            payload = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
            cursor.execute(
                "INSERT OR REPLACE INTO entries (work_id, normalized_url, entry_json) VALUES (?, ?, ?)",
                (work_id, normalized, payload),
            )
            if is_pdf_like:
                completed.add(work_id)
        self._conn.commit()
        self._completed_ids = frozenset(completed)

    @property
    def completed_work_ids(self) -> FrozenSet[str]:
        """Return work identifiers that completed successfully in the manifest."""

        return self._completed_ids

    def close(self) -> None:
        """Close the temporary SQLite database and clean up resources."""

        with self._lock:
            if self._closed:
                return
            if self._preload_on_close:
                self._preload_all_entries_unlocked()
                self._preload_on_close = False
            self._closed = True
            with contextlib.suppress(Exception):
                self._conn.close()
        with contextlib.suppress(Exception):
            shutil.rmtree(self._tempdir)

    def __getitem__(self, key: str) -> Dict[str, Any]:
        lookup_key = str(key)
        if lookup_key in self._cache:
            return self._cache[lookup_key]
        if lookup_key in self._missing:
            raise KeyError(lookup_key)
        rows: Tuple[Tuple[Any, Any], ...]
        with self._lock:
            if self._closed:
                rows = tuple()
            else:
                try:
                    cursor = self._conn.execute(
                        "SELECT normalized_url, entry_json FROM entries WHERE work_id = ?",
                        (lookup_key,),
                    )
                except sqlite3.OperationalError:
                    rows = tuple()
                else:
                    rows = tuple(cursor)
        if not rows:
            self._missing.add(lookup_key)
            raise KeyError(lookup_key)
        entries: Dict[str, Dict[str, Any]] = {}
        for normalized_url, entry_json in rows:
            if not normalized_url:
                continue
            try:
                entry = json.loads(entry_json)
            except json.JSONDecodeError:
                continue
            entries[str(normalized_url)] = entry
        if not entries:
            self._missing.add(lookup_key)
            raise KeyError(lookup_key)
        self._cache[lookup_key] = entries
        return entries

    def __iter__(self) -> Iterator[str]:
        seen: Set[str] = set(self._cache.keys())
        for key in seen:
            yield key
        rows: Tuple[str, ...]
        with self._lock:
            if self._closed:
                rows = tuple()
            else:
                try:
                    cursor = self._conn.execute(
                        "SELECT DISTINCT work_id FROM entries ORDER BY work_id"
                    )
                except sqlite3.OperationalError:
                    rows = tuple()
                else:
                    rows = tuple(str(row[0]) for row in cursor if row and row[0])
        for key in rows:
            if key not in seen:
                seen.add(key)
                yield key

    def __len__(self) -> int:
        with self._lock:
            if self._closed:
                return len(self._cache)
            try:
                row = self._conn.execute(
                    "SELECT COUNT(DISTINCT work_id) FROM entries"
                ).fetchone()
            except sqlite3.OperationalError:
                return len(self._cache)
        total = int(row[0]) if row and row[0] is not None else 0
        return max(total, len(self._cache))

    def __contains__(self, key: object) -> bool:  # type: ignore[override]
        lookup_key = str(key)
        if lookup_key in self._cache:
            return True
        if lookup_key in self._missing:
            return False
        rows: Tuple[Tuple[Any, Any], ...]
        with self._lock:
            if self._closed:
                rows = tuple()
            else:
                try:
                    cursor = self._conn.execute(
                        "SELECT 1 FROM entries WHERE work_id = ? LIMIT 1",
                        (lookup_key,),
                    )
                except sqlite3.OperationalError:
                    rows = tuple()
                else:
                    rows = tuple(cursor)
        if not rows:
            self._missing.add(lookup_key)
            return False
        return True

    def get(self, key: str, default: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        try:
            return self[key]
        except KeyError:
            return default

    def enable_preload_on_close(self) -> None:
        """Load all entries into memory when :meth:`close` is invoked."""

        self._preload_on_close = True

    def preload_all_entries(self) -> None:
        """Populate the in-memory cache with every manifest entry."""

        with self._lock:
            self._preload_all_entries_unlocked()

    def _preload_all_entries_unlocked(self) -> None:
        if self._closed:
            return
        try:
            cursor = self._conn.execute(
                "SELECT work_id, normalized_url, entry_json FROM entries ORDER BY work_id"
            )
        except sqlite3.OperationalError:
            return
        for work_id, normalized_url, entry_json in cursor:
            if not work_id or not normalized_url:
                continue
            try:
                entry = json.loads(entry_json)
            except json.JSONDecodeError:
                continue
            work_key = str(work_id)
            url_key = str(normalized_url)
            per_work = self._cache.setdefault(work_key, {})
            per_work[url_key] = entry

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        with contextlib.suppress(Exception):
            self.close()


class SqliteResumeLookup(Mapping[str, Dict[str, Any]]):
    """Lazy resume mapping that fetches manifest entries directly from SQLite."""

    def __init__(self, path: Path) -> None:
        if not path.exists():
            raise ValueError(f"SQLite resume cache {path!s} does not exist")
        self._path = path
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._closed = False
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._missing: Set[str] = set()
        self._preload_on_close = False

    def close(self) -> None:
        """Close the underlying SQLite connection."""

        with self._lock:
            if self._closed:
                return
            if self._preload_on_close:
                # Preload all entries before closing for offline access
                self._preload_all_entries_unlocked()
            self._closed = True
            self._conn.close()

    def __getitem__(self, key: str) -> Dict[str, Any]:
        lookup_key = str(key)
        if lookup_key in self._cache:
            return self._cache[lookup_key]
        if lookup_key in self._missing:
            raise KeyError(lookup_key)
        # If connection is closed, only return cached data
        if self._closed:
            raise KeyError(lookup_key)
        entries = self._fetch_work_entries(lookup_key)
        if entries is None:
            self._missing.add(lookup_key)
            raise KeyError(lookup_key)
        self._cache[lookup_key] = entries
        return entries

    def __iter__(self) -> Iterator[str]:
        seen: Set[str] = set()
        seen.update(self._cache.keys())
        for key in seen:
            yield key
        with self._lock:
            if self._closed:
                remaining: Tuple[str, ...] = tuple()
            else:
                try:
                    cursor = self._conn.execute("SELECT DISTINCT work_id FROM manifests")
                except sqlite3.OperationalError:
                    remaining = tuple()
                else:
                    remaining = tuple(str(row[0]) for row in cursor if row[0])
        for key in remaining:
            if key not in seen:
                seen.add(key)
                yield key

    def __len__(self) -> int:
        with self._lock:
            if self._closed:
                return len(self._cache)
            try:
                row = self._conn.execute("SELECT COUNT(DISTINCT work_id) FROM manifests").fetchone()
            except sqlite3.OperationalError:
                # Connection closed or table doesn't exist
                return len(self._cache)
        total = int(row[0]) if row and row[0] is not None else 0
        return max(total, len(self._cache))

    def __contains__(self, key: object) -> bool:  # type: ignore[override]
        lookup_key = str(key)
        if lookup_key in self._cache:
            return True
        if lookup_key in self._missing:
            return False
        # If connection is closed, only check cached data
        if self._closed:
            return False
        try:
            entries = self._fetch_work_entries(lookup_key)
        except KeyError:
            return False
        if entries is None:
            self._missing.add(lookup_key)
            return False
        self._cache[lookup_key] = entries
        return True

    def get(self, key: str, default: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        try:
            return self[key]
        except KeyError:
            return default

    def _fetch_work_entries(self, work_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            if self._closed:
                raise RuntimeError("SqliteResumeLookup has been closed")
            try:
                cursor = self._conn.execute(
                    (
                        "SELECT run_id, work_id, url, normalized_url, schema_version, classification, "
                        "reason, reason_detail, path, path_mtime_ns, sha256, content_length, etag, last_modified "
                        "FROM manifests WHERE work_id = ? ORDER BY normalized_url"
                    ),
                    (work_id,),
                )
            except sqlite3.OperationalError:
                rows: Tuple[Tuple[Any, ...], ...] = tuple()
            else:
                rows = tuple(cursor)

        if not rows:
            return None

        entries: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            parsed = _manifest_entry_from_sqlite_row(*row)
            if parsed is None:
                continue
            _, normalized, entry, _ = parsed
            entries[normalized] = entry
        return entries if entries else None

    def _fetch_work_entries_unlocked(self, work_id: str) -> Optional[Dict[str, Any]]:
        """Fetch work entries without acquiring lock (must be called with lock held)."""
        if self._closed:
            raise RuntimeError("SqliteResumeLookup has been closed")
        try:
            cursor = self._conn.execute(
                (
                    "SELECT run_id, work_id, url, normalized_url, schema_version, classification, "
                    "reason, reason_detail, path, path_mtime_ns, sha256, content_length, etag, last_modified "
                    "FROM manifests WHERE work_id = ? ORDER BY normalized_url"
                ),
                (work_id,),
            )
        except sqlite3.OperationalError:
            rows: Tuple[Tuple[Any, ...], ...] = tuple()
        else:
            rows = tuple(cursor)

        if not rows:
            return None

        entries: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            parsed = _manifest_entry_from_sqlite_row(*row)
            if parsed is None:
                continue
            _, normalized, entry, _ = parsed
            entries[normalized] = entry
        return entries if entries else None

    def _preload_all_entries_unlocked(self) -> None:
        """Preload all manifest entries into the cache (must be called with lock held)."""
        if self._closed:
            return
        try:
            cursor = self._conn.execute("SELECT DISTINCT work_id FROM manifests ORDER BY work_id")
            work_ids = [row[0] for row in cursor if row[0]]
        except sqlite3.OperationalError:
            return

        # Load each work_id into cache using lock-free fetch
        for work_id in work_ids:
            if work_id not in self._cache and work_id not in self._missing:
                try:
                    entries = self._fetch_work_entries_unlocked(work_id)
                    if entries is not None:
                        self._cache[work_id] = entries
                    else:
                        self._missing.add(work_id)
                except RuntimeError:
                    # Connection closed during iteration
                    break

    def enable_preload_on_close(self) -> None:
        """Enable preloading all entries when close() is called."""
        self._preload_on_close = True

    def preload_all_entries(self) -> None:
        """Preload all manifest entries into the cache for offline access after close."""
        with self._lock:
            self._preload_all_entries_unlocked()


def load_resume_completed_from_sqlite(sqlite_path: Path) -> Set[str]:
    """Return work identifiers completed according to the SQLite manifest cache."""

    if not sqlite_path.exists():
        return set()

    pdf_like_values = tuple(classification.value for classification in PDF_LIKE)
    try:
        conn = sqlite3.connect(sqlite_path)
    except sqlite3.Error:  # pragma: no cover - defensive guard
        return set()

    try:
        try:
            cursor = conn.execute(
                "SELECT DISTINCT work_id FROM manifests WHERE classification IN ({})".format(
                    ",".join("?" for _ in pdf_like_values)
                ),
                pdf_like_values,
            )
        except sqlite3.OperationalError:
            return set()
        return {str(row[0]) for row in cursor if row[0]}
    finally:
        conn.close()


def load_manifest_url_index(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    """Return a mapping of normalised URLs to manifest metadata from SQLite."""

    if not path or not path.exists():
        return {}
    conn = sqlite3.connect(path)
    try:
        try:
            cursor = conn.execute(
                "SELECT url, normalized_url, path, sha256, classification, etag, last_modified, content_length, path_mtime_ns "
                "FROM manifests ORDER BY timestamp"
            )
        except sqlite3.OperationalError:
            return {}
        mapping: Dict[str, Dict[str, Any]] = {}
        base_dir = path.parent
        for (
            url,
            normalized_url,
            stored_path,
            sha256,
            classification,
            etag,
            last_modified,
            content_length,
            path_mtime_ns,
        ) in cursor:
            if not url:
                continue
            normalized_value = normalized_url or normalize_url(url)
            normalized_path = normalize_manifest_path(stored_path, base=base_dir)
            mapping[normalized_value] = {
                "url": url,
                "path": normalized_path,
                "sha256": sha256,
                "classification": classification,
                "etag": etag,
                "last_modified": last_modified,
                "content_length": content_length,
                "mtime_ns": path_mtime_ns,
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

    normalized_path = normalize_manifest_path(path)

    path_mtime_ns: Optional[int] = None
    if normalized_path:
        try:
            path_mtime_ns = Path(normalized_path).stat().st_mtime_ns
        except OSError:
            path_mtime_ns = None

    reason_token = normalize_reason(reason)
    if reason_token is None and outcome:
        reason_token = normalize_reason(outcome_reason)
    reason_value = reason_token.value if isinstance(reason_token, ReasonCode) else reason_token

    detail_source = reason_detail if reason_detail is not None else outcome_detail
    detail_token = normalize_reason(detail_source)
    detail_value = detail_token.value if isinstance(detail_token, ReasonCode) else detail_token

    normalized_html_paths: List[str] = []
    for html_path in html_paths:
        normalized = normalize_manifest_path(html_path)
        if normalized is not None:
            normalized_html_paths.append(normalized)

    normalized_text_path = normalize_manifest_path(extracted_text_path)

    return ManifestEntry(
        schema_version=MANIFEST_SCHEMA_VERSION,
        timestamp=timestamp,
        run_id=run_id,
        work_id=getattr(artifact, "work_id"),
        title=getattr(artifact, "title"),
        publication_year=getattr(artifact, "publication_year"),
        resolver=resolver,
        url=url,
        path=normalized_path,
        path_mtime_ns=path_mtime_ns,
        classification=classification,
        content_type=content_type,
        reason=reason_value,
        reason_detail=detail_value,
        html_paths=normalized_html_paths,
        sha256=sha256,
        content_length=content_length,
        etag=etag,
        last_modified=last_modified,
        extracted_text_path=normalized_text_path,
        dry_run=dry_run,
    )


__all__ = [
    "AttemptSink",
    "ManifestEntry",
    "ManifestUrlIndex",
    "RunTelemetry",
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
    "iter_previous_manifest_entries",
    "looks_like_csv_resume_target",
    "load_resume_completed_from_sqlite",
    "load_previous_manifest",
    "load_manifest_url_index",
    "JsonlResumeLookup",
    "SqliteResumeLookup",
]
