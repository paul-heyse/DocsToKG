#!/usr/bin/env python3
"""
OpenAlex PDF Downloader CLI

This module implements the command-line interface responsible for downloading
open-access PDFs referenced by OpenAlex works. It combines resolver discovery,
content classification, manifest logging, and polite crawling behaviours into a
single executable entrypoint. The implementation aligns with the modular content
download architecture documented in the OpenSpec proposal and exposes hooks for
custom resolver configuration, dry-run execution, and manifest resume logic.

Key Features:
- Threaded resolver pipeline with conditional request caching.
- Thread-safe JSONL/CSV logging including manifest entries and attempt metrics.
- Streaming content hashing with corruption detection heuristics for PDFs.
- Centralised retry handling and polite header management for resolver requests.
- Single-request download path (no redundant HEAD probes) with classification via
  streamed sniff buffers.
- CLI flags for controlling topic selection, time ranges, resolver order, and
  polite crawling identifiers.
- Optional global URL deduplication and domain-level throttling controls for
  large-scale crawls.

Dependencies:
- `requests`: HTTP communication and connection pooling adapters.
- `pyalex`: Query construction for OpenAlex works and topics.
- `DocsToKG.ContentDownload` submodules: Resolver pipeline orchestration,
  conditional caching, and shared utilities.

Usage:
    python -m DocsToKG.ContentDownload.download_pyalex_pdfs \\
        --topic \"knowledge graphs\" --year-start 2020 --year-end 2023 \\
        --out ./pdfs --resolver-config download_config.yaml
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import unquote, urlsplit

import requests
from pyalex import Topics, Works
from pyalex import config as oa_config

from DocsToKG.ContentDownload import resolvers
from DocsToKG.ContentDownload.network import (
    CachedResult,
    ConditionalRequestHelper,
    ModifiedResult,
    create_session,
    request_with_retries,
)
from DocsToKG.ContentDownload.utils import dedupe, normalize_doi, normalize_pmcid, strip_prefix

ResolverPipeline = resolvers.ResolverPipeline
ResolverConfig = resolvers.ResolverConfig
ResolverMetrics = resolvers.ResolverMetrics
DownloadOutcome = resolvers.DownloadOutcome
AttemptRecord = resolvers.AttemptRecord
default_resolvers = resolvers.default_resolvers
clear_resolver_caches = resolvers.clear_resolver_caches

MAX_SNIFF_BYTES = 64 * 1024
LOGGER = logging.getLogger("DocsToKG.ContentDownload")


def _utc_timestamp() -> str:
    """Return the current time as an ISO 8601 UTC timestamp.

    Returns:
        Timestamp string formatted with a trailing ``'Z'`` suffix.
    """

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _has_pdf_eof(path: Path) -> bool:
    """Check whether a PDF file terminates with the ``%%EOF`` marker.

    Args:
        path: Path to the candidate PDF file.

    Returns:
        ``True`` if the file ends with ``%%EOF``; ``False`` otherwise.
    """

    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            offset = max(size - 1024, 0)
            handle.seek(offset)
            tail = handle.read().decode(errors="ignore")
            return "%%EOF" in tail
    except OSError:
        return False


def _head_precheck_candidate(
    session: requests.Session,
    url: str,
    timeout: float,
) -> bool:
    """Evaluate whether ``url`` is likely to return a PDF payload.

    The helper performs a single HEAD request with a tight timeout budget
    to avoid fetching large payloads unnecessarily. Tests rely on this
    behaviour to ensure dry-run execution does not trigger streaming
    downloads.

    Args:
        session: HTTP session used for the outbound HEAD request.
        url: Candidate download URL that should be validated.
        timeout: Per-request timeout budget, in seconds.

    Returns:
        ``True`` when the HEAD response suggests the URL returns a PDF;
        ``False`` when the response clearly indicates HTML or a missing file.
    """

    try:
        response = request_with_retries(
            session,
            "HEAD",
            url,
            max_retries=1,
            timeout=min(timeout, 5.0),
            allow_redirects=True,
        )
    except Exception:
        return True

    try:
        if response.status_code not in {200, 302, 304}:
            return False

        content_type = (response.headers.get("Content-Type") or "").lower()
        content_length = response.headers.get("Content-Length", "")

        if "text/html" in content_type:
            return False
        if content_length == "0":
            return False

        return True
    finally:
        response.close()


def slugify(text: str, keep: int = 80) -> str:
    """Create a filesystem-friendly slug for a work title.

    Args:
        text: Input string to normalize into a slug.
        keep: Maximum number of characters to retain.

    Returns:
        Sanitized slug string suitable for filenames.
    """
    text = re.sub(r"[^\w\s]+", "", text or "")
    text = re.sub(r"\s+", "_", text.strip())
    return text[:keep] or "untitled"


def ensure_dir(path: Path) -> None:
    """Create a directory if it does not already exist.

    Args:
        path: Directory path to create when absent.

    Returns:
        None

    Raises:
        OSError: If the directory cannot be created because of permissions.
    """
    path.mkdir(parents=True, exist_ok=True)


def _parse_domain_interval(value: str) -> Tuple[str, float]:
    """Parse ``DOMAIN=SECONDS`` CLI arguments for domain throttling.

    Args:
        value: Argument provided via ``--domain-min-interval``.

    Returns:
        Tuple containing the normalized domain name and interval seconds.

    Raises:
        argparse.ArgumentTypeError: If the argument is malformed or negative.
    """

    if "=" not in value:
        raise argparse.ArgumentTypeError("domain interval must use the format domain=seconds")
    domain, interval = value.split("=", 1)
    domain = domain.strip().lower()
    if not domain:
        raise argparse.ArgumentTypeError("domain component cannot be empty")
    try:
        seconds = float(interval)
    except ValueError as exc:  # pragma: no cover - defensive parsing guard
        raise argparse.ArgumentTypeError(
            f"invalid interval for domain '{domain}': {interval}"
        ) from exc
    if seconds < 0:
        raise argparse.ArgumentTypeError(f"interval for domain '{domain}' must be non-negative")
    return domain, seconds


def _make_session(headers: Dict[str, str]) -> requests.Session:
    """Create a :class:`requests.Session` configured for polite crawling.

    Adapter-level retries remain disabled so :func:`request_with_retries` fully
    controls backoff, ensuring deterministic retry counts across the pipeline.

    Args:
        headers (Dict[str, str]): Header dictionary returned by
            :func:`load_resolver_config`. The mapping must already include the
            project user agent and ``mailto`` contact address. A copy of the
            mapping is applied to the outgoing session so callers can reuse
            mutable dictionaries without side effects.

    Returns:
        requests.Session: Session with connection pooling enabled and retries
        disabled at the adapter level so the application layer governs backoff.

    Notes:
        Each worker should call this helper to obtain an isolated session instance.
        Example:

            >>> _make_session({\"User-Agent\": \"DocsToKGDownloader/1.0\", \"mailto\": \"ops@example.org\"})  # doctest: +ELLIPSIS
            <requests.sessions.Session object at ...>

        The returned session is safe for concurrent HTTP requests because
        :class:`requests.adapters.HTTPAdapter` manages a thread-safe connection
        pool. Avoid mutating shared session state (for example ``session.headers.update``)
        once the session is handed to worker threads.
    """

    # Delegate to shared network helper to keep adapter defaults aligned.
    return create_session(headers)


@dataclass
class ManifestEntry:
    """Structured record capturing the outcome of a resolver attempt.

    Attributes:
        timestamp: ISO timestamp when the manifest entry was created.
        work_id: OpenAlex work identifier associated with the download.
        title: Human-readable work title.
        publication_year: Publication year when available.
        resolver: Name of the resolver that produced the asset.
        url: Source URL of the downloaded artifact.
        path: Local filesystem path to the stored artifact.
        classification: Classification label describing the outcome (e.g., 'pdf').
        content_type: MIME type reported by the server.
        reason: Failure or status reason for non-successful attempts.
        html_paths: Paths to any captured HTML artifacts.
        sha256: SHA-256 digest of the downloaded content.
        content_length: Size of the artifact in bytes.
        etag: HTTP ETag header value if provided.
        last_modified: HTTP Last-Modified timestamp.
        extracted_text_path: Optional path to extracted text content.
        dry_run: Flag indicating whether the download was simulated.

    Examples:
        >>> ManifestEntry(
        ...     timestamp="2024-01-01T00:00:00Z",
        ...     work_id="W123",
        ...     title="Sample Work",
        ...     publication_year=2024,
        ...     resolver="unpaywall",
        ...     url="https://example.org/sample.pdf",
        ...     path="pdfs/sample.pdf",
        ...     classification="pdf",
        ...     content_type="application/pdf",
        ...     reason=None,
        ...     dry_run=False,
        ... )
    """

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


class JsonlLogger:
    """Structured logger that emits attempt, manifest, and summary JSONL records.

    Attributes:
        _path: Destination JSONL log path.
        _file: Underlying file handle used for writes.

    Examples:
        >>> logger = JsonlLogger(Path("logs/attempts.jsonl"))
        >>> logger.log_summary({"processed": 10})
        >>> logger.close()

    The logger serialises records outside a thread lock and performs atomic
    writes under the lock, ensuring well-formed output even when multiple
    threads share the instance. It also implements the context manager protocol
    for deterministic resource cleanup.
    """

    def __init__(self, path: Path) -> None:
        """Create a logger backing to the given JSONL file path.

        Args:
            path: Destination JSONL log file.

        Returns:
            None
        """
        self._path = path
        ensure_dir(path.parent)
        self._file = path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def _write(self, payload: Dict[str, Any]) -> None:
        """Append a JSON record to the log file ensuring timestamps are present.

        Args:
            payload: JSON-serializable mapping to write.

        Returns:
            None
        """
        payload.setdefault("timestamp", _utc_timestamp())
        line = json.dumps(payload, sort_keys=True) + "\n"
        with self._lock:
            self._file.write(line)
            self._file.flush()

    def log_attempt(self, record: AttemptRecord, *, timestamp: Optional[str] = None) -> None:
        """Record a resolver attempt entry.

        Args:
            record: Attempt metadata captured from the resolver pipeline.
            timestamp: Optional override timestamp (ISO format).

        Returns:
            None
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
                "status": record.status,
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

    def log(self, record: AttemptRecord) -> None:
        """Compatibility shim mapping to :meth:`log_attempt`.

        Args:
            record: Attempt record to forward to :meth:`log_attempt`.

        Returns:
            None
        """
        self.log_attempt(record)

    def log_manifest(self, entry: ManifestEntry) -> None:
        """Persist a manifest entry to the JSONL log.

        Args:
            entry: Manifest entry to write.

        Returns:
            None
        """
        self._write(
            {
                "record_type": "manifest",
                "timestamp": entry.timestamp,
                "work_id": entry.work_id,
                "title": entry.title,
                "publication_year": entry.publication_year,
                "resolver": entry.resolver,
                "url": entry.url,
                "path": entry.path,
                "classification": entry.classification,
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
        """Write a summary record to the log.

        Args:
            summary: Mapping containing summary metrics.

        Returns:
            None
        """
        payload = {
            "record_type": "summary",
            "timestamp": _utc_timestamp(),
        }
        payload.update(summary)
        self._write(payload)

    def close(self) -> None:
        """Close the underlying file handle.

        Args:
            self: Logger instance managing the JSONL file descriptor.

        Returns:
            None
        """
        with self._lock:
            if not self._file.closed:
                self._file.close()

    def __enter__(self) -> "JsonlLogger":
        """Return ``self`` when used as a context manager.

        Args:
            None

        Returns:
            Logger instance configured for context-managed usage.
        """

        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Close the file handle on context manager exit.

        Args:
            exc_type: Exception type raised within the context, if any.
            exc: Exception instance raised within the context.
            tb: Traceback associated with ``exc``.

        Returns:
            None
        """

        self.close()


class CsvAttemptLoggerAdapter:
    """Adapter that mirrors attempt records to CSV for backward compatibility.

    Attributes:
        _logger: Underlying :class:`JsonlLogger` instance.
        _file: CSV file handle used for writing.
        _writer: ``csv.DictWriter`` writing to :attr:`_file`.

    Examples:
        >>> adapter = CsvAttemptLoggerAdapter(JsonlLogger(Path("attempts.jsonl")), Path("attempts.csv"))
        >>> adapter.log_attempt(AttemptRecord(work_id="W1", resolver_name="unpaywall", resolver_order=1,
        ...                                   url="https://example", status="pdf", http_status=200,
        ...                                   content_type="application/pdf", elapsed_ms=120.0))
        >>> adapter.close()

    CSV writes are protected by a lock to ensure rows remain well formed when
    multiple worker threads log through the same adapter instance.
    """

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

    def __init__(self, logger: JsonlLogger, path: Path) -> None:
        """Initialise the adapter with JSONL and CSV targets.

        Args:
            logger: Underlying :class:`JsonlLogger` used for JSONL output.
            path: Destination CSV path used to persist attempt records.

        Returns:
            None

        Raises:
            OSError: If the CSV file cannot be opened for appending.
        """

        self._logger = logger
        ensure_dir(path.parent)
        exists = path.exists()
        self._file = path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.HEADER)
        self._lock = threading.Lock()
        if not exists:
            self._writer.writeheader()

    def log_attempt(self, record: AttemptRecord) -> None:
        """Write an attempt record to both JSONL and CSV outputs.

        Args:
            record: Attempt record to persist.

        Returns:
            None
        """
        ts = _utc_timestamp()
        self._logger.log_attempt(record, timestamp=ts)
        row = {
            "timestamp": ts,
            "work_id": record.work_id,
            "resolver_name": record.resolver_name,
            "resolver_order": record.resolver_order,
            "url": record.url,
            "status": record.status,
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

    def log_manifest(self, entry: ManifestEntry) -> None:
        """Forward manifest entries to the JSONL logger.

        Args:
            entry: Manifest entry to forward.

        Returns:
            None
        """
        self._logger.log_manifest(entry)

    def log_summary(self, summary: Dict[str, Any]) -> None:
        """Forward summary entries to the JSONL logger.

        Args:
            summary: Summary mapping to forward.

        Returns:
            None
        """
        self._logger.log_summary(summary)

    def log(self, record: AttemptRecord) -> None:
        """Compatibility shim mapping to :meth:`log_attempt`.

        Args:
            record: Attempt record to log.

        Returns:
            None
        """
        self.log_attempt(record)

    def close(self) -> None:
        """Close both the JSONL logger and the CSV file handle.

        Args:
            self: Adapter instance coordinating CSV and JSONL streams.

        Returns:
            None
        """
        self._logger.close()

    def __enter__(self) -> "CsvAttemptLoggerAdapter":
        """Return ``self`` when used as a context manager.

        Args:
            None

        Returns:
            Adapter instance configured for context-managed usage.
        """

        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Close the CSV file handle on context manager exit.

        Args:
            exc_type: Exception type raised within the context, if any.
            exc: Exception instance raised within the context.
            tb: Traceback associated with ``exc``.

        Returns:
            None
        """

        self.close()
        with self._lock:
            if not self._file.closed:
                self._file.close()


def load_previous_manifest(path: Optional[Path]) -> Tuple[Dict[str, Dict[str, Any]], Set[str]]:
    """Load manifest JSONL entries indexed by work identifier.

    Args:
        path: Path to a previous manifest JSONL log, or None.

    Returns:
        Tuple containing:
            - Mapping of work_id -> url -> manifest payloads
            - Set of work IDs already completed

    Raises:
        json.JSONDecodeError: If the manifest contains invalid JSON.
    """

    per_work: Dict[str, Dict[str, Any]] = {}
    completed: Set[str] = set()
    if not path or not path.exists():
        return per_work, completed

    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("record_type") and data.get("record_type") != "manifest":
                continue
            work_id = data.get("work_id")
            url = data.get("url")
            if not work_id or not url:
                continue
            per_work.setdefault(work_id, {})[url] = data
            classification = (data.get("classification") or "").lower()
            if classification in {"pdf", "pdf_unknown", "cached"}:
                completed.add(work_id)

    return per_work, completed


def build_manifest_entry(
    artifact: WorkArtifact,
    resolver: Optional[str],
    url: Optional[str],
    outcome: Optional[DownloadOutcome],
    html_paths: List[str],
    *,
    dry_run: bool,
    reason: Optional[str] = None,
) -> ManifestEntry:
    """Create a manifest entry summarizing a download attempt.

    Args:
        artifact: Work artifact providing metadata.
        resolver: Resolver name responsible for the download.
        url: URL that was attempted.
        outcome: Download outcome describing classification and metadata.
        html_paths: Any HTML paths captured during the attempt.
        dry_run: Whether this was a dry-run execution.
        reason: Optional reason string for failures.

    Returns:
        ManifestEntry populated with download metadata.
    """
    timestamp = _utc_timestamp()
    classification = outcome.classification if outcome else "miss"
    return ManifestEntry(
        timestamp=timestamp,
        work_id=artifact.work_id,
        title=artifact.title,
        publication_year=artifact.publication_year,
        resolver=resolver,
        url=url,
        path=outcome.path if outcome else None,
        classification=classification,
        content_type=outcome.content_type if outcome else None,
        reason=reason or (outcome.error if outcome else None),
        html_paths=html_paths,
        sha256=outcome.sha256 if outcome else None,
        content_length=outcome.content_length if outcome else None,
        etag=outcome.etag if outcome else None,
        last_modified=outcome.last_modified if outcome else None,
        extracted_text_path=outcome.extracted_text_path if outcome else None,
        dry_run=dry_run,
    )


def classify_payload(head_bytes: bytes, content_type: str, url: str) -> Optional[str]:
    """Classify a payload as PDF, HTML, or unknown based on heuristics.

    Args:
        head_bytes: Leading bytes from the HTTP payload.
        content_type: Content-Type header reported by the server.
        url: Source URL of the payload.

    Returns:
        Classification string ``\"pdf\"`` or ``\"html\"`` when detection succeeds,
        otherwise ``None``.

    Raises:
        UnicodeDecodeError: If heuristics attempt to decode malformed byte
            sequences while inspecting the payload prefix.
    """

    ctype = (content_type or "").lower()
    stripped = head_bytes.lstrip() if head_bytes else b""
    prefix = stripped[:64].lower()

    if prefix.startswith(b"<!doctype html") or prefix.startswith(b"<html"):
        return "html"
    if prefix.startswith(b"<head") or prefix.startswith(b"<body"):
        return "html"

    if stripped.startswith(b"%PDF") or b"%PDF" in head_bytes[:2048]:
        return "pdf"

    if "html" in ctype:
        return "html"
    if "pdf" in ctype:
        return "pdf"

    if url.lower().endswith(".pdf"):
        return "pdf"

    return None


def _extract_filename_from_disposition(disposition: Optional[str]) -> Optional[str]:
    """Return the filename component from a Content-Disposition header."""

    if not disposition:
        return None
    parts = [segment.strip() for segment in disposition.split(";") if segment.strip()]
    for part in parts:
        lower = part.lower()
        if lower.startswith("filename*="):
            try:
                value = part.split("=", 1)[1].strip()
            except IndexError:
                continue
            _, _, encoded = value.partition("''")
            candidate = unquote(encoded or value)
            candidate = candidate.strip('"')
            if candidate:
                return candidate
        if lower.startswith("filename="):
            try:
                candidate = part.split("=", 1)[1].strip()
            except IndexError:
                continue
            candidate = candidate.strip('"')
            if candidate:
                return candidate
    return None


def _infer_suffix(
    url: str,
    content_type: Optional[str],
    disposition: Optional[str],
    classification: str,
    default_suffix: str,
) -> str:
    """Infer a destination suffix from HTTP hints and classification heuristics.

    Args:
        url: Candidate download URL emitted by a resolver.
        content_type: Content-Type header returned by the response (if any).
        disposition: Raw Content-Disposition header for RFC 6266 parsing.
        classification: Downloader classification such as ``"pdf"`` or ``"html"``.
        default_suffix: Fallback extension to use when no signals are present.

    Returns:
        Lowercase file suffix (including leading dot) chosen from the strongest
        available signal. Preference order is:

        1. ``filename*`` / ``filename`` parameters in Content-Disposition.
        2. Content-Type heuristics (PDF/HTML).
        3. URL path suffix derived from :func:`urllib.parse.urlsplit`.
        4. Provided ``default_suffix``.
    """

    filename = _extract_filename_from_disposition(disposition)
    if filename:
        suffix = Path(filename).suffix
        if suffix:
            return suffix.lower()

    ctype = (content_type or "").split(";")[0].strip().lower()
    if ctype == "application/pdf" or (ctype.endswith("pdf") and classification.startswith("pdf")):
        return ".pdf"
    if ctype in {"text/html", "application/xhtml+xml"} or "html" in ctype:
        return ".html"

    if classification.startswith("pdf"):
        path_suffix = Path(urlsplit(url).path).suffix.lower()
        if path_suffix:
            return path_suffix
        return default_suffix

    if classification == "html":
        path_suffix = Path(urlsplit(url).path).suffix.lower()
        if path_suffix:
            return path_suffix
        return default_suffix

    return default_suffix


def _update_tail_buffer(buffer: bytearray, chunk: bytes, *, limit: int = 1024) -> None:
    """Maintain the trailing ``limit`` bytes of a streamed download."""

    if not chunk:
        return
    buffer.extend(chunk)
    if len(buffer) > limit:
        del buffer[:-limit]


@dataclass
class WorkArtifact:
    """Normalized artifact describing an OpenAlex work to process.

    Attributes:
        work_id: OpenAlex work identifier.
        title: Work title suitable for logging.
        publication_year: Publication year or None.
        doi: Canonical DOI string.
        pmid: PubMed identifier (normalized).
        pmcid: PubMed Central identifier (normalized).
        arxiv_id: Normalized arXiv identifier.
        landing_urls: Candidate landing page URLs.
        pdf_urls: Candidate PDF download URLs.
        open_access_url: Open access URL provided by OpenAlex.
        source_display_names: Source names for provenance.
        base_stem: Base filename stem for local artefacts.
        pdf_dir: Directory where PDFs are stored.
        html_dir: Directory where HTML assets are stored.
        failed_pdf_urls: URLs that failed during resolution.
        metadata: Arbitrary metadata collected during processing.

    Examples:
        >>> artifact = WorkArtifact(
        ...     work_id="W123",
        ...     title="Sample Work",
        ...     publication_year=2024,
        ...     doi="10.1234/example",
        ...     pmid=None,
        ...     pmcid=None,
        ...     arxiv_id=None,
        ...     landing_urls=["https://example.org"],
        ...     pdf_urls=[],
        ...     open_access_url=None,
        ...     source_display_names=["Example Source"],
        ...     base_stem="2024__Sample_Work__W123",
        ...     pdf_dir=Path("pdfs"),
        ...     html_dir=Path("html"),
        ... )
    """

    work_id: str
    title: str
    publication_year: Optional[int]
    doi: Optional[str]
    pmid: Optional[str]
    pmcid: Optional[str]
    arxiv_id: Optional[str]
    landing_urls: List[str]
    pdf_urls: List[str]
    open_access_url: Optional[str]
    source_display_names: List[str]
    base_stem: str
    pdf_dir: Path
    html_dir: Path
    failed_pdf_urls: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Define namespace mappings for output artefact directories.

        Args:
            self: Instance whose namespace mapping is being initialised.

        Returns:
            None
        """

        self.namespaces: Dict[str, Path] = {"pdf": self.pdf_dir, "html": self.html_dir}


class DownloadState(Enum):
    """State machine for streaming downloads.

    Attributes:
        PENDING: Payload type is being sniffed.
        WRITING: Payload bytes are being streamed to disk.

    Examples:
        >>> DownloadState.PENDING is DownloadState.WRITING
        False
    """

    PENDING = "pending"
    WRITING = "writing"


def _build_download_outcome(
    *,
    artifact: WorkArtifact,
    classification: Optional[str],
    dest_path: Optional[Path],
    response: requests.Response,
    elapsed_ms: float,
    flagged_unknown: bool,
    sha256: Optional[str],
    content_length: Optional[int],
    etag: Optional[str],
    last_modified: Optional[str],
    extracted_text_path: Optional[str],
    tail_bytes: Optional[bytes],
    dry_run: bool,
    head_precheck_passed: bool = False,
) -> DownloadOutcome:
    """Create a :class:`DownloadOutcome` applying PDF validation rules.

    The helper normalises classification labels, performs the terminal ``%%EOF``
    check for PDFs (skipping when running in ``--dry-run`` mode), and attaches
    bookkeeping metadata such as digests and conditional request headers.

    Args:
        artifact: Work metadata describing the current OpenAlex record.
        classification: Initial classification derived from content sniffing.
        dest_path: Final storage path for the artefact (if any).
        response: HTTP response object returned by :func:`request_with_retries`.
        elapsed_ms: Download duration in milliseconds.
        flagged_unknown: Whether heuristics flagged the payload as ambiguous.
        sha256: SHA-256 digest of the payload when computed.
        content_length: Size of the payload in bytes, if known.
        etag: ETag header value supplied by the origin.
        last_modified: Last-Modified header value supplied by the origin.
        extracted_text_path: Optional path to extracted HTML text artefacts.
        tail_bytes: Trailing bytes captured from the streamed download for
            corruption detection heuristics.
        dry_run: Indicates whether this execution runs in dry-run mode.

    Returns:
        DownloadOutcome capturing the normalized classification and metadata.
    """

    normalized = classification or "empty"
    if flagged_unknown and normalized == "pdf":
        normalized = "pdf_unknown"

    path_str = str(dest_path) if dest_path else None

    if normalized in {"pdf", "pdf_unknown"} and not dry_run and dest_path is not None:
        size_hint = content_length
        if size_hint is None:
            with contextlib.suppress(OSError):
                size_hint = dest_path.stat().st_size
        if size_hint is not None and size_hint < 1024 and not head_precheck_passed:
            # PDFs smaller than 1 KiB are overwhelmingly HTML error stubs or
            # truncated responses observed in production crawls.
            with contextlib.suppress(OSError):
                dest_path.unlink()
            return DownloadOutcome(
                classification="pdf_corrupt",
                path=None,
                http_status=response.status_code,
                content_type=response.headers.get("Content-Type"),
                elapsed_ms=elapsed_ms,
                error=None,
                sha256=None,
                content_length=None,
                etag=etag,
                last_modified=last_modified,
                extracted_text_path=extracted_text_path,
            )

        if tail_bytes and b"</html" in tail_bytes.lower():
            with contextlib.suppress(OSError):
                dest_path.unlink()
            return DownloadOutcome(
                classification="pdf_corrupt",
                path=None,
                http_status=response.status_code,
                content_type=response.headers.get("Content-Type"),
                elapsed_ms=elapsed_ms,
                error=None,
                sha256=None,
                content_length=None,
                etag=etag,
                last_modified=last_modified,
                extracted_text_path=extracted_text_path,
            )

        if not _has_pdf_eof(dest_path):
            with contextlib.suppress(OSError):
                dest_path.unlink()
            return DownloadOutcome(
                classification="pdf_corrupt",
                path=None,
                http_status=response.status_code,
                content_type=response.headers.get("Content-Type"),
                elapsed_ms=elapsed_ms,
                error=None,
                sha256=None,
                content_length=None,
                etag=etag,
                last_modified=last_modified,
                extracted_text_path=extracted_text_path,
            )

    return DownloadOutcome(
        classification=normalized,
        path=path_str,
        http_status=response.status_code,
        content_type=response.headers.get("Content-Type"),
        elapsed_ms=elapsed_ms,
        error=None,
        sha256=sha256,
        content_length=content_length,
        etag=etag,
        last_modified=last_modified,
        extracted_text_path=extracted_text_path,
    )


def _normalize_pmid(pmid: Optional[str]) -> Optional[str]:
    """Extract the numeric PubMed identifier or return ``None`` when absent.

    Args:
        pmid: Raw PubMed identifier string which may include prefixes.

    Returns:
        Normalised numeric PMCID string or ``None`` when not parsable.
    """

    if not pmid:
        return None
    pmid = pmid.strip()
    match = re.search(r"(\d+)", pmid)
    return match.group(1) if match else None


def _normalize_arxiv(arxiv_id: Optional[str]) -> Optional[str]:
    """Normalize arXiv identifiers by removing prefixes and whitespace.

    Args:
        arxiv_id: Raw arXiv identifier which may include URL or prefix.

    Returns:
        Canonical arXiv identifier without prefixes or whitespace.
    """

    if not arxiv_id:
        return None
    arxiv_id = strip_prefix(arxiv_id, "arxiv:") or arxiv_id
    arxiv_id = arxiv_id.replace("https://arxiv.org/abs/", "")
    return arxiv_id.strip()


def _collect_location_urls(work: Dict[str, Any]) -> Dict[str, List[str]]:
    """Return landing/PDF/source URL collections derived from OpenAlex metadata.

    Args:
        work: OpenAlex work payload as returned by the Works API.

    Returns:
        Dictionary containing ``landing``, ``pdf``, and ``sources`` URL lists.
    """

    landing_urls: List[str] = []
    pdf_urls: List[str] = []
    sources: List[str] = []

    def _append_location(loc: Optional[Dict[str, Any]]) -> None:
        """Accumulate location URLs from a single OpenAlex location record.

        Args:
            loc: Location dictionary as returned by OpenAlex (may be None).
        """

        if not isinstance(loc, dict):
            return
        landing = loc.get("landing_page_url")
        pdf = loc.get("pdf_url")
        source = (loc.get("source") or {}).get("display_name")
        if landing:
            landing_urls.append(landing)
        if pdf:
            pdf_urls.append(pdf)
        if source:
            sources.append(source)

    _append_location(work.get("best_oa_location"))
    _append_location(work.get("primary_location"))
    for loc in work.get("locations", []) or []:
        _append_location(loc)

    oa_url = (work.get("open_access") or {}).get("oa_url") or None
    if oa_url:
        pdf_urls.append(oa_url)

    return {
        "landing": dedupe(landing_urls),
        "pdf": dedupe(pdf_urls),
        "sources": dedupe(sources),
    }


def build_query(args: argparse.Namespace) -> Works:
    """Build a pyalex Works query based on CLI arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Configured Works query object ready for iteration.
    """
    query = Works()
    if args.topic_id:
        query = query.filter(topics={"id": args.topic_id})
    else:
        query = query.search(args.topic)

    query = query.filter(publication_year=f"{args.year_start}-{args.year_end}")
    if args.oa_only:
        query = query.filter(open_access={"is_oa": True})

    query = query.select(
        [
            "id",
            "title",
            "publication_year",
            "ids",
            "open_access",
            "best_oa_location",
            "primary_location",
            "locations",
        ]
    )

    query = query.sort(publication_date="desc")
    return query


def resolve_topic_id_if_needed(topic_text: Optional[str]) -> Optional[str]:
    """Resolve a textual topic label into an OpenAlex topic identifier.

    Args:
        topic_text: Free-form topic text supplied via CLI.

    Returns:
        OpenAlex topic identifier string if resolved, else None.
    """
    if not topic_text:
        return None
    try:
        hits = Topics().search(topic_text).get()
    except requests.RequestException as exc:  # pragma: no cover - network guard
        LOGGER.warning("Topic lookup failed for %s: %s", topic_text, exc)
        return None
    if not hits:
        return None
    resolved = hits[0].get("id")
    if resolved:
        LOGGER.info("Resolved topic '%s' -> %s", topic_text, resolved)
    return resolved


def create_artifact(work: Dict[str, Any], pdf_dir: Path, html_dir: Path) -> WorkArtifact:
    """Normalize an OpenAlex work into a WorkArtifact instance.

    Args:
        work: Raw OpenAlex work payload.
        pdf_dir: Directory where PDFs should be stored.
        html_dir: Directory where HTML resources should be stored.

    Returns:
        WorkArtifact describing the work and candidate URLs.

    Raises:
        KeyError: If required identifiers are missing from the work payload.
    """
    work_id = (work.get("id") or "W").rsplit("/", 1)[-1]
    title = work.get("title") or work.get("display_name") or ""
    year = work.get("publication_year")
    ids = work.get("ids") or {}
    doi = normalize_doi(ids.get("doi"))
    pmid = _normalize_pmid(ids.get("pmid"))
    pmcid = normalize_pmcid(ids.get("pmcid"))
    arxiv_id = _normalize_arxiv(ids.get("arxiv"))

    locations = _collect_location_urls(work)
    landing_urls = locations["landing"]
    pdf_urls = locations["pdf"]
    sources = locations["sources"]
    oa_url = (work.get("open_access") or {}).get("oa_url")

    base = f"{year or 'noyear'}__{slugify(title)}__{work_id}".strip("_")

    artifact = WorkArtifact(
        work_id=work_id,
        title=title,
        publication_year=year,
        doi=doi,
        pmid=pmid,
        pmcid=pmcid,
        arxiv_id=arxiv_id,
        landing_urls=landing_urls,
        pdf_urls=pdf_urls,
        open_access_url=oa_url,
        source_display_names=sources,
        base_stem=base,
        pdf_dir=pdf_dir,
        html_dir=html_dir,
        metadata={"openalex_id": work.get("id")},
    )
    return artifact


def download_candidate(
    session: requests.Session,
    artifact: WorkArtifact,
    url: str,
    referer: Optional[str],
    timeout: float,
    context: Optional[Dict[str, Any]] = None,
    head_precheck_passed: bool = False,
) -> DownloadOutcome:
    """Download a single candidate URL and classify the payload.

    Args:
        session: HTTP session capable of issuing retried requests via the
            centralised :func:`request_with_retries` helper.
        artifact: Work metadata and output directory handles for the current record.
        url: Candidate download URL discovered by a resolver.
        referer: Optional referer header override provided by the resolver.
        timeout: Per-request timeout in seconds.
        context: Execution context containing ``dry_run``, ``extract_html_text``,
            and ``previous`` manifest lookup data.

    Returns:
        DownloadOutcome describing the result of the download attempt including
        streaming hash metadata when available.

    Notes:
        A lightweight HEAD preflight is issued when the caller has not already
        validated the URL. This mirrors the resolver pipeline behaviour and
        keeps dry-run tests deterministic.

    Raises:
        OSError: If writing the downloaded payload to disk fails.
        TypeError: If conditional response parsing returns unexpected objects.
    """
    context = context or {}
    headers: Dict[str, str] = {}
    if referer:
        headers["Referer"] = referer

    dry_run = bool(context.get("dry_run", False))
    head_precheck_passed = head_precheck_passed or bool(context.get("head_precheck_passed", False))
    if not head_precheck_passed and not context.get("skip_head_precheck", False):
        head_precheck_passed = _head_precheck_candidate(session, url, timeout)
        context["head_precheck_passed"] = head_precheck_passed
    extract_html_text = bool(context.get("extract_html_text", False))
    previous_map: Dict[str, Dict[str, Any]] = context.get("previous", {})
    previous = previous_map.get(url, {})
    previous_etag = previous.get("etag")
    previous_last_modified = previous.get("last_modified")
    existing_path = previous.get("path")
    previous_sha = previous.get("sha256")
    previous_length = previous.get("content_length")

    cond_helper = ConditionalRequestHelper(
        prior_etag=previous_etag,
        prior_last_modified=previous_last_modified,
        prior_sha256=previous_sha,
        prior_content_length=previous_length,
        prior_path=existing_path,
    )
    headers.update(cond_helper.build_headers())

    start = time.monotonic()
    content_type_hint = ""
    try:
        with request_with_retries(
            session,
            "GET",
            url,
            stream=True,
            allow_redirects=True,
            timeout=timeout,
            headers=headers,
        ) as response:
            elapsed_ms = (time.monotonic() - start) * 1000.0
            if response.status_code == 304:
                cached = cond_helper.interpret_response(response)
                if not isinstance(cached, CachedResult):  # pragma: no cover - defensive
                    raise TypeError("Expected CachedResult for 304 response")
                return DownloadOutcome(
                    classification="cached",
                    path=cached.path,
                    http_status=response.status_code,
                    content_type=response.headers.get("Content-Type") or content_type_hint,
                    elapsed_ms=elapsed_ms,
                    error=None,
                    sha256=cached.sha256,
                    content_length=cached.content_length,
                    etag=cached.etag,
                    last_modified=cached.last_modified,
                    extracted_text_path=None,
                )

            if response.status_code != 200:
                return DownloadOutcome(
                    classification="http_error",
                    path=None,
                    http_status=response.status_code,
                    content_type=response.headers.get("Content-Type") or content_type_hint,
                    elapsed_ms=elapsed_ms,
                    error=None,
                    sha256=None,
                    content_length=None,
                    etag=None,
                    last_modified=None,
                    extracted_text_path=None,
                )

            modified_result: ModifiedResult = cond_helper.interpret_response(response)

            content_type = response.headers.get("Content-Type") or content_type_hint
            disposition = response.headers.get("Content-Disposition")
            sniff_buffer = bytearray()
            detected: Optional[str] = None
            flagged_unknown = False
            dest_path: Optional[Path] = None
            part_path: Optional[Path] = None
            handle = None
            state = DownloadState.PENDING
            hasher = hashlib.sha256() if not dry_run else None
            byte_count = 0
            tail_buffer = bytearray()

            try:
                for chunk in response.iter_content(chunk_size=1 << 15):
                    if not chunk:
                        continue
                    if state is DownloadState.PENDING:
                        sniff_buffer.extend(chunk)
                        detected = classify_payload(bytes(sniff_buffer), content_type, url)
                        if detected is None and len(sniff_buffer) >= MAX_SNIFF_BYTES:
                            detected = "pdf"
                            flagged_unknown = True

                        if detected is not None:
                            if dry_run:
                                break
                            default_suffix = ".html" if detected == "html" else ".pdf"
                            suffix = _infer_suffix(
                                url, content_type, disposition, detected, default_suffix
                            )
                            dest_dir = artifact.html_dir if detected == "html" else artifact.pdf_dir
                            dest_path = dest_dir / f"{artifact.base_stem}{suffix}"
                            ensure_dir(dest_path.parent)
                            part_path = dest_path.with_suffix(dest_path.suffix + ".part")
                            handle = part_path.open("wb")
                            initial_bytes = bytes(sniff_buffer)
                            if initial_bytes:
                                handle.write(initial_bytes)
                                if hasher:
                                    hasher.update(initial_bytes)
                                byte_count += len(initial_bytes)
                                _update_tail_buffer(tail_buffer, initial_bytes)
                            sniff_buffer.clear()
                            state = DownloadState.WRITING
                            continue
                    elif handle is not None:
                        handle.write(chunk)
                        if hasher:
                            hasher.update(chunk)
                        byte_count += len(chunk)
                        _update_tail_buffer(tail_buffer, chunk)

                if detected is None:
                    return DownloadOutcome(
                        classification="empty",
                        path=None,
                        http_status=response.status_code,
                        content_type=content_type,
                        elapsed_ms=elapsed_ms,
                    )
            finally:
                if handle is not None:
                    handle.close()

            if dry_run:
                return _build_download_outcome(
                    artifact=artifact,
                    classification=detected,
                    dest_path=None,
                    response=response,
                    elapsed_ms=elapsed_ms,
                    flagged_unknown=flagged_unknown,
                    sha256=None,
                    content_length=None,
                    etag=modified_result.etag,
                    last_modified=modified_result.last_modified,
                    extracted_text_path=None,
                    dry_run=True,
                    tail_bytes=None,
                    head_precheck_passed=head_precheck_passed,
                )

            sha256: Optional[str] = None
            content_length: Optional[int] = None
            if dest_path and hasher is not None:
                sha256 = hasher.hexdigest()
                content_length = byte_count
            if part_path and dest_path:
                os.replace(part_path, dest_path)
            if part_path:
                with contextlib.suppress(FileNotFoundError):
                    part_path.unlink()
            tail_snapshot: Optional[bytes] = bytes(tail_buffer) if tail_buffer else None

            extracted_text_path: Optional[str] = None
            if dest_path and detected == "html" and extract_html_text:
                try:
                    import trafilatura  # type: ignore
                except ImportError:
                    LOGGER.warning(
                        "HTML text extraction requested but trafilatura is not installed."
                    )
                else:
                    try:
                        text = trafilatura.extract(
                            dest_path.read_text(encoding="utf-8", errors="ignore")
                        )
                    except Exception as exc:  # pragma: no cover - third-party failure
                        LOGGER.warning("Failed to extract HTML text for %s: %s", dest_path, exc)
                    else:
                        if text:
                            text_path = dest_path.with_suffix(dest_path.suffix + ".txt")
                            text_path.write_text(text, encoding="utf-8")
                            extracted_text_path = str(text_path)

            return _build_download_outcome(
                artifact=artifact,
                classification=detected,
                dest_path=dest_path,
                response=response,
                elapsed_ms=elapsed_ms,
                flagged_unknown=flagged_unknown,
                sha256=sha256,
                content_length=content_length,
                etag=modified_result.etag,
                last_modified=modified_result.last_modified,
                extracted_text_path=extracted_text_path,
                dry_run=False,
                tail_bytes=tail_snapshot,
                head_precheck_passed=head_precheck_passed,
            )
    except requests.RequestException as exc:
        elapsed_ms = (time.monotonic() - start) * 1000.0
        return DownloadOutcome(
            classification="request_error",
            path=None,
            http_status=None,
            content_type=None,
            elapsed_ms=elapsed_ms,
            error=str(exc),
            sha256=None,
            content_length=None,
            etag=None,
            last_modified=None,
            extracted_text_path=None,
        )


def read_resolver_config(path: Path) -> Dict[str, Any]:
    """Read resolver configuration from JSON or YAML files.

    Args:
        path: Path to the configuration file.

    Returns:
        Parsed configuration mapping.

    Raises:
        RuntimeError: If YAML parsing is requested but PyYAML is unavailable.
    """
    text = path.read_text(encoding="utf-8")
    ext = path.suffix.lower()
    if ext in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - missing dependency
            raise RuntimeError(
                "Install PyYAML to load YAML resolver configs, or provide JSON."
            ) from exc
        return yaml.safe_load(text) or {}
    if ext in {".json", ".jsn"}:
        return json.loads(text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - missing dependency
            raise RuntimeError(
                "Unable to parse resolver config. Install PyYAML or supply JSON."
            ) from exc
        return yaml.safe_load(text) or {}


def apply_config_overrides(
    config: ResolverConfig,
    data: Dict[str, Any],
    resolver_names: Sequence[str],
) -> None:
    """Apply overrides from configuration data onto a ResolverConfig.

    Args:
        config: Resolver configuration object to mutate.
        data: Mapping loaded from a configuration file.
        resolver_names: Known resolver names to seed toggle defaults.

    Returns:
        None
    """
    for field_name in (
        "resolver_order",
        "resolver_toggles",
        "max_attempts_per_work",
        "timeout",
        "sleep_jitter",
        "polite_headers",
        "unpaywall_email",
        "core_api_key",
        "semantic_scholar_api_key",
        "doaj_api_key",
        "resolver_timeouts",
        "resolver_min_interval_s",
        "mailto",
    ):
        if field_name in data and data[field_name] is not None:
            setattr(config, field_name, data[field_name])

    if "resolver_rate_limits" in data and "resolver_min_interval_s" not in data:
        LOGGER.warning(
            "resolver_rate_limits deprecated, use resolver_min_interval_s",
        )
        legacy_limits = data.get("resolver_rate_limits") or {}
        config.resolver_min_interval_s.update(legacy_limits)

    for name in resolver_names:
        default_enabled = name not in {"openaire", "hal", "osf"}
        config.resolver_toggles.setdefault(name, default_enabled)


def load_resolver_config(
    args: argparse.Namespace,
    resolver_names: Sequence[str],
    resolver_order_override: Optional[List[str]] = None,
) -> ResolverConfig:
    """Construct resolver configuration combining CLI, config files, and env vars.

    Args:
        args: Parsed CLI arguments.
        resolver_names: Sequence of resolver names supported by the pipeline.
        resolver_order_override: Optional override list for resolver order.

    Returns:
        Populated ResolverConfig instance.

    Raises:
        FileNotFoundError: If the resolver configuration file does not exist.
        RuntimeError: If YAML parsing is requested but PyYAML is unavailable.
    """
    config = ResolverConfig()
    if args.resolver_config:
        config_data = read_resolver_config(Path(args.resolver_config))
        apply_config_overrides(config, config_data, resolver_names)

    # Environment fallbacks
    config.unpaywall_email = (
        args.unpaywall_email
        or config.unpaywall_email
        or os.getenv("UNPAYWALL_EMAIL")
        or args.mailto
    )
    config.core_api_key = args.core_api_key or config.core_api_key or os.getenv("CORE_API_KEY")
    config.semantic_scholar_api_key = (
        args.semantic_scholar_api_key or config.semantic_scholar_api_key or os.getenv("S2_API_KEY")
    )
    config.doaj_api_key = args.doaj_api_key or config.doaj_api_key or os.getenv("DOAJ_API_KEY")
    config.mailto = args.mailto or config.mailto

    if getattr(args, "max_resolver_attempts", None):
        config.max_attempts_per_work = args.max_resolver_attempts
    if getattr(args, "resolver_timeout", None):
        config.timeout = args.resolver_timeout
    if hasattr(args, "concurrent_resolvers") and args.concurrent_resolvers is not None:
        config.max_concurrent_resolvers = args.concurrent_resolvers

    if resolver_order_override:
        ordered = []
        for name in resolver_order_override:
            if name not in ordered:
                ordered.append(name)
        for name in resolver_names:
            if name not in ordered:
                ordered.append(name)
        config.resolver_order = ordered

    for name in resolver_names:
        default_enabled = name not in {"openaire", "hal", "osf"}
        config.resolver_toggles.setdefault(name, default_enabled)

    for disabled in getattr(args, "disable_resolver", []) or []:
        config.resolver_toggles[disabled] = False

    for enabled in getattr(args, "enable_resolver", []) or []:
        config.resolver_toggles[enabled] = True

    if hasattr(args, "global_url_dedup") and args.global_url_dedup is not None:
        config.enable_global_url_dedup = args.global_url_dedup

    if getattr(args, "domain_min_interval", None):
        domain_limits = dict(config.domain_min_interval_s)
        for domain, interval in args.domain_min_interval:
            domain_limits[domain] = interval
        config.domain_min_interval_s = domain_limits

    # Polite headers include mailto when available
    headers = dict(config.polite_headers)
    existing_mailto = headers.get("mailto")
    mailto_value = config.mailto or existing_mailto
    base_agent = headers.get("User-Agent") or "DocsToKGDownloader/1.0"
    if mailto_value:
        config.mailto = config.mailto or mailto_value
        headers["mailto"] = mailto_value
        user_agent = f"DocsToKGDownloader/1.0 (+{mailto_value}; mailto:{mailto_value})"
    else:
        headers.pop("mailto", None)
        user_agent = base_agent
    headers["User-Agent"] = user_agent
    if getattr(args, "accept", None):
        headers["Accept"] = args.accept
    config.polite_headers = headers

    if hasattr(args, "head_precheck") and args.head_precheck is not None:
        config.enable_head_precheck = args.head_precheck

    # Apply resolver rate defaults (Unpaywall recommends 1 request per second)
    config.resolver_min_interval_s.setdefault("unpaywall", 1.0)

    return config


def iterate_openalex(
    query: Works, per_page: int, max_results: Optional[int]
) -> Iterable[Dict[str, Any]]:
    """Iterate over OpenAlex works respecting pagination and limits.

    Args:
        query: Configured Works query instance.
        per_page: Number of results to request per page.
        max_results: Optional maximum number of works to yield.

    Yields:
        Work payload dictionaries returned by the OpenAlex API.

    Returns:
        Iterable yielding the same work payload dictionaries for convenience.
    """
    pager = query.paginate(per_page=per_page, n_max=None)
    retrieved = 0
    for page in pager:
        for work in page:
            yield work
            retrieved += 1
            if max_results and retrieved >= max_results:
                return


def process_one_work(
    work: Dict[str, Any],
    session: requests.Session,
    pdf_dir: Path,
    html_dir: Path,
    pipeline: ResolverPipeline,
    logger: JsonlLogger,
    metrics: ResolverMetrics,
    *,
    dry_run: bool,
    extract_html_text: bool,
    previous_lookup: Dict[str, Dict[str, Any]],
    resume_completed: Set[str],
) -> Dict[str, Any]:
    """Process a single OpenAlex work through the resolver pipeline.

    Args:
        work: OpenAlex work payload from :func:`iterate_openalex`.
        session: Requests session configured for resolver usage.
        pdf_dir: Directory where PDF artefacts are written.
        html_dir: Directory where HTML artefacts are written.
        pipeline: Resolver pipeline orchestrating downstream resolvers.
        logger: Structured attempt logger capturing manifest records.
        metrics: Resolver metrics collector.
        dry_run: When True, simulate downloads without writing files.
        extract_html_text: Whether to extract plaintext from HTML artefacts.
        previous_lookup: Mapping of work_id/URL to prior manifest entries.
        resume_completed: Set of work IDs already processed in resume mode.

    Returns:
        Dictionary summarizing the outcome (saved/html_only/skipped flags).

    Raises:
        requests.RequestException: Propagated if resolver HTTP requests fail
            unexpectedly outside guarded sections.
        Exception: Bubbling from resolver pipeline internals when not handled.
    """
    artifact = create_artifact(work, pdf_dir=pdf_dir, html_dir=html_dir)

    result = {"work_id": artifact.work_id, "saved": False, "html_only": False, "skipped": False}

    raw_previous = previous_lookup.get(artifact.work_id, {})
    previous_map = {
        url: {
            "etag": entry.get("etag"),
            "last_modified": entry.get("last_modified"),
            "path": entry.get("path"),
            "sha256": entry.get("sha256"),
            "content_length": entry.get("content_length"),
        }
        for url, entry in raw_previous.items()
    }
    download_context = {
        "dry_run": dry_run,
        "extract_html_text": extract_html_text,
        "previous": previous_map,
    }

    if artifact.work_id in resume_completed:
        LOGGER.info("Skipping %s (already completed)", artifact.work_id)
        skipped_outcome = DownloadOutcome(
            classification="skipped",
            path=None,
            http_status=None,
            content_type=None,
            elapsed_ms=None,
            error="resume-complete",
        )
        entry = build_manifest_entry(
            artifact,
            resolver="resume",
            url=None,
            outcome=skipped_outcome,
            html_paths=[],
            dry_run=dry_run,
            reason="resume-complete",
        )
        logger.log_manifest(entry)
        result["skipped"] = True
        return result

    existing_pdf = artifact.pdf_dir / f"{artifact.base_stem}.pdf"
    if not dry_run and existing_pdf.exists():
        existing_outcome = DownloadOutcome(
            classification="exists",
            path=str(existing_pdf),
            http_status=None,
            content_type=None,
            elapsed_ms=None,
            error="already-downloaded",
        )
        entry = build_manifest_entry(
            artifact,
            resolver="existing",
            url=None,
            outcome=existing_outcome,
            html_paths=[],
            dry_run=dry_run,
            reason="already-downloaded",
        )
        logger.log_manifest(entry)
        result["skipped"] = True
        return result

    pipeline_result = pipeline.run(session, artifact, context=download_context)
    html_paths_total = list(pipeline_result.html_paths)

    if pipeline_result.success and pipeline_result.outcome:
        entry = build_manifest_entry(
            artifact,
            resolver=pipeline_result.resolver_name,
            url=pipeline_result.url,
            outcome=pipeline_result.outcome,
            html_paths=html_paths_total,
            dry_run=dry_run,
        )
        logger.log_manifest(entry)
        if pipeline_result.outcome.is_pdf:
            result["saved"] = True
        elif pipeline_result.outcome.classification == "html":
            result["html_only"] = True
        return result

    reason = pipeline_result.reason or (
        pipeline_result.outcome.error if pipeline_result.outcome else "no-resolver-success"
    )
    outcome = pipeline_result.outcome or DownloadOutcome(
        classification="miss",
        path=None,
        http_status=None,
        content_type=None,
        elapsed_ms=None,
        error=reason,
    )
    logger.log(
        AttemptRecord(
            work_id=artifact.work_id,
            resolver_name="final",
            resolver_order=None,
            url=pipeline_result.url,
            status=outcome.classification,
            http_status=outcome.http_status,
            content_type=outcome.content_type,
            elapsed_ms=outcome.elapsed_ms,
            reason=reason,
            sha256=outcome.sha256,
            content_length=outcome.content_length,
            dry_run=dry_run,
        )
    )

    if html_paths_total:
        result["html_only"] = True

    entry = build_manifest_entry(
        artifact,
        resolver=pipeline_result.resolver_name,
        url=pipeline_result.url,
        outcome=outcome,
        html_paths=html_paths_total,
        dry_run=dry_run,
        reason=reason,
    )
    logger.log_manifest(entry)
    return result


def main() -> None:
    """Parse CLI arguments, configure resolvers, and execute downloads.

    The entrypoint wires together argument parsing, resolver configuration,
    logging setup, and the resolver pipeline orchestration documented in the
    modular content download specification. It is exposed both as the module's
    ``__main__`` handler and via `python -m`.

    Args:
        None

    Returns:
        None
    """
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Download OpenAlex PDFs for a topic and year range with resolvers.",
    )
    parser.add_argument("--topic", type=str, help="Free-text topic search.")
    parser.add_argument(
        "--topic-id",
        type=str,
        help="OpenAlex Topic ID (e.g., https://openalex.org/T12345). Overrides --topic.",
    )
    parser.add_argument("--year-start", type=int, required=True, help="Start year (inclusive).")
    parser.add_argument("--year-end", type=int, required=True, help="End year (inclusive).")
    parser.add_argument("--out", type=Path, default=Path("./pdfs"), help="Output folder for PDFs.")
    parser.add_argument(
        "--html-out",
        type=Path,
        default=None,
        help="Folder for HTML responses (default: sibling 'HTML').",
    )
    parser.add_argument("--manifest", type=Path, default=None, help="Path to manifest JSONL log.")
    parser.add_argument(
        "--mailto", type=str, default=None, help="Email for the OpenAlex polite pool."
    )
    parser.add_argument("--per-page", type=int, default=200, help="Results per page (1-200).")
    parser.add_argument("--max", type=int, default=None, help="Maximum works to process.")
    parser.add_argument("--oa-only", action="store_true", help="Only consider open-access works.")
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.05,
        help="Sleep seconds between works (sequential mode).",
    )

    # Resolver configuration
    parser.add_argument(
        "--resolver-config", type=str, default=None, help="Path to resolver config (YAML/JSON)."
    )
    parser.add_argument(
        "--resolver-order",
        type=str,
        default=None,
        help="Comma-separated resolver order override (e.g., 'unpaywall,crossref').",
    )
    parser.add_argument(
        "--unpaywall-email", type=str, default=None, help="Override Unpaywall email credential."
    )
    parser.add_argument("--core-api-key", type=str, default=None, help="CORE API key override.")
    parser.add_argument(
        "--semantic-scholar-api-key",
        type=str,
        default=None,
        help="Semantic Scholar Graph API key override.",
    )
    parser.add_argument("--doaj-api-key", type=str, default=None, help="DOAJ API key override.")
    parser.add_argument(
        "--disable-resolver",
        action="append",
        default=[],
        help="Disable a resolver by name (can be repeated).",
    )
    parser.add_argument(
        "--enable-resolver",
        action="append",
        default=[],
        help="Enable a resolver by name (can be repeated).",
    )
    parser.add_argument(
        "--max-resolver-attempts",
        type=int,
        default=None,
        help="Maximum resolver attempts per work.",
    )
    parser.add_argument(
        "--resolver-timeout",
        type=float,
        default=None,
        help="Default timeout (seconds) for resolver HTTP requests.",
    )
    parser.add_argument(
        "--concurrent-resolvers",
        type=int,
        default=None,
        help="Maximum resolver threads per work item (default: 1).",
    )
    parser.add_argument(
        "--log-path",
        dest="log_jsonl",
        type=Path,
        default=None,
        help="Deprecated alias for --manifest.",
    )
    parser.add_argument(
        "--log-csv",
        type=Path,
        default=None,
        help="Optional CSV attempts log output path.",
    )
    parser.add_argument(
        "--global-url-dedup",
        dest="global_url_dedup",
        action="store_true",
        help="Skip downloads when a URL was already fetched in this run.",
    )
    parser.add_argument(
        "--no-global-url-dedup",
        dest="global_url_dedup",
        action="store_false",
        help="Disable global URL deduplication (default).",
    )
    parser.add_argument(
        "--domain-min-interval",
        dest="domain_min_interval",
        type=_parse_domain_interval,
        action="append",
        default=[],
        metavar="DOMAIN=SECONDS",
        help=(
            "Enforce a minimum interval between requests to a domain. "
            "Repeat the option to configure multiple domains."
        ),
    )
    parser.add_argument(
        "--head-precheck",
        dest="head_precheck",
        action="store_true",
        help="Enable resolver HEAD preflight filtering (default).",
    )
    parser.add_argument(
        "--no-head-precheck",
        dest="head_precheck",
        action="store_false",
        help="Disable resolver HEAD preflight filtering.",
    )
    parser.add_argument(
        "--accept",
        type=str,
        default=None,
        help="Override the Accept header sent with resolver HTTP requests.",
    )
    parser.set_defaults(head_precheck=True, global_url_dedup=None)
    parser.add_argument(
        "--log-format",
        choices=["jsonl", "csv"],
        default="jsonl",
        help="Log format for attempts (default: jsonl).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1 for sequential).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Measure resolver coverage without writing files.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume from manifest JSONL log, skipping completed works.",
    )
    parser.add_argument(
        "--extract-html-text",
        action="store_true",
        help="Extract plaintext from HTML fallbacks (requires trafilatura).",
    )

    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.concurrent_resolvers is not None and args.concurrent_resolvers < 1:
        parser.error("--concurrent-resolvers must be >= 1")
    if not args.topic and not args.topic_id:
        parser.error("Provide --topic or --topic-id.")

    if args.mailto:
        oa_config.email = args.mailto

    topic_id = args.topic_id
    if not topic_id and args.topic:
        try:
            topic_id = resolve_topic_id_if_needed(args.topic)
        except Exception as exc:
            LOGGER.warning("Failed to resolve topic ID for '%s': %s", args.topic, exc)
            topic_id = None

    query = build_query(
        argparse.Namespace(
            topic=args.topic,
            topic_id=topic_id,
            year_start=args.year_start,
            year_end=args.year_end,
            oa_only=args.oa_only,
        )
    )

    pdf_dir = args.out
    html_dir = args.html_out or (pdf_dir.parent / "HTML")
    ensure_dir(pdf_dir)
    ensure_dir(html_dir)

    resolver_instances = default_resolvers()
    resolver_names = [resolver.name for resolver in resolver_instances]
    resolver_order_override: Optional[List[str]] = None
    if args.resolver_order:
        resolver_order_override = [
            name.strip() for name in args.resolver_order.split(",") if name.strip()
        ]
        if not resolver_order_override:
            parser.error("--resolver-order requires at least one resolver name.")
        unknown = [name for name in resolver_order_override if name not in resolver_names]
        if unknown:
            parser.error(f"Unknown resolver(s) in --resolver-order: {', '.join(unknown)}")
        resolver_order_override.extend(
            name for name in resolver_names if name not in resolver_order_override
        )

    config = load_resolver_config(args, resolver_names, resolver_order_override)

    manifest_path = args.manifest or args.log_jsonl or (pdf_dir / "manifest.jsonl")
    if manifest_path.suffix != ".jsonl":
        manifest_path = manifest_path.with_suffix(".jsonl")
    csv_path = args.log_csv
    if args.log_format == "csv":
        csv_path = csv_path or manifest_path.with_suffix(".csv")

    summary: Dict[str, Any] = {}
    summary_record: Dict[str, Any] = {}
    processed = 0
    saved = 0
    html_only = 0
    skipped = 0

    with JsonlLogger(manifest_path) as base_logger:
        attempt_logger: Any = base_logger
        csv_adapter: Optional[CsvAttemptLoggerAdapter] = None
        if csv_path:
            csv_adapter = CsvAttemptLoggerAdapter(base_logger, csv_path)
            attempt_logger = csv_adapter

        resume_lookup, resume_completed = load_previous_manifest(args.resume_from)
        if args.resume_from:
            clear_resolver_caches()

        metrics = ResolverMetrics()
        pipeline = ResolverPipeline(
            resolvers=resolver_instances,
            config=config,
            download_func=download_candidate,
            logger=attempt_logger,
            metrics=metrics,
        )

        def _session_factory() -> requests.Session:
            """Build a fresh requests session configured with polite headers."""

    summary_record: Dict[str, Any] = {}

    with contextlib.ExitStack() as stack:
        base_logger = stack.enter_context(JsonlLogger(manifest_path))
        attempt_logger: Any = base_logger
        csv_path = args.log_csv
        if args.log_format == "csv":
            csv_path = csv_path or manifest_path.with_suffix(".csv")
        if csv_path:
            attempt_logger = stack.enter_context(CsvAttemptLoggerAdapter(base_logger, csv_path))

        resume_lookup, resume_completed = load_previous_manifest(args.resume_from)
        if args.resume_from:
            clear_resolver_caches()

        metrics = ResolverMetrics()
        pipeline = ResolverPipeline(
            resolvers=resolver_instances,
            config=config,
            download_func=download_candidate,
            logger=attempt_logger,
            metrics=metrics,
        )

        def _session_factory() -> requests.Session:
            """Build a fresh requests session configured with polite headers."""

            return _make_session(config.polite_headers)
            return _make_session(config.polite_headers)

        def _record_result(res: Dict[str, Any]) -> None:
            """Update aggregate counters based on a single work result."""

            nonlocal processed, saved, html_only, skipped
            processed += 1
            if res.get("saved"):
                saved += 1
            if res.get("html_only"):
                html_only += 1
            if res.get("skipped"):
                skipped += 1

        try:
            if args.workers == 1:
                session = _session_factory()
                try:
                    for work in iterate_openalex(
                        query, per_page=args.per_page, max_results=args.max
                    ):
                        result = process_one_work(
                            work,
                            session,
                            pdf_dir,
                            html_dir,
                            pipeline,
                            attempt_logger,
                            metrics,
                            dry_run=args.dry_run,
                            extract_html_text=args.extract_html_text,
                            previous_lookup=resume_lookup,
                            resume_completed=resume_completed,
                        )
                        _record_result(result)
                        if args.sleep > 0:
                            time.sleep(args.sleep)
                finally:
                    if hasattr(session, "close"):
                        session.close()
            else:
                with ThreadPoolExecutor(max_workers=args.workers) as executor:
                    futures = []

                    def _submit_work(work_item: Dict[str, Any]) -> None:
                        """Submit a work item to the executor for asynchronous processing."""

                        def _runner() -> Dict[str, Any]:
                            """Process a single work item within a worker-managed session."""

                            session = _session_factory()
                            try:
                                return process_one_work(
                                    work_item,
                                    session,
                                    pdf_dir,
                                    html_dir,
                                    pipeline,
                                    attempt_logger,
                                    metrics,
                                    dry_run=args.dry_run,
                                    extract_html_text=args.extract_html_text,
                                    previous_lookup=resume_lookup,
                                    resume_completed=resume_completed,
                                )
                            finally:
                                if hasattr(session, "close"):
                                    session.close()

                        futures.append(executor.submit(_runner))

                    for work in iterate_openalex(
                        query, per_page=args.per_page, max_results=args.max
                    ):
                        _submit_work(work)

                    for future in as_completed(futures):
                        _record_result(future.result())
        except Exception:
            raise
        else:
            summary = metrics.summary()
            summary_record = {
                "processed": processed,
                "saved": saved,
                "html_only": html_only,
                "skipped": skipped,
                "resolvers": summary,
            }
            try:
                attempt_logger.log_summary(summary_record)
            except Exception:  # pragma: no cover - defensive logging safeguard
                LOGGER.warning("Failed to log summary record", exc_info=True)
            metrics_path = manifest_path.with_suffix(".metrics.json")
            try:
                ensure_dir(metrics_path.parent)
                metrics_path.write_text(
                    json.dumps(summary_record, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            except Exception:
                LOGGER.warning("Failed to write metrics sidecar %s", metrics_path, exc_info=True)
        finally:
            if csv_adapter is not None:
                csv_adapter.close()

    print(
        f"\nDone. Processed {processed} works, saved {saved} PDFs, HTML-only {html_only}, skipped {skipped}."
    )
    if args.dry_run:
        print("DRY RUN: no files written, resolver coverage only.")
    print("Resolver summary:")
    for key, values in summary.items():
        print(f"  {key}: {values}")

    if not summary_record:
        summary_record = {
            "processed": processed,
            "saved": saved,
            "html_only": html_only,
            "skipped": skipped,
            "resolvers": summary,
        }
    LOGGER.info("resolver_run_summary %s", json.dumps(summary_record, sort_keys=True))


if __name__ == "__main__":
    main()
