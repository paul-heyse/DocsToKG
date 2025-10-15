#!/usr/bin/env python3
"""Download PDFs for OpenAlex works with a configurable resolver stack."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import inspect
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import requests
from pyalex import Topics, Works
from pyalex import config as oa_config
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from DocsToKG.ContentDownload.resolvers import (
    AttemptRecord,
    DownloadOutcome,
    ResolverConfig,
    ResolverMetrics,
    ResolverPipeline,
    clear_resolver_caches,
    default_resolvers,
)
from DocsToKG.ContentDownload.utils import (
    dedupe,
    normalize_doi,
    normalize_pmcid,
    strip_prefix,
)

MAX_SNIFF_BYTES = 64 * 1024
SUCCESS_STATUSES = {"pdf", "pdf_unknown"}

LOGGER = logging.getLogger("DocsToKG.ContentDownload")


def _accepts_argument(func: Callable[..., Any], name: str) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return True

    for parameter in signature.parameters.values():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            return True
        if parameter.name == name:
            return True
    return False


def _has_pdf_eof(path: Path) -> bool:
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
        path: Directory path to create.

    Returns:
        None
    """
    path.mkdir(parents=True, exist_ok=True)


def _make_session(headers: Dict[str, str]) -> requests.Session:
    """Create a retry-enabled :class:`requests.Session` with polite headers.

    The session mounts an :class:`urllib3.util.retry.Retry` adapter that retries
    transient HTTP failures (429/502/503/504) up to five times with
    exponential backoff and `Retry-After` support. Callers should pass the
    polite header set returned by :func:`load_resolver_config` so that each
    worker advertises the required contact information.
    """

    session = requests.Session()
    if hasattr(session, "headers") and isinstance(session.headers, dict):
        session.headers.update(headers)
    retry = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 502, 503, 504],
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    if hasattr(session, "mount"):
        session.mount("http://", adapter)
        session.mount("https://", adapter)
    return session


def _make_session_for_worker(headers: Dict[str, str]) -> requests.Session:
    """Factory helper for per-worker sessions."""

    return _make_session(headers)


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
    """Structured logger that emits attempt, manifest, and summary JSONL records."""

    def __init__(self, path: Path) -> None:
        """Create a logger backing to the given JSONL file path."""
        self._path = path
        ensure_dir(path.parent)
        self._file = path.open("a", encoding="utf-8")

    def _write(self, payload: Dict[str, Any]) -> None:
        """Append a JSON record to the log file ensuring timestamps are present."""
        payload.setdefault("timestamp", datetime.utcnow().isoformat() + "Z")
        self._file.write(json.dumps(payload, sort_keys=True) + "\n")
        self._file.flush()

    def log_attempt(self, record: AttemptRecord, *, timestamp: Optional[str] = None) -> None:
        """Record a resolver attempt entry.

        Args:
            record: Attempt metadata captured from the resolver pipeline.
            timestamp: Optional override timestamp (ISO format).
        """
        ts = timestamp or (datetime.utcnow().isoformat() + "Z")
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
                "reason": record.reason,
                "metadata": record.metadata,
                "sha256": record.sha256,
                "content_length": record.content_length,
                "dry_run": record.dry_run,
            }
        )

    def log(self, record: AttemptRecord) -> None:
        """Compatibility shim mapping to :meth:`log_attempt`."""
        self.log_attempt(record)

    def log_manifest(self, entry: ManifestEntry) -> None:
        """Persist a manifest entry to the JSONL log."""
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
        """Write a summary record to the log."""
        payload = {
            "record_type": "summary",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        payload.update(summary)
        self._write(payload)

    def close(self) -> None:
        """Close the underlying file handle."""
        self._file.close()


class CsvAttemptLoggerAdapter:
    """Adapter that mirrors attempt records to CSV for backward compatibility."""

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
        "reason",
        "sha256",
        "content_length",
        "dry_run",
        "metadata",
    ]

    def __init__(self, logger: JsonlLogger, path: Path) -> None:
        self._logger = logger
        ensure_dir(path.parent)
        exists = path.exists()
        self._file = path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.HEADER)
        if not exists:
            self._writer.writeheader()

    def log_attempt(self, record: AttemptRecord) -> None:
        """Write an attempt record to both JSONL and CSV outputs."""
        ts = datetime.utcnow().isoformat() + "Z"
        self._logger.log_attempt(record, timestamp=ts)
        self._writer.writerow(
            {
                "timestamp": ts,
                "work_id": record.work_id,
                "resolver_name": record.resolver_name,
                "resolver_order": record.resolver_order,
                "url": record.url,
                "status": record.status,
                "http_status": record.http_status,
                "content_type": record.content_type,
                "elapsed_ms": record.elapsed_ms,
                "reason": record.reason,
                "sha256": record.sha256,
                "content_length": record.content_length,
                "dry_run": record.dry_run,
                "metadata": json.dumps(record.metadata, sort_keys=True) if record.metadata else "",
            }
        )
        self._file.flush()

    def log_manifest(self, entry: ManifestEntry) -> None:
        """Forward manifest entries to the JSONL logger."""
        self._logger.log_manifest(entry)

    def log_summary(self, summary: Dict[str, Any]) -> None:
        """Forward summary entries to the JSONL logger."""
        self._logger.log_summary(summary)

    def log(self, record: AttemptRecord) -> None:
        """Compatibility shim mapping to :meth:`log_attempt`."""
        self.log_attempt(record)

    def close(self) -> None:
        """Close both the JSONL logger and the CSV file handle."""
        self._logger.close()
        self._file.close()


def load_previous_manifest(path: Optional[Path]) -> Tuple[Dict[str, Dict[str, Any]], Set[str]]:
    """Load manifest JSONL entries indexed by work identifier.

    Args:
        path: Path to a previous manifest JSONL log, or None.

    Returns:
        Tuple containing:
            - Mapping of work_id -> url -> manifest payloads
            - Set of work IDs already completed
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
    timestamp = datetime.utcnow().isoformat() + "Z"
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
    """Return 'pdf', 'html', or None if undecided.

    Args:
        head_bytes: Leading bytes from the HTTP payload.
        content_type: Content-Type header reported by the server.
        url: Source URL of the payload.

    Returns:
        Classification string or None when unknown.
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


@dataclass
class WorkArtifact:
    """Normalized artifact describing an OpenAlex work to process."""

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
        self.namespaces: Dict[str, Path] = {"pdf": self.pdf_dir, "html": self.html_dir}


class DownloadState(Enum):
    """State machine for streaming downloads."""

    PENDING = "pending"
    WRITING = "writing"


def build_query(args: argparse.Namespace) -> Works:
    """Build a pyalex Works query based on CLI arguments.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Configured Works query object ready for iteration.
    """


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
    dry_run: bool,
) -> DownloadOutcome:
    """Create a :class:`DownloadOutcome` applying PDF validation rules.

    The helper normalises classification labels, performs the terminal ``%%EOF``
    check for PDFs (skipping when running in ``--dry-run`` mode), and attaches
    bookkeeping metadata such as digests and conditional request headers.
    """

    normalized = classification or "empty"
    if flagged_unknown and normalized == "pdf":
        normalized = "pdf_unknown"

    path_str = str(dest_path) if dest_path else None

    if normalized in {"pdf", "pdf_unknown"} and not dry_run and dest_path is not None:
        if not _has_pdf_eof(dest_path):
            with contextlib.suppress(OSError):
                dest_path.unlink()
            return DownloadOutcome(
                classification="pdf_corrupt",
                path=None,
                http_status=response.status_code,
                content_type=response.headers.get("Content-Type"),
                elapsed_ms=elapsed_ms,
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
        sha256=sha256,
        content_length=content_length,
        etag=etag,
        last_modified=last_modified,
        extracted_text_path=extracted_text_path,
    )


def _normalize_pmid(pmid: Optional[str]) -> Optional[str]:
    if not pmid:
        return None
    pmid = pmid.strip()
    match = re.search(r"(\d+)", pmid)
    return match.group(1) if match else None


def _normalize_arxiv(arxiv_id: Optional[str]) -> Optional[str]:
    if not arxiv_id:
        return None
    arxiv_id = strip_prefix(arxiv_id, "arxiv:") or arxiv_id
    arxiv_id = arxiv_id.replace("https://arxiv.org/abs/", "")
    return arxiv_id.strip()


def _collect_location_urls(work: Dict[str, Any]) -> Dict[str, List[str]]:
    landing_urls: List[str] = []
    pdf_urls: List[str] = []
    sources: List[str] = []

    def append_location(loc: Optional[Dict[str, Any]]) -> None:
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

    append_location(work.get("best_oa_location"))
    append_location(work.get("primary_location"))
    for loc in work.get("locations", []) or []:
        append_location(loc)

    oa_url = (work.get("open_access") or {}).get("oa_url") or None
    if oa_url:
        pdf_urls.append(oa_url)

    return {
        "landing": dedupe(landing_urls),
        "pdf": dedupe(pdf_urls),
        "sources": dedupe(sources),
    }


def build_query(args: argparse.Namespace) -> Works:
    """Build a pyalex Works query based on CLI arguments."""
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
) -> DownloadOutcome:
    """Download a single candidate URL and classify the outcome.

    Args:
        session: Prepared requests session with retry/backoff.
        artifact: Work artifact being downloaded.
        url: Candidate URL to fetch.
        referer: Optional referer header value.
        timeout: Request timeout in seconds.
        context: Optional context including dry-run and previous manifest.

    Returns:
        DownloadOutcome describing the attempt.
    """
    context = context or {}
    headers: Dict[str, str] = {}
    if referer:
        headers["Referer"] = referer

    dry_run = bool(context.get("dry_run", False))
    extract_html_text = bool(context.get("extract_html_text", False))
    previous_map: Dict[str, Dict[str, Any]] = context.get("previous", {})
    previous = previous_map.get(url, {})
    previous_etag = previous.get("etag")
    previous_last_modified = previous.get("last_modified")
    existing_path = previous.get("path")
    previous_sha = previous.get("sha256")
    previous_length = previous.get("content_length")

    if previous_etag:
        headers["If-None-Match"] = previous_etag
    if previous_last_modified:
        headers["If-Modified-Since"] = previous_last_modified

    start = time.monotonic()
    content_type_hint = ""
    try:
        head_headers = dict(headers)
        head_headers.pop("If-None-Match", None)
        head_headers.pop("If-Modified-Since", None)
        with contextlib.suppress(requests.RequestException):
            head = session.head(
                url,
                allow_redirects=True,
                timeout=timeout,
                headers=head_headers,
            )
            if head.status_code < 400:
                content_type_hint = head.headers.get("Content-Type", "") or ""
            head.close()

        with session.get(
            url,
            stream=True,
            allow_redirects=True,
            timeout=timeout,
            headers=headers,
        ) as response:
            elapsed_ms = (time.monotonic() - start) * 1000.0
            if response.status_code == 304:
                return DownloadOutcome(
                    classification="cached",
                    path=existing_path,
                    http_status=response.status_code,
                    content_type=response.headers.get("Content-Type") or content_type_hint,
                    elapsed_ms=elapsed_ms,
                    sha256=previous_sha,
                    content_length=previous_length,
                    etag=previous_etag,
                    last_modified=previous_last_modified,
                )

            if response.status_code != 200:
                return DownloadOutcome(
                    classification="http_error",
                    path=None,
                    http_status=response.status_code,
                    content_type=response.headers.get("Content-Type"),
                    elapsed_ms=elapsed_ms,
                )

            content_type = response.headers.get("Content-Type") or content_type_hint
            sniff_buffer = bytearray()
            detected: Optional[str] = None
            flagged_unknown = False
            dest_path: Optional[Path] = None
            part_path: Optional[Path] = None
            handle = None
            state = DownloadState.PENDING

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
                            if detected == "html":
                                dest_path = artifact.html_dir / f"{artifact.base_stem}.html"
                            else:
                                dest_path = artifact.pdf_dir / f"{artifact.base_stem}.pdf"
                            ensure_dir(dest_path.parent)
                            part_path = dest_path.with_suffix(dest_path.suffix + ".part")
                            handle = part_path.open("wb")
                            handle.write(sniff_buffer)
                            sniff_buffer.clear()
                            state = DownloadState.WRITING
                            continue
                    elif handle is not None:
                        handle.write(chunk)

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
                    etag=response.headers.get("ETag") or previous_etag,
                    last_modified=response.headers.get("Last-Modified") or previous_last_modified,
                    extracted_text_path=None,
                    dry_run=True,
                )

            sha256: Optional[str] = None
            content_length: Optional[int] = None
            if part_path and dest_path:
                sha = hashlib.sha256()
                try:
                    with part_path.open("rb") as tmp:
                        for chunk in iter(lambda: tmp.read(1 << 20), b""):
                            if not chunk:
                                break
                            sha.update(chunk)
                    sha256 = sha.hexdigest()
                    content_length = part_path.stat().st_size
                    os.replace(part_path, dest_path)
                finally:
                    with contextlib.suppress(FileNotFoundError):
                        if part_path.exists() and dest_path.exists():
                            part_path.unlink()

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
                etag=response.headers.get("ETag"),
                last_modified=response.headers.get("Last-Modified"),
                extracted_text_path=extracted_text_path,
                dry_run=False,
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
        )


def read_resolver_config(path: Path) -> Dict[str, Any]:
    """Read resolver configuration from JSON or YAML files."""
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

    if args.max_resolver_attempts:
        config.max_attempts_per_work = args.max_resolver_attempts
    if args.resolver_timeout:
        config.timeout = args.resolver_timeout

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

    for disabled in args.disable_resolver or []:
        config.resolver_toggles[disabled] = False

    # Polite headers include mailto when available
    headers = dict(config.polite_headers)
    headers.pop("mailto", None)
    base_agent = headers.get("User-Agent") or "DocsToKGDownloader/1.0"
    if config.mailto:
        user_agent = f"DocsToKGDownloader/1.0 (+{config.mailto}; mailto:{config.mailto})"
    else:
        user_agent = base_agent
    headers["User-Agent"] = user_agent
    config.polite_headers = headers

    # Apply resolver rate defaults (Unpaywall recommends 1 request per second)
    config.resolver_min_interval_s.setdefault("unpaywall", 1.0)

    return config


def iterate_openalex(
    query: Works, per_page: int, max_results: Optional[int]
) -> Iterable[Dict[str, Any]]:
    pager = query.paginate(per_page=per_page, n_max=None)
    retrieved = 0
    for page in pager:
        for work in page:
            yield work
            retrieved += 1
            if max_results and retrieved >= max_results:
                return


def attempt_openalex_candidates(
    session: requests.Session,
    artifact: WorkArtifact,
    logger: JsonlLogger,
    metrics: ResolverMetrics,
    context: Optional[Dict[str, Any]] = None,
) -> Optional[Tuple[DownloadOutcome, str]]:
    candidates = list(artifact.pdf_urls)
    if artifact.open_access_url:
        candidates.append(artifact.open_access_url)

    call_context = context or {}
    accepts_context = _accepts_argument(download_candidate, "context")
    seen = set()
    html_paths: List[str] = []
    for url in candidates:
        if not url or url in seen:
            continue
        seen.add(url)
        call_kwargs = {"context": call_context} if accepts_context else {}
        outcome = download_candidate(
            session,
            artifact,
            url,
            referer=None,
            timeout=30.0,
            **call_kwargs,
        )
        logger.log(
            AttemptRecord(
                work_id=artifact.work_id,
                resolver_name="openalex",
                resolver_order=0,
                url=url,
                status=outcome.classification,
                http_status=outcome.http_status,
                content_type=outcome.content_type,
                elapsed_ms=outcome.elapsed_ms,
                reason=outcome.error,
                metadata={"source": "openalex"},
                sha256=outcome.sha256,
                content_length=outcome.content_length,
                dry_run=bool(call_context.get("dry_run", False)),
            )
        )
        metrics.record_attempt("openalex", outcome)
        if outcome.classification == "html" and outcome.path:
            html_paths.append(outcome.path)
        if outcome.is_pdf:
            if html_paths:
                artifact.metadata.setdefault("openalex_html_paths", []).extend(html_paths)
            return outcome, url
        if outcome.classification not in SUCCESS_STATUSES and url:
            artifact.failed_pdf_urls.append(url)
    if not seen:
        metrics.record_skip("openalex", "no-candidates")
    if html_paths:
        artifact.metadata.setdefault("openalex_html_paths", []).extend(html_paths)
    return None


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

    openalex_result = attempt_openalex_candidates(
        session,
        artifact,
        logger,
        metrics,
        download_context,
    )

    if openalex_result and openalex_result[0].is_pdf:
        outcome, url = openalex_result
        html_paths = artifact.metadata.get("openalex_html_paths", [])
        entry = build_manifest_entry(
            artifact,
            resolver="openalex",
            url=url,
            outcome=outcome,
            html_paths=html_paths,
            dry_run=dry_run,
        )
        logger.log_manifest(entry)
        result["saved"] = True
        return result

    html_paths_total = list(artifact.metadata.get("openalex_html_paths", []))

    pipeline_result = pipeline.run(session, artifact, context=download_context)
    html_paths_total.extend(pipeline_result.html_paths)

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
    parser.add_argument(
        "--oa-only", action="store_true", help="Only consider open-access works."
    )
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
    base_logger = JsonlLogger(manifest_path)
    attempt_logger: Any = base_logger
    csv_path = args.log_csv
    if args.log_format == "csv":
        csv_path = csv_path or manifest_path.with_suffix(".csv")
    if csv_path:
        attempt_logger = CsvAttemptLoggerAdapter(base_logger, csv_path)

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

    def session_factory() -> requests.Session:
        return _make_session_for_worker(config.polite_headers)

    processed = 0
    saved = 0
    html_only = 0
    skipped = 0

    def record_result(res: Dict[str, Any]) -> None:
        nonlocal processed, saved, html_only, skipped
        processed += 1
        if res.get("saved"):
            saved += 1
        if res.get("html_only"):
            html_only += 1
        if res.get("skipped"):
            skipped += 1

    summary: Dict[str, Any] = {}
    try:
        if args.workers == 1:
            session = session_factory()
            try:
                for work in iterate_openalex(query, per_page=args.per_page, max_results=args.max):
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
                    record_result(result)
                    if args.sleep > 0:
                        time.sleep(args.sleep)
            finally:
                if hasattr(session, "close"):
                    session.close()
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = []

                def submit_work(work_item: Dict[str, Any]) -> None:
                    def runner() -> Dict[str, Any]:
                        session = session_factory()
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

                    futures.append(executor.submit(runner))

                for work in iterate_openalex(query, per_page=args.per_page, max_results=args.max):
                    submit_work(work)

                for future in as_completed(futures):
                    record_result(future.result())
    except Exception:
        attempt_logger.close()
        raise
    else:
        summary = metrics.summary()
        attempt_logger.close()

    print(
        f"\nDone. Processed {processed} works, saved {saved} PDFs, HTML-only {html_only}, skipped {skipped}."
    )
    if args.dry_run:
        print("DRY RUN: no files written, resolver coverage only.")
    print("Resolver summary:")
    for key, values in summary.items():
        print(f"  {key}: {values}")

    LOGGER.info(
        "resolver_run_summary %s",
        json.dumps(
            {
                "processed": processed,
                "saved": saved,
                "html_only": html_only,
                "skipped": skipped,
                "summary": summary,
            },
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    main()
