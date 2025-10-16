#!/usr/bin/env python3
# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.download_pyalex_pdfs",
#   "purpose": "OpenAlex PDF download CLI and supporting utilities",
#   "sections": [
#     {"id": "_utc_timestamp", "name": "_utc_timestamp", "anchor": "function-_utc_timestamp", "kind": "function"},
#     {"id": "_has_pdf_eof", "name": "_has_pdf_eof", "anchor": "function-_has_pdf_eof", "kind": "function"},
#     {"id": "slugify", "name": "slugify", "anchor": "function-slugify", "kind": "function"},
#     {"id": "ensure_dir", "name": "ensure_dir", "anchor": "function-ensure_dir", "kind": "function"},
#     {"id": "_parse_domain_interval", "name": "_parse_domain_interval", "anchor": "function-_parse_domain_interval", "kind": "function"},
#     {"id": "_make_session", "name": "_make_session", "anchor": "function-_make_session", "kind": "function"},
#     {"id": "JsonlSink", "name": "JsonlSink", "anchor": "class-JsonlSink", "kind": "class"},
#     {"id": "CsvSink", "name": "CsvSink", "anchor": "class-CsvSink", "kind": "class"},
#     {"id": "MultiSink", "name": "MultiSink", "anchor": "class-MultiSink", "kind": "class"},
#     {"id": "load_previous_manifest", "name": "load_previous_manifest", "anchor": "function-load_previous_manifest", "kind": "function"},
#     {"id": "build_manifest_entry", "name": "build_manifest_entry", "anchor": "function-build_manifest_entry", "kind": "function"},
#     {"id": "classify_payload", "name": "classify_payload", "anchor": "function-classify_payload", "kind": "function"},
#     {"id": "_extract_filename_from_disposition", "name": "_extract_filename_from_disposition", "anchor": "function-_extract_filename_from_disposition", "kind": "function"},
#     {"id": "_infer_suffix", "name": "_infer_suffix", "anchor": "function-_infer_suffix", "kind": "function"},
#     {"id": "_update_tail_buffer", "name": "_update_tail_buffer", "anchor": "function-_update_tail_buffer", "kind": "function"},
#     {"id": "WorkArtifact", "name": "WorkArtifact", "anchor": "class-WorkArtifact", "kind": "class"},
#     {"id": "DownloadState", "name": "DownloadState", "anchor": "class-DownloadState", "kind": "class"},
#     {"id": "_build_download_outcome", "name": "_build_download_outcome", "anchor": "function-_build_download_outcome", "kind": "function"},
#     {"id": "_normalize_pmid", "name": "_normalize_pmid", "anchor": "function-_normalize_pmid", "kind": "function"},
#     {"id": "_normalize_arxiv", "name": "_normalize_arxiv", "anchor": "function-_normalize_arxiv", "kind": "function"},
#     {"id": "_collect_location_urls", "name": "_collect_location_urls", "anchor": "function-_collect_location_urls", "kind": "function"},
#     {"id": "build_query", "name": "build_query", "anchor": "function-build_query", "kind": "function"},
#     {"id": "resolve_topic_id_if_needed", "name": "resolve_topic_id_if_needed", "anchor": "function-resolve_topic_id_if_needed", "kind": "function"},
#     {"id": "create_artifact", "name": "create_artifact", "anchor": "function-create_artifact", "kind": "function"},
#     {"id": "download_candidate", "name": "download_candidate", "anchor": "function-download_candidate", "kind": "function"},
#     {"id": "read_resolver_config", "name": "read_resolver_config", "anchor": "function-read_resolver_config", "kind": "function"},
#     {"id": "_seed_resolver_toggle_defaults", "name": "_seed_resolver_toggle_defaults", "anchor": "function-_seed_resolver_toggle_defaults", "kind": "function"},
#     {"id": "apply_config_overrides", "name": "apply_config_overrides", "anchor": "function-apply_config_overrides", "kind": "function"},
#     {"id": "load_resolver_config", "name": "load_resolver_config", "anchor": "function-load_resolver_config", "kind": "function"},
#     {"id": "iterate_openalex", "name": "iterate_openalex", "anchor": "function-iterate_openalex", "kind": "function"},
#     {"id": "process_one_work", "name": "process_one_work", "anchor": "function-process_one_work", "kind": "function"},
#     {"id": "main", "name": "main", "anchor": "function-main", "kind": "function"}
#   ]
# }
# === /NAVMAP ===

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
import hashlib
import json
import logging
import os
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from functools import lru_cache
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)
from urllib.parse import unquote, urlsplit

import requests
from pyalex import Topics, Works
from pyalex import config as oa_config

from DocsToKG.ContentDownload import resolvers
from DocsToKG.ContentDownload.classifications import Classification, PDF_LIKE
from DocsToKG.ContentDownload.network import (
    CachedResult,
    ConditionalRequestHelper,
    ModifiedResult,
    create_session,
    head_precheck,
    request_with_retries,
)
from DocsToKG.ContentDownload.classifier import (
    classify_payload,
    _extract_filename_from_disposition,
    _infer_suffix,
)
from DocsToKG.ContentDownload.utils import (
    dedupe,
    normalize_doi,
    normalize_pmcid,
    normalize_pmid as _normalize_pmid,
    normalize_arxiv as _normalize_arxiv,
    strip_prefix,
    slugify,
)
from DocsToKG.ContentDownload.telemetry import (
    AttemptSink,
    CsvSink,
    JsonlSink,
    ManifestEntry,
    MultiSink,
)

ResolverPipeline = resolvers.ResolverPipeline
ResolverConfig = resolvers.ResolverConfig
ResolverMetrics = resolvers.ResolverMetrics
DownloadOutcome = resolvers.DownloadOutcome
AttemptRecord = resolvers.AttemptRecord
default_resolvers = resolvers.default_resolvers

# --- Globals ---

__all__ = (
    "AttemptSink",
    "CsvSink",
    "DownloadState",
    "JsonlSink",
    "ManifestEntry",
    "MultiSink",
    "WorkArtifact",
    "apply_config_overrides",
    "build_manifest_entry",
    "build_query",
    "classify_payload",
    "create_artifact",
    "download_candidate",
    "ensure_dir",
    "iterate_openalex",
    "load_previous_manifest",
    "load_resolver_config",
    "main",
    "process_one_work",
    "read_resolver_config",
    "resolve_topic_id_if_needed",
    "slugify",
)

LOGGER = logging.getLogger("DocsToKG.ContentDownload")


# --- Pipeline Helpers ---

def _utc_timestamp() -> str:
    """Return the current time as an ISO 8601 UTC timestamp.

    Returns:
        Timestamp string formatted with a trailing ``'Z'`` suffix.
    """

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _has_pdf_eof(path: Path, *, window_bytes: int = 2048) -> bool:
    """Check whether a PDF file terminates with the ``%%EOF`` marker.

    Args:
        path: Path to the candidate PDF file.
        window_bytes: Number of trailing bytes to scan for the EOF marker.

    Returns:
        ``True`` if the file ends with ``%%EOF``; ``False`` otherwise.
    """

    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            window = max(int(window_bytes), 0)
            offset = max(size - window, 0)
            handle.seek(offset)
            tail = handle.read().decode(errors="ignore")
            return "%%EOF" in tail
    except OSError:
        return False


def _update_tail_buffer(buffer: bytearray, chunk: bytes, *, limit: int = 1024) -> None:
    """Maintain the trailing ``limit`` bytes of a streamed download."""

    if not chunk:
        return
    buffer.extend(chunk)
    if len(buffer) > limit:
        del buffer[:-limit]


# --- Public Functions ---

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


def _build_download_outcome(
    *,
    artifact: WorkArtifact,
    classification: Optional[Classification | str],
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
    min_pdf_bytes: int = 1024,
    tail_check_bytes: int = 2048,
) -> DownloadOutcome:
    if isinstance(classification, Classification):
        classification_code = classification
    else:
        classification_code = Classification.from_wire(classification)
    if flagged_unknown and classification_code is Classification.PDF:
        classification_code = Classification.PDF_UNKNOWN

    path_str = str(dest_path) if dest_path else None

    if classification_code in PDF_LIKE and not dry_run and dest_path is not None:
        size_hint = content_length
        if size_hint is None:
            with contextlib.suppress(OSError):
                size_hint = dest_path.stat().st_size
        if (
            size_hint is not None
            and min_pdf_bytes > 0
            and size_hint < min_pdf_bytes
            and not head_precheck_passed
        ):
            with contextlib.suppress(OSError):
                dest_path.unlink()
            return DownloadOutcome(
                classification=Classification.PDF_CORRUPT,
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
                classification=Classification.PDF_CORRUPT,
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

        if not _has_pdf_eof(dest_path, window_bytes=tail_check_bytes):
            with contextlib.suppress(OSError):
                dest_path.unlink()
            return DownloadOutcome(
                classification=Classification.PDF_CORRUPT,
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
        classification=classification_code,
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


def _make_session(
    headers: Dict[str, str],
    *,
    pool_connections: int = 64,
    pool_maxsize: int = 128,
) -> requests.Session:
    """Create a :class:`requests.Session` configured for polite crawling.

    Adapter-level retries remain disabled so :func:`request_with_retries` fully
    controls backoff, ensuring deterministic retry counts across the pipeline.

    Args:
        headers (Dict[str, str]): Header dictionary returned by
            :func:`load_resolver_config`. The mapping must already include the
            project user agent and ``mailto`` contact address. A copy of the
            mapping is applied to the outgoing session so callers can reuse
            mutable dictionaries without side effects.
        pool_connections: Minimum pool size shared across HTTP and HTTPS adapters.
        pool_maxsize: Upper bound for per-host connections retained in the pool.

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
    return create_session(
        headers,
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize,
    )


# --- Logging Sinks ---

# --- Manifest Utilities ---

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
        ValueError: If entries omit required fields or use deprecated schemas.
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
            data = json.loads(line)
            record_type = data.get("record_type")
            if record_type is None:
                raise ValueError(
                    "Legacy manifest entries without record_type are no longer supported."
                )
            if record_type != "manifest":
                continue
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
            per_work.setdefault(work_id, {})[url] = data
            raw_classification = data.get("classification")
            classification_text = (raw_classification or "").strip()
            if not classification_text:
                raise ValueError("Manifest entries must declare a classification.")
            classification_code = Classification.from_wire(classification_text)
            data["classification"] = classification_code.value
            if classification_code in PDF_LIKE:
                completed.add(work_id)

    return per_work, completed


# --- Download Pipeline ---

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
    classification = outcome.classification.value if outcome else Classification.MISS.value
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


@lru_cache(maxsize=128)
def _lookup_topic_id(topic_text: str) -> Optional[str]:
    """Cached helper to resolve an OpenAlex topic identifier."""
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


def resolve_topic_id_if_needed(topic_text: Optional[str]) -> Optional[str]:
    """Resolve a textual topic label into an OpenAlex topic identifier.

    Args:
        topic_text: Free-form topic text supplied via CLI.

    Returns:
        OpenAlex topic identifier string if resolved, else None.
    """
    if not topic_text:
        return None
    normalized = topic_text.strip()
    if not normalized:
        return None
    return _lookup_topic_id(normalized)


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
    sniff_limit = max(int(context.get("sniff_bytes", 64 * 1024)), 0)
    min_pdf_bytes = max(int(context.get("min_pdf_bytes", 1024)), 0)
    tail_window_bytes = max(int(context.get("tail_check_bytes", 2048)), 0)
    max_bytes_raw = context.get("max_bytes")
    max_bytes: Optional[int]
    try:
        max_bytes = int(max_bytes_raw) if max_bytes_raw is not None else None
    except (TypeError, ValueError):
        max_bytes = None
    if max_bytes is not None and max_bytes <= 0:
        max_bytes = None
    headers: Dict[str, str] = {}
    if referer:
        headers["Referer"] = referer

    dry_run = bool(context.get("dry_run", False))
    head_precheck_passed = head_precheck_passed or bool(context.get("head_precheck_passed", False))
    if not head_precheck_passed and not context.get("skip_head_precheck", False):
        head_precheck_passed = head_precheck(session, url, timeout)
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
                    classification=Classification.CACHED,
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
                    classification=Classification.HTTP_ERROR,
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
            content_type_lower = content_type.lower() if content_type else ""
            content_length_header = response.headers.get("Content-Length")
            content_length_hint: Optional[int] = None
            if content_length_header:
                try:
                    content_length_hint = int(content_length_header.strip())
                except (TypeError, ValueError):
                    content_length_hint = None
            if max_bytes and content_length_hint is not None and content_length_hint > max_bytes:
                LOGGER.warning(
                    "Aborting download due to max-bytes limit",
                    extra={
                        "extra_fields": {
                            "url": url,
                            "work_id": artifact.work_id,
                            "content_length": content_length_hint,
                            "max_bytes": max_bytes,
                        }
                    },
                )
                elapsed_now = (time.monotonic() - start) * 1000.0
                classification_limit = (
                    Classification.HTML_TOO_LARGE
                    if "html" in content_type_lower
                    else Classification.PAYLOAD_TOO_LARGE
                )
                return DownloadOutcome(
                    classification=classification_limit,
                    path=None,
                    http_status=response.status_code,
                    content_type=content_type,
                    elapsed_ms=elapsed_now,
                    error=f"content-length {content_length_hint} exceeds max_bytes {max_bytes}",
                    sha256=None,
                    content_length=content_length_hint,
                    etag=modified_result.etag,
                    last_modified=modified_result.last_modified,
                    extracted_text_path=None,
                )
            sniff_buffer = bytearray()
            detected: Optional[Classification] = None
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
                        if (
                            detected is None
                            and sniff_limit
                            and len(sniff_buffer) >= sniff_limit
                        ):
                            detected = Classification.PDF
                            flagged_unknown = True

                        if detected is not None:
                            if dry_run:
                                break
                            default_suffix = ".html" if detected == Classification.HTML else ".pdf"
                            suffix = _infer_suffix(
                                url, content_type, disposition, detected, default_suffix
                            )
                            dest_dir = artifact.html_dir if detected == Classification.HTML else artifact.pdf_dir
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
                                _update_tail_buffer(
                                    tail_buffer, initial_bytes, limit=tail_window_bytes
                                )
                            sniff_buffer.clear()
                            state = DownloadState.WRITING
                            continue
                    elif handle is not None:
                        handle.write(chunk)
                        if hasher:
                            hasher.update(chunk)
                        byte_count += len(chunk)
                        _update_tail_buffer(tail_buffer, chunk, limit=tail_window_bytes)

                        if max_bytes and byte_count > max_bytes:
                            LOGGER.warning(
                                "Aborting download during stream due to max-bytes limit",
                                extra={
                                    "extra_fields": {
                                        "url": url,
                                        "work_id": artifact.work_id,
                                        "bytes_downloaded": byte_count,
                                        "max_bytes": max_bytes,
                                    }
                                },
                            )
                            if handle is not None:
                                handle.close()
                                handle = None
                            if part_path:
                                with contextlib.suppress(FileNotFoundError):
                                    part_path.unlink()
                            classification_limit = (
                                Classification.HTML_TOO_LARGE
                                if detected is Classification.HTML
                                else Classification.PAYLOAD_TOO_LARGE
                            )
                            elapsed_limit = (time.monotonic() - start) * 1000.0
                            return DownloadOutcome(
                                classification=classification_limit,
                                path=None,
                                http_status=response.status_code,
                                content_type=content_type,
                                elapsed_ms=elapsed_limit,
                                error=f"download exceeded max_bytes {max_bytes}",
                                sha256=None,
                                content_length=byte_count,
                                etag=modified_result.etag,
                                last_modified=modified_result.last_modified,
                                extracted_text_path=None,
                            )

                if detected is None:
                    return DownloadOutcome(
                        classification=Classification.MISS,
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
                    min_pdf_bytes=min_pdf_bytes,
                    tail_check_bytes=tail_window_bytes,
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
            if dest_path and detected == Classification.HTML and extract_html_text:
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
                min_pdf_bytes=min_pdf_bytes,
                tail_check_bytes=tail_window_bytes,
            )
    except requests.RequestException as exc:
        elapsed_ms = (time.monotonic() - start) * 1000.0
        return DownloadOutcome(
            classification=Classification.REQUEST_ERROR,
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


def _seed_resolver_toggle_defaults(config: ResolverConfig, resolver_names: Sequence[str]) -> None:
    """Ensure resolver toggles include defaults for every known resolver."""

    for name in resolver_names:
        default_enabled = resolvers.DEFAULT_RESOLVER_TOGGLES.get(name, True)
        config.resolver_toggles.setdefault(name, default_enabled)


def apply_config_overrides(
    config: ResolverConfig,
    data: Dict[str, Any],
    resolver_names: Sequence[str],
) -> None:
    """Apply overrides from configuration data onto a ResolverConfig.

    Args:
        config: Resolver configuration object to mutate.
        data: Mapping loaded from a configuration file.
        resolver_names: Known resolver names. Defaults are applied after overrides.

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
        "resolver_head_precheck",
    ):
        if field_name in data and data[field_name] is not None:
            setattr(config, field_name, data[field_name])

    if "resolver_rate_limits" in data:
        raise ValueError(
            "resolver_rate_limits is no longer supported. "
            "Rename entries to resolver_min_interval_s."
        )

    # Resolver toggle defaults are applied once after all overrides via
    # ``_seed_resolver_toggle_defaults`` to ensure a single source of truth.


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

    for disabled in getattr(args, "disable_resolver", []) or []:
        config.resolver_toggles[disabled] = False

    for enabled in getattr(args, "enable_resolver", []) or []:
        config.resolver_toggles[enabled] = True

    _seed_resolver_toggle_defaults(config, resolver_names)

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
    accept_override = getattr(args, "accept", None)
    if accept_override:
        headers["Accept"] = accept_override
    elif not headers.get("Accept"):
        headers["Accept"] = "application/pdf, text/html;q=0.9, */*;q=0.8"
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
    logger: AttemptSink,
    metrics: ResolverMetrics,
    *,
    dry_run: bool,
    list_only: bool,
    extract_html_text: bool,
    previous_lookup: Dict[str, Dict[str, Any]],
    resume_completed: Set[str],
    max_bytes: Optional[int],
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
        list_only: When True, record candidate URLs without fetching content.
            extract_html_text: Whether to extract plaintext from HTML artefacts.
            previous_lookup: Mapping of work_id/URL to prior manifest entries.
            resume_completed: Set of work IDs already processed in resume mode.
            max_bytes: Optional size limit per download in bytes.

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
    previous_map: Dict[str, Dict[str, Any]] = {}
    for url, entry in raw_previous.items():
        if not isinstance(entry, dict):
            continue
        etag = entry.get("etag")
        last_modified = entry.get("last_modified")
        path = entry.get("path")
        sha256 = entry.get("sha256")
        content_length = entry.get("content_length")
        content_length_value: Optional[int]
        if isinstance(content_length, str):
            try:
                content_length_value = int(content_length)
            except ValueError:
                content_length_value = None
        else:
            content_length_value = content_length

        if etag or last_modified:
            missing_fields = []
            if not path:
                missing_fields.append("path")
            if not sha256:
                missing_fields.append("sha256")
            if content_length_value is None:
                missing_fields.append("content_length")
            if missing_fields:
                LOGGER.warning(
                    "resume-metadata-incomplete: dropping cached metadata",
                    extra={
                        "extra_fields": {
                            "work_id": artifact.work_id,
                            "url": url,
                            "missing_fields": missing_fields,
                        }
                    },
                )
                continue
        previous_map[url] = {
            "etag": etag,
            "last_modified": last_modified,
            "path": path,
            "sha256": sha256,
            "content_length": content_length_value,
        }
    download_context = {
        "dry_run": dry_run,
        "extract_html_text": extract_html_text,
        "previous": previous_map,
        "list_only": list_only,
        "max_bytes": max_bytes,
    }

    if artifact.work_id in resume_completed:
        LOGGER.info("Skipping %s (already completed)", artifact.work_id)
        skipped_outcome = DownloadOutcome(
            classification=Classification.SKIPPED,
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
            classification=Classification.EXISTS,
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

    if list_only:
        result["skipped"] = True
        return result

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
        elif pipeline_result.outcome.classification is Classification.HTML:
            result["html_only"] = True
        return result

    reason = pipeline_result.reason or (
        pipeline_result.outcome.error if pipeline_result.outcome else "no-resolver-success"
    )
    outcome = pipeline_result.outcome or DownloadOutcome(
        classification=Classification.MISS,
        path=None,
        http_status=None,
        content_type=None,
        elapsed_ms=None,
        error=reason,
    )
    logger.log_attempt(
        AttemptRecord(
            work_id=artifact.work_id,
            resolver_name="final",
            resolver_order=None,
            url=pipeline_result.url,
            status=outcome.classification.value,
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


# --- CLI Entry Points ---

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
    parser.add_argument(
        "--staging",
        action="store_true",
        help="Create timestamped run directories under --out with separate PDF and HTML folders.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to manifest JSONL log.",
    )
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
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=None,
        help="Maximum bytes to download per request before aborting (default: unlimited).",
    )

    resolver_group = parser.add_argument_group("Resolver settings")
    resolver_group.add_argument(
        "--resolver-config", type=str, default=None, help="Path to resolver config (YAML/JSON)."
    )
    resolver_group.add_argument(
        "--resolver-order",
        type=str,
        default=None,
        help="Comma-separated resolver order override (e.g., 'unpaywall,crossref').",
    )
    resolver_group.add_argument(
        "--unpaywall-email", type=str, default=None, help="Override Unpaywall email credential."
    )
    resolver_group.add_argument(
        "--core-api-key", type=str, default=None, help="CORE API key override."
    )
    resolver_group.add_argument(
        "--semantic-scholar-api-key",
        type=str,
        default=None,
        help="Semantic Scholar Graph API key override.",
    )
    resolver_group.add_argument(
        "--doaj-api-key", type=str, default=None, help="DOAJ API key override."
    )
    resolver_group.add_argument(
        "--disable-resolver",
        action="append",
        default=[],
        help="Disable a resolver by name (can be repeated).",
    )
    resolver_group.add_argument(
        "--enable-resolver",
        action="append",
        default=[],
        help="Enable a resolver by name (can be repeated).",
    )
    resolver_group.add_argument(
        "--max-resolver-attempts",
        type=int,
        default=None,
        help="Maximum resolver attempts per work.",
    )
    resolver_group.add_argument(
        "--resolver-timeout",
        type=float,
        default=None,
        help="Default timeout (seconds) for resolver HTTP requests.",
    )
    resolver_group.add_argument(
        "--concurrent-resolvers",
        type=int,
        default=None,
        help="Maximum resolver threads per work item (default: 1).",
    )
    resolver_group.add_argument(
        "--global-url-dedup",
        dest="global_url_dedup",
        action="store_true",
        help="Skip downloads when a URL was already fetched in this run.",
    )
    resolver_group.add_argument(
        "--no-global-url-dedup",
        dest="global_url_dedup",
        action="store_false",
        help="Disable global URL deduplication (default).",
    )
    resolver_group.add_argument(
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
    resolver_group.add_argument(
        "--head-precheck",
        dest="head_precheck",
        action="store_true",
        help="Enable resolver HEAD preflight filtering (default).",
    )
    resolver_group.add_argument(
        "--no-head-precheck",
        dest="head_precheck",
        action="store_false",
        help="Disable resolver HEAD preflight filtering.",
    )
    resolver_group.add_argument(
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
        "--log-csv",
        type=Path,
        default=None,
        help="Optional CSV attempts log output path.",
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
        "--list-only",
        action="store_true",
        help="Discover candidate URLs but do not fetch content.",
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
    if args.max_bytes is not None and args.max_bytes <= 0:
        parser.error("--max-bytes must be a positive integer")

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

    base_pdf_dir = args.out
    manifest_override = args.manifest
    if args.staging:
        run_dir = base_pdf_dir / datetime.now(UTC).strftime("%Y%m%d_%H%M")
        pdf_dir = run_dir / "PDF"
        html_dir = run_dir / "HTML"
        manifest_path = run_dir / "manifest.jsonl"
        if args.html_out:
            LOGGER.info("Staging mode overrides --html-out; using %s", html_dir)
        if manifest_override:
            LOGGER.info("Staging mode overrides --manifest; writing to %s", manifest_path)
    else:
        pdf_dir = base_pdf_dir
        html_dir = args.html_out or (pdf_dir.parent / "HTML")
        manifest_path = manifest_override or (pdf_dir / "manifest.jsonl")
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

    try:
        config = load_resolver_config(args, resolver_names, resolver_order_override)
    except ValueError as exc:
        parser.error(str(exc))
    concurrency_product = max(args.workers, 1) * max(config.max_concurrent_resolvers, 1)
    if concurrency_product > 32:
        LOGGER.warning(
            "High parallelism detected (workers x concurrent_resolvers = %s). "
            "Ensure resolver and domain rate limits are configured appropriately.",
            concurrency_product,
        )

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

    with contextlib.ExitStack() as stack:
        jsonl_sink = stack.enter_context(JsonlSink(manifest_path))
        if csv_path:
            csv_sink = stack.enter_context(CsvSink(csv_path))
            attempt_logger = MultiSink([jsonl_sink, csv_sink])
        else:
            attempt_logger = jsonl_sink

        resume_lookup, resume_completed = load_previous_manifest(args.resume_from)
        metrics = ResolverMetrics()
        pipeline = ResolverPipeline(
            resolvers=resolver_instances,
            config=config,
            download_func=download_candidate,
            logger=attempt_logger,
            metrics=metrics,
        )

        def _session_factory() -> requests.Session:
            """Return a new :class:`requests.Session` using the run's polite headers.

            The factory is invoked by worker threads to obtain an isolated session
            that inherits the resolver configuration's polite identification
            headers. Creating sessions through this helper ensures each worker
            reuses the shared retry configuration while keeping connection pools
            thread-local.
            """

            pool_connections = max(64, concurrency_product * 4)
            pool_maxsize = max(128, concurrency_product * 8)
            return _make_session(
                config.polite_headers,
                pool_connections=pool_connections,
                pool_maxsize=pool_maxsize,
            )

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
                            list_only=args.list_only,
                            extract_html_text=args.extract_html_text,
                            previous_lookup=resume_lookup,
                            resume_completed=resume_completed,
                            max_bytes=args.max_bytes,
                        )
                        _record_result(result)
                        if args.sleep > 0:
                            time.sleep(args.sleep)
                finally:
                    if hasattr(session, "close"):
                        session.close()
            else:
                with ThreadPoolExecutor(max_workers=args.workers) as executor:
                    in_flight: List[Future[Dict[str, Any]]] = []
                    max_in_flight = max(args.workers * 2, 1)

                    def _submit(work_item: Dict[str, Any]) -> Future[Dict[str, Any]]:
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
                                    list_only=args.list_only,
                                    extract_html_text=args.extract_html_text,
                                    previous_lookup=resume_lookup,
                                    resume_completed=resume_completed,
                                    max_bytes=args.max_bytes,
                                )
                            finally:
                                if hasattr(session, "close"):
                                    session.close()

                        return executor.submit(_runner)

                    for work in iterate_openalex(
                        query, per_page=args.per_page, max_results=args.max
                    ):
                        if len(in_flight) >= max_in_flight:
                            done, pending = wait(set(in_flight), return_when=FIRST_COMPLETED)
                            for completed_future in done:
                                _record_result(completed_future.result())
                            in_flight = list(pending)
                        in_flight.append(_submit(work))

                    if in_flight:
                        for future in as_completed(list(in_flight)):
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
    print(
        f"\nDone. Processed {processed} works, saved {saved} PDFs, HTML-only {html_only}, skipped {skipped}."
    )
    if args.dry_run:
        print("DRY RUN: no files written, resolver coverage only.")
    print("Resolver summary:")
    for key in ("attempts", "successes", "html", "skips", "failures"):
        values = summary.get(key, {})
        if values:
            print(f"  {key}: {values}")
    latency_summary = summary.get("latency_ms", {})
    if latency_summary:
        print("  latency_ms:")
        for resolver_name, stats in latency_summary.items():
            mean_ms = stats.get("mean_ms", 0.0)
            p95_ms = stats.get("p95_ms", 0.0)
            max_ms = stats.get("max_ms", 0.0)
            count = stats.get("count", 0)
            print(
                f"    {resolver_name}: count={count} mean={mean_ms:.1f}ms p95={p95_ms:.1f}ms max={max_ms:.1f}ms"
            )
    status_counts = summary.get("status_counts", {})
    if status_counts:
        print("  status_counts:")
        for resolver_name, counts in status_counts.items():
            print(f"    {resolver_name}: {counts}")
    error_reasons = summary.get("error_reasons", {})
    if error_reasons:
        print("  top_error_reasons:")
        for resolver_name, items in error_reasons.items():
            formatted = ", ".join(f"{entry['reason']} ({entry['count']})" for entry in items)
            print(f"    {resolver_name}: {formatted}")

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
