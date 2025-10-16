#!/usr/bin/env python3
# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.download_pyalex_pdfs",
#   "purpose": "OpenAlex PDF download CLI and supporting utilities",
#   "sections": [
#     {
#       "id": "has-pdf-eof",
#     {
#       "id": "ensure-dir",
#       "name": "ensure_dir",
#       "anchor": "function-ensure-dir",
#       "kind": "function"
#     },
#     {
#       "id": "workartifact",
#       "name": "WorkArtifact",
#       "anchor": "class-workartifact",
#       "kind": "class"
#     },
#     {
#       "id": "downloadstate",
#       "name": "DownloadState",
#       "anchor": "class-downloadstate",
#       "kind": "class"
#     },
#     {
#       "id": "build-download-outcome",
#       "name": "_build_download_outcome",
#       "anchor": "function-build-download-outcome",
#       "kind": "function"
#     },
#     {
#       "id": "parse-domain-interval",
#       "name": "_parse_domain_interval",
#       "anchor": "function-parse-domain-interval",
#       "kind": "function"
#     },
#     {
#       "id": "robotscache",
#       "name": "RobotsCache",
#       "anchor": "class-robotscache",
#       "kind": "class"
#     },
#     {
#       "id": "parse-budget",
#       "name": "_parse_budget",
#       "anchor": "function-parse-budget",
#       "kind": "function"
#     },
#     {
#       "id": "collect-location-urls",
#       "name": "_collect_location_urls",
#       "anchor": "function-collect-location-urls",
#       "kind": "function"
#     },
#     {
#       "id": "build-query",
#       "name": "build_query",
#       "anchor": "function-build-query",
#       "kind": "function"
#     },
#     {
#       "id": "lookup-topic-id",
#       "name": "_lookup_topic_id",
#       "anchor": "function-lookup-topic-id",
#       "kind": "function"
#     },
#     {
#       "id": "resolve-topic-id-if-needed",
#       "name": "resolve_topic_id_if_needed",
#       "anchor": "function-resolve-topic-id-if-needed",
#       "kind": "function"
#     },
#     {
#       "id": "create-artifact",
#       "name": "create_artifact",
#       "anchor": "function-create-artifact",
#       "kind": "function"
#     },
#     {
#       "id": "download-candidate",
#       "name": "download_candidate",
#       "anchor": "function-download-candidate",
#       "kind": "function"
#     },
#     {
#       "id": "iterate-openalex",
#       "name": "iterate_openalex",
#       "anchor": "function-iterate-openalex",
#       "kind": "function"
#     },
#     {
#       "id": "process-one-work",
#       "name": "process_one_work",
#       "anchor": "function-process-one-work",
#       "kind": "function"
#     },
#     {
#       "id": "main",
#       "name": "main",
#       "anchor": "function-main",
#       "kind": "function"
#     }
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
import shutil
import threading
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
)
from urllib.parse import urlparse, urlsplit
from urllib.robotparser import RobotFileParser

import requests
from pyalex import Topics, Works
from pyalex import config as oa_config

from DocsToKG.ContentDownload import resolvers
from DocsToKG.ContentDownload.classifications import PDF_LIKE, Classification, ReasonCode
from DocsToKG.ContentDownload.classifier import (
    _infer_suffix,
    classify_payload,
    has_pdf_eof,
    tail_contains_html,
    update_tail_buffer,
)
from DocsToKG.ContentDownload.config import (
    apply_config_overrides,
    load_resolver_config,
    read_resolver_config,
)
from DocsToKG.ContentDownload.network import (
    CachedResult,
    ConditionalRequestHelper,
    ModifiedResult,
    create_session,
    head_precheck,
    request_with_retries,
)
from DocsToKG.ContentDownload.telemetry import (
    MANIFEST_SCHEMA_VERSION,
    AttemptSink,
    CsvSink,
    JsonlSink,
    LastAttemptCsvSink,
    ManifestEntry,
    ManifestIndexSink,
    MultiSink,
    SqliteSink,
    SummarySink,
    build_manifest_entry,
    load_manifest_url_index,
    load_previous_manifest,
)
from DocsToKG.ContentDownload.utils import (
    dedupe,
    normalize_doi,
    normalize_pmcid,
    normalize_url,
    slugify,
)
from DocsToKG.ContentDownload.utils import (
    normalize_arxiv as _normalize_arxiv,
)
from DocsToKG.ContentDownload.utils import (
    normalize_pmid as _normalize_pmid,
)

ResolverPipeline = resolvers.ResolverPipeline
ResolverConfig = resolvers.ResolverConfig
ResolverMetrics = resolvers.ResolverMetrics
DownloadOutcome = resolvers.DownloadOutcome
AttemptRecord = resolvers.AttemptRecord
default_resolvers = resolvers.default_resolvers

# --- Globals ---

DEFAULT_SNIFF_BYTES = 64 * 1024
DEFAULT_MIN_PDF_BYTES = 1024
DEFAULT_TAIL_CHECK_BYTES = 2048

__all__ = (
    "AttemptSink",
    "CsvSink",
    "DEFAULT_MIN_PDF_BYTES",
    "DEFAULT_SNIFF_BYTES",
    "DEFAULT_TAIL_CHECK_BYTES",
    "DownloadState",
    "JsonlSink",
    "LastAttemptCsvSink",
    "ManifestEntry",
    "ManifestIndexSink",
    "MultiSink",
    "MANIFEST_SCHEMA_VERSION",
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

    reason_code: Optional[ReasonCode] = None
    reason_detail: Optional[str] = None
    if flagged_unknown and classification_code is Classification.PDF:
        reason_code = ReasonCode.PDF_SNIFF_UNKNOWN
        reason_detail = "pdf-sniff-unknown"

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
                classification=Classification.MISS,
                path=None,
                http_status=response.status_code,
                content_type=response.headers.get("Content-Type"),
                elapsed_ms=elapsed_ms,
                reason=ReasonCode.PDF_TOO_SMALL,
                reason_detail="pdf-too-small",
                sha256=None,
                content_length=None,
                etag=etag,
                last_modified=last_modified,
                extracted_text_path=extracted_text_path,
            )

        if tail_contains_html(tail_bytes):
            with contextlib.suppress(OSError):
                dest_path.unlink()
            return DownloadOutcome(
                classification=Classification.MISS,
                path=None,
                http_status=response.status_code,
                content_type=response.headers.get("Content-Type"),
                elapsed_ms=elapsed_ms,
                reason=ReasonCode.HTML_TAIL_DETECTED,
                reason_detail="html-tail-detected",
                sha256=None,
                content_length=None,
                etag=etag,
                last_modified=last_modified,
                extracted_text_path=extracted_text_path,
            )

        if not has_pdf_eof(dest_path, window_bytes=tail_check_bytes):
            with contextlib.suppress(OSError):
                dest_path.unlink()
            return DownloadOutcome(
                classification=Classification.MISS,
                path=None,
                http_status=response.status_code,
                content_type=response.headers.get("Content-Type"),
                elapsed_ms=elapsed_ms,
                reason=ReasonCode.PDF_EOF_MISSING,
                reason_detail="pdf-eof-missing",
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
        reason=reason_code,
        reason_detail=reason_detail,
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


class RobotsCache:
    """Cache robots.txt policies per host and evaluate allowed URLs."""

    def __init__(self, user_agent: str) -> None:
        self._user_agent = user_agent
        self._parsers: Dict[str, RobotFileParser] = {}
        self._lock = threading.Lock()

    def is_allowed(self, session: requests.Session, url: str, timeout: float) -> bool:
        """Return ``False`` when robots.txt forbids fetching ``url``."""

        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return True
        origin = f"{parsed.scheme}://{parsed.netloc}"
        with self._lock:
            parser = self._parsers.get(origin)
        if parser is None:
            parser = self._fetch(session, origin, timeout)
            with self._lock:
                self._parsers[origin] = parser

        path = parsed.path or "/"
        if parsed.params:
            path = f"{path};{parsed.params}"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        try:
            return parser.can_fetch(self._user_agent, path or "/")
        except Exception:
            return True

    def _fetch(self, session: requests.Session, origin: str, timeout: float) -> RobotFileParser:
        """Fetch and parse the robots.txt policy for ``origin``."""

        robots_url = origin.rstrip("/") + "/robots.txt"
        parser = RobotFileParser()
        parser.set_url(robots_url)
        try:
            with request_with_retries(
                session,
                "GET",
                robots_url,
                timeout=min(timeout, 5.0),
                allow_redirects=True,
                max_retries=1,
            ) as response:
                if response.status_code and response.status_code >= 400:
                    parser.parse([])
                else:
                    body = response.text or ""
                    parser.parse(body.splitlines())
        except Exception:
            parser.parse([])
        return parser


def _parse_budget(value: str) -> Tuple[str, int]:
    """Parse ``requests=N`` or ``bytes=N`` budget specifications."""

    if "=" not in value:
        raise argparse.ArgumentTypeError("budget must use the format kind=value")
    kind, raw_amount = value.split("=", 1)
    key = kind.strip().lower()
    if key not in {"requests", "bytes"}:
        raise argparse.ArgumentTypeError("budget kind must be 'requests' or 'bytes'")
    amount_text = raw_amount.strip().lower().replace(",", "").replace("_", "")
    multiplier = 1
    if amount_text.endswith("kb"):
        multiplier = 1024
        amount_text = amount_text[:-2]
    elif amount_text.endswith("mb"):
        multiplier = 1024**2
        amount_text = amount_text[:-2]
    elif amount_text.endswith("gb"):
        multiplier = 1024**3
        amount_text = amount_text[:-2]
    elif amount_text.endswith("k"):
        multiplier = 1000
        amount_text = amount_text[:-1]
    elif amount_text.endswith("m"):
        multiplier = 1000**2
        amount_text = amount_text[:-1]
    elif amount_text.endswith("g"):
        multiplier = 1000**3
        amount_text = amount_text[:-1]
    try:
        amount = int(amount_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid budget value: {raw_amount}") from exc
    if amount <= 0:
        raise argparse.ArgumentTypeError("budget value must be positive")
    return key, amount * multiplier


def _apply_content_addressed_storage(dest_path: Path, sha256: str) -> Path:
    """Move `dest_path` into a content-addressed location and create a symlink."""

    hashed_dir = dest_path.parent / sha256[:2]
    hashed_dir.mkdir(parents=True, exist_ok=True)
    hashed_path = hashed_dir / f"{sha256}{dest_path.suffix}"
    if dest_path.exists():
        if hashed_path.exists():
            with contextlib.suppress(OSError):
                dest_path.unlink()
        else:
            os.replace(dest_path, hashed_path)
    if not hashed_path.exists():
        return dest_path
    try:
        with contextlib.suppress(FileNotFoundError):
            dest_path.unlink()
        dest_path.symlink_to(hashed_path)
    except OSError:
        with contextlib.suppress(OSError):
            if dest_path.exists():
                dest_path.unlink()
        shutil.copy2(hashed_path, dest_path)
    return hashed_path


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
    sniff_limit = max(int(context.get("sniff_bytes", DEFAULT_SNIFF_BYTES)), 0)
    min_pdf_bytes = max(int(context.get("min_pdf_bytes", DEFAULT_MIN_PDF_BYTES)), 0)
    tail_window_bytes = max(int(context.get("tail_check_bytes", DEFAULT_TAIL_CHECK_BYTES)), 0)
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
    base_headers = dict(headers)
    accept_overrides = context.get("host_accept_overrides", {})
    accept_value: Optional[str] = None
    if isinstance(accept_overrides, dict):
        host_for_accept = urlsplit(url).netloc.lower()
        if host_for_accept:
            accept_value = accept_overrides.get(host_for_accept)
            if accept_value is None and host_for_accept.startswith("www."):
                accept_value = accept_overrides.get(host_for_accept[4:])
    if accept_value:
        base_headers["Accept"] = str(accept_value)

    dry_run = bool(context.get("dry_run", False))
    robots_checker: Optional[RobotsCache] = context.get("robots_checker")
    if robots_checker is not None:
        robots_allowed = robots_checker.is_allowed(session, url, timeout)
        if not robots_allowed:
            LOGGER.info(
                "robots-disallowed",
                extra={
                    "extra_fields": {
                        "url": url,
                        "work_id": artifact.work_id,
                    }
                },
            )
            return DownloadOutcome(
                classification=Classification.SKIPPED,
                path=None,
                http_status=None,
                content_type=None,
                elapsed_ms=0.0,
                reason=ReasonCode.ROBOTS_DISALLOWED,
                reason_detail="robots-disallowed",
            )
    head_precheck_passed = head_precheck_passed or bool(context.get("head_precheck_passed", False))
    if not head_precheck_passed and not context.get("skip_head_precheck", False):
        head_precheck_passed = head_precheck(session, url, timeout)
        context["head_precheck_passed"] = head_precheck_passed
    extract_html_text = bool(context.get("extract_html_text", False))
    previous_map: Dict[str, Dict[str, Any]] = context.get("previous", {})
    global_index: Dict[str, Dict[str, Any]] = context.get("global_manifest_index", {})
    normalized_url = normalize_url(url)
    previous = previous_map.get(normalized_url) or global_index.get(normalized_url, {})
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
    content_type_hint = ""
    attempt_conditional = True
    logged_conditional_downgrade = False

    start = time.monotonic()
    try:
        while True:
            headers = dict(base_headers)
            if attempt_conditional:
                headers.update(cond_helper.build_headers())

            start = time.monotonic()
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
                    if not attempt_conditional:
                        LOGGER.warning(
                            "Received HTTP 304 for %s without conditional headers; treating as http_error.",
                            url,
                        )
                        return DownloadOutcome(
                            classification=Classification.HTTP_ERROR,
                            path=None,
                            http_status=response.status_code,
                            content_type=response.headers.get("Content-Type") or content_type_hint,
                            elapsed_ms=elapsed_ms,
                            reason=ReasonCode.UNEXPECTED_304,
                            reason_detail="unexpected-304",
                            sha256=None,
                            content_length=None,
                            etag=None,
                            last_modified=None,
                            extracted_text_path=None,
                        )

                    try:
                        cached = cond_helper.interpret_response(response)
                    except (FileNotFoundError, ValueError) as exc:
                        if not logged_conditional_downgrade:
                            LOGGER.warning(
                                "Conditional cache invalid for %s: %s. Refetching without conditional headers.",
                                url,
                                exc,
                                extra={
                                    "reason": "conditional-cache-invalid",
                                    "url": url,
                                    "work_id": artifact.work_id,
                                },
                            )
                            logged_conditional_downgrade = True
                        attempt_conditional = False
                        cond_helper = ConditionalRequestHelper()
                        continue

                    if not isinstance(cached, CachedResult):  # pragma: no cover - defensive
                        raise TypeError("Expected CachedResult for 304 response")
                    return DownloadOutcome(
                        classification=Classification.CACHED,
                        path=cached.path,
                        http_status=response.status_code,
                        content_type=response.headers.get("Content-Type") or content_type_hint,
                        elapsed_ms=elapsed_ms,
                        reason=ReasonCode.CONDITIONAL_NOT_MODIFIED,
                        reason_detail="not-modified",
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
                        reason=ReasonCode.HTTP_STATUS,
                        reason_detail=f"status-{response.status_code}",
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
                if (
                    max_bytes
                    and content_length_hint is not None
                    and content_length_hint > max_bytes
                ):
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
                        reason=ReasonCode.MAX_BYTES_HEADER,
                        reason_detail=(
                            f"content-length {content_length_hint} exceeds max_bytes {max_bytes}"
                        ),
                        sha256=None,
                        content_length=content_length_hint,
                        etag=modified_result.etag,
                        last_modified=modified_result.last_modified,
                        extracted_text_path=None,
                    )
                sniff_buffer = bytearray()
                detected: Classification = Classification.UNKNOWN
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
                            candidate = classify_payload(bytes(sniff_buffer), content_type, url)
                            if candidate is not Classification.UNKNOWN:
                                detected = candidate
                            if (
                                detected is Classification.UNKNOWN
                                and sniff_limit
                                and len(sniff_buffer) >= sniff_limit
                            ):
                                detected = Classification.PDF
                                flagged_unknown = True

                            if detected is not Classification.UNKNOWN:
                                if dry_run:
                                    break
                                default_suffix = (
                                    ".html" if detected == Classification.HTML else ".pdf"
                                )
                                suffix = _infer_suffix(
                                    url, content_type, disposition, detected, default_suffix
                                )
                                dest_dir = (
                                    artifact.html_dir
                                    if detected == Classification.HTML
                                    else artifact.pdf_dir
                                )
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
                                    update_tail_buffer(
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
                            update_tail_buffer(tail_buffer, chunk, limit=tail_window_bytes)

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
                                    reason=ReasonCode.MAX_BYTES_STREAM,
                                    reason_detail=f"download exceeded max_bytes {max_bytes}",
                                    sha256=None,
                                    content_length=byte_count,
                                    etag=modified_result.etag,
                                    last_modified=modified_result.last_modified,
                                    extracted_text_path=None,
                                )

                    if detected is Classification.UNKNOWN:
                        return DownloadOutcome(
                            classification=Classification.MISS,
                            path=None,
                            http_status=response.status_code,
                            content_type=content_type,
                            elapsed_ms=elapsed_ms,
                            reason=ReasonCode.UNKNOWN,
                            reason_detail="classifier-unknown",
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
            if (
                dest_path
                and context.get("content_addressed")
                and sha256
                and detected is Classification.PDF
            ):
                dest_path = _apply_content_addressed_storage(dest_path, sha256)
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
            classification=Classification.HTTP_ERROR,
            path=None,
            http_status=None,
            content_type=None,
            elapsed_ms=elapsed_ms,
            reason=ReasonCode.REQUEST_EXCEPTION,
            reason_detail=str(exc),
            sha256=None,
            content_length=None,
            etag=None,
            last_modified=None,
            extracted_text_path=None,
        )


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
    sniff_bytes: int,
    min_pdf_bytes: int,
    tail_check_bytes: int,
    robots_checker: Optional[RobotsCache] = None,
    content_addressed: bool = False,
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
        sniff_bytes: Number of leading bytes to buffer for payload inference.
        min_pdf_bytes: Minimum PDF size accepted when HEAD prechecks fail.
        tail_check_bytes: Tail window size used to detect embedded HTML payloads.
        robots_checker: Cache enforcing robots.txt policies when enabled.
        content_addressed: Whether to store PDFs under content-addressed paths.

    Returns:
        Dictionary summarizing the outcome (saved/html_only/skipped flags).

    Raises:
        requests.RequestException: Propagated if resolver HTTP requests fail
            unexpectedly outside guarded sections.
        Exception: Bubbling from resolver pipeline internals when not handled.
    """
    artifact = create_artifact(work, pdf_dir=pdf_dir, html_dir=html_dir)

    result = {
        "work_id": artifact.work_id,
        "saved": False,
        "html_only": False,
        "skipped": False,
        "downloaded_bytes": 0,
    }

    raw_previous = previous_lookup.get(artifact.work_id, {})
    previous_map: Dict[str, Dict[str, Any]] = {}
    for previous_url, entry in raw_previous.items():
        if not isinstance(entry, dict):
            continue
        normalized_url = normalize_url(previous_url)
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
                            "url": previous_url,
                            "missing_fields": missing_fields,
                        }
                    },
                )
                continue
        previous_map[normalized_url] = {
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
        "sniff_bytes": sniff_bytes,
        "min_pdf_bytes": min_pdf_bytes,
        "tail_check_bytes": tail_check_bytes,
        "robots_checker": robots_checker,
        "content_addressed": content_addressed,
    }

    if artifact.work_id in resume_completed:
        LOGGER.info("Skipping %s (already completed)", artifact.work_id)
        skipped_outcome = DownloadOutcome(
            classification=Classification.SKIPPED,
            path=None,
            http_status=None,
            content_type=None,
            elapsed_ms=None,
            reason=ReasonCode.RESUME_COMPLETE,
            reason_detail="resume-complete",
        )
        entry = build_manifest_entry(
            artifact,
            resolver="resume",
            url=None,
            outcome=skipped_outcome,
            html_paths=[],
            dry_run=dry_run,
            reason=ReasonCode.RESUME_COMPLETE,
            reason_detail="resume-complete",
        )
        logger.log_manifest(entry)
        result["skipped"] = True
        return result

    existing_pdf = artifact.pdf_dir / f"{artifact.base_stem}.pdf"
    if not dry_run and existing_pdf.exists():
        existing_outcome = DownloadOutcome(
            classification=Classification.CACHED,
            path=str(existing_pdf),
            http_status=None,
            content_type=None,
            elapsed_ms=None,
            reason=ReasonCode.ALREADY_DOWNLOADED,
            reason_detail="already-downloaded",
        )
        entry = build_manifest_entry(
            artifact,
            resolver="existing",
            url=None,
            outcome=existing_outcome,
            html_paths=[],
            dry_run=dry_run,
            reason=ReasonCode.ALREADY_DOWNLOADED,
            reason_detail="already-downloaded",
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
        length_hint = pipeline_result.outcome.content_length
        bytes_downloaded = 0
        if isinstance(length_hint, str):
            try:
                bytes_downloaded = int(length_hint)
            except ValueError:
                bytes_downloaded = 0
        elif isinstance(length_hint, int):
            bytes_downloaded = max(length_hint, 0)
        elif length_hint is not None:
            try:
                bytes_downloaded = int(length_hint)
            except (TypeError, ValueError):
                bytes_downloaded = 0
        if bytes_downloaded <= 0 and pipeline_result.outcome.path:
            with contextlib.suppress(OSError):
                bytes_downloaded = int(Path(pipeline_result.outcome.path).stat().st_size)
        result["downloaded_bytes"] = max(bytes_downloaded, 0)
        return result

    reason_code = pipeline_result.reason
    reason_detail = pipeline_result.reason_detail
    if pipeline_result.outcome and pipeline_result.outcome.reason and not reason_code:
        reason_code = pipeline_result.outcome.reason
    if pipeline_result.outcome and pipeline_result.outcome.reason_detail and not reason_detail:
        reason_detail = pipeline_result.outcome.reason_detail
    if reason_code is None:
        reason_code = ReasonCode.UNKNOWN
    if reason_detail is None:
        reason_detail = "no-resolver-success"
    outcome = pipeline_result.outcome or DownloadOutcome(
        classification=Classification.MISS,
        path=None,
        http_status=None,
        content_type=None,
        elapsed_ms=None,
        reason=reason_code,
        reason_detail=reason_detail,
    )
    bytes_hint = outcome.content_length
    if bytes_hint is not None:
        try:
            result["downloaded_bytes"] = max(int(bytes_hint), 0)
        except (TypeError, ValueError):
            pass
    logger.log_attempt(
        AttemptRecord(
            work_id=artifact.work_id,
            resolver_name="final",
            resolver_order=None,
            url=pipeline_result.url,
            status=outcome.classification,
            http_status=outcome.http_status,
            content_type=outcome.content_type,
            elapsed_ms=outcome.elapsed_ms,
            reason=reason_code,
            reason_detail=reason_detail,
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
        reason=reason_code,
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
        "--content-addressed",
        action="store_true",
        help="Store PDFs using content-addressed paths with friendly symlinks.",
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
    parser.add_argument(
        "--ignore-robots",
        action="store_true",
        help="Bypass robots.txt checks (defaults to respecting policies).",
    )
    parser.add_argument(
        "--budget",
        action="append",
        default=[],
        metavar="KIND=VALUE",
        help="Stop after reaching a budget (e.g., requests=1000, bytes=5GB). Option may repeat.",
    )

    classifier_group = parser.add_argument_group("Classifier settings")
    classifier_group.add_argument(
        "--sniff-bytes",
        type=int,
        default=DEFAULT_SNIFF_BYTES,
        help=f"Bytes to buffer before inferring payload type (default: {DEFAULT_SNIFF_BYTES}).",
    )
    classifier_group.add_argument(
        "--min-pdf-bytes",
        type=int,
        default=DEFAULT_MIN_PDF_BYTES,
        help=f"Minimum PDF size required when HEAD precheck fails (default: {DEFAULT_MIN_PDF_BYTES}).",
    )
    classifier_group.add_argument(
        "--tail-check-bytes",
        type=int,
        default=DEFAULT_TAIL_CHECK_BYTES,
        help=f"Tail window size used to detect embedded HTML (default: {DEFAULT_TAIL_CHECK_BYTES}).",
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
    for field_name in ("sniff_bytes", "min_pdf_bytes", "tail_check_bytes"):
        value = getattr(args, field_name, None)
        if value is not None and value < 0:
            parser.error(f"--{field_name.replace('_', '-')} must be non-negative")

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

    sqlite_path = manifest_path.with_suffix(".sqlite3")
    previous_url_index: Dict[str, Dict[str, Any]] = {}
    if config.enable_global_url_dedup:
        previous_url_index = load_manifest_url_index(sqlite_path)
    persistent_seen_urls: Set[str] = {
        url
        for url, meta in previous_url_index.items()
        if meta.get("path")
        and Path(str(meta["path"])).exists()
        and str(meta.get("classification", "")).lower()
        in {Classification.PDF.value, Classification.CACHED.value}
    }

    budget_requests: Optional[int] = None
    budget_bytes: Optional[int] = None
    robots_checker: Optional[RobotsCache] = None
    if not args.ignore_robots:
        user_agent = config.polite_headers.get("User-Agent", "DocsToKGDownloader/1.0")
        robots_checker = RobotsCache(user_agent)
    for spec in args.budget or []:
        try:
            kind, amount = _parse_budget(spec)
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))
        if kind == "requests":
            budget_requests = amount if budget_requests is None else min(budget_requests, amount)
        else:
            budget_bytes = amount if budget_bytes is None else min(budget_bytes, amount)

    summary: Dict[str, Any] = {}
    summary_record: Dict[str, Any] = {}
    processed = 0
    saved = 0
    html_only = 0
    skipped = 0
    total_downloaded_bytes = 0
    stop_due_to_budget = False

    with contextlib.ExitStack() as stack:
        sinks: List[AttemptSink] = []
        jsonl_sink = stack.enter_context(JsonlSink(manifest_path))
        sinks.append(jsonl_sink)
        index_path = manifest_path.with_suffix(".index.json")
        index_sink = stack.enter_context(ManifestIndexSink(index_path))
        sinks.append(index_sink)
        last_attempt_path = manifest_path.with_suffix(".last.csv")
        last_attempt_sink = stack.enter_context(LastAttemptCsvSink(last_attempt_path))
        sinks.append(last_attempt_sink)
        sqlite_sink = stack.enter_context(SqliteSink(sqlite_path))
        sinks.append(sqlite_sink)
        summary_path = manifest_path.with_suffix(".summary.json")
        summary_sink = stack.enter_context(SummarySink(summary_path))
        sinks.append(summary_sink)
        if csv_path:
            csv_sink = stack.enter_context(CsvSink(csv_path))
            sinks.append(csv_sink)
        attempt_logger: AttemptSink
        if len(sinks) == 1:
            attempt_logger = sinks[0]
        else:
            attempt_logger = MultiSink(sinks)

        resume_lookup, resume_completed = load_previous_manifest(args.resume_from)
        metrics = ResolverMetrics()
        pipeline = ResolverPipeline(
            resolvers=resolver_instances,
            config=config,
            download_func=download_candidate,
            logger=attempt_logger,
            metrics=metrics,
            initial_seen_urls=persistent_seen_urls if config.enable_global_url_dedup else None,
            global_manifest_index=previous_url_index if config.enable_global_url_dedup else {},
        )

        def _session_factory() -> requests.Session:
            """Return a new :class:`requests.Session` using the run's polite headers.

            The factory is invoked by worker threads to obtain an isolated session
            that inherits the resolver configuration's polite identification
            headers. Creating sessions through this helper ensures each worker
            reuses the shared retry configuration while keeping connection pools
            thread-local.
            """

            pool_connections = min(max(64, concurrency_product * 4), 512)
            pool_maxsize = min(max(128, concurrency_product * 8), 1024)
            return create_session(
                config.polite_headers,
                pool_connections=pool_connections,
                pool_maxsize=pool_maxsize,
            )

        def _record_result(res: Dict[str, Any]) -> None:
            """Update aggregate counters based on a single work result."""

            nonlocal processed, saved, html_only, skipped, total_downloaded_bytes
            processed += 1
            if res.get("saved"):
                saved += 1
            if res.get("html_only"):
                html_only += 1
            if res.get("skipped"):
                skipped += 1
            downloaded = res.get("downloaded_bytes") or 0
            try:
                total_downloaded_bytes += int(downloaded)
            except (TypeError, ValueError):
                pass

        def _should_stop() -> bool:
            if budget_requests is not None and processed >= budget_requests:
                return True
            if budget_bytes is not None and total_downloaded_bytes >= budget_bytes:
                return True
            return False

        try:
            if args.workers == 1:
                session = _session_factory()
                try:
                    for work in iterate_openalex(
                        query, per_page=args.per_page, max_results=args.max
                    ):
                        if stop_due_to_budget:
                            break
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
                            sniff_bytes=args.sniff_bytes,
                            min_pdf_bytes=args.min_pdf_bytes,
                            tail_check_bytes=args.tail_check_bytes,
                            robots_checker=robots_checker,
                            content_addressed=args.content_addressed,
                        )
                        _record_result(result)
                        if not stop_due_to_budget and _should_stop():
                            stop_due_to_budget = True
                            break
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
                                    sniff_bytes=args.sniff_bytes,
                                    min_pdf_bytes=args.min_pdf_bytes,
                                    tail_check_bytes=args.tail_check_bytes,
                                    robots_checker=robots_checker,
                                    content_addressed=args.content_addressed,
                                )
                            finally:
                                if hasattr(session, "close"):
                                    session.close()

                        return executor.submit(_runner)

                    for work in iterate_openalex(
                        query, per_page=args.per_page, max_results=args.max
                    ):
                        if stop_due_to_budget:
                            break
                        if len(in_flight) >= max_in_flight:
                            done, pending = wait(set(in_flight), return_when=FIRST_COMPLETED)
                            for completed_future in done:
                                _record_result(completed_future.result())
                                if not stop_due_to_budget and _should_stop():
                                    stop_due_to_budget = True
                            in_flight = list(pending)
                            if stop_due_to_budget:
                                break
                        if stop_due_to_budget:
                            break
                        in_flight.append(_submit(work))

                    if in_flight:
                        for future in as_completed(list(in_flight)):
                            _record_result(future.result())
                            if not stop_due_to_budget and _should_stop():
                                stop_due_to_budget = True
        except Exception:
            raise
        else:
            if stop_due_to_budget:
                LOGGER.info(
                    "Stopping due to budget exhaustion",
                    extra={
                        "extra_fields": {
                            "budget_requests": budget_requests,
                            "budget_bytes": budget_bytes,
                            "processed": processed,
                            "bytes_downloaded": total_downloaded_bytes,
                        }
                    },
                )
            summary = metrics.summary()
            reason_totals: Dict[str, int] = defaultdict(int)
            for items in summary.get("error_reasons", {}).values():
                for entry in items:
                    reason_totals[entry["reason"]] += entry["count"]
            classification_totals: Dict[str, int] = defaultdict(int)
            for counts in summary.get("status_counts", {}).values():
                for status, count in counts.items():
                    classification_totals[status] += count
            summary_record = {
                "processed": processed,
                "saved": saved,
                "html_only": html_only,
                "skipped": skipped,
                "bytes_downloaded": total_downloaded_bytes,
                "budget": {
                    "requests": budget_requests,
                    "bytes": budget_bytes,
                    "requests_consumed": processed,
                    "bytes_consumed": total_downloaded_bytes,
                    "exhausted": stop_due_to_budget,
                },
                "classification_totals": dict(classification_totals),
                "reason_totals": dict(reason_totals),
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
    print(f"Total bytes downloaded {total_downloaded_bytes}.")
    if stop_due_to_budget:
        print("Budget exhausted; halting further work.")
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
            "bytes_downloaded": total_downloaded_bytes,
            "budget": {
                "requests": budget_requests,
                "bytes": budget_bytes,
                "requests_consumed": processed,
                "bytes_consumed": total_downloaded_bytes,
                "exhausted": stop_due_to_budget,
            },
            "resolvers": summary,
        }
    LOGGER.info("resolver_run_summary %s", json.dumps(summary_record, sort_keys=True))


if __name__ == "__main__":
    main()
