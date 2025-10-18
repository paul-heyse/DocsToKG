"""Download orchestration helpers for the content acquisition pipeline.

This module coordinates the streaming download workflow, tying together
resolver outputs, HTTP policy enforcement, and telemetry reporting. It exposes
utilities that transform resolver candidates into stored artifacts while
respecting retry budgets, robots.txt directives, and classification rules.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Set, Union
from urllib.parse import urlparse, urlsplit
from urllib.robotparser import RobotFileParser

import requests

from DocsToKG.ContentDownload.core import (
    DEFAULT_MIN_PDF_BYTES,
    DEFAULT_SNIFF_BYTES,
    DEFAULT_TAIL_CHECK_BYTES,
    PDF_LIKE,
    Classification,
    DownloadContext,
    ReasonCode,
    WorkArtifact,
    _infer_suffix,
    atomic_write,
    atomic_write_text,
    classify_payload,
    dedupe,
    has_pdf_eof,
    normalize_doi,
    normalize_pmcid,
    normalize_reason,
    normalize_url,
    slugify,
    tail_contains_html,
    update_tail_buffer,
)
from DocsToKG.ContentDownload.core import normalize_arxiv as _normalize_arxiv
from DocsToKG.ContentDownload.core import normalize_pmid as _normalize_pmid
from DocsToKG.ContentDownload.errors import (
    log_download_failure,
)
from DocsToKG.ContentDownload.networking import (
    CachedResult,
    ConditionalRequestHelper,
    ContentPolicyViolation,
    ModifiedResult,
    head_precheck,
    parse_retry_after_header,
    request_with_retries,
)
from DocsToKG.ContentDownload.pipeline import (
    AttemptRecord,
    DownloadOutcome,
    ResolverMetrics,
    ResolverPipeline,
)
from DocsToKG.ContentDownload.telemetry import RunTelemetry

__all__ = [
    "ensure_dir",
    "DownloadOptions",
    "DownloadState",
    "RobotsCache",
    "create_artifact",
    "download_candidate",
    "process_one_work",
]

LOGGER = logging.getLogger("DocsToKG.ContentDownload")


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
class DownloadOptions:
    """Stable collection of per-run download settings applied to each work item."""

    dry_run: bool
    list_only: bool
    extract_html_text: bool
    run_id: str
    previous_lookup: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    resume_completed: Set[str] = field(default_factory=set)
    max_bytes: Optional[int] = None
    sniff_bytes: int = DEFAULT_SNIFF_BYTES
    min_pdf_bytes: int = DEFAULT_MIN_PDF_BYTES
    tail_check_bytes: int = DEFAULT_TAIL_CHECK_BYTES
    robots_checker: Optional["RobotsCache"] = None
    content_addressed: bool = False


class DownloadState(Enum):
    """State machine for streaming downloads."""

    PENDING = "pending"
    WRITING = "writing"


class _MaxBytesExceeded(RuntimeError):
    """Internal signal raised when the stream exceeds the configured byte budget."""


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
    retry_after: Optional[float] = None,
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

    if reason_code is None and classification_code in PDF_LIKE:
        reason_code = ReasonCode.CONDITIONAL_NOT_MODIFIED

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
                retry_after=retry_after,
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
                retry_after=retry_after,
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
                retry_after=retry_after,
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
        retry_after=retry_after,
    )


def _validate_cached_artifact(result: CachedResult) -> bool:
    """Return ``True`` when cached artefact metadata matches on-disk state."""

    if not result.path:
        LOGGER.warning(
            "Cached artifact missing path; cannot validate cache entry.",
            extra={"reason": "cached-path-missing"},
        )
        return False

    cached_path = Path(result.path)
    if not cached_path.exists():
        LOGGER.warning(
            "Cached artifact missing at %s; cannot reuse prior download.",
            cached_path,
            extra={"reason": "cached-missing", "path": result.path},
        )
        return False

    try:
        actual_size = cached_path.stat().st_size
    except OSError as exc:
        LOGGER.warning(
            "Unable to stat cached artifact at %s: %s",
            cached_path,
            exc,
            extra={"reason": "cached-stat-failed", "path": result.path},
        )
        return False
    if result.content_length is not None and actual_size != result.content_length:
        LOGGER.warning(
            "Cached content length mismatch: expected %s got %s",
            result.content_length,
            actual_size,
            extra={
                "reason": "cached-length-mismatch",
                "expected": result.content_length,
                "actual": actual_size,
                "path": result.path,
            },
        )
        return False

    if result.sha256:
        try:
            hasher = hashlib.sha256()
            with cached_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8192), b""):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
        except OSError as exc:
            LOGGER.warning(
                "Unable to read cached artifact at %s: %s",
                cached_path,
                exc,
                extra={"reason": "cached-read-failed", "path": result.path},
            )
            return False
        if digest != result.sha256:
            LOGGER.warning(
                "Cached SHA256 mismatch: expected %s got %s",
                result.sha256,
                digest,
                extra={
                    "reason": "cached-digest-mismatch",
                    "expected": result.sha256,
                    "actual": digest,
                    "path": result.path,
                },
            )
            return False

    return True


class RobotsCache:
    """Cache robots.txt policies per host and evaluate allowed URLs with TTL."""

    def __init__(self, user_agent: str, *, ttl_seconds: int = 3600) -> None:
        self._user_agent = user_agent
        self._parsers: Dict[str, RobotFileParser] = {}
        self._fetched_at: Dict[str, float] = {}
        self._ttl = float(ttl_seconds)
        self._lock = threading.Lock()

    def is_allowed(self, session: requests.Session, url: str, timeout: float) -> bool:
        """Return ``False`` when robots.txt forbids fetching ``url``."""

        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return True
        origin = f"{parsed.scheme}://{parsed.netloc}"
        parser = self._lookup_parser(session, origin, timeout)

        path = parsed.path or "/"
        if parsed.params:
            path = f"{path};{parsed.params}"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        try:
            return parser.can_fetch(self._user_agent, path or "/")
        except Exception:
            return True

    def _lookup_parser(
        self,
        session: requests.Session,
        origin: str,
        timeout: float,
    ) -> RobotFileParser:
        now = time.time()
        with self._lock:
            parser = self._parsers.get(origin)
            fetched_at = self._fetched_at.get(origin, 0.0)

        if parser is None or (self._ttl > 0 and (now - fetched_at) >= self._ttl):
            parser = self._fetch(session, origin, timeout)
            with self._lock:
                self._parsers[origin] = parser
                self._fetched_at[origin] = now

        return parser

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


def _cohort_order_for(artifact: WorkArtifact) -> List[str]:
    """Return a resolver order tailored to the artifact's identifiers."""

    sources = {s.strip() for s in (artifact.source_display_names or []) if s}
    paywalled_publishers = {
        "Elsevier",
        "Wiley",
        "IEEE",
        "ACM",
        "Taylor & Francis",
        "Springer",
        "SAGE",
    }

    if getattr(artifact, "pmcid", None):
        return [
            "pmc",
            "europe_pmc",
            "unpaywall",
            "crossref",
            "landing_page",
            "wayback",
        ]

    if getattr(artifact, "arxiv_id", None):
        return [
            "arxiv",
            "unpaywall",
            "crossref",
            "landing_page",
            "wayback",
        ]

    if sources & paywalled_publishers:
        return [
            "unpaywall",
            "core",
            "doaj",
            "crossref",
            "landing_page",
            "wayback",
        ]

    try:
        from DocsToKG.ContentDownload import pipeline as _pipeline

        default_order = getattr(_pipeline, "DEFAULT_RESOLVER_ORDER", None)
        if default_order:
            return list(default_order)
    except Exception:  # pragma: no cover - defensive
        pass

    return []


def create_artifact(
    work: Dict[str, Any],
    pdf_dir: Path,
    html_dir: Path,
    xml_dir: Path,
) -> WorkArtifact:
    """Normalize an OpenAlex work into a WorkArtifact instance.

    Args:
        work: Raw OpenAlex work payload.
        pdf_dir: Directory where PDFs should be stored.
        html_dir: Directory where HTML resources should be stored.
        xml_dir: Directory where XML resources should be stored.

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
        xml_dir=xml_dir,
        metadata={"openalex_id": work.get("id")},
    )
    return artifact


def download_candidate(
    session: requests.Session,
    artifact: WorkArtifact,
    url: str,
    referer: Optional[str],
    timeout: float,
    context: Optional[Union[DownloadContext, Mapping[str, Any]]] = None,
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
        context: :class:`DownloadContext` (or mapping convertible to it) containing
            ``dry_run``, ``extract_html_text``, prior manifest lookups, and optional
            ``progress_callback``.

            Progress callback signature: ``callback(bytes_downloaded, total_bytes, url)``
            where ``total_bytes`` may be ``None`` if Content-Length is unavailable.

    Returns:
        DownloadOutcome describing the result of the download attempt including
        streaming hash metadata when available.

    Notes:
        A lightweight HEAD preflight is issued when the caller has not already
        validated the URL. This mirrors the resolver pipeline behaviour and
        keeps dry-run tests deterministic.

        Progress callbacks are invoked approximately every 128KB to balance
        responsiveness with performance overhead.

    Raises:
        OSError: If writing the downloaded payload to disk fails.
        TypeError: If conditional response parsing returns unexpected objects.
    """
    ctx = DownloadContext.from_mapping(context)
    sniff_limit = ctx.sniff_bytes
    min_pdf_bytes = ctx.min_pdf_bytes
    tail_window_bytes = ctx.tail_check_bytes

    # Extract progress callback if provided
    progress_callback = ctx.progress_callback
    progress_update_interval = 128 * 1024  # Update progress every 128KB

    max_bytes: Optional[int] = ctx.max_bytes
    parsed_url = urlsplit(url)
    domain_policies: Dict[str, Dict[str, Any]] = ctx.domain_content_rules
    host_key = (parsed_url.hostname or parsed_url.netloc or "").lower()
    content_policy: Optional[Dict[str, Any]] = None
    if domain_policies and host_key:
        content_policy = domain_policies.get(host_key)
        if content_policy is None and host_key.startswith("www."):
            content_policy = domain_policies.get(host_key[4:])
    policy_max_bytes: Optional[int] = None
    if content_policy:
        raw_policy_limit = content_policy.get("max_bytes")
        if isinstance(raw_policy_limit, int) and raw_policy_limit > 0:
            policy_max_bytes = raw_policy_limit
            if max_bytes is None or raw_policy_limit < max_bytes:
                max_bytes = raw_policy_limit
    headers: Dict[str, str] = {}
    if referer:
        headers["Referer"] = referer
    base_headers = dict(headers)

    # Support for resuming partial downloads via HTTP Range requests
    # Note: Full resume support requires atomic_write to handle append mode
    # and rehashing from partial file state. Currently infrastructure is in place
    # but feature is disabled by default pending append-mode implementation.
    enable_resume = ctx.enable_range_resume
    resume_bytes_offset: Optional[int] = None
    resume_probe_path: Optional[Path] = None
    accept_overrides = ctx.host_accept_overrides
    accept_value: Optional[str] = None
    if isinstance(accept_overrides, dict):
        host_for_accept = (parsed_url.netloc or "").lower()
        if host_for_accept:
            accept_value = accept_overrides.get(host_for_accept)
            if accept_value is None and host_for_accept.startswith("www."):
                accept_value = accept_overrides.get(host_for_accept[4:])
    if accept_value:
        base_headers["Accept"] = str(accept_value)

    dry_run = ctx.dry_run
    robots_checker: Optional[RobotsCache] = ctx.robots_checker
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
    head_precheck_passed = head_precheck_passed or ctx.head_precheck_passed
    if not head_precheck_passed and not ctx.skip_head_precheck:
        head_precheck_passed = head_precheck(session, url, timeout, content_policy=content_policy)
        ctx.head_precheck_passed = head_precheck_passed
    extract_html_text = ctx.extract_html_text
    previous_map = ctx.previous
    global_index = ctx.global_manifest_index
    normalized_url = normalize_url(url)
    previous = previous_map.get(normalized_url) or global_index.get(normalized_url, {})
    previous_etag = previous.get("etag")
    previous_last_modified = previous.get("last_modified")
    existing_path = previous.get("path")
    previous_sha = previous.get("sha256")
    previous_length = previous.get("content_length")
    if isinstance(previous_length, str):
        try:
            previous_length = int(previous_length)
        except ValueError:
            previous_length = None

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
            resume_probe_path = None
            if (
                enable_resume
                and existing_path
                and not attempt_conditional
                and resume_bytes_offset is None
            ):
                try:
                    existing_file = Path(existing_path)
                    candidate_file = existing_file
                    if not candidate_file.exists():
                        candidate_part = existing_file.with_suffix(existing_file.suffix + ".part")
                        if candidate_part.exists():
                            candidate_file = candidate_part
                    if candidate_file.exists() and candidate_file.is_file():
                        file_size = candidate_file.stat().st_size
                        if (
                            file_size > 0
                            and isinstance(previous_length, int)
                            and file_size < previous_length
                        ):
                            resume_bytes_offset = file_size
                            resume_probe_path = candidate_file
                            LOGGER.debug(
                                "Resuming partial download from byte %d for %s (source=%s)",
                                resume_bytes_offset,
                                url,
                                candidate_file,
                            )
                except (OSError, ValueError) as exc:
                    LOGGER.debug("Could not check for partial download resume: %s", exc)
                    resume_bytes_offset = None

            retry_after_hint: Optional[float] = None
            headers = dict(base_headers)
            if attempt_conditional:
                headers.update(cond_helper.build_headers())
            elif resume_bytes_offset is not None and resume_bytes_offset > 0:
                # Add Range header to resume from offset
                headers["Range"] = f"bytes={resume_bytes_offset}-"

            start = time.monotonic()
            try:
                response_cm = request_with_retries(
                    session,
                    "GET",
                    url,
                    stream=True,
                    allow_redirects=True,
                    timeout=timeout,
                    headers=headers,
                    content_policy=content_policy,
                )
            except ContentPolicyViolation as exc:
                elapsed_ms = (time.monotonic() - start) * 1000.0
                violation = exc.violation
                reason_code = (
                    ReasonCode.DOMAIN_DISALLOWED_MIME
                    if violation == "content-type"
                    else ReasonCode.DOMAIN_MAX_BYTES
                )
                LOGGER.info(
                    "domain-content-policy-blocked",
                    extra={
                        "extra_fields": {
                            "url": url,
                            "work_id": artifact.work_id,
                            "violation": violation,
                            "detail": exc.detail,
                        }
                    },
                )
                return DownloadOutcome(
                    classification=Classification.SKIPPED,
                    path=None,
                    http_status=None,
                    content_type=exc.content_type,
                    elapsed_ms=elapsed_ms,
                    reason=reason_code,
                    reason_detail=exc.detail or violation,
                    sha256=None,
                    content_length=exc.content_length,
                    etag=None,
                    last_modified=None,
                    extracted_text_path=None,
                    retry_after=None,
                )

            with response_cm as response:
                elapsed_ms = (time.monotonic() - start) * 1000.0
                status_code = response.status_code or 0
                if status_code >= 400:
                    retry_after_hint = parse_retry_after_header(response)
                if response.status_code == 412:
                    if attempt_conditional:
                        if not logged_conditional_downgrade:
                            LOGGER.warning(
                                "Conditional validators rejected for %s; retrying without conditional headers.",
                                url,
                                extra={
                                    "reason": "conditional-precondition-failed",
                                    "url": url,
                                    "work_id": artifact.work_id,
                                },
                            )
                            logged_conditional_downgrade = True
                        attempt_conditional = False
                        cond_helper = ConditionalRequestHelper()
                        continue
                    return DownloadOutcome(
                        classification=Classification.HTTP_ERROR,
                        path=None,
                        http_status=response.status_code,
                        content_type=response.headers.get("Content-Type") or content_type_hint,
                        elapsed_ms=elapsed_ms,
                        reason=None,
                        reason_detail="precondition-failed",
                        sha256=None,
                        content_length=None,
                        etag=None,
                        last_modified=None,
                        extracted_text_path=None,
                        retry_after=retry_after_hint,
                    )
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
                            retry_after=retry_after_hint,
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
                    if not _validate_cached_artifact(cached):
                        if not logged_conditional_downgrade:
                            LOGGER.warning(
                                "Conditional cache invalid for %s; refetching without conditional headers.",
                                url,
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
                        retry_after=retry_after_hint,
                    )

                # Handle 206 Partial Content for resume support
                if response.status_code == 206:
                    if resume_bytes_offset is None or resume_bytes_offset <= 0:
                        LOGGER.warning(
                            "Received HTTP 206 for %s without Range request; treating as error.",
                            url,
                        )
                        return DownloadOutcome(
                            classification=Classification.HTTP_ERROR,
                            path=None,
                            http_status=response.status_code,
                            content_type=response.headers.get("Content-Type") or content_type_hint,
                            elapsed_ms=elapsed_ms,
                            reason=ReasonCode.UNKNOWN,
                            reason_detail="unexpected-206-partial-content",
                            sha256=None,
                            content_length=None,
                            etag=None,
                            last_modified=None,
                            extracted_text_path=None,
                            retry_after=retry_after_hint,
                        )
                    LOGGER.info(
                        "Resuming download from byte %d for %s",
                        resume_bytes_offset,
                        url,
                        extra={
                            "reason": "resume-complete",
                            "url": url,
                            "work_id": artifact.work_id,
                            "resume_offset": resume_bytes_offset,
                        },
                    )
                    # Continue with normal download flow, but append mode below

                elif response.status_code == 416:
                    if resume_bytes_offset:
                        LOGGER.warning(
                            "Server rejected resume Range request for %s at offset %d; retrying full download.",
                            url,
                            resume_bytes_offset,
                            extra={
                                "reason": "resume-range-not-satisfiable",
                                "url": url,
                                "work_id": artifact.work_id,
                                "resume_offset": resume_bytes_offset,
                            },
                        )
                    cleanup_targets: List[Path] = []
                    if existing_path:
                        cleanup_targets.append(Path(existing_path))
                        cleanup_targets.append(
                            Path(existing_path).with_suffix(Path(existing_path).suffix + ".part")
                        )
                    if resume_probe_path:
                        cleanup_targets.append(resume_probe_path)
                    seen: Set[Path] = set()
                    for candidate in cleanup_targets:
                        if candidate in seen:
                            continue
                        seen.add(candidate)
                        try:
                            candidate.unlink()
                        except FileNotFoundError:
                            continue
                        except OSError as exc:
                            LOGGER.debug("Failed to remove partial file %s: %s", candidate, exc)
                    existing_path = None
                    resume_bytes_offset = None
                    resume_probe_path = None
                    attempt_conditional = False
                    cond_helper = ConditionalRequestHelper()
                    continue

                elif response.status_code != 200:
                    return DownloadOutcome(
                        classification=Classification.HTTP_ERROR,
                        path=None,
                        http_status=response.status_code,
                        content_type=response.headers.get("Content-Type") or content_type_hint,
                        elapsed_ms=elapsed_ms,
                        reason=None,
                        reason_detail=None,
                        sha256=None,
                        content_length=None,
                        etag=None,
                        last_modified=None,
                        extracted_text_path=None,
                        retry_after=retry_after_hint,
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

                # Size warning for unexpectedly large downloads
                size_warning_threshold = ctx.size_warning_threshold
                if (
                    size_warning_threshold
                    and content_length_hint
                    and content_length_hint > size_warning_threshold
                ):
                    size_mb = content_length_hint / (1024 * 1024)
                    threshold_mb = size_warning_threshold / (1024 * 1024)
                    LOGGER.warning(
                        "Large download detected: %.1f MB (threshold: %.1f MB)",
                        size_mb,
                        threshold_mb,
                        extra={
                            "extra_fields": {
                                "url": url,
                                "work_id": artifact.work_id,
                                "content_length": content_length_hint,
                                "threshold": size_warning_threshold,
                                "size_mb": size_mb,
                            }
                        },
                    )
                    # Check if we should skip large downloads
                    if ctx.skip_large_downloads:
                        return DownloadOutcome(
                            classification=Classification.SKIPPED,
                            path=None,
                            http_status=response.status_code,
                            content_type=content_type,
                            elapsed_ms=elapsed_ms,
                            reason=ReasonCode.DOMAIN_MAX_BYTES,
                            reason_detail=f"file-too-large: {size_mb:.1f}MB exceeds threshold {threshold_mb:.1f}MB",
                            sha256=None,
                            content_length=content_length_hint,
                            etag=modified_result.etag,
                            last_modified=modified_result.last_modified,
                            extracted_text_path=None,
                            retry_after=retry_after_hint,
                        )
                if (
                    max_bytes is not None
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
                    policy_triggered = (
                        policy_max_bytes is not None and max_bytes == policy_max_bytes
                    )
                    reason_code = (
                        ReasonCode.DOMAIN_MAX_BYTES
                        if policy_triggered
                        else ReasonCode.MAX_BYTES_HEADER
                    )
                    reason_detail = (
                        f"content-length {content_length_hint} exceeds domain max_bytes {policy_max_bytes}"
                        if policy_triggered and policy_max_bytes is not None
                        else f"content-length {content_length_hint} exceeds max_bytes {max_bytes}"
                    )
                    return DownloadOutcome(
                        classification=classification_limit,
                        path=None,
                        http_status=response.status_code,
                        content_type=content_type,
                        elapsed_ms=elapsed_now,
                        reason=reason_code,
                        reason_detail=reason_detail,
                        sha256=None,
                        content_length=content_length_hint,
                        etag=modified_result.etag,
                        last_modified=modified_result.last_modified,
                        extracted_text_path=None,
                        retry_after=retry_after_hint,
                    )
                hasher = hashlib.sha256() if not dry_run else None
                byte_count = 0
                # Get configurable chunk size, defaulting to 32KB
                chunk_size = int(ctx.chunk_size or (1 << 15))
                content_iter = response.iter_content(chunk_size=chunk_size)
                sniff_buffer = bytearray()
                prefetched: List[bytes] = []
                detected: Classification = Classification.UNKNOWN
                flagged_unknown = False
                last_classification_size = 0

                # Optimization: Only classify when we have new meaningful data
                # Most formats can be detected in first 1-4KB
                min_classify_interval = 4096  # Re-classify every 4KB if still unknown

                for chunk in content_iter:
                    if not chunk:
                        continue
                    sniff_buffer.extend(chunk)
                    prefetched.append(chunk)

                    # Optimize: Only classify when buffer has grown enough or first time
                    buffer_size = len(sniff_buffer)
                    should_classify = last_classification_size == 0 or (  # First chunk
                        detected is Classification.UNKNOWN
                        and buffer_size >= last_classification_size + min_classify_interval
                    )

                    if should_classify:
                        candidate = classify_payload(bytes(sniff_buffer), content_type, url)
                        last_classification_size = buffer_size
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
                        break
                else:
                    return DownloadOutcome(
                        classification=Classification.MISS,
                        path=None,
                        http_status=response.status_code,
                        content_type=content_type,
                        elapsed_ms=elapsed_ms,
                        reason=ReasonCode.UNKNOWN,
                        reason_detail="classifier-unknown",
                        retry_after=retry_after_hint,
                    )

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
                    retry_after=retry_after_hint,
                )

            if detected == Classification.HTML:
                default_suffix = ".html"
                dest_dir = artifact.html_dir
            elif detected == Classification.XML:
                default_suffix = ".xml"
                dest_dir = artifact.xml_dir
            else:
                default_suffix = ".pdf"
                dest_dir = artifact.pdf_dir
            suffix = _infer_suffix(url, content_type, disposition, detected, default_suffix)
            dest_path = dest_dir / f"{artifact.base_stem}{suffix}"
            ensure_dir(dest_path.parent)

            # Optimization: Only maintain tail buffer for PDFs (used for corruption detection)
            # Skip for HTML/XML to reduce CPU overhead
            is_pdf = detected == Classification.PDF
            tail_buffer = bytearray() if is_pdf else None

            def _stream_chunks() -> Iterable[bytes]:
                """Optimized streaming: write prefetched chunks inline to avoid double storage."""
                nonlocal byte_count
                last_progress_update = 0

                for chunk in prefetched:
                    if not chunk:
                        continue
                    byte_count += len(chunk)
                    # Only update tail buffer for PDFs
                    if is_pdf and tail_buffer is not None:
                        update_tail_buffer(tail_buffer, chunk, limit=tail_window_bytes)
                    if max_bytes is not None and byte_count > max_bytes:
                        raise _MaxBytesExceeded()

                    # Progress tracking: update every 128KB to balance overhead
                    if (
                        progress_callback
                        and callable(progress_callback)
                        and byte_count - last_progress_update >= progress_update_interval
                    ):
                        try:
                            progress_callback(byte_count, content_length_hint, url)
                            last_progress_update = byte_count
                        except Exception as exc:
                            # Don't let callback failures break downloads
                            LOGGER.debug("Progress callback failed: %s", exc)

                    yield chunk

                for chunk in content_iter:
                    if not chunk:
                        continue
                    byte_count += len(chunk)
                    # Only update tail buffer for PDFs
                    if is_pdf and tail_buffer is not None:
                        update_tail_buffer(tail_buffer, chunk, limit=tail_window_bytes)
                    if max_bytes is not None and byte_count > max_bytes:
                        raise _MaxBytesExceeded()

                    # Progress tracking: update every 128KB
                    if (
                        progress_callback
                        and callable(progress_callback)
                        and byte_count - last_progress_update >= progress_update_interval
                    ):
                        try:
                            progress_callback(byte_count, content_length_hint, url)
                            last_progress_update = byte_count
                        except Exception as exc:
                            LOGGER.debug("Progress callback failed: %s", exc)

                    yield chunk

                # Final progress update at completion
                if progress_callback and callable(progress_callback) and byte_count > 0:
                    try:
                        progress_callback(byte_count, content_length_hint, url)
                    except Exception as exc:
                        LOGGER.debug("Final progress callback failed: %s", exc)

            try:
                atomic_write(
                    dest_path,
                    _stream_chunks(),
                    hasher=hasher,
                    keep_partial_on_error=True,
                )
            except _MaxBytesExceeded:
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
                classification_limit = (
                    Classification.HTML_TOO_LARGE
                    if detected is Classification.HTML
                    else Classification.PAYLOAD_TOO_LARGE
                )
                elapsed_limit = (time.monotonic() - start) * 1000.0
                policy_triggered = (
                    policy_max_bytes is not None
                    and max_bytes is not None
                    and max_bytes == policy_max_bytes
                )
                reason_code = (
                    ReasonCode.DOMAIN_MAX_BYTES if policy_triggered else ReasonCode.MAX_BYTES_STREAM
                )
                reason_detail = (
                    f"download exceeded domain max_bytes {policy_max_bytes}"
                    if policy_triggered and policy_max_bytes is not None
                    else f"download exceeded max_bytes {max_bytes}"
                )
                return DownloadOutcome(
                    classification=classification_limit,
                    path=None,
                    http_status=response.status_code,
                    content_type=content_type,
                    elapsed_ms=elapsed_limit,
                    reason=reason_code,
                    reason_detail=reason_detail,
                    sha256=None,
                    content_length=byte_count,
                    etag=modified_result.etag,
                    last_modified=modified_result.last_modified,
                    extracted_text_path=None,
                    retry_after=retry_after_hint,
                )
            except (requests.exceptions.ChunkedEncodingError, AttributeError) as exc:
                LOGGER.warning(
                    "Streaming download failed for %s: %s",
                    url,
                    exc,
                    extra={"extra_fields": {"work_id": artifact.work_id}},
                )
                seen_attempts = ctx.stream_retry_attempts
                if seen_attempts < 1:
                    ctx.stream_retry_attempts = seen_attempts + 1
                    attempt_conditional = False
                    LOGGER.info("Retrying download for %s after stream failure", url)
                    continue
                elapsed_err = (time.monotonic() - start) * 1000.0
                return DownloadOutcome(
                    classification=Classification.HTTP_ERROR,
                    path=None,
                    http_status=response.status_code,
                    content_type=content_type,
                    elapsed_ms=elapsed_err,
                    reason=ReasonCode.REQUEST_EXCEPTION,
                    reason_detail=f"stream-error: {exc}",
                    sha256=None,
                    content_length=None,
                    etag=modified_result.etag,
                    last_modified=modified_result.last_modified,
                    extracted_text_path=None,
                    retry_after=retry_after_hint,
                )

            sha256: Optional[str] = None
            content_length: Optional[int] = None
            if hasher is not None:
                sha256 = hasher.hexdigest()
                content_length = byte_count
            if dest_path and ctx.content_addressed and sha256 and detected is Classification.PDF:
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
                            atomic_write_text(text_path, text)
                            extracted_text_path = str(text_path)

            ctx.stream_retry_attempts = 0
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
                retry_after=retry_after_hint,
            )
    except requests.RequestException as exc:
        elapsed_ms = (time.monotonic() - start) * 1000.0

        # Log failure with actionable error message
        http_status = (
            getattr(exc.response, "status_code", None) if hasattr(exc, "response") else None
        )
        log_download_failure(
            LOGGER,
            url,
            artifact.work_id,
            http_status=http_status,
            reason_code="request_exception",
            error_details=str(exc),
            exception=exc,
        )

        return DownloadOutcome(
            classification=Classification.HTTP_ERROR,
            path=None,
            http_status=http_status,
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


def process_one_work(
    work: Union[WorkArtifact, Dict[str, Any]],
    session: requests.Session,
    pdf_dir: Path,
    html_dir: Path,
    xml_dir: Path,
    pipeline: ResolverPipeline,
    logger: RunTelemetry,
    metrics: ResolverMetrics,
    *,
    options: DownloadOptions,
    session_factory: Optional[Callable[[], requests.Session]] = None,
) -> Dict[str, Any]:
    """Process a single work artifact through the resolver pipeline.

    Args:
        work: Either a preconstructed :class:`WorkArtifact` or a raw OpenAlex
            work payload. Raw payloads are normalised via :func:`create_artifact`.
        session: Requests session configured for resolver usage.
        pdf_dir: Directory where PDF artefacts are written.
        html_dir: Directory where HTML artefacts are written.
        xml_dir: Directory where XML artefacts are written.
        pipeline: Resolver pipeline orchestrating downstream resolvers.
        logger: Structured attempt logger capturing manifest records.
        metrics: Resolver metrics collector.
        options: :class:`DownloadOptions` describing download behaviour for the work.
        session_factory: Optional callable returning a thread-local session for
            resolver execution when concurrency is enabled.

    Returns:
        Dictionary summarizing the outcome (saved/html_only/skipped flags).

    Raises:
        requests.RequestException: Propagated if resolver HTTP requests fail
            unexpectedly outside guarded sections.
        Exception: Bubbling from resolver pipeline internals when not handled.
    """
    if isinstance(work, WorkArtifact):
        artifact = work
    else:
        artifact = create_artifact(work, pdf_dir=pdf_dir, html_dir=html_dir, xml_dir=xml_dir)
    run_id = options.run_id
    dry_run = options.dry_run
    list_only = options.list_only
    extract_html_text = options.extract_html_text
    previous_lookup = options.previous_lookup
    resume_completed = options.resume_completed
    max_bytes = options.max_bytes
    sniff_bytes = options.sniff_bytes
    min_pdf_bytes = options.min_pdf_bytes
    tail_check_bytes = options.tail_check_bytes
    robots_checker = options.robots_checker
    content_addressed = options.content_addressed

    result = {
        "work_id": artifact.work_id,
        "saved": False,
        "html_only": False,
        "xml_only": False,
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
    download_context = DownloadContext(
        dry_run=dry_run,
        extract_html_text=extract_html_text,
        previous=previous_map,
        list_only=list_only,
        max_bytes=max_bytes,
        sniff_bytes=sniff_bytes,
        min_pdf_bytes=min_pdf_bytes,
        tail_check_bytes=tail_check_bytes,
        robots_checker=robots_checker,
        content_addressed=content_addressed,
    )
    download_context.mark_explicit(
        "dry_run",
        "extract_html_text",
        "previous",
        "list_only",
        "max_bytes",
        "sniff_bytes",
        "min_pdf_bytes",
        "tail_check_bytes",
        "robots_checker",
        "content_addressed",
    )

    cohort_order = _cohort_order_for(artifact)
    if cohort_order:
        download_context.resolver_order = list(dict.fromkeys(cohort_order))
        download_context.mark_explicit("resolver_order")

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
        logger.record_manifest(
            artifact,
            resolver="resume",
            url=None,
            outcome=skipped_outcome,
            html_paths=[],
            dry_run=dry_run,
            run_id=run_id,
            reason=ReasonCode.RESUME_COMPLETE,
            reason_detail="resume-complete",
        )
        result["skipped"] = True
        return result

    existing_pdf = artifact.pdf_dir / f"{artifact.base_stem}.pdf"
    existing_xml = artifact.xml_dir / f"{artifact.base_stem}.xml"
    if not dry_run and (existing_pdf.exists() or existing_xml.exists()):
        cached_path = existing_pdf if existing_pdf.exists() else existing_xml
        existing_outcome = DownloadOutcome(
            classification=Classification.CACHED,
            path=str(cached_path) if cached_path else None,
            http_status=None,
            content_type=None,
            elapsed_ms=None,
            reason=ReasonCode.ALREADY_DOWNLOADED,
            reason_detail="already-downloaded",
        )
        logger.record_manifest(
            artifact,
            resolver="existing",
            url=None,
            outcome=existing_outcome,
            html_paths=[],
            dry_run=dry_run,
            run_id=run_id,
            reason=ReasonCode.ALREADY_DOWNLOADED,
            reason_detail="already-downloaded",
        )
        result["skipped"] = True
        return result

    pipeline_result = pipeline.run(
        session,
        artifact,
        context=download_context,
        session_factory=session_factory,
    )
    html_paths_total = list(pipeline_result.html_paths)

    if list_only:
        result["skipped"] = True
        return result

    if pipeline_result.success and pipeline_result.outcome:
        logger.record_pipeline_result(
            artifact,
            pipeline_result,
            dry_run=dry_run,
            run_id=run_id,
        )
        if pipeline_result.outcome.is_pdf:
            result["saved"] = True
            LOGGER.info(
                "downloaded %s via %s -> %s",
                artifact.work_id,
                pipeline_result.resolver_name,
                pipeline_result.outcome.path or pipeline_result.url,
            )
        elif pipeline_result.outcome.classification is Classification.HTML:
            result["html_only"] = True
        elif pipeline_result.outcome.classification is Classification.XML:
            result["xml_only"] = True
            LOGGER.info(
                "downloaded %s (XML) via %s -> %s",
                artifact.work_id,
                pipeline_result.resolver_name,
                pipeline_result.outcome.path or pipeline_result.url,
            )
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

    reason_token = normalize_reason(pipeline_result.reason)
    if reason_token is None and pipeline_result.outcome:
        reason_token = normalize_reason(pipeline_result.outcome.reason)
    if reason_token is None:
        reason_token = ReasonCode.UNKNOWN

    detail_token = normalize_reason(pipeline_result.reason_detail)
    if detail_token is None and pipeline_result.outcome:
        detail_token = normalize_reason(pipeline_result.outcome.reason_detail)
    if detail_token is None:
        detail_token = "no-resolver-success"

    outcome = pipeline_result.outcome
    if outcome is None:
        outcome = DownloadOutcome(
            classification=Classification.MISS,
            path=None,
            http_status=None,
            content_type=None,
            elapsed_ms=None,
            reason=reason_token,
            reason_detail=detail_token,
        )
    else:
        if reason_token is not None:
            outcome.reason = reason_token
        detail_value = detail_token.value if isinstance(detail_token, ReasonCode) else detail_token
        outcome.reason_detail = detail_value

    bytes_hint = outcome.content_length
    if bytes_hint is not None:
        try:
            result["downloaded_bytes"] = max(int(bytes_hint), 0)
        except (TypeError, ValueError):
            pass

    reason_value = reason_token.value if isinstance(reason_token, ReasonCode) else reason_token
    detail_value = detail_token.value if isinstance(detail_token, ReasonCode) else detail_token

    logger.log_attempt(
        AttemptRecord(
            run_id=run_id,
            work_id=artifact.work_id,
            resolver_name="final",
            resolver_order=None,
            url=pipeline_result.url,
            status=outcome.classification,
            http_status=outcome.http_status,
            content_type=outcome.content_type,
            elapsed_ms=outcome.elapsed_ms,
            reason=reason_value,
            reason_detail=detail_value,
            sha256=outcome.sha256,
            content_length=outcome.content_length,
            dry_run=dry_run,
            retry_after=outcome.retry_after,
        )
    )

    if html_paths_total:
        result["html_only"] = True

    logger.record_manifest(
        artifact,
        resolver=pipeline_result.resolver_name,
        url=pipeline_result.url,
        outcome=outcome,
        html_paths=html_paths_total,
        dry_run=dry_run,
        run_id=run_id,
        reason=reason_token,
        reason_detail=detail_value,
    )

    return result
