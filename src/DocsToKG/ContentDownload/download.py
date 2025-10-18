"""Download orchestration helpers for the content acquisition pipeline.

This module coordinates the streaming download workflow, tying together
resolver outputs, HTTP policy enforcement, and telemetry reporting. It exposes
utilities that transform resolver candidates into stored artifacts while
respecting retry policies, robots.txt directives, and classification rules.
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
from functools import lru_cache
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Set,
    Tuple,
    Union,
)
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
    normalize_arxiv,
    normalize_doi,
    normalize_pmcid,
    normalize_pmid,
    normalize_reason,
    normalize_url,
    slugify,
    tail_contains_html,
    update_tail_buffer,
)
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
from DocsToKG.ContentDownload.download_execution import (
    finalize_candidate_download,
    prepare_candidate_download,
    stream_candidate_payload,
)
from DocsToKG.ContentDownload.telemetry import RunTelemetry

__all__ = [
    "ensure_dir",
    "DownloadOptions",
    "DownloadState",
    "ValidationResult",
    "ResumeDecision",
    "DownloadStrategy",
    "DownloadStrategyContext",
    "PdfDownloadStrategy",
    "HtmlDownloadStrategy",
    "XmlDownloadStrategy",
    "get_strategy_for_classification",
    "validate_classification",
    "handle_resume_logic",
    "cleanup_sidecar_files",
    "build_download_outcome",
    "prepare_candidate_download",
    "stream_candidate_payload",
    "finalize_candidate_download",
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
    sniff_bytes: int = DEFAULT_SNIFF_BYTES
    min_pdf_bytes: int = DEFAULT_MIN_PDF_BYTES
    tail_check_bytes: int = DEFAULT_TAIL_CHECK_BYTES
    robots_checker: Optional["RobotsCache"] = None
    content_addressed: bool = False
    verify_cache_digest: bool = False


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validating an artifact classification."""

    is_valid: bool
    classification: Classification
    expected: Optional[Classification] = None
    reason: Optional[ReasonCode | str] = None
    detail: Optional[str] = None


@dataclass
class ResumeDecision:
    """Decision container describing resume handling for a work artifact."""

    should_skip: bool
    previous_map: Dict[str, Dict[str, Any]]
    outcome: Optional[DownloadOutcome] = None
    resolver: Optional[str] = None
    reason: Optional[ReasonCode | str] = None
    reason_detail: Optional[str] = None
    html_paths: List[str] = field(default_factory=list)


@dataclass
class DownloadStrategyContext:
    """Mutable state shared between strategy phases for a single download."""

    download_context: DownloadContext
    dest_path: Optional[Path] = None
    content_type: Optional[str] = None
    disposition: Optional[str] = None
    elapsed_ms: float = 0.0
    flagged_unknown: bool = False
    sha256: Optional[str] = None
    content_length: Optional[int] = None
    etag: Optional[str] = None
    last_modified: Optional[str] = None
    extracted_text_path: Optional[str] = None
    tail_bytes: Optional[bytes] = None
    head_precheck_passed: bool = False
    min_pdf_bytes: int = DEFAULT_MIN_PDF_BYTES
    tail_check_bytes: int = DEFAULT_TAIL_CHECK_BYTES
    retry_after: Optional[float] = None
    classification_hint: Optional[Classification] = None
    response: Optional[requests.Response] = None
    skip_outcome: Optional[DownloadOutcome] = None


class DownloadStrategy(Protocol):
    """Protocol implemented by artifact-specific download strategies."""

    def should_download(
        self,
        artifact: WorkArtifact,
        context: DownloadStrategyContext,
    ) -> bool:
        """Decide whether the current artifact warrants a fresh download attempt.

        Args:
            artifact: The work artifact under consideration.
            context: Mutable state shared across download strategy phases.

        Returns:
            True when the strategy should perform a download, False when it should
            short-circuit the workflow (for example, due to resume state).
        """
        ...

    def process_response(
        self,
        response: requests.Response,
        artifact: WorkArtifact,
        context: DownloadStrategyContext,
    ) -> Classification:
        """Process the HTTP response to derive classification metadata.

        Args:
            response: Raw HTTP response returned by the download request.
            artifact: Work artifact metadata associated with the response.
            context: Shared strategy state that accumulates response details.

        Returns:
            The classification that should be assigned to the downloaded artifact.
        """
        ...

    def finalize_artifact(
        self,
        artifact: WorkArtifact,
        classification: Classification,
        context: DownloadStrategyContext,
    ) -> DownloadOutcome:
        """Assemble the final download outcome after all processing steps.

        Args:
            artifact: Work artifact being finalized.
            classification: Final classification selected for the artifact.
            context: Strategy context containing response metadata and paths.

        Returns:
            Structured outcome describing the finalized download results.
        """
        ...


class _BaseDownloadStrategy:
    classification: Classification

    def should_download(
        self,
        artifact: WorkArtifact,
        context: DownloadStrategyContext,
    ) -> bool:
        """Evaluate whether to start a download for the provided artifact.

        Args:
            artifact: Work artifact describing the requested content.
            context: Mutable strategy context shared across download phases.

        Returns:
            False when the download is skipped due to configuration, True otherwise.
        """
        if context.download_context.list_only:
            context.skip_outcome = DownloadOutcome(
                classification=Classification.SKIPPED,
                path=None,
                http_status=None,
                content_type=None,
                elapsed_ms=0.0,
                reason=ReasonCode.LIST_ONLY,
                reason_detail="list-only",
            )
            return False
        return True

    def process_response(
        self,
        response: requests.Response,
        artifact: WorkArtifact,
        context: DownloadStrategyContext,
    ) -> Classification:
        """Capture the response and select a classification if not already set.

        Args:
            response: HTTP response received for the download attempt.
            artifact: Work artifact metadata being processed.
            context: Strategy context that caches response metadata.

        Returns:
            Classification derived either from the response or preconfigured hints.
        """
        context.response = response
        if context.classification_hint is None:
            context.classification_hint = self.classification
        return context.classification_hint

    def finalize_artifact(
        self,
        artifact: WorkArtifact,
        classification: Classification,
        context: DownloadStrategyContext,
    ) -> DownloadOutcome:
        """Convert the accumulated strategy context into a download outcome.

        Args:
            artifact: Work artifact associated with the download.
            classification: Final classification selected for the artifact.
            context: Strategy context populated during prior phases.

        Returns:
            Structured `DownloadOutcome` describing results and metadata.
        """
        return build_download_outcome(
            artifact=artifact,
            classification=classification,
            dest_path=context.dest_path,
            response=context.response or requests.Response(),
            elapsed_ms=context.elapsed_ms,
            flagged_unknown=context.flagged_unknown,
            sha256=context.sha256,
            content_length=context.content_length,
            etag=context.etag,
            last_modified=context.last_modified,
            extracted_text_path=context.extracted_text_path,
            dry_run=context.download_context.dry_run,
            tail_bytes=context.tail_bytes,
            head_precheck_passed=context.head_precheck_passed,
            min_pdf_bytes=context.min_pdf_bytes,
            tail_check_bytes=context.tail_check_bytes,
            retry_after=context.retry_after,
            options=context.download_context,
        )


class PdfDownloadStrategy(_BaseDownloadStrategy):
    """Download strategy that enforces PDF-specific processing rules."""

    classification = Classification.PDF


class HtmlDownloadStrategy(_BaseDownloadStrategy):
    """Download strategy tailored for HTML artifacts."""

    classification = Classification.HTML


class XmlDownloadStrategy(_BaseDownloadStrategy):
    """Download strategy tailored for XML artifacts."""

    classification = Classification.XML


def get_strategy_for_classification(classification: Classification) -> DownloadStrategy:
    """Return a strategy implementation for the provided classification."""

    if classification in PDF_LIKE:
        return PdfDownloadStrategy()
    if classification is Classification.HTML:
        return HtmlDownloadStrategy()
    if classification is Classification.XML:
        return XmlDownloadStrategy()
    return PdfDownloadStrategy()


class DownloadState(Enum):
    """State machine for streaming downloads."""

    PENDING = "pending"
    WRITING = "writing"


_DIGEST_CACHE_MAXSIZE = 256


@lru_cache(maxsize=_DIGEST_CACHE_MAXSIZE)
def _cached_sha256(signature: Tuple[str, int, int]) -> Optional[str]:
    """Compute and cache SHA-256 digests keyed by path, size, and mtime."""

    path_str, _, _ = signature
    cached_path = Path(path_str)
    hasher = hashlib.sha256()
    try:
        with cached_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                hasher.update(chunk)
    except OSError:
        return None
    return hasher.hexdigest()


def _validate_cached_artifact(
    result: CachedResult,
    *,
    verify_digest: bool = False,
) -> Tuple[bool, str]:
    """Return validation success flag and mode (``fast_path`` or ``digest``)."""

    validation_mode = "fast_path"

    if not result.path:
        LOGGER.warning(
            "Cached artifact missing path; cannot validate cache entry.",
            extra={"reason": "cached-path-missing"},
        )
        return False, validation_mode

    cached_path = Path(result.path)
    if not cached_path.exists():
        LOGGER.warning(
            "Cached artifact missing at %s; cannot reuse prior download.",
            cached_path,
            extra={"reason": "cached-missing", "path": result.path},
        )
        return False, validation_mode

    try:
        stat_result = cached_path.stat()
    except OSError as exc:
        LOGGER.warning(
            "Unable to stat cached artifact at %s: %s",
            cached_path,
            exc,
            extra={"reason": "cached-stat-failed", "path": result.path},
        )
        return False, validation_mode

    actual_size = stat_result.st_size
    actual_mtime_ns = getattr(stat_result, "st_mtime_ns", None)
    if actual_mtime_ns is None:
        actual_mtime_ns = int(stat_result.st_mtime * 1_000_000_000)

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
        return False, validation_mode

    recorded_mtime_ns = getattr(result, "recorded_mtime_ns", None)
    requires_digest = verify_digest

    if not requires_digest:
        if recorded_mtime_ns is None:
            requires_digest = True
        elif actual_mtime_ns == recorded_mtime_ns:
            return True, validation_mode
        else:
            requires_digest = True

    if not requires_digest:
        return True, validation_mode

    validation_mode = "digest"

    if not result.sha256:
        LOGGER.warning(
            "Cached artifact missing sha256; cannot verify digest for %s.",
            cached_path,
            extra={"reason": "cached-digest-missing", "path": result.path},
        )
        return False, validation_mode

    digest = _cached_sha256((str(cached_path), actual_size, actual_mtime_ns))
    if digest is None:
        LOGGER.warning(
            "Unable to read cached artifact at %s during digest verification.",
            cached_path,
            extra={"reason": "cached-read-failed", "path": result.path},
        )
        return False, validation_mode

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
        return False, validation_mode

    return True, validation_mode


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


def validate_classification(
    classification: Union[Classification, str, None],
    artifact: WorkArtifact,
    options: Union[DownloadOptions, DownloadContext],
) -> ValidationResult:
    """Validate resolver classification against configured expectations."""

    normalized = Classification.from_wire(classification)
    expected: Optional[Classification] = Classification.PDF
    html_allowed = getattr(options, "extract_html_text", False)
    xml_dir = getattr(artifact, "xml_dir", None)

    if normalized in PDF_LIKE:
        expected = Classification.PDF
    elif normalized is Classification.HTML:
        expected = Classification.HTML
        if not html_allowed:
            return ValidationResult(
                is_valid=False,
                classification=normalized,
                expected=Classification.PDF,
                reason=ReasonCode.UNKNOWN,
                detail="html-disabled",
            )
    elif normalized is Classification.XML:
        expected = Classification.XML if xml_dir else Classification.PDF
    else:
        return ValidationResult(
            is_valid=False,
            classification=normalized,
            expected=expected,
            reason=ReasonCode.UNKNOWN,
            detail=f"unexpected-{normalized.value}",
        )

    return ValidationResult(
        is_valid=True,
        classification=normalized,
        expected=expected,
    )


def handle_resume_logic(
    artifact: WorkArtifact,
    previous_index: Mapping[str, Dict[str, Any]],
    options: DownloadOptions,
) -> ResumeDecision:
    """Normalise previous attempts and detect early skip conditions."""

    previous_map: Dict[str, Dict[str, Any]] = {}
    for previous_url, entry in previous_index.items():
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

    if artifact.work_id in options.resume_completed:
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
        return ResumeDecision(
            should_skip=True,
            previous_map=previous_map,
            outcome=skipped_outcome,
            resolver="resume",
            reason=ReasonCode.RESUME_COMPLETE,
            reason_detail="resume-complete",
        )

    if not options.dry_run:
        existing_pdf = artifact.pdf_dir / f"{artifact.base_stem}.pdf"
        existing_xml = artifact.xml_dir / f"{artifact.base_stem}.xml"
        cached_path = existing_pdf if existing_pdf.exists() else existing_xml
        if cached_path.exists():
            outcome = DownloadOutcome(
                classification=Classification.CACHED,
                path=str(cached_path),
                http_status=None,
                content_type=None,
                elapsed_ms=None,
                reason=ReasonCode.ALREADY_DOWNLOADED,
                reason_detail="already-downloaded",
            )
            return ResumeDecision(
                should_skip=True,
                previous_map=previous_map,
                outcome=outcome,
                resolver="existing",
                reason=ReasonCode.ALREADY_DOWNLOADED,
                reason_detail="already-downloaded",
            )

    return ResumeDecision(should_skip=False, previous_map=previous_map)


def cleanup_sidecar_files(
    artifact: WorkArtifact,
    classification: Classification,
    options: Union[DownloadOptions, DownloadContext],
) -> None:
    """Remove temporary sidecar files for the given artifact classification."""

    if classification in PDF_LIKE:
        dest_path = artifact.pdf_dir / f"{artifact.base_stem}.pdf"
    elif classification is Classification.HTML:
        dest_path = artifact.html_dir / f"{artifact.base_stem}.html"
    elif classification is Classification.XML:
        dest_path = artifact.xml_dir / f"{artifact.base_stem}.xml"
    else:
        return

    part_path = dest_path.with_suffix(dest_path.suffix + ".part")
    with contextlib.suppress(FileNotFoundError):
        part_path.unlink()

    dry_run = getattr(options, "dry_run", False)
    if dry_run:
        return

    with contextlib.suppress(FileNotFoundError):
        dest_path.unlink()


def build_download_outcome(
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
    dry_run: bool,
    tail_bytes: Optional[bytes],
    head_precheck_passed: bool,
    min_pdf_bytes: int,
    tail_check_bytes: int,
    retry_after: Optional[float],
    options: Optional[Union[DownloadOptions, DownloadContext]] = None,
) -> DownloadOutcome:
    """Compose a :class:`DownloadOutcome` with shared validation logic."""

    classification_code = (
        classification
        if isinstance(classification, Classification)
        else Classification.from_wire(classification)
    )

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

    options_obj = options or DownloadContext()
    validation = validate_classification(classification_code, artifact, options_obj)
    if not validation.is_valid:
        detail = validation.detail or "classification-invalid"
        return DownloadOutcome(
            classification=Classification.MISS,
            path=None,
            http_status=response.status_code,
            content_type=response.headers.get("Content-Type"),
            elapsed_ms=elapsed_ms,
            reason=ReasonCode.UNKNOWN,
            reason_detail=detail,
            sha256=None,
            content_length=content_length,
            etag=etag,
            last_modified=last_modified,
            extracted_text_path=extracted_text_path,
            retry_after=retry_after,
        )

    status_code = getattr(response, "status_code", None)
    try:
        normalized_status = int(status_code) if status_code is not None else None
    except (TypeError, ValueError):
        normalized_status = None

    conditional_hit = normalized_status == 304 or classification_code is Classification.CACHED

    reason_code: Optional[ReasonCode] = None
    reason_detail: Optional[str] = None
    if flagged_unknown and classification_code is Classification.PDF:
        reason_code = ReasonCode.PDF_SNIFF_UNKNOWN
        reason_detail = "pdf-sniff-unknown"
    elif conditional_hit:
        reason_code = ReasonCode.CONDITIONAL_NOT_MODIFIED
        if reason_detail is None:
            reason_detail = "not-modified"

    path_str = str(dest_path) if dest_path else None

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
    pmid = normalize_pmid(ids.get("pmid"))
    pmcid = normalize_pmcid(ids.get("pmcid"))
    arxiv_id = normalize_arxiv(ids.get("arxiv"))

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
    preflight = prepare_candidate_download(
        session=session,
        artifact=artifact,
        url=url,
        referer=referer,
        timeout=timeout,
        context_payload=context,
        head_precheck_passed=head_precheck_passed,
    )

    if preflight.early_outcome is not None:
        return preflight.early_outcome

    stream_result = stream_candidate_payload(
        session=session,
        artifact=artifact,
        url=url,
        timeout=timeout,
        preflight=preflight,
        cleanup_sidecar=cleanup_sidecar_files,
        validate_cached_artifact=_validate_cached_artifact,
        strategy_selector=get_strategy_for_classification,
        strategy_context_factory=DownloadStrategyContext,
        content_address_factory=_apply_content_addressed_storage,
    )

    return finalize_candidate_download(
        artifact=artifact,
        stream_result=stream_result,
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
    strategy_selector: Callable[
        [Classification], DownloadStrategy
    ] = get_strategy_for_classification,
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
    if not isinstance(raw_previous, Mapping):
        raw_previous = {}

    resume_decision = handle_resume_logic(artifact, raw_previous, options)
    previous_map = resume_decision.previous_map
    download_context = DownloadContext(
        dry_run=dry_run,
        extract_html_text=extract_html_text,
        previous=previous_map,
        list_only=list_only,
        sniff_bytes=sniff_bytes,
        min_pdf_bytes=min_pdf_bytes,
        tail_check_bytes=tail_check_bytes,
        robots_checker=robots_checker,
        content_addressed=content_addressed,
        verify_cache_digest=options.verify_cache_digest,
        global_manifest_index=getattr(pipeline, "_global_manifest_index", {}),
    )
    download_context.mark_explicit(
        "dry_run",
        "extract_html_text",
        "previous",
        "list_only",
        "sniff_bytes",
        "min_pdf_bytes",
        "tail_check_bytes",
        "robots_checker",
        "content_addressed",
    )
    download_context.extra["strategy_factory"] = strategy_selector

    cohort_order = _cohort_order_for(artifact)
    if cohort_order:
        download_context.resolver_order = list(dict.fromkeys(cohort_order))
        download_context.mark_explicit("resolver_order")

    if resume_decision.should_skip:
        skipped_outcome = resume_decision.outcome or DownloadOutcome(
            classification=Classification.SKIPPED,
            path=None,
            http_status=None,
            content_type=None,
            elapsed_ms=None,
            reason=resume_decision.reason or ReasonCode.UNKNOWN,
            reason_detail=resume_decision.reason_detail or "resume-skip",
        )
        logger.record_manifest(
            artifact,
            resolver=resume_decision.resolver,
            url=None,
            outcome=skipped_outcome,
            html_paths=[],
            dry_run=dry_run,
            run_id=run_id,
            reason=resume_decision.reason,
            reason_detail=resume_decision.reason_detail,
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
        validation = validate_classification(
            pipeline_result.outcome.classification,
            artifact,
            options,
        )
        if not validation.is_valid:
            pipeline_result.outcome.reason = validation.reason or ReasonCode.UNKNOWN
            pipeline_result.outcome.reason_detail = validation.detail or "classification-invalid"
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
