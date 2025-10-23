# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.download",
#   "purpose": "Download orchestration helpers for the content acquisition pipeline.",
#   "sections": [
#     {
#       "id": "ensure-dir",
#       "name": "ensure_dir",
#       "anchor": "function-ensure-dir",
#       "kind": "function"
#     },
#     {
#       "id": "validationresult",
#       "name": "ValidationResult",
#       "anchor": "class-validationresult",
#       "kind": "class"
#     },
#     {
#       "id": "resumedecision",
#       "name": "ResumeDecision",
#       "anchor": "class-resumedecision",
#       "kind": "class"
#     },
#     {
#       "id": "downloadstrategycontext",
#       "name": "DownloadStrategyContext",
#       "anchor": "class-downloadstrategycontext",
#       "kind": "class"
#     },
#     {
#       "id": "downloadpreflightplan",
#       "name": "DownloadPreflightPlan",
#       "anchor": "class-downloadpreflightplan",
#       "kind": "class"
#     },
#     {
#       "id": "downloadstreamresult",
#       "name": "DownloadStreamResult",
#       "anchor": "class-downloadstreamresult",
#       "kind": "class"
#     },
#     {
#       "id": "prepare-candidate-download",
#       "name": "prepare_candidate_download",
#       "anchor": "function-prepare-candidate-download",
#       "kind": "function"
#     },
#     {
#       "id": "stream-candidate-payload",
#       "name": "stream_candidate_payload",
#       "anchor": "function-stream-candidate-payload",
#       "kind": "function"
#     },
#     {
#       "id": "finalize-candidate-download",
#       "name": "finalize_candidate_download",
#       "anchor": "function-finalize-candidate-download",
#       "kind": "function"
#     },
#     {
#       "id": "downloadstrategy",
#       "name": "DownloadStrategy",
#       "anchor": "class-downloadstrategy",
#       "kind": "class"
#     },
#     {
#       "id": "basedownloadstrategy",
#       "name": "_BaseDownloadStrategy",
#       "anchor": "class-basedownloadstrategy",
#       "kind": "class"
#     },
#     {
#       "id": "pdfdownloadstrategy",
#       "name": "PdfDownloadStrategy",
#       "anchor": "class-pdfdownloadstrategy",
#       "kind": "class"
#     },
#     {
#       "id": "htmldownloadstrategy",
#       "name": "HtmlDownloadStrategy",
#       "anchor": "class-htmldownloadstrategy",
#       "kind": "class"
#     },
#     {
#       "id": "xmldownloadstrategy",
#       "name": "XmlDownloadStrategy",
#       "anchor": "class-xmldownloadstrategy",
#       "kind": "class"
#     },
#     {
#       "id": "get-strategy-for-classification",
#       "name": "get_strategy_for_classification",
#       "anchor": "function-get-strategy-for-classification",
#       "kind": "function"
#     },
#     {
#       "id": "downloadstate",
#       "name": "DownloadState",
#       "anchor": "class-downloadstate",
#       "kind": "class"
#     },
#     {
#       "id": "cached-sha256",
#       "name": "_cached_sha256",
#       "anchor": "function-cached-sha256",
#       "kind": "function"
#     },
#     {
#       "id": "validate-cached-artifact",
#       "name": "_validate_cached_artifact",
#       "anchor": "function-validate-cached-artifact",
#       "kind": "function"
#     },
#     {
#       "id": "robotscache",
#       "name": "RobotsCache",
#       "anchor": "class-robotscache",
#       "kind": "class"
#     },
#     {
#       "id": "apply-content-addressed-storage",
#       "name": "_apply_content_addressed_storage",
#       "anchor": "function-apply-content-addressed-storage",
#       "kind": "function"
#     },
#     {
#       "id": "streaming-finalization-enabled",
#       "name": "_streaming_finalization_enabled",
#       "anchor": "function-streaming-finalization-enabled",
#       "kind": "function"
#     },
#     {
#       "id": "try-streaming-finalization",
#       "name": "_try_streaming_finalization",
#       "anchor": "function-try-streaming-finalization",
#       "kind": "function"
#     },
#     {
#       "id": "validate-classification",
#       "name": "validate_classification",
#       "anchor": "function-validate-classification",
#       "kind": "function"
#     },
#     {
#       "id": "handle-resume-logic",
#       "name": "handle_resume_logic",
#       "anchor": "function-handle-resume-logic",
#       "kind": "function"
#     },
#     {
#       "id": "cleanup-sidecar-files",
#       "name": "cleanup_sidecar_files",
#       "anchor": "function-cleanup-sidecar-files",
#       "kind": "function"
#     },
#     {
#       "id": "build-download-outcome",
#       "name": "build_download_outcome",
#       "anchor": "function-build-download-outcome",
#       "kind": "function"
#     },
#     {
#       "id": "collect-location-urls",
#       "name": "_collect_location_urls",
#       "anchor": "function-collect-location-urls",
#       "kind": "function"
#     },
#     {
#       "id": "cohort-order-for",
#       "name": "_cohort_order_for",
#       "anchor": "function-cohort-order-for",
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
#       "id": "process-one-work",
#       "name": "process_one_work",
#       "anchor": "function-process-one-work",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Download orchestration helpers for the content acquisition pipeline.

Responsibilities
----------------
- Translate resolver candidates into persisted artifacts by enforcing robots
  policies, executing conditional requests, classifying payloads, and routing
  them through strategy-specific handlers (PDF/HTML/XML).
- Maintain the :class:`DownloadState` lifecycle, including cache reuse,
  duplicate suppression, digest verification, content-addressed promotion, and
  directory preparation.
- Provide hooks into telemetry (:mod:`DocsToKG.ContentDownload.telemetry`) so
  every attempt captures consistent manifest records and retry metadata.
- Coordinate resume-aware behaviour (JSONL, CSV, SQLite) and global URL dedupe
  decisions before the pipeline emits additional resolver traffic.
- Surface helpers such as :func:`download_candidate` and
  :func:`process_one_work` that the runner and tests can call directly.

Design Notes
------------
- The module keeps streaming IO and checksum calculations in one place so it
  can be unit-tested with fake responses and file systems.
- Robots handling is encapsulated by :class:`RobotsCache`, allowing callers to
  flip behaviour (respect/ignore) without touching networking code.
- Resume and dedupe heuristics operate on normalised manifest metadata so runs
  remain idempotent even when log rotation or CSV-only logging is enabled.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import shutil
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import (
    Any,
    Protocol,
)
from urllib.parse import urlparse, urlsplit
from urllib.robotparser import RobotFileParser

import httpcore
import httpx

from DocsToKG.ContentDownload import locks
from DocsToKG.ContentDownload.api.types import (
    AttemptRecord,
    DownloadOutcome,
)
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
    slugify,
    tail_contains_html,
    update_tail_buffer,
)
from DocsToKG.ContentDownload.errors import (
    RateLimitError,
    log_download_failure,
)
from DocsToKG.ContentDownload.httpx_transport import get_http_client
from DocsToKG.ContentDownload.networking import (
    BreakerOpenError,
    CachedResult,
    ConditionalRequestHelper,
    ContentPolicyViolation,
    ModifiedResult,
    head_precheck,
    parse_retry_after_header,
    request_with_retries,
)
from DocsToKG.ContentDownload.telemetry import RunTelemetry, normalize_manifest_path
from DocsToKG.ContentDownload.urls import canonical_for_index, canonical_for_request

# New work orchestration enabled by default - no backward compatibility fallback
ENABLE_IDEMPOTENCY = True
ENABLE_FALLBACK_STRATEGY = True


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


_RUN_EXTRA_REF_KEY = "_run_extra"
_RANGE_RESUME_WARNING_KEY = "range_resume_warning_emitted"


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


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validating an artifact classification."""

    is_valid: bool
    classification: Classification
    expected: Classification | None = None
    reason: ReasonCode | str | None = None
    detail: str | None = None


@dataclass
class ResumeDecision:
    """Decision container describing resume handling for a work artifact."""

    should_skip: bool
    previous_map: dict[str, dict[str, Any]]
    outcome: DownloadOutcome | None = None
    resolver: str | None = None
    reason: ReasonCode | str | None = None
    reason_detail: str | None = None
    html_paths: list[str] = field(default_factory=list)


@dataclass
class DownloadStrategyContext:
    """Mutable state shared between strategy phases for a single download."""

    download_context: DownloadContext
    dest_path: Path | None = None
    content_type: str | None = None
    disposition: str | None = None
    elapsed_ms: float = 0.0
    flagged_unknown: bool = False
    sha256: str | None = None
    content_length: int | None = None
    etag: str | None = None
    last_modified: str | None = None
    extracted_text_path: str | None = None
    tail_bytes: bytes | None = None
    head_precheck_passed: bool = False
    min_pdf_bytes: int = DEFAULT_MIN_PDF_BYTES
    tail_check_bytes: int = DEFAULT_TAIL_CHECK_BYTES
    retry_after: float | None = None
    classification_hint: Classification | None = None
    response: httpx.Response | None = None
    canonical_url: str | None = None
    canonical_index: str | None = None
    original_url: str | None = None
    skip_outcome: DownloadOutcome | None = None


@dataclass
class DownloadPreflightPlan:
    """Prepared inputs required to stream a candidate download."""

    client: httpx.Client
    artifact: WorkArtifact
    url: str
    timeout: float
    context: DownloadContext
    base_headers: dict[str, str]
    content_policy: dict[str, Any] | None
    canonical_url: str
    canonical_index: str
    original_url: str | None
    origin_host: str | None
    cond_helper: ConditionalRequestHelper
    attempt_conditional: bool = True
    head_precheck_passed: bool = False
    extract_html_text: bool = False
    sniff_limit: int = DEFAULT_SNIFF_BYTES
    min_pdf_bytes: int = DEFAULT_MIN_PDF_BYTES
    tail_window_bytes: int = DEFAULT_TAIL_CHECK_BYTES
    progress_callback: Callable[[int, int | None, str], None] | None = None
    progress_update_interval: int = 128 * 1024
    content_type_hint: str = ""
    skip_outcome: DownloadOutcome | None = None
    telemetry: Any | None = None
    run_id: str | None = None


@dataclass
class DownloadStreamResult:
    """Structured payload returned by :func:`stream_candidate_payload`."""

    outcome: DownloadOutcome | None = None
    response: httpx.Response | None = None
    classification: Classification | None = None
    strategy: DownloadStrategy | None = None
    strategy_context: DownloadStrategyContext | None = None
    elapsed_ms: float = 0.0
    retry_after: float | None = None


def prepare_candidate_download(
    client: httpx.Client | None,
    artifact: WorkArtifact,
    url: str,
    referer: str | None,
    timeout: float,
    ctx: DownloadContext,
    *,
    head_precheck_passed: bool = False,
    original_url: str | None = None,
    origin_host: str | None = None,
    telemetry: Any | None = None,
    run_id: str | None = None,
) -> DownloadPreflightPlan:
    """Prepare request metadata prior to streaming the download."""

    http_client = client or get_http_client()
    sniff_limit = ctx.sniff_bytes
    min_pdf_bytes = ctx.min_pdf_bytes
    tail_window_bytes = ctx.tail_check_bytes
    progress_callback = ctx.progress_callback
    progress_update_interval = 128 * 1024

    effective_original = original_url or url
    canonical_index = canonical_for_index(effective_original)
    request_url = canonical_for_request(url, role="artifact", origin_host=origin_host)
    url = request_url

    parsed_url = urlsplit(url)
    domain_policies: dict[str, dict[str, Any]] = ctx.domain_content_rules
    host_key = (parsed_url.hostname or parsed_url.netloc or "").lower()
    content_policy: dict[str, Any] | None = None
    if domain_policies and host_key:
        content_policy = domain_policies.get(host_key)
        if content_policy is None and host_key.startswith("www."):
            content_policy = domain_policies.get(host_key[4:])

    headers: dict[str, str] = {}
    if referer:
        headers["Referer"] = referer

    accept_overrides = ctx.host_accept_overrides
    accept_value: str | None = None
    if isinstance(accept_overrides, dict):
        host_for_accept = (parsed_url.netloc or "").lower()
        if host_for_accept:
            accept_value = accept_overrides.get(host_for_accept)
            if accept_value is None and host_for_accept.startswith("www."):
                accept_value = accept_overrides.get(host_for_accept[4:])
    if accept_value:
        headers["Accept"] = str(accept_value)

    http_client = client or get_http_client()

    robots_checker: RobotsCache | None = ctx.robots_checker
    skip_outcome: DownloadOutcome | None = None
    if robots_checker is not None:
        robots_allowed = robots_checker.is_allowed(http_client, url, timeout)
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
            skip_outcome = DownloadOutcome(
                classification=Classification.SKIPPED,
                path=None,
                http_status=None,
                content_type=None,
                elapsed_ms=0.0,
                reason=ReasonCode.ROBOTS_DISALLOWED,
                reason_detail="robots-disallowed",
                canonical_url=url,
                canonical_index=canonical_index,
                original_url=effective_original,
            )

    head_precheck_state = head_precheck_passed or ctx.head_precheck_passed
    if skip_outcome is None and not head_precheck_state and not ctx.skip_head_precheck:
        head_precheck_state = head_precheck(
            http_client, url, timeout, content_policy=content_policy
        )
        ctx.head_precheck_passed = head_precheck_state

    extract_html_text = ctx.extract_html_text
    previous_map = ctx.previous
    global_index = ctx.global_manifest_index
    previous = previous_map.get(canonical_index) or global_index.get(canonical_index, {})
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
    previous_mtime_ns = previous.get("mtime_ns")
    if isinstance(previous_mtime_ns, str):
        try:
            previous_mtime_ns = int(previous_mtime_ns)
        except ValueError:
            previous_mtime_ns = None

    cond_helper = ConditionalRequestHelper(
        prior_etag=previous_etag,
        prior_last_modified=previous_last_modified,
        prior_sha256=previous_sha,
        prior_content_length=previous_length,
        prior_path=existing_path,
        prior_mtime_ns=previous_mtime_ns,
    )

    # Use deferred import to avoid circular dependency
    from DocsToKG.ContentDownload.breakers_loader import _normalize_host_key

    plan_origin_host = (_normalize_host_key(origin_host) if origin_host else host_key) or None

    return DownloadPreflightPlan(
        client=http_client,
        artifact=artifact,
        url=url,
        timeout=timeout,
        context=ctx,
        base_headers=dict(headers),
        content_policy=content_policy,
        canonical_url=canonical_index,
        canonical_index=canonical_index,
        original_url=effective_original,
        origin_host=plan_origin_host,
        cond_helper=cond_helper,
        attempt_conditional=True,
        head_precheck_passed=head_precheck_state,
        extract_html_text=extract_html_text,
        sniff_limit=sniff_limit,
        min_pdf_bytes=min_pdf_bytes,
        tail_window_bytes=tail_window_bytes,
        progress_callback=progress_callback,
        progress_update_interval=progress_update_interval,
        content_type_hint="",
        skip_outcome=skip_outcome,
        telemetry=telemetry,
        run_id=run_id,
    )


def stream_candidate_payload(plan: DownloadPreflightPlan) -> DownloadStreamResult:
    """Execute the streaming phase for a prepared download plan."""

    ctx = plan.context
    attempt_conditional = plan.attempt_conditional
    cond_helper = plan.cond_helper
    content_policy = plan.content_policy
    base_headers = plan.base_headers
    content_type_hint = plan.content_type_hint
    head_precheck_passed = plan.head_precheck_passed
    extract_html_text = plan.extract_html_text
    sniff_limit = plan.sniff_limit
    min_pdf_bytes = plan.min_pdf_bytes
    tail_window_bytes = plan.tail_window_bytes
    progress_callback = plan.progress_callback
    progress_update_interval = plan.progress_update_interval

    client = plan.client
    artifact = plan.artifact
    url = plan.url
    timeout = plan.timeout

    dry_run = ctx.dry_run
    logged_conditional_downgrade = False
    start = time.monotonic()

    try:
        while True:
            retry_after_hint: float | None = None
            headers = dict(base_headers)
            if attempt_conditional:
                headers.update(cond_helper.build_headers())

            start_request = time.monotonic()
            try:
                retry_after_cap = None
                context_extra = getattr(plan.context, "extra", None)
                if isinstance(context_extra, Mapping):
                    raw_cap = context_extra.get("retry_after_cap")
                    if isinstance(raw_cap, (int, float)):
                        retry_after_cap = float(raw_cap)
                response_cm = request_with_retries(
                    client,
                    "GET",
                    url,
                    role="artifact",
                    stream=True,
                    allow_redirects=True,
                    timeout=timeout,
                    headers=headers,
                    retry_after_cap=retry_after_cap,
                    content_policy=content_policy,
                    original_url=plan.original_url,
                    origin_host=plan.origin_host,
                    telemetry=plan.telemetry,
                    run_id=plan.run_id,
                )
            except BreakerOpenError as exc:
                elapsed_ms = (time.monotonic() - start_request) * 1000.0
                breaker_meta = getattr(exc, "breaker_meta", None)
                metadata: dict[str, Any] = {}
                if isinstance(breaker_meta, Mapping):
                    metadata["breaker"] = dict(breaker_meta)
                return DownloadStreamResult(
                    outcome=DownloadOutcome(
                        classification=Classification.SKIPPED,
                        path=None,
                        http_status=None,
                        content_type=None,
                        elapsed_ms=elapsed_ms,
                        reason=ReasonCode.BREAKER_OPEN,
                        reason_detail="breaker-open",
                        sha256=None,
                        content_length=None,
                        etag=None,
                        last_modified=None,
                        extracted_text_path=None,
                        retry_after=None,
                        metadata=metadata if metadata else {},
                        canonical_url=plan.canonical_url,
                        canonical_index=plan.canonical_index,
                        original_url=plan.original_url,
                    )
                )
            except ContentPolicyViolation as exc:
                elapsed_ms = (time.monotonic() - start_request) * 1000.0
                violation = exc.violation
                reason_code = (
                    ReasonCode.DOMAIN_DISALLOWED_MIME
                    if violation == "content-type"
                    else ReasonCode.UNKNOWN
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
                cleanup_sidecar_files(artifact, Classification.PDF, ctx)
                return DownloadStreamResult(
                    outcome=DownloadOutcome(
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
                        canonical_url=plan.canonical_url,
                        canonical_index=plan.canonical_index,
                        original_url=plan.original_url,
                    )
                )

            response_context: contextlib.AbstractContextManager[httpx.Response]
            if hasattr(response_cm, "__enter__") and hasattr(response_cm, "__exit__"):
                response_context = response_cm  # type: ignore[assignment]
            else:
                response_context = contextlib.nullcontext(response_cm)

            with response_context as response:
                elapsed_ms = (time.monotonic() - start_request) * 1000.0
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
                    return DownloadStreamResult(
                        outcome=DownloadOutcome(
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
                            canonical_url=plan.canonical_url,
                            canonical_index=plan.canonical_index,
                            original_url=plan.original_url,
                        )
                    )

                if response.status_code == 304:
                    if not attempt_conditional:
                        LOGGER.warning(
                            "Received HTTP 304 for %s without conditional headers; treating as http_error.",
                            url,
                        )
                        return DownloadStreamResult(
                            outcome=DownloadOutcome(
                                classification=Classification.HTTP_ERROR,
                                path=None,
                                http_status=response.status_code,
                                content_type=response.headers.get("Content-Type")
                                or content_type_hint,
                                elapsed_ms=elapsed_ms,
                                reason=ReasonCode.UNEXPECTED_304,
                                reason_detail="unexpected-304",
                                sha256=None,
                                content_length=None,
                                etag=None,
                                last_modified=None,
                                extracted_text_path=None,
                                retry_after=retry_after_hint,
                                canonical_url=plan.canonical_url,
                                canonical_index=plan.canonical_index,
                                original_url=plan.original_url,
                            )
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

                    if not isinstance(cached, CachedResult):
                        raise TypeError("Expected CachedResult for 304 response")
                    is_valid_cache, validation_mode = _validate_cached_artifact(
                        cached,
                        verify_digest=ctx.verify_cache_digest,
                    )
                    if not is_valid_cache:
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
                    outcome = DownloadOutcome(
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
                        canonical_url=plan.canonical_url,
                        canonical_index=plan.canonical_index,
                        original_url=plan.original_url,
                    )
                    outcome.metadata["cache_validation_mode"] = validation_mode
                    return DownloadStreamResult(outcome=outcome)

                if response.status_code == 206:
                    LOGGER.warning(
                        "Received HTTP 206 for %s without Range request; treating as error.",
                        url,
                    )
                    return DownloadStreamResult(
                        outcome=DownloadOutcome(
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
                            canonical_url=plan.canonical_url,
                            canonical_index=plan.canonical_index,
                            original_url=plan.original_url,
                        )
                    )

                elif response.status_code != 200:
                    return DownloadStreamResult(
                        outcome=DownloadOutcome(
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
                            canonical_url=plan.canonical_url,
                            canonical_index=plan.canonical_index,
                            original_url=plan.original_url,
                        )
                    )

                modified_result: ModifiedResult = cond_helper.interpret_response(response)

                content_type = response.headers.get("Content-Type") or content_type_hint
                disposition = response.headers.get("Content-Disposition")
                content_length_header = response.headers.get("Content-Length")
                content_length_hint: int | None = None
                if content_length_header:
                    try:
                        content_length_hint = int(content_length_header.strip())
                    except (TypeError, ValueError):
                        content_length_hint = None

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

                    # Skip large downloads if configured to do so
                    if ctx.skip_large_downloads:
                        elapsed_ms = (time.monotonic() - start) * 1000.0
                        return DownloadStreamResult(
                            outcome=DownloadOutcome(
                                classification=Classification.SKIPPED,
                                path=None,
                                http_status=response.status_code,
                                content_type=response.headers.get("Content-Type"),
                                elapsed_ms=elapsed_ms,
                                reason=ReasonCode.SKIP_LARGE_DOWNLOAD,
                                reason_detail=f"skipped-large-download-{size_mb:.1f}mb",
                                sha256=None,
                                content_length=content_length_hint,
                                etag=response.headers.get("ETag"),
                                last_modified=response.headers.get("Last-Modified"),
                                extracted_text_path=None,
                                canonical_url=plan.canonical_url,
                                canonical_index=plan.canonical_index,
                                original_url=plan.original_url,
                            )
                        )

                hasher = hashlib.sha256() if not dry_run else None
                byte_count = 0
                chunk_size = int(ctx.chunk_size or (1 << 15))
                iter_bytes_fn = getattr(response, "iter_bytes", None)
                if callable(iter_bytes_fn):
                    content_iter = iter_bytes_fn(chunk_size=chunk_size)
                else:
                    iter_content_fn = getattr(response, "iter_content", None)
                    if callable(iter_content_fn):
                        content_iter = iter_content_fn(chunk_size=chunk_size)
                    else:
                        body = getattr(response, "content", b"")
                        if isinstance(body, bytes):
                            if not body:
                                content_iter = iter(())
                            else:
                                content_iter = iter([body])
                        else:
                            content_iter = iter(body or [])
                sniff_buffer = bytearray()
                prefetched: list[bytes] = []
                detected: Classification = Classification.UNKNOWN
                flagged_unknown = False
                last_classification_size = 0
                min_classify_interval = 4096

                for chunk in content_iter:
                    if not chunk:
                        continue
                    sniff_buffer.extend(chunk)
                    prefetched.append(chunk)

                    buffer_size = len(sniff_buffer)
                    should_classify = last_classification_size == 0 or (
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
                    return DownloadStreamResult(
                        outcome=DownloadOutcome(
                            classification=Classification.MISS,
                            path=None,
                            http_status=response.status_code,
                            content_type=content_type,
                            elapsed_ms=elapsed_ms,
                            reason=ReasonCode.UNKNOWN,
                            reason_detail="classifier-unknown",
                            retry_after=retry_after_hint,
                            canonical_url=plan.canonical_url,
                            canonical_index=plan.canonical_index,
                            original_url=plan.original_url,
                        )
                    )

                strategy_factory = getattr(ctx, "extra", {}).get("strategy_factory")
                if not callable(strategy_factory):
                    strategy_factory = get_strategy_for_classification
                strategy = strategy_factory(detected)
                strategy_context = DownloadStrategyContext(
                    download_context=ctx,
                    canonical_url=plan.canonical_url,
                    canonical_index=plan.canonical_index,
                    original_url=plan.original_url,
                    content_type=content_type,
                    disposition=disposition,
                    elapsed_ms=elapsed_ms,
                    flagged_unknown=flagged_unknown,
                    min_pdf_bytes=min_pdf_bytes,
                    tail_check_bytes=tail_window_bytes,
                    head_precheck_passed=head_precheck_passed,
                    retry_after=retry_after_hint,
                    classification_hint=detected,
                )

                if dry_run:
                    if not strategy.should_download(artifact, strategy_context):
                        return DownloadStreamResult(
                            outcome=strategy_context.skip_outcome
                            or DownloadOutcome(
                                classification=Classification.SKIPPED,
                                path=None,
                                http_status=None,
                                content_type=None,
                                elapsed_ms=elapsed_ms,
                                reason=ReasonCode.UNKNOWN,
                                reason_detail="strategy-skip",
                                canonical_url=plan.canonical_url,
                                canonical_index=plan.canonical_index,
                                original_url=plan.original_url,
                            )
                        )
                    strategy_context.retry_after = retry_after_hint
                    processed_classification = strategy.process_response(
                        response, artifact, strategy_context
                    )
                    strategy_context.elapsed_ms = elapsed_ms
                    strategy_context.etag = modified_result.etag
                    strategy_context.last_modified = modified_result.last_modified
                    strategy_context.flagged_unknown = flagged_unknown
                    return DownloadStreamResult(
                        classification=processed_classification,
                        response=response,
                        strategy=strategy,
                        strategy_context=strategy_context,
                        elapsed_ms=elapsed_ms,
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
                strategy_context.dest_path = dest_path
                strategy_context.content_type = content_type
                strategy_context.disposition = disposition
                if not strategy.should_download(artifact, strategy_context):
                    return DownloadStreamResult(
                        outcome=strategy_context.skip_outcome
                        or DownloadOutcome(
                            classification=Classification.SKIPPED,
                            path=None,
                            http_status=None,
                            content_type=None,
                            elapsed_ms=elapsed_ms,
                            reason=ReasonCode.UNKNOWN,
                            reason_detail="strategy-skip",
                            canonical_url=plan.canonical_url,
                            canonical_index=plan.canonical_index,
                            original_url=plan.original_url,
                        )
                    )
                ensure_dir(dest_path.parent)

                is_pdf = detected == Classification.PDF
                tail_buffer = bytearray() if is_pdf else None

                def _stream_chunks() -> Iterable[bytes]:
                    nonlocal byte_count
                    last_progress_update = 0

                    for chunk in prefetched:
                        if not chunk:
                            continue
                        byte_count += len(chunk)
                        if is_pdf and tail_buffer is not None:
                            update_tail_buffer(tail_buffer, chunk, limit=tail_window_bytes)
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

                    for chunk in content_iter:
                        if not chunk:
                            continue
                        byte_count += len(chunk)
                        if is_pdf and tail_buffer is not None:
                            update_tail_buffer(tail_buffer, chunk, limit=tail_window_bytes)
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

                    if progress_callback and callable(progress_callback) and byte_count > 0:
                        try:
                            progress_callback(byte_count, content_length_hint, url)
                        except Exception as exc:
                            LOGGER.debug("Final progress callback failed: %s", exc)

                sha256: str | None = None
                content_length: int | None = None
                tail_snapshot: bytes | None = None
                extracted_text_path: str | None = None
                artifact_lock_cm: contextlib.AbstractContextManager[None]
                if dest_path and not dry_run:
                    artifact_lock_cm = locks.artifact_lock(dest_path)
                else:
                    artifact_lock_cm = contextlib.nullcontext()

                with artifact_lock_cm:
                    try:
                        atomic_write(
                            dest_path,
                            _stream_chunks(),
                            hasher=hasher,
                            keep_partial_on_error=True,
                        )
                    except (httpx.HTTPError, httpcore.ProtocolError, AttributeError) as exc:
                        LOGGER.warning(
                            "Streaming download failed for %s: %s",
                            url,
                            exc,
                            extra={"extra_fields": {"work_id": artifact.work_id}},
                        )
                        classification_hint = (
                            detected
                            if detected
                            in {Classification.PDF, Classification.HTML, Classification.XML}
                            else Classification.PDF
                        )
                        cleanup_sidecar_files(
                            artifact,
                            classification_hint,
                            ctx,
                        )
                        seen_attempts = ctx.stream_retry_attempts
                        if seen_attempts < 1:
                            ctx.stream_retry_attempts = seen_attempts + 1
                            attempt_conditional = False
                            LOGGER.info("Retrying download for %s after stream failure", url)
                            continue
                        elapsed_err = (time.monotonic() - start) * 1000.0
                        return DownloadStreamResult(
                            outcome=DownloadOutcome(
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
                                canonical_url=plan.canonical_url,
                                canonical_index=plan.canonical_index,
                                original_url=plan.original_url,
                            )
                        )

                    if hasher is not None:
                        sha256 = hasher.hexdigest()
                        content_length = byte_count
                    if (
                        dest_path
                        and ctx.content_addressed
                        and sha256
                        and detected is Classification.PDF
                    ):
                        dest_path = _apply_content_addressed_storage(dest_path, sha256)
                    tail_snapshot = bytes(tail_buffer) if tail_buffer else None

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
                            except Exception as exc:
                                LOGGER.warning(
                                    "Failed to extract HTML text for %s: %s", dest_path, exc
                                )
                            else:
                                if text:
                                    text_path = dest_path.with_suffix(dest_path.suffix + ".txt")
                                    atomic_write_text(text_path, text)
                                    extracted_text_path = str(text_path)

                ctx.stream_retry_attempts = 0
                strategy_context.dest_path = dest_path
                strategy_context.elapsed_ms = elapsed_ms
                strategy_context.sha256 = sha256
                strategy_context.content_length = content_length
                strategy_context.etag = modified_result.etag
                strategy_context.last_modified = modified_result.last_modified
                strategy_context.extracted_text_path = extracted_text_path
                strategy_context.tail_bytes = tail_snapshot
                strategy_context.flagged_unknown = flagged_unknown
                strategy_context.retry_after = retry_after_hint
                processed_classification = strategy.process_response(
                    response, artifact, strategy_context
                )
                return DownloadStreamResult(
                    classification=processed_classification,
                    response=response,
                    strategy=strategy,
                    strategy_context=strategy_context,
                    elapsed_ms=elapsed_ms,
                    retry_after=retry_after_hint,
                )
    except RateLimitError as exc:
        elapsed_ms = (time.monotonic() - start) * 1000.0
        log_download_failure(
            LOGGER,
            url,
            artifact.work_id,
            http_status=None,
            reason_code=ReasonCode.RATE_LIMITED.value,
            error_details=str(exc),
            exception=exc,
        )

        limiter_metadata: dict[str, Any] = {
            "backend": exc.backend,
            "mode": exc.mode,
            "wait_ms": exc.waited_ms,
            "role": exc.role,
            "blocked": True,
        }
        if exc.host:
            limiter_metadata["host"] = exc.host
        if exc.next_allowed_at is not None:
            limiter_metadata["next_allowed_at"] = exc.next_allowed_at.isoformat()
        if exc.retry_after is not None:
            limiter_metadata["retry_after"] = exc.retry_after
        outcome_metadata: dict[str, Any] = {
            "rate_limiter": limiter_metadata,
        }
        if exc.details:
            outcome_metadata["rate_limit_details"] = exc.details

        return DownloadStreamResult(
            outcome=DownloadOutcome(
                classification=Classification.SKIPPED,
                path=None,
                http_status=None,
                content_type=None,
                elapsed_ms=elapsed_ms,
                reason=ReasonCode.RATE_LIMITED,
                reason_detail="rate_limited",
                sha256=None,
                content_length=None,
                etag=None,
                last_modified=None,
                extracted_text_path=None,
                retry_after=exc.retry_after,
                metadata=outcome_metadata,
                error=str(exc),
                canonical_url=plan.canonical_url,
                canonical_index=plan.canonical_index,
                original_url=plan.original_url,
            )
        )

    except httpx.HTTPError as exc:
        elapsed_ms = (time.monotonic() - start) * 1000.0
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

        return DownloadStreamResult(
            outcome=DownloadOutcome(
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
                canonical_url=plan.canonical_url,
                canonical_index=plan.canonical_index,
                original_url=plan.original_url,
            )
        )


def finalize_candidate_download(
    plan: DownloadPreflightPlan, stream: DownloadStreamResult
) -> DownloadOutcome:
    """Combine streaming results into a finalized :class:`DownloadOutcome`."""

    if stream.outcome is not None:
        return stream.outcome

    if stream.strategy is None or stream.strategy_context is None or stream.response is None:
        raise ValueError("Stream result missing required strategy context")

    classification = stream.classification or Classification.UNKNOWN
    # =========================================================================
    # Phase 3: Streaming Finalization Integration (Atomic File Operations)
    # =========================================================================
    streaming_finalization = None
    if _streaming_finalization_enabled():
        try:
            streaming_finalization = _try_streaming_finalization(
                stream.strategy,
                plan.artifact,
                classification,
                stream.strategy_context,
            )
            if streaming_finalization is not None:
                LOGGER.debug(
                    "streaming_finalization_complete",
                    extra={"extra_fields": {"classification": classification.value}},
                )
                return streaming_finalization
        except Exception as e:
            LOGGER.debug("streaming_finalization_failed", extra={"extra_fields": {"error": str(e)}})

    return stream.strategy.finalize_artifact(
        plan.artifact,
        classification,
        stream.strategy_context,
    )


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
        response: httpx.Response,
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
        response: httpx.Response,
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
            response=context.response,
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
            canonical_url=context.canonical_url,
            canonical_index=context.canonical_index,
            original_url=context.original_url,
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
def _cached_sha256(signature: tuple[str, int, int]) -> str | None:
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
) -> tuple[bool, str]:
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
        self._parsers: dict[str, RobotFileParser] = {}
        self._fetched_at: dict[str, float] = {}
        self._ttl = float(ttl_seconds)
        self._lock = threading.Lock()

    def is_allowed(self, client: httpx.Client, url: str, timeout: float) -> bool:
        """Return ``False`` when robots.txt forbids fetching ``url``."""

        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return True
        origin = f"{parsed.scheme}://{parsed.netloc}"
        parser = self._lookup_parser(client, origin, timeout)

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
        client: httpx.Client,
        origin: str,
        timeout: float,
    ) -> RobotFileParser:
        now = time.time()
        with self._lock:
            parser = self._parsers.get(origin)
            fetched_at = self._fetched_at.get(origin, 0.0)

        if parser is None or (self._ttl > 0 and (now - fetched_at) >= self._ttl):
            parser = self._fetch(client, origin, timeout)
            with self._lock:
                self._parsers[origin] = parser
                self._fetched_at[origin] = now

        return parser

    def _fetch(self, client: httpx.Client, origin: str, timeout: float) -> RobotFileParser:
        """Fetch and parse the robots.txt policy for ``origin``."""

        robots_url = origin.rstrip("/") + "/robots.txt"
        parser = RobotFileParser()
        parser.set_url(robots_url)
        try:
            response_cm = request_with_retries(
                client,
                "GET",
                robots_url,
                role="metadata",
                timeout=min(timeout, 5.0),
                allow_redirects=True,
                max_retries=1,
                max_retry_duration=min(timeout, 5.0),
                backoff_max=min(timeout, 5.0),
                retry_after_cap=min(timeout, 5.0),
            )
            if hasattr(response_cm, "__enter__") and hasattr(response_cm, "__exit__"):
                response_context: contextlib.AbstractContextManager[httpx.Response] = response_cm  # type: ignore[assignment]
            else:
                response_context = contextlib.nullcontext(response_cm)

            with response_context as response:
                if response.status_code and response.status_code >= 400:
                    parser.parse([])
                else:
                    body = response.text or ""
                    parser.parse(body.splitlines())
        except Exception:
            # Robots failures degrade to an empty policy so downloads continue.
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


# ============================================================================
# Phase 3: Streaming Finalization Helpers (Atomic File Operations)
# ============================================================================


def _streaming_finalization_enabled() -> bool:
    """Check if streaming finalization is enabled."""
    try:
        from DocsToKG.ContentDownload.streaming_integration import streaming_enabled

        return streaming_enabled()
    except ImportError:
        return False


def _try_streaming_finalization(
    strategy: DownloadStrategy,
    artifact: WorkArtifact,
    classification: Classification,
    context: DownloadStrategyContext,
) -> DownloadOutcome | None:
    """Try finalization using streaming module for atomic operations."""
    try:
        from DocsToKG.ContentDownload.streaming_integration import use_streaming_for_finalization

        # Check if streaming finalization should be used
        mock_outcome = DownloadOutcome(
            classification=classification,
            path=str(context.dest_path) if context.dest_path else None,
            http_status=None,
            content_type=context.content_type,
            elapsed_ms=context.elapsed_ms,
            reason=None,
            reason_detail=None,
        )

        if not use_streaming_for_finalization(mock_outcome):
            return None

        # Use existing strategy finalization (streaming validates atomically)
        return strategy.finalize_artifact(artifact, classification, context)

    except (ImportError, AttributeError, Exception):
        return None


def validate_classification(
    classification: Classification | str | None,
    artifact: WorkArtifact,
    options: DownloadOptions | DownloadContext,
) -> ValidationResult:
    """Validate resolver classification against configured expectations."""

    normalized = Classification.from_wire(classification)
    expected: Classification | None = Classification.PDF
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
    previous_index: Mapping[str, dict[str, Any]],
    options: DownloadContext,
) -> ResumeDecision:
    """Normalise previous attempts and detect early skip conditions."""

    previous_map: dict[str, dict[str, Any]] = {}
    for previous_url, entry in previous_index.items():
        if not isinstance(entry, dict):
            continue
        canonical_index = canonical_for_index(previous_url)
        etag = entry.get("etag")
        last_modified = entry.get("last_modified")
        path = entry.get("path")
        sha256 = entry.get("sha256")
        content_length = entry.get("content_length")
        content_length_value: int | None
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

        previous_map[canonical_index] = {
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
    options: DownloadOptions | DownloadContext,
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

    if getattr(options, "dry_run", False):
        return

    with contextlib.suppress(FileNotFoundError):
        dest_path.unlink()


def build_download_outcome(
    *,
    artifact: WorkArtifact,
    classification: Classification | str | None,
    dest_path: Path | None,
    response: httpx.Response | None,
    elapsed_ms: float,
    flagged_unknown: bool,
    sha256: str | None,
    content_length: int | None,
    etag: str | None,
    last_modified: str | None,
    extracted_text_path: str | None,
    dry_run: bool,
    tail_bytes: bytes | None,
    head_precheck_passed: bool,
    min_pdf_bytes: int,
    tail_check_bytes: int,
    retry_after: float | None,
    options: DownloadOptions | DownloadContext | None = None,
    canonical_url: str | None = None,
    canonical_index: str | None = None,
    original_url: str | None = None,
) -> DownloadOutcome:
    """Compose a :class:`DownloadOutcome` with shared validation logic."""

    status_code = response.status_code if response is not None else None
    headers = response.headers if response is not None else httpx.Headers()

    breaker_host_state: str | None = None
    breaker_resolver_state: str | None = None
    breaker_open_remaining_ms: int | None = None
    breaker_recorded: str | None = None
    if response is not None:
        extensions = getattr(response, "extensions", None)
        if isinstance(extensions, Mapping):
            host_state_val = extensions.get("breaker_host_state")
            if isinstance(host_state_val, str):
                breaker_host_state = host_state_val
            resolver_state_val = extensions.get("breaker_resolver_state")
            if isinstance(resolver_state_val, str):
                breaker_resolver_state = resolver_state_val
            remaining_val = extensions.get("breaker_open_remaining_ms")
            if isinstance(remaining_val, (int, float)):
                breaker_open_remaining_ms = int(remaining_val)
            recorded_val = extensions.get("breaker_recorded")
            if isinstance(recorded_val, str):
                breaker_recorded = recorded_val

    breaker_kwargs = {
        "breaker_host_state": breaker_host_state,
        "breaker_resolver_state": breaker_resolver_state,
        "breaker_open_remaining_ms": breaker_open_remaining_ms,
        "breaker_recorded": breaker_recorded,
    }

    canonical_value = canonical_url
    if canonical_value is None and response is not None:
        try:
            canonical_value = str(response.request.url)
        except AttributeError:
            canonical_value = None
    canonical_index_value = canonical_index or canonical_value
    original_value = original_url or canonical_value

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
                http_status=status_code,
                content_type=headers.get("Content-Type"),
                elapsed_ms=elapsed_ms,
                reason=ReasonCode.PDF_TOO_SMALL,
                reason_detail="pdf-too-small",
                sha256=None,
                content_length=None,
                etag=etag,
                last_modified=last_modified,
                extracted_text_path=extracted_text_path,
                retry_after=retry_after,
                canonical_url=canonical_value,
                canonical_index=canonical_index_value,
                original_url=original_value,
                **breaker_kwargs,
            )

        if tail_contains_html(tail_bytes):
            with contextlib.suppress(OSError):
                dest_path.unlink()
            return DownloadOutcome(
                classification=Classification.MISS,
                path=None,
                http_status=status_code,
                content_type=headers.get("Content-Type"),
                elapsed_ms=elapsed_ms,
                reason=ReasonCode.HTML_TAIL_DETECTED,
                reason_detail="html-tail-detected",
                sha256=None,
                content_length=None,
                etag=etag,
                last_modified=last_modified,
                extracted_text_path=extracted_text_path,
                retry_after=retry_after,
                canonical_url=canonical_value,
                canonical_index=canonical_index_value,
                original_url=original_value,
                **breaker_kwargs,
            )

        if not has_pdf_eof(dest_path, window_bytes=tail_check_bytes):
            with contextlib.suppress(OSError):
                dest_path.unlink()
            return DownloadOutcome(
                classification=Classification.MISS,
                path=None,
                http_status=status_code,
                content_type=headers.get("Content-Type"),
                elapsed_ms=elapsed_ms,
                reason=ReasonCode.PDF_EOF_MISSING,
                reason_detail="pdf-eof-missing",
                sha256=None,
                content_length=None,
                etag=etag,
                last_modified=last_modified,
                extracted_text_path=extracted_text_path,
                retry_after=retry_after,
                canonical_url=canonical_value,
                canonical_index=canonical_index_value,
                original_url=original_value,
                **breaker_kwargs,
            )

    if isinstance(options, DownloadContext):
        validation_context = options
    else:
        # Convert DownloadOptions or other mapping to DownloadContext
        validation_context = DownloadContext.from_mapping(
            {
                k: getattr(options, k, None)
                for k in [
                    "dry_run",
                    "list_only",
                    "extract_html_text",
                    "previous",
                    "sniff_bytes",
                    "min_pdf_bytes",
                    "tail_check_bytes",
                    "robots_checker",
                    "content_addressed",
                    "verify_cache_digest",
                    "domain_content_rules",
                    "host_accept_overrides",
                    "progress_callback",
                    "skip_head_precheck",
                    "head_precheck_passed",
                    "global_manifest_index",
                    "size_warning_threshold",
                    "chunk_size",
                    "stream_retry_attempts",
                    "extra",
                ]
            }
        )

    validation = validate_classification(classification_code, artifact, validation_context)
    if not validation.is_valid:
        detail = validation.detail or "classification-invalid"
        return DownloadOutcome(
            classification=Classification.MISS,
            path=None,
            http_status=status_code,
            content_type=headers.get("Content-Type"),
            elapsed_ms=elapsed_ms,
            reason=ReasonCode.UNKNOWN,
            reason_detail=detail,
            sha256=None,
            content_length=content_length,
            etag=etag,
            last_modified=last_modified,
            extracted_text_path=extracted_text_path,
            retry_after=retry_after,
            canonical_url=canonical_value,
            canonical_index=canonical_index_value,
            original_url=original_value,
            **breaker_kwargs,
        )

    try:
        normalized_status = int(status_code) if status_code is not None else None
    except (TypeError, ValueError):
        normalized_status = None

    conditional_hit = normalized_status == 304 or classification_code is Classification.CACHED

    reason_code: ReasonCode | None = None
    reason_detail: str | None = None
    if flagged_unknown and classification_code is Classification.PDF:
        reason_code = ReasonCode.PDF_SNIFF_UNKNOWN
        reason_detail = "pdf-sniff-unknown"
    elif conditional_hit:
        reason_code = ReasonCode.CONDITIONAL_NOT_MODIFIED
        if reason_detail is None:
            reason_detail = "not-modified"

    path_str = normalize_manifest_path(dest_path) if dest_path else None
    normalized_text_path = (
        normalize_manifest_path(extracted_text_path) if extracted_text_path else None
    )

    metadata: dict[str, Any] = {}
    if response is not None:
        req_meta = response.request.extensions.get("docs_network_meta")
        if isinstance(req_meta, Mapping):
            metadata["network"] = dict(req_meta)
            rl_snapshot = {
                "wait_ms": req_meta.get("rate_limiter_wait_ms"),
                "backend": req_meta.get("rate_limiter_backend"),
                "mode": req_meta.get("rate_limiter_mode"),
                "role": req_meta.get("rate_limiter_role"),
            }
            cleaned = {k: v for k, v in rl_snapshot.items() if v is not None}
            if cleaned:
                metadata["rate_limiter"] = cleaned
            breaker_snapshot = req_meta.get("breaker")
            if isinstance(breaker_snapshot, Mapping):
                metadata["breaker"] = dict(breaker_snapshot)

    breaker_host_state: str | None = None
    breaker_resolver_state: str | None = None
    breaker_recorded: str | None = None
    breaker_remaining_ms: int | None = None
    breaker_snapshot = metadata.get("breaker")
    if isinstance(breaker_snapshot, Mapping):
        host_state_value = breaker_snapshot.get("host_state") or breaker_snapshot.get("host")
        if isinstance(host_state_value, str):
            breaker_host_state = host_state_value
        resolver_state_value = breaker_snapshot.get("resolver_state") or breaker_snapshot.get(
            "resolver"
        )
        if isinstance(resolver_state_value, str):
            breaker_resolver_state = resolver_state_value
        recorded_value = breaker_snapshot.get("recorded")
        if isinstance(recorded_value, str):
            breaker_recorded = recorded_value
        remaining_value = breaker_snapshot.get("cooldown_remaining_ms")
        if isinstance(remaining_value, (int, float)):
            breaker_remaining_ms = int(remaining_value)

    breaker_meta = {
        "host_state": breaker_host_state,
        "resolver_state": breaker_resolver_state,
        "remaining_ms": breaker_open_remaining_ms,
        "recorded": breaker_recorded,
    }
    breaker_cleaned = {k: v for k, v in breaker_meta.items() if v is not None}
    if breaker_cleaned:
        metadata["breaker"] = breaker_cleaned

    return DownloadOutcome(
        classification=classification_code,
        path=path_str,
        http_status=status_code,
        content_type=headers.get("Content-Type"),
        elapsed_ms=elapsed_ms,
        reason=reason_code,
        reason_detail=reason_detail,
        sha256=sha256,
        content_length=content_length,
        etag=etag,
        last_modified=last_modified,
        extracted_text_path=normalized_text_path,
        retry_after=retry_after,
        metadata=metadata,
        canonical_url=canonical_value,
        canonical_index=canonical_index_value,
        original_url=original_value,
        **breaker_kwargs,
    )


def _collect_location_urls(work: dict[str, Any]) -> dict[str, list[str]]:
    """Return landing/PDF/source URL collections derived from OpenAlex metadata.

    Args:
        work: OpenAlex work payload as returned by the Works API.

    Returns:
        Dictionary containing ``landing``, ``pdf``, and ``sources`` URL lists.
    """

    landing_urls: list[str] = []
    pdf_urls: list[str] = []
    sources: list[str] = []

    def _append_location(loc: dict[str, Any] | None) -> None:
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


def _cohort_order_for(artifact: WorkArtifact) -> list[str]:
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
    work: dict[str, Any],
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
    client: httpx.Client | None,
    artifact: WorkArtifact,
    url: str,
    referer: str | None,
    timeout: float,
    context: DownloadContext | Mapping[str, Any] | None = None,
    *,
    original_url: str | None = None,
    origin_host: str | None = None,
    head_precheck_passed: bool = False,
    telemetry: Any | None = None,
    run_id: str | None = None,
) -> DownloadOutcome:
    """Download a single candidate URL and classify the payload.

    Phase diagram::

        [prepare_candidate_download]
                  |--(robots/cache skip)--> outcome
                  v
        [stream_candidate_payload]
                  |--(cached/error)-------> outcome
                  v
        [finalize_candidate_download] ---> outcome

    Each helper owns a single responsibility: preflight assembles request
    context, streaming performs network I/O, and finalization converts the
    structured stream result into a manifest-ready :class:`DownloadOutcome`.
    """

    ctx = DownloadContext.from_mapping(context)
    plan = prepare_candidate_download(
        client,
        artifact,
        url,
        referer,
        timeout,
        ctx,
        head_precheck_passed=head_precheck_passed,
        original_url=original_url,
        origin_host=origin_host,
        telemetry=telemetry,
        run_id=run_id,
    )
    if plan.skip_outcome is not None:
        return plan.skip_outcome

    stream_result = stream_candidate_payload(plan)
    return finalize_candidate_download(plan, stream_result)


def process_one_work(
    work: WorkArtifact | dict[str, Any],
    client: httpx.Client | None,
    pdf_dir: Path,
    html_dir: Path,
    xml_dir: Path,
    pipeline: ResolverPipeline,
    logger: RunTelemetry,
    metrics: ResolverMetrics,
    *,
    options: DownloadContext,
    strategy_selector: Callable[
        [Classification], DownloadStrategy
    ] = get_strategy_for_classification,
) -> dict[str, Any]:
    """Process a single work artifact through the resolver pipeline.

    Args:
        work: Either a preconstructed :class:`WorkArtifact` or a raw OpenAlex
            work payload. Raw payloads are normalised via :func:`create_artifact`.
        client: Optional HTTPX client to use for resolver/download operations. When
            ``None``, the shared client is used.
        pdf_dir: Directory where PDF artefacts are written.
        html_dir: Directory where HTML artefacts are written.
        xml_dir: Directory where XML artefacts are written.
        pipeline: Resolver pipeline orchestrating downstream resolvers.
        logger: Structured attempt logger capturing manifest records.
        metrics: Resolver metrics collector.
        options: :class:`DownloadContext` describing download behaviour for the work.

    Returns:
        Dictionary summarizing the outcome (saved/html_only/skipped flags).

    Raises:
        httpx.HTTPError: Propagated if resolver HTTP requests fail
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
    previous_lookup = options.previous_lookup

    result = {
        "work_id": artifact.work_id,
        "saved": False,
        "html_only": False,
        "xml_only": False,
        "skipped": False,
        "downloaded_bytes": 0,
    }

    # Plan job for idempotency tracking if enabled
    job_id: str | None = None
    idempotency_conn: Any | None = None
    if ENABLE_IDEMPOTENCY:
        try:
            # Attempt to get telemetry database connection from logger
            telemetry_conn = getattr(logger, "conn", None)
            if telemetry_conn is not None:
                from DocsToKG.ContentDownload.job_planning import plan_job_if_absent

                # Canonicalize the URL for consistency
                canonical_url = (
                    artifact.candidate_urls[0] if artifact.candidate_urls else artifact.work_id
                )
                job_id = plan_job_if_absent(
                    telemetry_conn,
                    work_id=artifact.work_id,
                    artifact_id=artifact.artifact_id,
                    canonical_url=canonical_url,
                )
                idempotency_conn = telemetry_conn
                LOGGER.debug(f"Planned job {job_id} for artifact {artifact.artifact_id}")
        except Exception as e:  # pylint: disable=broad-except
            LOGGER.debug(f"Failed to plan idempotency job: {e}")
            job_id = None
            idempotency_conn = None

    raw_previous = previous_lookup.get(artifact.work_id, {})
    if not isinstance(raw_previous, Mapping):
        raw_previous = {}

    resume_decision = handle_resume_logic(artifact, raw_previous, options)
    previous_map = resume_decision.previous_map
    download_context = options.to_context(
        {
            "previous": previous_map,
            "global_manifest_index": getattr(pipeline, "_global_manifest_index", {}),
        }
    )
    download_context.mark_explicit(
        "dry_run",
        "extract_html_text",
        "list_only",
        "sniff_bytes",
        "min_pdf_bytes",
        "tail_check_bytes",
        "robots_checker",
        "content_addressed",
        "previous",
        "global_manifest_index",
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

    active_client = client or get_http_client()

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
        active_client,
        artifact,
        context=download_context,
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

    # Mark job as successful if idempotency is enabled
    if ENABLE_IDEMPOTENCY and job_id and idempotency_conn and pipeline_result.success:
        try:
            from DocsToKG.ContentDownload.job_state import advance_state

            # Advance to FINALIZED state for successful downloads
            advance_state(
                idempotency_conn,
                job_id=job_id,
                to_state="FINALIZED",
                allowed_from=("PLANNED", "LEASED", "HEAD_DONE", "RESUME_OK", "STREAMING"),
            )
            LOGGER.debug(f"Job {job_id} marked as FINALIZED")
        except Exception as e:  # pylint: disable=broad-except
            LOGGER.debug(f"Failed to mark job as finalized: {e}")

    return result
