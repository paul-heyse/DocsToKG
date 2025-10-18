"""Helper utilities covering the modular download execution pipeline.

The functions in this module break the legacy ``download_candidate`` workflow
into focused phases that can be invoked and tested independently:

``prepare_candidate_download`` -> cache/preflight setup
``stream_candidate_payload`` -> HTTP execution and persistence
``finalize_candidate_download`` -> strategy-driven outcome assembly
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import requests

from DocsToKG.ContentDownload.core import (
    DEFAULT_MIN_PDF_BYTES,
    DEFAULT_SNIFF_BYTES,
    DEFAULT_TAIL_CHECK_BYTES,
    Classification,
    DownloadContext,
    PDF_LIKE,
    ReasonCode,
    WorkArtifact,
    atomic_write,
    atomic_write_text,
    classify_payload,
    normalize_url,
    update_tail_buffer,
)
from DocsToKG.ContentDownload.errors import log_download_failure
from DocsToKG.ContentDownload.networking import (
    CachedResult,
    ConditionalRequestHelper,
    ContentPolicyViolation,
    ModifiedResult,
    head_precheck,
    parse_retry_after_header,
    request_with_retries,
)
from DocsToKG.ContentDownload.pipeline import DownloadOutcome

LOGGER = logging.getLogger("DocsToKG.ContentDownload")


@dataclass
class DownloadPreflightResult:
    """Container describing reusable state for the streaming phase."""

    context: DownloadContext
    sniff_limit: int = DEFAULT_SNIFF_BYTES
    min_pdf_bytes: int = DEFAULT_MIN_PDF_BYTES
    tail_check_bytes: int = DEFAULT_TAIL_CHECK_BYTES
    progress_callback: Optional[Callable[[int, Optional[int], str], None]] = None
    progress_interval: int = 128 * 1024
    content_policy: Optional[Dict[str, Any]] = None
    base_headers: Dict[str, str] = field(default_factory=dict)
    dry_run: bool = False
    extract_html_text: bool = False
    head_precheck_passed: bool = False
    cond_helper: ConditionalRequestHelper = field(default_factory=ConditionalRequestHelper)
    attempt_conditional: bool = True
    logged_conditional_downgrade: bool = False
    enable_resume: bool = False
    resume_bytes_offset: Optional[int] = None
    existing_path: Optional[str] = None
    previous_length: Optional[int] = None
    previous_sha: Optional[str] = None
    previous_etag: Optional[str] = None
    previous_last_modified: Optional[str] = None
    previous_mtime_ns: Optional[int] = None
    content_type_hint: str = ""
    accept_value: Optional[str] = None
    early_outcome: Optional[DownloadOutcome] = None


@dataclass
class StreamExecutionResult:
    """Result returned from the streaming phase."""

    outcome: Optional[DownloadOutcome]
    strategy: Optional[Any] = None
    strategy_context: Optional[Any] = None
    classification: Optional[Classification] = None


def prepare_candidate_download(
    *,
    session: requests.Session,
    artifact: WorkArtifact,
    url: str,
    referer: Optional[str],
    timeout: float,
    context_payload: Optional[Mapping[str, Any]] = None,
    head_precheck_passed: bool = False,
) -> DownloadPreflightResult:
    """Prepare shared download state before streaming the payload."""

    ctx = DownloadContext.from_mapping(context_payload)
    sniff_limit = ctx.sniff_bytes
    min_pdf_bytes = ctx.min_pdf_bytes
    tail_window_bytes = ctx.tail_check_bytes

    progress_callback = ctx.progress_callback
    progress_update_interval = 128 * 1024  # update progress every 128KB

    parsed_url = requests.utils.urlparse(url)
    domain_policies: Dict[str, Dict[str, Any]] = ctx.domain_content_rules
    host_key = (parsed_url.hostname or parsed_url.netloc or "").lower()
    content_policy: Optional[Dict[str, Any]] = None
    if domain_policies and host_key:
        content_policy = domain_policies.get(host_key)
        if content_policy is None and host_key.startswith("www."):
            content_policy = domain_policies.get(host_key[4:])

    headers: Dict[str, str] = {}
    if referer:
        headers["Referer"] = referer

    accept_overrides = ctx.host_accept_overrides
    accept_value: Optional[str] = None
    if isinstance(accept_overrides, Mapping):
        host_for_accept = (parsed_url.netloc or "").lower()
        if host_for_accept:
            accept_value = accept_overrides.get(host_for_accept)
            if accept_value is None and host_for_accept.startswith("www."):
                accept_value = accept_overrides.get(host_for_accept[4:])
    if accept_value:
        headers["Accept"] = str(accept_value)

    resume_requested = bool(getattr(ctx, "enable_range_resume", False))
    if resume_requested:
        ctx.extra["resume_disabled"] = True
        if not ctx.extra.get("resume_warning_logged"):
            LOGGER.warning(
                "Range resume requested for %s; feature is deprecated and will be ignored.",
                url,
                extra={
                    "reason": "resume-disabled",
                    "extra_fields": {
                        "url": url,
                        "work_id": artifact.work_id,
                    },
                },
            )
            ctx.extra["resume_warning_logged"] = True
    else:
        ctx.extra.pop("resume_disabled", None)
    ctx.enable_range_resume = False
    enable_resume = False

    dry_run = ctx.dry_run
    robots_checker = ctx.robots_checker
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
            outcome = DownloadOutcome(
                classification=Classification.SKIPPED,
                path=None,
                http_status=None,
                content_type=None,
                elapsed_ms=0.0,
                reason=ReasonCode.ROBOTS_DISALLOWED,
                reason_detail="robots-disallowed",
            )
            return DownloadPreflightResult(
                context=ctx,
                sniff_limit=sniff_limit,
                min_pdf_bytes=min_pdf_bytes,
                tail_check_bytes=tail_window_bytes,
                progress_callback=progress_callback,
                progress_interval=progress_update_interval,
                content_policy=content_policy,
                base_headers=headers,
                dry_run=dry_run,
                extract_html_text=ctx.extract_html_text,
                head_precheck_passed=head_precheck_passed,
                early_outcome=outcome,
            )

    head_precheck_state = head_precheck_passed or ctx.head_precheck_passed
    if not head_precheck_state and not ctx.skip_head_precheck:
        head_precheck_state = head_precheck(
            session, url, timeout, content_policy=content_policy
        )
        ctx.head_precheck_passed = head_precheck_state

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

    return DownloadPreflightResult(
        context=ctx,
        sniff_limit=sniff_limit,
        min_pdf_bytes=min_pdf_bytes,
        tail_check_bytes=tail_window_bytes,
        progress_callback=progress_callback,
        progress_interval=progress_update_interval,
        content_policy=content_policy,
        base_headers=headers,
        dry_run=dry_run,
        extract_html_text=ctx.extract_html_text,
        head_precheck_passed=head_precheck_state,
        cond_helper=cond_helper,
        attempt_conditional=True,
        logged_conditional_downgrade=False,
        enable_resume=enable_resume,
        resume_bytes_offset=None,
        existing_path=existing_path,
        previous_length=previous_length,
        previous_sha=previous_sha,
        previous_etag=previous_etag,
        previous_last_modified=previous_last_modified,
        previous_mtime_ns=previous_mtime_ns,
        content_type_hint="",
        accept_value=accept_value,
        early_outcome=None,
    )


def stream_candidate_payload(
    *,
    session: requests.Session,
    artifact: WorkArtifact,
    url: str,
    timeout: float,
    preflight: DownloadPreflightResult,
    cleanup_sidecar: Callable[[WorkArtifact, Classification, DownloadContext], None],
    validate_cached_artifact: Callable[[CachedResult, bool], Tuple[bool, str]],
    strategy_selector: Callable[[Classification], Any],
    strategy_context_factory: Callable[..., Any],
    content_address_factory: Optional[Callable[[Path, str], Path]] = None,
) -> StreamExecutionResult:
    """Execute the HTTP download and persist the payload to disk."""

    ctx = preflight.context
    attempt_conditional = preflight.attempt_conditional
    cond_helper = preflight.cond_helper
    logged_conditional_downgrade = preflight.logged_conditional_downgrade
    enable_resume = preflight.enable_resume
    resume_bytes_offset = preflight.resume_bytes_offset
    existing_path = preflight.existing_path
    previous_length = preflight.previous_length
    sniff_limit = preflight.sniff_limit
    min_pdf_bytes = preflight.min_pdf_bytes
    tail_window_bytes = preflight.tail_check_bytes
    progress_callback = preflight.progress_callback
    progress_update_interval = preflight.progress_interval
    content_policy = preflight.content_policy
    dry_run = preflight.dry_run
    extract_html_text = preflight.extract_html_text
    head_precheck_passed = preflight.head_precheck_passed

    base_headers = dict(preflight.base_headers)
    content_type_hint = preflight.content_type_hint

    outer_start = time.monotonic()

    try:
        while True:
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
                outcome = DownloadOutcome(
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
                return StreamExecutionResult(outcome=outcome)
            except requests.RequestException as exc:
                elapsed_ms = (time.monotonic() - start) * 1000.0
                http_status = getattr(exc.response, "status_code", None) if hasattr(exc, "response") else None
                log_download_failure(
                    LOGGER,
                    url,
                    artifact.work_id,
                    http_status=http_status,
                    reason_code="request_exception",
                    error_details=str(exc),
                    exception=exc,
                )
                outcome = DownloadOutcome(
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
                return StreamExecutionResult(outcome=outcome)

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
                    outcome = DownloadOutcome(
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
                    return StreamExecutionResult(outcome=outcome)

                if response.status_code == 304:
                    if not attempt_conditional:
                        LOGGER.warning(
                            "Received HTTP 304 for %s without conditional headers; treating as http_error.",
                            url,
                        )
                        outcome = DownloadOutcome(
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
                        return StreamExecutionResult(outcome=outcome)

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
                    is_valid_cache, validation_mode = validate_cached_artifact(
                        cached,
                        ctx.verify_cache_digest,
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
                    )
                    outcome.metadata["cache_validation_mode"] = validation_mode
                    if ctx.extra.get("resume_disabled"):
                        outcome.metadata.setdefault("resume_disabled", True)
                    return StreamExecutionResult(outcome=outcome)

                if response.status_code == 206:
                    if resume_bytes_offset is None or resume_bytes_offset <= 0:
                        LOGGER.warning(
                            "Received HTTP 206 for %s without Range request; treating as error.",
                            url,
                        )
                        outcome = DownloadOutcome(
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
                        return StreamExecutionResult(outcome=outcome)
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
                    cleanup_sidecar(artifact, Classification.PDF, ctx)
                    existing_path = None
                    resume_bytes_offset = None
                    attempt_conditional = False
                    cond_helper = ConditionalRequestHelper()
                    continue

                content_type = response.headers.get("Content-Type") or content_type_hint
                content_type_hint = content_type or content_type_hint
                disposition = response.headers.get("Content-Disposition")
                content_length_hint_raw = response.headers.get("Content-Length")
                content_length_hint: Optional[int]
                if isinstance(content_length_hint_raw, str):
                    try:
                        content_length_hint = int(content_length_hint_raw)
                    except ValueError:
                        content_length_hint = None
                else:
                    content_length_hint = None

                if content_length_hint and content_length_hint <= 0:
                    content_length_hint = None

                size_warning_threshold = ctx.size_warning_threshold
                if (
                    size_warning_threshold
                    and content_length_hint
                    and content_length_hint > size_warning_threshold
                ):
                    threshold_mb = size_warning_threshold / (1024 * 1024)
                    LOGGER.warning(
                        "download-size-warning",
                        extra={
                            "extra_fields": {
                                "url": url,
                                "work_id": artifact.work_id,
                                "content_length": content_length_hint,
                                "threshold": size_warning_threshold,
                            }
                        },
                    )
                    if ctx.skip_large_downloads:
                        outcome = DownloadOutcome(
                            classification=Classification.SKIPPED,
                            path=None,
                            http_status=response.status_code,
                            content_type=content_type,
                            elapsed_ms=elapsed_ms,
                            reason=ReasonCode.CONTENT_TOO_LARGE,
                            reason_detail=f"content-length>{threshold_mb:.2f}MB",
                            sha256=None,
                            content_length=content_length_hint,
                            etag=None,
                            last_modified=None,
                            extracted_text_path=None,
                            retry_after=retry_after_hint,
                        )
                        cleanup_sidecar(artifact, Classification.PDF, ctx)
                        return StreamExecutionResult(outcome=outcome)

                modified_result = ModifiedResult.from_response(response, cond_helper)
                cond_helper.update_from_response(response, modified_result)

                if attempt_conditional and not modified_result.is_modified:
                    outcome = DownloadOutcome(
                        classification=Classification.CACHED,
                        path=existing_path,
                        http_status=response.status_code,
                        content_type=content_type,
                        elapsed_ms=elapsed_ms,
                        reason=ReasonCode.CONDITIONAL_NOT_MODIFIED,
                        reason_detail="not-modified",
                        sha256=modified_result.sha256,
                        content_length=modified_result.content_length,
                        etag=modified_result.etag,
                        last_modified=modified_result.last_modified,
                        extracted_text_path=None,
                        retry_after=retry_after_hint,
                    )
                    if ctx.extra.get("resume_disabled"):
                        outcome.metadata.setdefault("resume_disabled", True)
                    return StreamExecutionResult(outcome=outcome)

                chunk_size = int(ctx.chunk_size or (1 << 15))
                content_iter = response.iter_content(chunk_size=chunk_size)
                sniff_buffer = bytearray()
                prefetched: List[bytes] = []
                byte_count = resume_bytes_offset or 0
                hasher = hashlib.sha256() if not dry_run else None

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
                    outcome = DownloadOutcome(
                        classification=Classification.MISS,
                        path=None,
                        http_status=response.status_code,
                        content_type=content_type,
                        elapsed_ms=elapsed_ms,
                        reason=ReasonCode.UNKNOWN,
                        reason_detail="classifier-unknown",
                        retry_after=retry_after_hint,
                    )
                    return StreamExecutionResult(outcome=outcome)

                strategy_factory = getattr(ctx, "extra", {}).get("strategy_factory")
                if not callable(strategy_factory):
                    strategy_factory = strategy_selector
                strategy = strategy_factory(detected)
                strategy_context = strategy_context_factory(
                    download_context=ctx,
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
                        outcome = strategy_context.skip_outcome or DownloadOutcome(
                            classification=Classification.SKIPPED,
                            path=None,
                            http_status=None,
                            content_type=None,
                            elapsed_ms=elapsed_ms,
                            reason=ReasonCode.UNKNOWN,
                            reason_detail="strategy-skip",
                        )
                        return StreamExecutionResult(outcome=outcome)
                    strategy_context.retry_after = retry_after_hint
                    processed_classification = strategy.process_response(
                        response, artifact, strategy_context
                    )
                    strategy_context.elapsed_ms = elapsed_ms
                    strategy_context.etag = modified_result.etag
                    strategy_context.last_modified = modified_result.last_modified
                    strategy_context.flagged_unknown = flagged_unknown
                    return StreamExecutionResult(
                        outcome=None,
                        strategy=strategy,
                        strategy_context=strategy_context,
                        classification=processed_classification,
                    )

                if detected == Classification.HTML:
                    dest_dir = artifact.html_dir
                    default_suffix = ".html"
                elif detected == Classification.XML:
                    dest_dir = artifact.xml_dir
                    default_suffix = ".xml"
                else:
                    dest_dir = artifact.pdf_dir
                    default_suffix = ".pdf"

                from DocsToKG.ContentDownload.core import _infer_suffix

                suffix = _infer_suffix(url, content_type, disposition, detected, default_suffix)
                dest_path = dest_dir / f"{artifact.base_stem}{suffix}"
                strategy_context.dest_path = dest_path
                strategy_context.content_type = content_type
                strategy_context.disposition = disposition
                if not strategy.should_download(artifact, strategy_context):
                    outcome = strategy_context.skip_outcome or DownloadOutcome(
                        classification=Classification.SKIPPED,
                        path=None,
                        http_status=None,
                        content_type=None,
                        elapsed_ms=elapsed_ms,
                        reason=ReasonCode.UNKNOWN,
                        reason_detail="strategy-skip",
                    )
                    return StreamExecutionResult(outcome=outcome)

                dest_path.parent.mkdir(parents=True, exist_ok=True)

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

                try:
                    atomic_write(
                        dest_path,
                        _stream_chunks(),
                        hasher=hasher,
                        keep_partial_on_error=True,
                    )
                except (requests.exceptions.ChunkedEncodingError, AttributeError) as exc:
                    LOGGER.warning(
                        "Streaming download failed for %s: %s",
                        url,
                        exc,
                        extra={"extra_fields": {"work_id": artifact.work_id}},
                    )
                    if dest_path is not None and not dry_run:
                        classification_hint = (
                            detected
                            if detected in {Classification.PDF, Classification.HTML, Classification.XML}
                            else Classification.PDF
                        )
                        cleanup_sidecar(artifact, classification_hint, ctx)
                    seen_attempts = ctx.stream_retry_attempts
                    if seen_attempts < 1:
                        ctx.stream_retry_attempts = seen_attempts + 1
                        attempt_conditional = False
                        LOGGER.info("Retrying download for %s after stream failure", url)
                        continue
                    elapsed_err = (time.monotonic() - start) * 1000.0
                    outcome = DownloadOutcome(
                        classification=Classification.HTTP_ERROR,
                        path=None,
                        http_status=response.status_code,
                        content_type=content_type,
                        elapsed_ms=elapsed_err,
                        reason=ReasonCode.REQUEST_EXCEPTION,
                        reason_detail=f"stream-error: {exc}",
                    )
                    return StreamExecutionResult(outcome=outcome)

                sha256: Optional[str] = None
                content_length: Optional[int] = None
                if hasher is not None:
                    sha256 = hasher.hexdigest()
                    content_length = byte_count
                if (
                    dest_path
                    and ctx.content_addressed
                    and sha256
                    and detected in PDF_LIKE
                    and callable(content_address_factory)
                ):
                    dest_path = content_address_factory(dest_path, sha256)

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
                        except Exception as exc:  # pragma: no cover - external dependency
                            LOGGER.warning("Failed to extract HTML text for %s: %s", dest_path, exc)
                        else:
                            if text:
                                text_path = dest_path.with_suffix(dest_path.suffix + ".txt")
                                atomic_write_text(text_path, text)
                                extracted_text_path = str(text_path)

                ctx.stream_retry_attempts = 0
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
                return StreamExecutionResult(
                    outcome=None,
                    strategy=strategy,
                    strategy_context=strategy_context,
                    classification=processed_classification,
                )
    except requests.RequestException as exc:
        elapsed_ms = (time.monotonic() - outer_start) * 1000.0
        http_status = getattr(exc.response, "status_code", None) if hasattr(exc, "response") else None
        log_download_failure(
            LOGGER,
            url,
            artifact.work_id,
            http_status=http_status,
            reason_code="request_exception",
            error_details=str(exc),
            exception=exc,
        )
        outcome = DownloadOutcome(
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
        return StreamExecutionResult(outcome=outcome)

    raise RuntimeError("stream_candidate_payload exited without returning a result")


def finalize_candidate_download(
    *,
    artifact: WorkArtifact,
    stream_result: StreamExecutionResult,
) -> DownloadOutcome:
    """Assemble the final outcome from the streaming phase result."""

    if stream_result.outcome is not None:
        return stream_result.outcome
    if (
        stream_result.strategy is None
        or stream_result.strategy_context is None
        or stream_result.classification is None
    ):
        raise RuntimeError("Streaming result missing strategy context for finalization.")
    return stream_result.strategy.finalize_artifact(
        artifact,
        stream_result.classification,
        stream_result.strategy_context,
    )
