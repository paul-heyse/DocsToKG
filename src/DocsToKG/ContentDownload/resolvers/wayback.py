# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.resolvers.wayback",
#   "purpose": "Wayback Machine resolver with CDX-first discovery algorithm",
#   "sections": [
#     {
#       "id": "waybackresolver",
#       "name": "WaybackResolver",
#       "anchor": "class-waybackresolver",
#       "kind": "class"
#     },
#     {
#       "id": "wayback-discovery",
#       "name": "_discover_snapshots",
#       "anchor": "function-wayback-discovery",
#       "kind": "function"
#     },
#     {
#       "id": "wayback-availability",
#       "name": "_check_availability",
#       "anchor": "function-wayback-availability",
#       "kind": "function"
#     },
#     {
#       "id": "wayback-cdx",
#       "name": "_query_cdx",
#       "anchor": "function-wayback-cdx",
#       "kind": "function"
#     },
#     {
#       "id": "wayback-html-parse",
#       "name": "_parse_html_for_pdf",
#       "anchor": "function-wayback-html-parse",
#       "kind": "function"
#     },
#     {
#       "id": "wayback-verify-pdf",
#       "name": "_verify_pdf_snapshot",
#       "anchor": "function-wayback-verify-pdf",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===
"""Resolver that queries the Internet Archive Wayback Machine with CDX-first discovery."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urljoin, urlparse

import httpx

from DocsToKG.ContentDownload.networking import request_with_retries
from DocsToKG.ContentDownload.telemetry_wayback import (
    AttemptContext,
    AttemptResult,
    CandidateDecision,
    DiscoveryStage,
    ModeSelected,
    PdfDiscoveryMethod,
    SkipReason,
    TelemetryWayback,
    create_telemetry_with_failsafe,
)
from DocsToKG.ContentDownload.urls import canonical_for_index

from .base import (
    RegisteredResolver,
    ResolverEvent,
    ResolverEventReason,
    ResolverResult,
    find_pdf_via_anchor,
    find_pdf_via_link,
    find_pdf_via_meta,
)

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.core import WorkArtifact
    from DocsToKG.ContentDownload.pipeline import ResolverConfig


LOGGER = logging.getLogger(__name__)


@dataclass
class _DiscoveryOutcome:
    """Internal container summarising snapshot discovery results."""

    url: Optional[str]
    metadata: Dict[str, Any]
    mode: ModeSelected
    attempt_result: AttemptResult
    skip_reason: Optional[SkipReason]
    skip_details: Optional[str]
    candidates_scanned: int


def _coerce_bool(value: Any) -> Optional[bool]:
    """Best-effort coercion of arbitrary metadata into an optional boolean."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "1", "yes", "y"}:
            return True
        if token in {"false", "0", "no", "n"}:
            return False
    return None


def _coerce_int(value: Any) -> Optional[int]:
    """Return ``value`` coerced to ``int`` when possible."""

    try:
        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str) and value.strip():
            return int(float(value))
    except (TypeError, ValueError):
        return None
    return None


def _extract_revalidated(meta: Mapping[str, Any]) -> Optional[bool]:
    """Return cache revalidation metadata when present."""

    cache_meta = meta.get("cache_metadata")
    if isinstance(cache_meta, Mapping):
        value = cache_meta.get("revalidated")
        coerced = _coerce_bool(value)
        if coerced is not None:
            return coerced
    return _coerce_bool(meta.get("revalidated"))


def _distance_to_pub_year(
    publication_year: Optional[int], timestamp: Optional[str]
) -> Optional[int]:
    """Return capture-year offset relative to ``publication_year`` when known."""

    if publication_year is None or not timestamp:
        return None
    match = re.match(r"(\d{4})", str(timestamp))
    if not match:
        return None
    try:
        capture_year = int(match.group(1))
    except ValueError:
        return None
    return capture_year - publication_year


class WaybackResolver(RegisteredResolver):
    """Fallback resolver that queries the Internet Archive Wayback Machine with CDX-first discovery."""

    name = "wayback"

    def __init__(
        self,
        *,
        telemetry: Optional[TelemetryWayback] = None,
        telemetry_factory: Optional[Callable[..., TelemetryWayback]] = None,
        telemetry_sinks: Optional[Iterable[Any]] = None,
        telemetry_jsonl_fallback: Optional[Path] = None,
        telemetry_kwargs: Optional[Mapping[str, Any]] = None,
        run_id: Optional[str] = None,
    ) -> None:
        self._telemetry_factory = telemetry_factory or create_telemetry_with_failsafe
        self._telemetry_kwargs: Dict[str, Any] = dict(telemetry_kwargs or {})
        self._telemetry_sinks: Sequence[Any] = list(telemetry_sinks or [])
        self._telemetry_jsonl_fallback = telemetry_jsonl_fallback
        self._run_id = run_id
        self._telemetry = telemetry
        if self._telemetry is None and run_id is not None:
            self._telemetry = self._telemetry_factory(  # type: ignore[call-arg]
                run_id,
                list(self._telemetry_sinks),
                jsonl_fallback_path=self._telemetry_jsonl_fallback,
                **self._telemetry_kwargs,
            )

    def configure_telemetry(
        self,
        *,
        run_id: str,
        sinks: Iterable[Any],
        jsonl_fallback_path: Optional[Path] = None,
        telemetry_factory_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """(Re)configure telemetry wiring for the resolver instance."""

        kwargs = dict(self._telemetry_kwargs)
        if telemetry_factory_kwargs:
            kwargs.update(telemetry_factory_kwargs)
        self._run_id = run_id
        self._telemetry_sinks = list(sinks)
        self._telemetry_jsonl_fallback = jsonl_fallback_path
        self._telemetry = self._telemetry_factory(  # type: ignore[call-arg]
            run_id,
            list(self._telemetry_sinks),
            jsonl_fallback_path=jsonl_fallback_path,
            **kwargs,
        )
        self._telemetry_kwargs = kwargs

    def _resolve_artifact_id(self, artifact: "WorkArtifact") -> str:
        metadata = getattr(artifact, "metadata", {}) or {}
        candidate = metadata.get("artifact_id")
        if isinstance(candidate, str) and candidate.strip():
            return candidate
        base_stem = getattr(artifact, "base_stem", None)
        if isinstance(base_stem, str) and base_stem:
            return base_stem
        return getattr(artifact, "work_id", "artifact")

    def is_enabled(self, config: "ResolverConfig", artifact: "WorkArtifact") -> bool:
        """Return ``True`` when prior resolver attempts have failed.

        Args:
            config: Resolver configuration (unused for enablement).
            artifact: Work record containing failed PDF URLs.

        Returns:
            bool: Whether the Wayback resolver should run.
        """
        return bool(artifact.failed_pdf_urls)

    def iter_urls(
        self,
        client: httpx.Client,
        config: "ResolverConfig",
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Query the Wayback Machine for archived versions of failed URLs using CDX-first discovery.

        Args:
            client: HTTPX client for HTTP calls.
            config: Resolver configuration providing timeouts and headers.
            artifact: Work metadata listing failed PDF URLs to retry.

        Yields:
            ResolverResult: Archived download URLs or diagnostic events.
        """
        if not artifact.failed_pdf_urls:
            yield ResolverResult(
                url=None,
                event=ResolverEvent.SKIPPED,
                event_reason=ResolverEventReason.NO_FAILED_URLS,
            )
            return

        # Get Wayback-specific configuration
        wayback_config = getattr(config, "wayback_config", {})
        year_window = wayback_config.get("year_window", 2)
        max_snapshots = wayback_config.get("max_snapshots", 8)
        min_pdf_bytes = wayback_config.get("min_pdf_bytes", 4096)
        html_parse_enabled = wayback_config.get("html_parse", True)

        telemetry = getattr(self, "_telemetry", None)
        artifact_id = self._resolve_artifact_id(artifact)

        for original_url in artifact.failed_pdf_urls:
            ctx: Optional[AttemptContext] = None
            attempt_ended = False
            mode_selected = ModeSelected.NONE
            attempt_result = AttemptResult.SKIPPED_NO_SNAPSHOT
            skip_reason: Optional[SkipReason] = SkipReason.NO_SNAPSHOT
            skip_details: Optional[str] = None
            candidates_scanned = 0

            try:
                canonical_url = canonical_for_index(original_url)
                publication_year = getattr(artifact, "publication_year", None)

                if telemetry is not None:
                    ctx = telemetry.emit_attempt_start(
                        artifact.work_id,
                        artifact_id,
                        original_url=original_url,
                        canonical_url=canonical_url,
                        publication_year=publication_year,
                    )

                outcome = self._discover_snapshots(
                    client,
                    config,
                    original_url,
                    canonical_url,
                    publication_year,
                    year_window,
                    max_snapshots,
                    min_pdf_bytes,
                    html_parse_enabled,
                    telemetry=telemetry,
                    ctx=ctx,
                )

                mode_selected = outcome.mode
                attempt_result = outcome.attempt_result
                skip_reason = outcome.skip_reason
                skip_details = outcome.skip_details
                candidates_scanned = outcome.candidates_scanned

                if outcome.url:
                    result_metadata = {
                        "original": original_url,
                        "canonical": canonical_url,
                        "source": "wayback",
                        **outcome.metadata,
                    }
                    if telemetry is not None and ctx is not None:
                        memento_ts = (
                            outcome.metadata.get("memento_ts")
                            or outcome.metadata.get("html_snapshot_ts")
                            or outcome.metadata.get("availability_timestamp")
                            or ""
                        )
                        telemetry.emit_emit(
                            ctx,
                            emitted_url=outcome.url,
                            memento_ts=str(memento_ts),
                            source_mode=outcome.mode,
                            http_ct_expected=str(
                                outcome.metadata.get("content_type", "application/pdf")
                            ),
                        )
                        telemetry.emit_attempt_end(
                            ctx,
                            mode_selected=outcome.mode,
                            result=outcome.attempt_result,
                            candidates_scanned=outcome.candidates_scanned,
                            extra={
                                "discovery_method": outcome.metadata.get("discovery_method"),
                            },
                        )
                        attempt_ended = True
                    yield ResolverResult(url=outcome.url, metadata=result_metadata)
                    continue

                skip_token = skip_reason.value if skip_reason else "no_snapshot"
                result_metadata = {
                    "original": original_url,
                    "canonical": canonical_url,
                    "reason": skip_token,
                    "details": skip_details,
                }
                if telemetry is not None and ctx is not None and skip_reason is not None:
                    telemetry.emit_skip(ctx, reason=skip_reason, details=skip_details)
                    telemetry.emit_attempt_end(
                        ctx,
                        mode_selected=mode_selected,
                        result=attempt_result,
                        candidates_scanned=candidates_scanned,
                        extra={
                            "skip_reason": skip_token,
                            "skip_details": skip_details,
                        },
                    )
                    attempt_ended = True
                yield ResolverResult(
                    url=None,
                    event=ResolverEvent.SKIPPED,
                    event_reason=ResolverEventReason.NO_FAILED_URLS,
                    metadata=result_metadata,
                )

            except httpx.TimeoutException as exc:
                attempt_result = AttemptResult.TIMEOUT
                skip_reason = SkipReason.TIMEOUT
                skip_details = str(exc)
                LOGGER.exception("Wayback resolver timeout for %s", original_url)
                if telemetry is not None and ctx is not None:
                    telemetry.emit_skip(ctx, reason=skip_reason, details=skip_details)
                    telemetry.emit_attempt_end(
                        ctx,
                        mode_selected=mode_selected,
                        result=attempt_result,
                        candidates_scanned=candidates_scanned,
                        extra={"skip_reason": skip_reason.value, "skip_details": skip_details},
                    )
                    attempt_ended = True
                yield ResolverResult(
                    url=None,
                    event=ResolverEvent.ERROR,
                    event_reason=ResolverEventReason.REQUEST_TIMEOUT,
                    metadata={
                        "original": original_url,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                )

            except httpx.HTTPError as exc:
                attempt_result = AttemptResult.ERROR_HTTP
                skip_reason = SkipReason.HTTP_ERROR
                skip_details = str(exc)
                LOGGER.exception("HTTP error while resolving Wayback snapshot for %s", original_url)
                if telemetry is not None and ctx is not None:
                    telemetry.emit_skip(ctx, reason=skip_reason, details=skip_details)
                    telemetry.emit_attempt_end(
                        ctx,
                        mode_selected=mode_selected,
                        result=attempt_result,
                        candidates_scanned=candidates_scanned,
                        extra={"skip_reason": skip_reason.value, "skip_details": skip_details},
                    )
                    attempt_ended = True
                yield ResolverResult(
                    url=None,
                    event=ResolverEvent.ERROR,
                    event_reason=ResolverEventReason.HTTP_ERROR,
                    metadata={
                        "original": original_url,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                )

            except Exception as exc:  # pragma: no cover - defensive logging
                attempt_result = AttemptResult.ERROR_HTTP
                skip_reason = SkipReason.HTTP_ERROR
                skip_details = str(exc)
                LOGGER.exception("Unexpected Wayback resolver error for %s", original_url)
                if telemetry is not None and ctx is not None:
                    telemetry.emit_skip(ctx, reason=skip_reason, details=skip_details)
                    telemetry.emit_attempt_end(
                        ctx,
                        mode_selected=mode_selected,
                        result=attempt_result,
                        candidates_scanned=candidates_scanned,
                        extra={"skip_reason": skip_reason.value, "skip_details": skip_details},
                    )
                    attempt_ended = True
                yield ResolverResult(
                    url=None,
                    event=ResolverEvent.ERROR,
                    event_reason=ResolverEventReason.UNEXPECTED_ERROR,
                    metadata={
                        "original": original_url,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                )

            finally:
                if telemetry is not None and ctx is not None and not attempt_ended:
                    extra: Dict[str, Any] = {}
                    if skip_reason is not None:
                        extra["skip_reason"] = skip_reason.value
                    if skip_details:
                        extra["skip_details"] = skip_details
                    telemetry.emit_attempt_end(
                        ctx,
                        mode_selected=mode_selected,
                        result=attempt_result,
                        candidates_scanned=candidates_scanned,
                        extra=extra or None,
                    )

    def _discover_snapshots(
        self,
        client: httpx.Client,
        config: "ResolverConfig",
        original_url: str,
        canonical_url: str,
        publication_year: Optional[int],
        year_window: int,
        max_snapshots: int,
        min_pdf_bytes: int,
        html_parse_enabled: bool,
        *,
        telemetry: Optional[TelemetryWayback] = None,
        ctx: Optional[AttemptContext] = None,
    ) -> _DiscoveryOutcome:
        """Discover the best PDF snapshot using CDX-first approach."""

        metadata: Dict[str, Any] = {"discovery_method": "none"}
        candidates_scanned = 0
        mode_selected = ModeSelected.NONE
        attempt_result = AttemptResult.SKIPPED_NO_SNAPSHOT
        skip_reason: Optional[SkipReason] = SkipReason.NO_SNAPSHOT
        skip_details: Optional[str] = None
        seen_non_pdf = False

        availability_url, availability_metadata = self._check_availability(
            client,
            config,
            original_url,
            telemetry=telemetry,
            ctx=ctx,
        )
        metadata.update(availability_metadata)

        if availability_url:
            if telemetry is not None and ctx is not None:
                telemetry.emit_candidate(
                    ctx,
                    archive_url=availability_url,
                    memento_ts=str(availability_metadata.get("availability_timestamp", "")),
                    statuscode=_coerce_int(availability_metadata.get("availability_status")),
                    mimetype="application/pdf",
                    source_stage=DiscoveryStage.AVAILABILITY,
                    decision=CandidateDecision.HEAD_CHECK,
                    distance_to_pub_year=_distance_to_pub_year(
                        publication_year,
                        availability_metadata.get("availability_timestamp"),
                    ),
                )

            passed, check_info = self._verify_pdf_snapshot(
                client,
                config,
                availability_url,
                min_pdf_bytes,
            )
            candidates_scanned += 1
            metadata["content_type"] = check_info.get("content_type")
            metadata["content_length"] = check_info.get("content_length")

            if telemetry is not None and ctx is not None:
                telemetry.emit_pdf_check(
                    ctx,
                    archive_pdf_url=availability_url,
                    head_status=check_info.get("head_status"),
                    content_type=check_info.get("content_type"),
                    content_length=check_info.get("content_length"),
                    is_pdf_signature=check_info.get("is_pdf_signature"),
                    min_bytes_pass=check_info.get("min_bytes_pass"),
                    decision=check_info.get("decision", CandidateDecision.HEAD_CHECK),
                )

            if passed:
                metadata["discovery_method"] = "availability"
                mode_selected = ModeSelected.PDF_DIRECT
                attempt_result = AttemptResult.EMITTED_PDF
                skip_reason = None
                skip_details = None
                return _DiscoveryOutcome(
                    url=availability_url,
                    metadata=metadata,
                    mode=mode_selected,
                    attempt_result=attempt_result,
                    skip_reason=skip_reason,
                    skip_details=skip_details,
                    candidates_scanned=candidates_scanned,
                )

            skip_reason = check_info.get("skip_reason") or skip_reason
            skip_details = check_info.get("skip_details") or skip_details
            seen_non_pdf = seen_non_pdf or skip_reason in {
                SkipReason.NON_PDF,
                SkipReason.BELOW_MIN_SIZE,
            }

        cdx_snapshots = self._query_cdx(
            client,
            config,
            original_url,
            publication_year,
            year_window,
            max_snapshots,
            telemetry=telemetry,
            ctx=ctx,
        )

        for snapshot in cdx_snapshots:
            snapshot_url = snapshot.get("archive_url")
            if not snapshot_url:
                continue

            mimetype = str(snapshot.get("mimetype", ""))
            statuscode = _coerce_int(snapshot.get("statuscode"))
            timestamp = str(snapshot.get("timestamp", ""))

            if mimetype == "application/pdf" and statuscode == 200:
                if telemetry is not None and ctx is not None:
                    telemetry.emit_candidate(
                        ctx,
                        archive_url=snapshot_url,
                        memento_ts=timestamp,
                        statuscode=statuscode,
                        mimetype=mimetype,
                        source_stage=DiscoveryStage.CDX,
                        decision=CandidateDecision.HEAD_CHECK,
                        distance_to_pub_year=_distance_to_pub_year(publication_year, timestamp),
                    )

                passed, check_info = self._verify_pdf_snapshot(
                    client,
                    config,
                    snapshot_url,
                    min_pdf_bytes,
                )
                candidates_scanned += 1
                metadata["content_type"] = check_info.get("content_type")
                metadata["content_length"] = check_info.get("content_length")

                if telemetry is not None and ctx is not None:
                    telemetry.emit_pdf_check(
                        ctx,
                        archive_pdf_url=snapshot_url,
                        head_status=check_info.get("head_status"),
                        content_type=check_info.get("content_type"),
                        content_length=check_info.get("content_length"),
                        is_pdf_signature=check_info.get("is_pdf_signature"),
                        min_bytes_pass=check_info.get("min_bytes_pass"),
                        decision=check_info.get("decision", CandidateDecision.HEAD_CHECK),
                    )

                if passed:
                    metadata.update(
                        {
                            "discovery_method": "cdx_pdf_direct",
                            "memento_ts": timestamp,
                            "statuscode": str(statuscode) if statuscode is not None else None,
                            "mimetype": mimetype,
                        }
                    )
                    mode_selected = ModeSelected.PDF_DIRECT
                    attempt_result = AttemptResult.EMITTED_PDF
                    skip_reason = None
                    skip_details = None
                    return _DiscoveryOutcome(
                        url=snapshot_url,
                        metadata=metadata,
                        mode=mode_selected,
                        attempt_result=attempt_result,
                        skip_reason=skip_reason,
                        skip_details=skip_details,
                        candidates_scanned=candidates_scanned,
                    )

                skip_reason = check_info.get("skip_reason") or skip_reason
                skip_details = check_info.get("skip_details") or skip_details
                seen_non_pdf = seen_non_pdf or skip_reason in {
                    SkipReason.NON_PDF,
                    SkipReason.BELOW_MIN_SIZE,
                }
                continue

            if html_parse_enabled and mimetype in {"text/html", "application/xhtml+xml"}:
                if telemetry is not None and ctx is not None:
                    telemetry.emit_candidate(
                        ctx,
                        archive_url=snapshot_url,
                        memento_ts=timestamp,
                        statuscode=statuscode,
                        mimetype=mimetype,
                        source_stage=DiscoveryStage.CDX,
                        decision=CandidateDecision.HEAD_CHECK,
                        distance_to_pub_year=_distance_to_pub_year(publication_year, timestamp),
                    )

                pdf_url, discovery_method, html_meta = self._parse_html_for_pdf(
                    client,
                    config,
                    snapshot_url,
                    telemetry=telemetry,
                    ctx=ctx,
                )

                if pdf_url:
                    passed, check_info = self._verify_pdf_snapshot(
                        client,
                        config,
                        pdf_url,
                        min_pdf_bytes,
                    )
                    candidates_scanned += 1
                    metadata["content_type"] = check_info.get("content_type")
                    metadata["content_length"] = check_info.get("content_length")

                    if telemetry is not None and ctx is not None:
                        telemetry.emit_pdf_check(
                            ctx,
                            archive_pdf_url=pdf_url,
                            head_status=check_info.get("head_status"),
                            content_type=check_info.get("content_type"),
                            content_length=check_info.get("content_length"),
                            is_pdf_signature=check_info.get("is_pdf_signature"),
                            min_bytes_pass=check_info.get("min_bytes_pass"),
                            decision=check_info.get("decision", CandidateDecision.HEAD_CHECK),
                        )

                    if passed:
                        metadata.update(
                            {
                                "discovery_method": "cdx_html_parse",
                                "html_snapshot_ts": timestamp,
                                "discovered_pdf_url": pdf_url,
                                "html_http_status": html_meta.get("html_http_status"),
                                "html_from_cache": html_meta.get("from_cache"),
                                "html_revalidated": html_meta.get("revalidated"),
                                "pdf_discovery_method": discovery_method.value
                                if discovery_method
                                else None,
                            }
                        )
                        mode_selected = ModeSelected.HTML_PARSE
                        attempt_result = AttemptResult.EMITTED_PDF_FROM_HTML
                        skip_reason = None
                        skip_details = None
                        return _DiscoveryOutcome(
                            url=pdf_url,
                            metadata=metadata,
                            mode=mode_selected,
                            attempt_result=attempt_result,
                            skip_reason=skip_reason,
                            skip_details=skip_details,
                            candidates_scanned=candidates_scanned,
                        )

                    skip_reason = check_info.get("skip_reason") or skip_reason
                    skip_details = check_info.get("skip_details") or skip_details
                    seen_non_pdf = True
                else:
                    seen_non_pdf = True
                continue

            seen_non_pdf = True

        if seen_non_pdf and skip_reason in {SkipReason.NO_SNAPSHOT, None}:
            skip_reason = SkipReason.ALL_NON_PDF

        if skip_reason is None:
            attempt_result = attempt_result
        elif skip_reason == SkipReason.NO_SNAPSHOT:
            attempt_result = AttemptResult.SKIPPED_NO_SNAPSHOT
        elif skip_reason == SkipReason.ALL_NON_PDF or skip_reason == SkipReason.NON_PDF:
            attempt_result = AttemptResult.SKIPPED_NON_PDF
        elif skip_reason == SkipReason.BELOW_MIN_SIZE:
            attempt_result = AttemptResult.SKIPPED_BELOW_MIN_SIZE
        elif skip_reason == SkipReason.BLOCKED_OFFLINE:
            attempt_result = AttemptResult.SKIPPED_BLOCKED_OFFLINE
        elif skip_reason == SkipReason.TIMEOUT:
            attempt_result = AttemptResult.TIMEOUT
        elif skip_reason == SkipReason.CDX_ERROR:
            attempt_result = AttemptResult.ERROR_CDX
        elif skip_reason == SkipReason.HTTP_ERROR:
            attempt_result = AttemptResult.ERROR_HTTP

        metadata["discovery_method"] = metadata.get("discovery_method", "none")
        return _DiscoveryOutcome(
            url=None,
            metadata=metadata,
            mode=mode_selected,
            attempt_result=attempt_result,
            skip_reason=skip_reason,
            skip_details=skip_details,
            candidates_scanned=candidates_scanned,
        )

    def _check_availability(
        self,
        client: httpx.Client,
        config: "ResolverConfig",
        url: str,
        *,
        telemetry: Optional[TelemetryWayback] = None,
        ctx: Optional[AttemptContext] = None,
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """Check Wayback Availability API for a quick snapshot."""

        metadata: Dict[str, Any] = {"availability_checked": True, "availability_available": False}
        http_status: Optional[int] = None
        meta_info: Dict[str, Any] = {}
        error_message: Optional[str] = None
        request_url = httpx.URL("https://archive.org/wayback/available").copy_add_params({"url": url})
        response: Optional[httpx.Response] = None

        try:
            response = request_with_retries(
                client,
                "get",
                str(request_url),
                role="metadata",
                timeout=config.get_timeout(self.name),
                headers=config.polite_headers,
                retry_after_cap=config.retry_after_cap,
            )
            http_status = getattr(response, "status_code", None)
            meta_info = dict(response.request.extensions.get("docs_network_meta", {}))  # type: ignore[attr-defined]
            response.raise_for_status()

            data = response.json()
            closest = (data.get("archived_snapshots") or {}).get("closest") or {}

            if closest.get("available") and closest.get("url"):
                metadata.update(
                    {
                        "availability_available": True,
                        "availability_timestamp": closest.get("timestamp"),
                        "availability_status": closest.get("status"),
                    }
                )
                return closest["url"], metadata

        except Exception as exc:
            LOGGER.debug("Availability check failed for %s: %s", url, exc)
            metadata["availability_error"] = str(exc)
            error_message = str(exc)

        finally:
            if telemetry is not None and ctx is not None:
                telemetry.emit_discovery_availability(
                    ctx,
                    query_url=str(request_url),
                    year_window=None,
                    http_status=http_status,
                    from_cache=_coerce_bool(meta_info.get("from_cache")),
                    revalidated=_extract_revalidated(meta_info),
                    rate_delay_ms=_coerce_int(meta_info.get("rate_limiter_wait_ms")),
                    retry_after_s=_coerce_int(meta_info.get("retry_after")),
                    retry_count=_coerce_int(meta_info.get("attempt")),
                    error=error_message,
                )
            if response is not None:
                response.close()

        return None, metadata

    def _query_cdx(
        self,
        client: httpx.Client,
        config: "ResolverConfig",
        url: str,
        publication_year: Optional[int],
        year_window: int,
        max_snapshots: int,
        *,
        telemetry: Optional[TelemetryWayback] = None,
        ctx: Optional[AttemptContext] = None,
    ) -> List[Dict[str, Any]]:
        """Query CDX API for snapshots."""

        snapshots: List[Dict[str, Any]] = []
        params = {
            "url": url,
            "output": "json",
            "limit": max_snapshots,
            "filter": "statuscode:200",
        }

        year_window_repr: Optional[str] = None
        if publication_year is not None:
            start_year = publication_year - year_window
            end_year = publication_year + year_window
            params["from"] = f"{start_year}0101000000"
            params["to"] = f"{end_year}1231235959"
            year_window_repr = f"-{year_window}..+{year_window}"

        request_url = httpx.URL("https://web.archive.org/cdx/search/cdx").copy_add_params(params)
        response: Optional[httpx.Response] = None
        http_status: Optional[int] = None
        error_message: Optional[str] = None
        meta_info: Dict[str, Any] = {}

        try:
            response = request_with_retries(
                client,
                "get",
                str(request_url),
                role="metadata",
                timeout=config.get_timeout(self.name),
                headers=config.polite_headers,
                retry_after_cap=config.retry_after_cap,
            )
            http_status = getattr(response, "status_code", None)
            meta_info = dict(response.request.extensions.get("docs_network_meta", {}))  # type: ignore[attr-defined]
            response.raise_for_status()

            data = response.json()
            if not data or len(data) < 2:
                return snapshots

            headers = data[0]
            for row in data[1:]:
                snapshot = dict(zip(headers, row))
                snapshots.append(snapshot)

            snapshots.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        except Exception as exc:
            LOGGER.debug("CDX query failed for %s: %s", url, exc)
            error_message = str(exc)

        finally:
            returned = len(snapshots) if snapshots else None
            first_ts = snapshots[0].get("timestamp") if snapshots else None
            last_ts = snapshots[-1].get("timestamp") if snapshots else None
            if telemetry is not None and ctx is not None:
                telemetry.emit_discovery_cdx(
                    ctx,
                    query_url=str(request_url),
                    year_window=year_window_repr,
                    limit=max_snapshots,
                    http_status=http_status,
                    returned=returned,
                    first_ts=str(first_ts) if first_ts else None,
                    last_ts=str(last_ts) if last_ts else None,
                    from_cache=_coerce_bool(meta_info.get("from_cache")),
                    revalidated=_extract_revalidated(meta_info),
                    rate_delay_ms=_coerce_int(meta_info.get("rate_limiter_wait_ms")),
                    retry_after_s=_coerce_int(meta_info.get("retry_after")),
                    retry_count=_coerce_int(meta_info.get("attempt")),
                    error=error_message,
                )
            if response is not None:
                response.close()

        return snapshots

    def _parse_html_for_pdf(
        self,
        client: httpx.Client,
        config: "ResolverConfig",
        html_url: str,
        *,
        telemetry: Optional[TelemetryWayback] = None,
        ctx: Optional[AttemptContext] = None,
    ) -> Tuple[Optional[str], Optional[PdfDiscoveryMethod], Dict[str, Any]]:
        """Parse archived HTML page to find PDF links."""

        response: Optional[httpx.Response] = None
        html_http_status: Optional[int] = None
        meta_info: Dict[str, Any] = {}
        html_bytes: Optional[int] = None
        error_message: Optional[str] = None
        discovered_pdf: Optional[str] = None
        discovery_method: Optional[PdfDiscoveryMethod] = None

        try:
            response = request_with_retries(
                client,
                "get",
                html_url,
                role="landing",
                timeout=config.get_timeout(self.name),
                headers=config.polite_headers,
                retry_after_cap=config.retry_after_cap,
            )
            html_http_status = getattr(response, "status_code", None)
            meta_info = dict(response.request.extensions.get("docs_network_meta", {}))  # type: ignore[attr-defined]
            response.raise_for_status()

            content_type = response.headers.get("content-type", "").lower()
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                return (
                    None,
                    None,
                    {
                        "html_http_status": html_http_status,
                        "from_cache": _coerce_bool(meta_info.get("from_cache")),
                        "revalidated": _extract_revalidated(meta_info),
                        "html_bytes": html_bytes,
                        "error": error_message,
                    },
                )

            html_content = response.text
            html_bytes = len(response.content or b"")

            try:
                from bs4 import BeautifulSoup
            except ImportError:
                LOGGER.debug("BeautifulSoup not available for HTML parsing")
                return (
                    None,
                    None,
                    {
                        "html_http_status": html_http_status,
                        "from_cache": _coerce_bool(meta_info.get("from_cache")),
                        "revalidated": _extract_revalidated(meta_info),
                        "html_bytes": html_bytes,
                        "error": "beautifulsoup_missing",
                    },
                )

            soup = BeautifulSoup(html_content, "html.parser")

            discovered_pdf = find_pdf_via_meta(soup, html_url)
            if discovered_pdf:
                discovery_method = PdfDiscoveryMethod.META
            else:
                discovered_pdf = find_pdf_via_link(soup, html_url)
                if discovered_pdf:
                    discovery_method = PdfDiscoveryMethod.LINK
                else:
                    discovered_pdf = find_pdf_via_anchor(soup, html_url)
                    if discovered_pdf:
                        discovery_method = PdfDiscoveryMethod.ANCHOR

        except Exception as exc:
            LOGGER.debug("HTML parsing failed for %s: %s", html_url, exc)
            error_message = str(exc)

        finally:
            if telemetry is not None and ctx is not None:
                telemetry.emit_html_parse(
                    ctx,
                    archive_html_url=html_url,
                    html_http_status=html_http_status,
                    from_cache=_coerce_bool(meta_info.get("from_cache")),
                    revalidated=_extract_revalidated(meta_info),
                    html_bytes=html_bytes,
                    pdf_link_found=bool(discovered_pdf),
                    pdf_discovery_method=discovery_method,
                    discovered_pdf_url=discovered_pdf,
                )
            if response is not None:
                response.close()

        return (
            discovered_pdf,
            discovery_method,
            {
                "html_http_status": html_http_status,
                "from_cache": _coerce_bool(meta_info.get("from_cache")),
                "revalidated": _extract_revalidated(meta_info),
                "html_bytes": html_bytes,
                "error": error_message,
            },
        )

    def _verify_pdf_snapshot(
        self,
        client: httpx.Client,
        config: "ResolverConfig",
        url: str,
        min_bytes: int,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Verify that a snapshot URL points to a valid PDF."""

        response: Optional[httpx.Response] = None
        head_status: Optional[int] = None
        content_type: Optional[str] = None
        content_length: Optional[int] = None
        is_pdf_signature: Optional[bool] = None
        min_bytes_pass: Optional[bool] = None
        decision: CandidateDecision = CandidateDecision.HEAD_CHECK
        skip_reason: Optional[SkipReason] = None
        skip_details: Optional[str] = None

        try:
            response = request_with_retries(
                client,
                "head",
                url,
                role="artifact",
                timeout=config.get_timeout(self.name),
                headers=config.polite_headers,
                retry_after_cap=config.retry_after_cap,
            )
            head_status = getattr(response, "status_code", None)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "").lower()
            if "application/pdf" not in content_type:
                decision = CandidateDecision.SKIPPED_MIME
                skip_reason = SkipReason.NON_PDF
                return False, {
                    "head_status": head_status,
                    "content_type": content_type,
                    "content_length": content_length,
                    "is_pdf_signature": is_pdf_signature,
                    "min_bytes_pass": min_bytes_pass,
                    "decision": decision,
                    "skip_reason": skip_reason,
                    "skip_details": None,
                }

            content_length = _coerce_int(response.headers.get("content-length"))
            if content_length is not None:
                if content_length < min_bytes:
                    decision = CandidateDecision.SKIPPED_MIME
                    skip_reason = SkipReason.BELOW_MIN_SIZE
                    min_bytes_pass = False
                    return False, {
                        "head_status": head_status,
                        "content_type": content_type,
                        "content_length": content_length,
                        "is_pdf_signature": is_pdf_signature,
                        "min_bytes_pass": min_bytes_pass,
                        "decision": decision,
                        "skip_reason": skip_reason,
                        "skip_details": None,
                    }
                min_bytes_pass = True

            if content_length is not None and content_length < min_bytes * 2:
                get_response: Optional[httpx.Response] = None
                try:
                    get_response = request_with_retries(
                        client,
                        "get",
                        url,
                        role="artifact",
                        timeout=config.get_timeout(self.name),
                        headers=config.polite_headers,
                        retry_after_cap=config.retry_after_cap,
                    )
                    get_response.raise_for_status()
                    chunk = (get_response.content or b"")[:8]
                    is_pdf_signature = chunk.startswith(b"%PDF-")
                    if not is_pdf_signature:
                        decision = CandidateDecision.SKIPPED_MIME
                        skip_reason = SkipReason.NON_PDF
                        return False, {
                            "head_status": head_status,
                            "content_type": content_type,
                            "content_length": content_length,
                            "is_pdf_signature": is_pdf_signature,
                            "min_bytes_pass": min_bytes_pass,
                            "decision": decision,
                            "skip_reason": skip_reason,
                            "skip_details": None,
                        }
                finally:
                    if get_response is not None:
                        get_response.close()

            return True, {
                "head_status": head_status,
                "content_type": content_type,
                "content_length": content_length,
                "is_pdf_signature": is_pdf_signature,
                "min_bytes_pass": min_bytes_pass,
                "decision": decision,
                "skip_reason": skip_reason,
                "skip_details": skip_details,
            }

        except httpx.TimeoutException as exc:
            skip_reason = SkipReason.TIMEOUT
            skip_details = str(exc)
            decision = CandidateDecision.HEAD_CHECK
            LOGGER.debug("PDF verification timeout for %s: %s", url, exc)
            return False, {
                "head_status": head_status,
                "content_type": content_type,
                "content_length": content_length,
                "is_pdf_signature": is_pdf_signature,
                "min_bytes_pass": min_bytes_pass,
                "decision": decision,
                "skip_reason": skip_reason,
                "skip_details": skip_details,
            }
        except httpx.HTTPStatusError as exc:
            skip_reason = SkipReason.HTTP_ERROR
            skip_details = str(exc)
            decision = CandidateDecision.SKIPPED_STATUS
            head_status = getattr(exc.response, "status_code", head_status)
            LOGGER.debug("PDF verification HTTP error for %s: %s", url, exc)
            return False, {
                "head_status": head_status,
                "content_type": content_type,
                "content_length": content_length,
                "is_pdf_signature": is_pdf_signature,
                "min_bytes_pass": min_bytes_pass,
                "decision": decision,
                "skip_reason": skip_reason,
                "skip_details": skip_details,
            }
        except httpx.HTTPError as exc:
            skip_reason = SkipReason.HTTP_ERROR
            skip_details = str(exc)
            decision = CandidateDecision.HEAD_CHECK
            LOGGER.debug("PDF verification request error for %s: %s", url, exc)
            return False, {
                "head_status": head_status,
                "content_type": content_type,
                "content_length": content_length,
                "is_pdf_signature": is_pdf_signature,
                "min_bytes_pass": min_bytes_pass,
                "decision": decision,
                "skip_reason": skip_reason,
                "skip_details": skip_details,
            }
        except Exception as exc:
            skip_reason = SkipReason.HTTP_ERROR
            skip_details = str(exc)
            decision = CandidateDecision.HEAD_CHECK
            LOGGER.debug("PDF verification failed for %s: %s", url, exc)
            return False, {
                "head_status": head_status,
                "content_type": content_type,
                "content_length": content_length,
                "is_pdf_signature": is_pdf_signature,
                "min_bytes_pass": min_bytes_pass,
                "decision": decision,
                "skip_reason": skip_reason,
                "skip_details": skip_details,
            }
        finally:
            if response is not None:
                response.close()
