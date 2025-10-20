Here’s a drop-in, **agent-ready** telemetry scaffold you can add to `src/DocsToKG/ContentDownload/telemetry_wayback.py`. It’s designed to be very explicit (junior-dev-friendly), strongly typed, and easy to wire into your resolver. It implements the exact event streams we specified:

* `wayback_attempt` (start/end)
* `wayback_discovery` (availability, cdx)
* `wayback_candidate` (evaluated candidate)
* `wayback_html_parse`
* `wayback_pdf_check`
* `wayback_emit`
* `wayback_skip`

It uses a pluggable **sink** interface so you can fan out to JSONL, SQLite, or any other backend without changing resolver code.

---

```python
# File: src/DocsToKG/ContentDownload/telemetry_wayback.py
"""
Telemetry helpers for the Wayback last-chance resolver.

This module centralizes *all* logging/metrics for Wayback discovery and emission so the
resolver stays simple and business-logic focused. It’s intentionally verbose and typed
to be junior-dev-friendly.

Integration points:
- Call `emit_attempt_start()` when the Wayback resolver begins evaluating a failed URL.
- Use `emit_discovery_*()` during Availability/CDX queries.
- Use `emit_candidate()` when you pick a snapshot to evaluate.
- If you switch to the HTML-parse path, call `emit_html_parse()`.
- Before emitting a PDF, call `emit_pdf_check()` with HEAD/sniff results.
- On success, call `emit_emit()`.
- On any terminal skip/failure, call `emit_skip()`.
- Always finish with `emit_attempt_end()`.

All events are enriched with a shared "envelope": run/work/artifact/attempt IDs, wall time,
and a monotonic clock delta to calculate durations independent of wall-clock skew.
"""

from __future__ import annotations

import json
import uuid
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol


# ────────────────────────────────────────────────────────────────────────────────
# Sink interface(s)
# ────────────────────────────────────────────────────────────────────────────────

class TelemetrySink(Protocol):
    """A sink consumes a single event dict (already envelope-enriched)."""

    def emit(self, event: Mapping[str, Any]) -> None:
        ...


class JsonlSink:
    """
    Very simple JSONL sink for local runs. For multi-process safety,
    wrap calls in your project’s file locks (e.g., locks.manifest_lock()).
    """

    def __init__(self, path: Path, ensure_parent: bool = True) -> None:
        self.path = Path(path)
        if ensure_parent:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: Mapping[str, Any]) -> None:
        line = json.dumps(event, separators=(",", ":"), ensure_ascii=False)
        # NOTE: Use your centralized file locks here if multiple writers exist.
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


# ────────────────────────────────────────────────────────────────────────────────
# Enums to reduce typos and keep cardinality tight
# ────────────────────────────────────────────────────────────────────────────────

class AttemptResult(str, Enum):
    EMITTED_PDF = "emitted_pdf"
    EMITTED_PDF_FROM_HTML = "emitted_pdf_from_html"
    SKIPPED_NO_SNAPSHOT = "skipped_no_snapshot"
    SKIPPED_NON_PDF = "skipped_non_pdf"
    SKIPPED_BELOW_MIN_SIZE = "skipped_below_min_size"
    SKIPPED_BLOCKED_OFFLINE = "skipped_blocked_offline"
    ERROR_HTTP = "error_http"
    ERROR_CDX = "error_cdx"
    TIMEOUT = "timeout"


class ModeSelected(str, Enum):
    PDF_DIRECT = "pdf_direct"
    HTML_PARSE = "html_parse"
    NONE = "none"


class DiscoveryStage(str, Enum):
    AVAILABILITY = "availability"
    CDX = "cdx"


class CandidateDecision(str, Enum):
    HEAD_CHECK = "head_check"
    SKIPPED_STATUS = "skipped_status"
    SKIPPED_MIME = "skipped_mime"


class PdfDiscoveryMethod(str, Enum):
    META = "meta"
    LINK = "link"
    ANCHOR = "anchor"


class SkipReason(str, Enum):
    NO_SNAPSHOT = "no_snapshot"
    ALL_NON_PDF = "all_non_pdf"
    BELOW_MIN_SIZE = "below_min_size"
    NON_PDF = "non_pdf"
    BLOCKED_OFFLINE = "blocked_offline"
    TIMEOUT = "timeout"
    CDX_ERROR = "cdx_error"
    HTTP_ERROR = "http_error"


# ────────────────────────────────────────────────────────────────────────────────
# Context captured per attempt (for durations & correlation)
# ────────────────────────────────────────────────────────────────────────────────

@dataclass
class AttemptContext:
    run_id: str
    work_id: str
    artifact_id: str
    attempt_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_wall: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    start_monotonic: float = field(default_factory=time.monotonic)
    original_url: str = ""
    canonical_url: str = ""
    publication_year: Optional[int] = None

    def monotonic_ms_since_start(self) -> int:
        return int((time.monotonic() - self.start_monotonic) * 1000)


# ────────────────────────────────────────────────────────────────────────────────
# Telemetry emitter
# ────────────────────────────────────────────────────────────────────────────────

class TelemetryWayback:
    """
    Pluggable telemetry emitter for the Wayback resolver.

    Parameters
    ----------
    run_id:
        Execution/run correlation id.
    sinks:
        A list of sinks to fan out events (e.g., [JsonlSink(Path(...))]).
    schema_version:
        Free-form version tag to bump if you adjust shapes.
    now_fn / monotonic_fn:
        Hooks for tests to control time.
    resolver_name:
        Defaults to "wayback"; override only if you fork the resolver’s name.
    """

    def __init__(
        self,
        run_id: str,
        sinks: Iterable[TelemetrySink],
        *,
        schema_version: str = "1",
        now_fn = lambda: datetime.now(timezone.utc),
        monotonic_fn = time.monotonic,
        resolver_name: str = "wayback",
    ) -> None:
        self.run_id = run_id
        self.sinks = list(sinks)
        self.schema_version = schema_version
        self._now = now_fn
        self._mono = monotonic_fn
        self.resolver_name = resolver_name

    # ── Envelope helpers ────────────────────────────────────────────────────────

    def _envelope(self, ctx: AttemptContext) -> Dict[str, Any]:
        return {
            "schema": self.schema_version,
            "resolver": self.resolver_name,
            "run_id": ctx.run_id,
            "work_id": ctx.work_id,
            "artifact_id": ctx.artifact_id,
            "attempt_id": ctx.attempt_id,
            "ts": self._now().isoformat(),
            "monotonic_ms": ctx.monotonic_ms_since_start(),
        }

    def _emit(self, ctx: AttemptContext, event_type: str, body: Mapping[str, Any]) -> None:
        event = {"event_type": event_type, **self._envelope(ctx), **body}
        for s in self.sinks:
            s.emit(event)

    # ── Attempt lifecycle ───────────────────────────────────────────────────────

    def emit_attempt_start(
        self,
        work_id: str,
        artifact_id: str,
        *,
        original_url: str,
        canonical_url: str,
        publication_year: Optional[int] = None,
    ) -> AttemptContext:
        """
        Start a Wayback resolver attempt for a single failed URL.
        Return an AttemptContext; pass it to all subsequent emit_* calls.
        """
        ctx = AttemptContext(
            run_id=self.run_id,
            work_id=work_id,
            artifact_id=artifact_id,
            original_url=original_url,
            canonical_url=canonical_url,
            publication_year=publication_year,
        )
        self._emit(
            ctx,
            "wayback_attempt",
            {
                "event": "start",
                "original_url": original_url,
                "canonical_url": canonical_url,
                "publication_year": publication_year,
            },
        )
        return ctx

    def emit_attempt_end(
        self,
        ctx: AttemptContext,
        *,
        mode_selected: ModeSelected,
        result: AttemptResult,
        candidates_scanned: int,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        End the attempt with a final result.
        """
        body = {
            "event": "end",
            "mode_selected": mode_selected.value,
            "result": result.value,
            "candidates_scanned": int(candidates_scanned),
            "total_duration_ms": ctx.monotonic_ms_since_start(),
        }
        if extra:
            body.update(extra)
        self._emit(ctx, "wayback_attempt", body)

    # ── Discovery (Availability / CDX) ─────────────────────────────────────────

    def emit_discovery_availability(
        self,
        ctx: AttemptContext,
        *,
        query_url: str,
        year_window: Optional[str] = None,  # e.g., "-2..+2"
        http_status: Optional[int] = None,
        from_cache: Optional[bool] = None,
        revalidated: Optional[bool] = None,
        rate_delay_ms: Optional[int] = None,
        retry_after_s: Optional[int] = None,
        retry_count: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        self._emit(
            ctx,
            "wayback_discovery",
            {
                "stage": DiscoveryStage.AVAILABILITY.value,
                "query_url": query_url,
                "year_window": year_window,
                "http_status": http_status,
                "from_cache": from_cache,
                "revalidated": revalidated,
                "rate_delay_ms": rate_delay_ms,
                "retry_after_s": retry_after_s,
                "retry_count": retry_count,
                "error": error,
            },
        )

    def emit_discovery_cdx(
        self,
        ctx: AttemptContext,
        *,
        query_url: str,
        year_window: Optional[str],
        limit: int,
        http_status: Optional[int],
        returned: Optional[int],
        first_ts: Optional[str],
        last_ts: Optional[str],
        from_cache: Optional[bool],
        revalidated: Optional[bool],
        rate_delay_ms: Optional[int],
        retry_after_s: Optional[int],
        retry_count: Optional[int],
        error: Optional[str] = None,
    ) -> None:
        self._emit(
            ctx,
            "wayback_discovery",
            {
                "stage": DiscoveryStage.CDX.value,
                "query_url": query_url,
                "year_window": year_window,
                "limit": limit,
                "returned": returned,
                "first_ts": first_ts,
                "last_ts": last_ts,
                "http_status": http_status,
                "from_cache": from_cache,
                "revalidated": revalidated,
                "rate_delay_ms": rate_delay_ms,
                "retry_after_s": retry_after_s,
                "retry_count": retry_count,
                "error": error,
            },
        )

    # ── Candidate (evaluate one or two best snapshots) ─────────────────────────

    def emit_candidate(
        self,
        ctx: AttemptContext,
        *,
        archive_url: str,
        memento_ts: str,
        statuscode: Optional[int],
        mimetype: Optional[str],
        source_stage: DiscoveryStage,
        decision: CandidateDecision,
        distance_to_pub_year: Optional[int] = None,
    ) -> None:
        self._emit(
            ctx,
            "wayback_candidate",
            {
                "archive_url": archive_url,
                "memento_ts": memento_ts,
                "statuscode": statuscode,
                "mimetype": mimetype,
                "source_stage": source_stage.value,
                "decision": decision.value,
                "distance_to_pub_year": distance_to_pub_year,
            },
        )

    # ── HTML path (archived landing page → PDF) ────────────────────────────────

    def emit_html_parse(
        self,
        ctx: AttemptContext,
        *,
        archive_html_url: str,
        html_http_status: Optional[int],
        from_cache: Optional[bool],
        revalidated: Optional[bool],
        html_bytes: Optional[int],
        pdf_link_found: bool,
        pdf_discovery_method: Optional[PdfDiscoveryMethod] = None,
        discovered_pdf_url: Optional[str] = None,
    ) -> None:
        self._emit(
            ctx,
            "wayback_html_parse",
            {
                "archive_html_url": archive_html_url,
                "html_http_status": html_http_status,
                "from_cache": from_cache,
                "revalidated": revalidated,
                "html_bytes": html_bytes,
                "pdf_link_found": pdf_link_found,
                "pdf_discovery_method": pdf_discovery_method.value if pdf_discovery_method else None,
                "discovered_pdf_url": discovered_pdf_url,
            },
        )

    # ── Archived PDF verification ──────────────────────────────────────────────

    def emit_pdf_check(
        self,
        ctx: AttemptContext,
        *,
        archive_pdf_url: str,
        head_status: Optional[int],
        content_type: Optional[str],
        content_length: Optional[int],
        is_pdf_signature: Optional[bool],
        min_bytes_pass: Optional[bool],
        decision: CandidateDecision,  # expected: "emit" is expressed via emit_emit()
    ) -> None:
        self._emit(
            ctx,
            "wayback_pdf_check",
            {
                "archive_pdf_url": archive_pdf_url,
                "head_status": head_status,
                "content_type": content_type,
                "content_length": content_length,
                "is_pdf_signature": is_pdf_signature,
                "min_bytes_pass": min_bytes_pass,
                "decision": decision.value,
            },
        )

    # ── Success emission ───────────────────────────────────────────────────────

    def emit_emit(
        self,
        ctx: AttemptContext,
        *,
        emitted_url: str,
        memento_ts: str,
        source_mode: ModeSelected,  # PDF_DIRECT or HTML_PARSE
        http_ct_expected: str = "application/pdf",
    ) -> None:
        self._emit(
            ctx,
            "wayback_emit",
            {
                "emitted_url": emitted_url,
                "memento_ts": memento_ts,
                "source_mode": source_mode.value,
                "http_ct_expected": http_ct_expected,
            },
        )

    # ── Terminal skip / error ──────────────────────────────────────────────────

    def emit_skip(
        self,
        ctx: AttemptContext,
        *,
        reason: SkipReason,
        details: Optional[str] = None,
    ) -> None:
        self._emit(
            ctx,
            "wayback_skip",
            {
                "reason": reason.value,
                "details": details,
            },
        )
```

---

## How to wire this in your resolver (example flow)

```python
# inside WaybackResolver.resolve(...)
tele = TelemetryWayback(run_id, sinks=[JsonlSink(Path(run_dir) / "telemetry/wayback.jsonl")])

ctx = tele.emit_attempt_start(
    work_id=work.id,
    artifact_id=artifact.id,
    original_url=failed_pdf_url,
    canonical_url=canonical_for_index(failed_pdf_url),
    publication_year=artifact.publication_year,
)

# 1) Availability (fast-path)
tele.emit_discovery_availability(
    ctx,
    query_url=failed_pdf_url,
    year_window="-2..+2",
    http_status=avail_status,
    from_cache=avail_from_cache,
    revalidated=avail_revalidated,
    rate_delay_ms=avail_rate_delay,
    retry_after_s=avail_retry_after,
    retry_count=avail_retry_count,
    error=None,
)

# 2) CDX scanning
tele.emit_discovery_cdx(
    ctx,
    query_url=failed_pdf_url,
    year_window="-2..+2",
    limit=8,
    http_status=200,
    returned=len(cdx_rows),
    first_ts=cdx_rows[0]["timestamp"] if cdx_rows else None,
    last_ts=cdx_rows[-1]["timestamp"] if cdx_rows else None,
    from_cache=cdx_from_cache,
    revalidated=cdx_revalidated,
    rate_delay_ms=cdx_rate_delay,
    retry_after_s=cdx_retry_after,
    retry_count=cdx_retry_count,
)

# 3) Candidate evaluation
tele.emit_candidate(
    ctx,
    archive_url=candidate_url,
    memento_ts=candidate_ts,
    statuscode=candidate_status,
    mimetype=candidate_mime,
    source_stage=DiscoveryStage.CDX,
    decision=CandidateDecision.HEAD_CHECK,
    distance_to_pub_year=abs(candidate_year - artifact.publication_year) if artifact.publication_year else None,
)

# 4) HTML parse path (only if needed)
tele.emit_html_parse(
    ctx,
    archive_html_url=archived_landing_url,
    html_http_status=200,
    from_cache=True,
    revalidated=False,
    html_bytes=len(html_bytes),
    pdf_link_found=True,
    pdf_discovery_method=PdfDiscoveryMethod.META,
    discovered_pdf_url=archived_pdf_url,
)

# 5) Archived PDF verification
tele.emit_pdf_check(
    ctx,
    archive_pdf_url=archived_pdf_url,
    head_status=200,
    content_type="application/pdf",
    content_length=123456,
    is_pdf_signature=True,
    min_bytes_pass=True,
    decision=CandidateDecision.HEAD_CHECK,  # Final decision manifested by emit_emit() next
)

# 6) Success + close
tele.emit_emit(
    ctx,
    emitted_url=archived_pdf_url,
    memento_ts=candidate_ts,
    source_mode=ModeSelected.HTML_PARSE,
)
tele.emit_attempt_end(
    ctx,
    mode_selected=ModeSelected.HTML_PARSE,
    result=AttemptResult.EMITTED_PDF_FROM_HTML,
    candidates_scanned=len(cdx_rows),
)
```

---

## Practical notes / “gotchas” to avoid

* **One context per attempt.** Don’t reuse `AttemptContext` across different failed URLs; create a new one each time you start evaluating a distinct candidate URL.
* **Enums → `.value`.** If you serialize enums manually, remember to pass `.value`; this class handles that internally.
* **Lock your sinks.** If multiple workers may write to the same JSONL/SQLite sinks, wrap the sink’s write in your `locks.py` helpers.
* **Low cardinality.** Keep `details` short; avoid dumping raw HTML or full CDX rows (log counts and timestamps instead).
* **Pin shapes.** The `schema` field lets you rev the format safely later; bump it if you add/remove keys.

If you want, I can also provide a tiny `SQLiteSink` with table DDLs (`wayback_attempts`, `wayback_emits`, etc.) matching the spec, but the JSONL sink above is usually enough to start—and easy to aggregate later.
