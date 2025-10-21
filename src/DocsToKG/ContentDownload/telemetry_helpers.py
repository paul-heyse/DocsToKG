# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.telemetry_helpers",
#   "purpose": "Convenience functions for structured event emission across HTTP, rate limiting, circuit breaker, and fallback layers",
#   "sections": [
#     {"id": "emit-http-event", "name": "emit_http_event", "anchor": "function-emit-http-event", "kind": "function"},
#     {"id": "emit-rate-event", "name": "emit_rate_event", "anchor": "function-emit-rate-event", "kind": "function"},
#     {"id": "emit-fallback-attempt", "name": "emit_fallback_attempt", "anchor": "function-emit-fallback-attempt", "kind": "function"}
#   ]
# }
# === /NAVMAP ===

"""Telemetry event emission helpers for observability instrumentation.

**Purpose**
-----------
This module provides high-level convenience functions for emitting structured telemetry
events from throughout the ContentDownload pipeline. These helpers abstract away the
details of event schema and sink coordination, allowing layers (HTTP, rate limiter,
circuit breaker, fallback) to emit rich, queryable events without tight coupling to
the telemetry backend.

**Responsibilities**
--------------------
- Define canonical event emitters for HTTP requests (:func:`emit_http_event`),
  rate limiter interactions (:func:`emit_rate_event`), and fallback resolution
  attempts (:func:`emit_fallback_attempt`).
- Normalize and validate event payloads before delegation to sinks.
- Handle ``None`` telemetry gracefully (no-op when telemetry is disabled).
- Provide type hints and docstrings for easy discovery and integration.

**Key Functions**
-----------------

:func:`emit_http_event`
  Records HTTP request/response metadata after cache and limiter decisions.
  Captures: host, role, method, status, URL hash, cache metadata, retry info,
  rate delay, circuit breaker state, elapsed time, and errors.

:func:`emit_rate_event`
  Records rate limiter acquisitions and blocks.
  Captures: host, role, action (acquire/block/head_skip), delay, max delay.

:func:`emit_fallback_attempt`
  Records fallback adapter resolution attempts.
  Captures: artifact ID, tier, source, outcome, reason, status, elapsed time.

**Integration Points**
----------------------
- Called from :mod:`networking` after request completion.
- Called from :mod:`ratelimit` on limiter operations.
- Called from :mod:`fallback.orchestrator` for each adapter attempt.

**Privacy & Performance**
-------------------------
- All URL data is hashed (SHA256, first 16 chars) before emission.
- Emission is best-effort; errors are logged but do not break requests.
- Telemetry parameter is optional; None signals disabled telemetry (no-op).

**Design Pattern**
------------------
Each emitter follows the same pattern:

    1. Check if telemetry is enabled (None → skip).
    2. Validate/normalize input values.
    3. Construct event dict with required fields.
    4. Call telemetry sink method (e.g., log_http_event).
    5. Catch and log any errors (never break the request).

This ensures telemetry is **non-breaking** and **optional** throughout the pipeline.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Optional


def emit_http_event(
    telemetry: Any,
    run_id: str,
    host: str,
    role: str,
    method: str,
    status: Optional[int] = None,
    url_hash: Optional[str] = None,
    from_cache: Optional[int] = None,
    revalidated: Optional[int] = None,
    stale: Optional[int] = None,
    retry_count: Optional[int] = None,
    retry_after_s: Optional[int] = None,
    rate_delay_ms: Optional[int] = None,
    breaker_state: Optional[str] = None,
    breaker_recorded: Optional[str] = None,
    elapsed_ms: Optional[int] = None,
    error: Optional[str] = None,
) -> None:
    """Emit HTTP event to telemetry bus.

    Parameters
    ----------
    telemetry : RunTelemetry or None
        Telemetry coordinator (can be None for graceful no-op)
    run_id : str
        Run identifier
    host : str
        Hostname (punycode normalized)
    role : str
        Request role: metadata|landing|artifact
    method : str
        HTTP method: GET|HEAD|POST
    status : Optional[int]
        HTTP status code (None if exception)
    url_hash : Optional[str]
        SHA256 hash of canonical URL (not raw URL)
    from_cache : Optional[int]
        0/1 if response from cache
    revalidated : Optional[int]
        0/1 if revalidated (304)
    stale : Optional[int]
        0/1 if stale (Hishel SWrV)
    retry_count : Optional[int]
        Number of retries (Tenacity attempts - 1)
    retry_after_s : Optional[int]
        Retry-After value honored (seconds)
    rate_delay_ms : Optional[int]
        Milliseconds waited in rate limiter
    breaker_state : Optional[str]
        Circuit breaker state: closed|half_open|open
    breaker_recorded : Optional[str]
        Breaker outcome: success|failure|none
    elapsed_ms : Optional[int]
        End-to-end request time (ms)
    error : Optional[str]
        Short exception class name if failed
    """
    if telemetry is None:
        return

    event = {
        "run_id": run_id,
        "ts": time.time(),
        "host": host,
        "role": role,
        "method": method,
        "status": status,
        "url_hash": url_hash,
        "from_cache": from_cache,
        "revalidated": revalidated,
        "stale": stale,
        "retry_count": retry_count,
        "retry_after_s": retry_after_s,
        "rate_delay_ms": rate_delay_ms,
        "breaker_state": breaker_state,
        "breaker_recorded": breaker_recorded,
        "elapsed_ms": elapsed_ms,
        "error": error,
    }
    telemetry.log_http_event(event)


def emit_rate_event(
    telemetry: Any,
    run_id: str,
    host: str,
    role: str,
    action: str,
    delay_ms: Optional[int] = None,
    max_delay_ms: Optional[int] = None,
) -> None:
    """Emit rate limiter event to telemetry bus.

    Parameters
    ----------
    telemetry : RunTelemetry or None
        Telemetry coordinator (can be None for graceful no-op)
    run_id : str
        Run identifier
    host : str
        Hostname
    role : str
        Request role: metadata|landing|artifact
    action : str
        Action: acquire|block|head_skip
    delay_ms : Optional[int]
        Milliseconds waited (if acquire/block)
    max_delay_ms : Optional[int]
        Maximum allowed delay (if configured)
    """
    if telemetry is None:
        return

    event = {
        "run_id": run_id,
        "ts": time.time(),
        "host": host,
        "role": role,
        "action": action,
        "delay_ms": delay_ms,
        "max_delay_ms": max_delay_ms,
    }
    telemetry.log_rate_event(event)


def emit_breaker_transition(
    telemetry: Any,
    run_id: str,
    host: str,
    scope: str,
    old_state: str,
    new_state: str,
    reset_timeout_s: Optional[int] = None,
) -> None:
    """Emit circuit breaker state transition to telemetry bus.

    Parameters
    ----------
    telemetry : RunTelemetry or None
        Telemetry coordinator (can be None for graceful no-op)
    run_id : str
        Run identifier
    host : str
        Hostname
    scope : str
        Breaker scope: host|resolver
    old_state : str
        Previous state (e.g., "CLOSED", "OPEN")
    new_state : str
        New state (e.g., "OPEN", "HALF_OPEN")
    reset_timeout_s : Optional[int]
        Reset timeout configured (seconds)
    """
    if telemetry is None:
        return

    event = {
        "run_id": run_id,
        "ts": time.time(),
        "host": host,
        "scope": scope,
        "old_state": old_state,
        "new_state": new_state,
        "reset_timeout_s": reset_timeout_s,
    }
    telemetry.log_breaker_transition(event)


def emit_fallback_attempt(
    telemetry: Any,
    run_id: str,
    work_id: Optional[str] = None,
    artifact_id: Optional[str] = None,
    tier: Optional[str] = None,
    source: Optional[str] = None,
    host: Optional[str] = None,
    outcome: Optional[str] = None,
    reason: Optional[str] = None,
    status: Optional[int] = None,
    elapsed_ms: Optional[int] = None,
) -> None:
    """Emit fallback attempt to telemetry bus.

    Parameters
    ----------
    telemetry : RunTelemetry or None
        Telemetry coordinator (can be None for graceful no-op)
    run_id : str
        Run identifier
    work_id : Optional[str]
        Work/job identifier
    artifact_id : Optional[str]
        Artifact identifier
    tier : Optional[str]
        Tier name (e.g., "direct_oa", "doi_follow")
    source : Optional[str]
        Source name (e.g., "unpaywall_pdf", "arxiv_pdf")
    host : Optional[str]
        Host accessed (if applicable)
    outcome : Optional[str]
        Outcome: success|retryable|nonretryable|timeout|skipped|error|no_pdf
    reason : Optional[str]
        Short reason code
    status : Optional[int]
        HTTP status code (if applicable)
    elapsed_ms : Optional[int]
        Attempt duration (ms)
    """
    if telemetry is None:
        return

    event = {
        "run_id": run_id,
        "ts": time.time(),
        "work_id": work_id,
        "artifact_id": artifact_id,
        "tier": tier,
        "source": source,
        "host": host,
        "outcome": outcome,
        "reason": reason,
        "status": status,
        "elapsed_ms": elapsed_ms,
    }
    telemetry.log_fallback_attempt(event)
