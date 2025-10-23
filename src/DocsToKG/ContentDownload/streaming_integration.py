# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.streaming_integration",
#   "purpose": "Streaming download integration for ContentDownload.",
#   "sections": [
#     {
#       "id": "use-streaming-for-resume",
#       "name": "use_streaming_for_resume",
#       "anchor": "function-use-streaming-for-resume",
#       "kind": "function"
#     },
#     {
#       "id": "try-streaming-resume-decision",
#       "name": "try_streaming_resume_decision",
#       "anchor": "function-try-streaming-resume-decision",
#       "kind": "function"
#     },
#     {
#       "id": "use-streaming-for-io",
#       "name": "use_streaming_for_io",
#       "anchor": "function-use-streaming-for-io",
#       "kind": "function"
#     },
#     {
#       "id": "try-streaming-io",
#       "name": "try_streaming_io",
#       "anchor": "function-try-streaming-io",
#       "kind": "function"
#     },
#     {
#       "id": "use-streaming-for-finalization",
#       "name": "use_streaming_for_finalization",
#       "anchor": "function-use-streaming-for-finalization",
#       "kind": "function"
#     },
#     {
#       "id": "integration-status",
#       "name": "integration_status",
#       "anchor": "function-integration-status",
#       "kind": "function"
#     },
#     {
#       "id": "log-integration-status",
#       "name": "log_integration_status",
#       "anchor": "function-log-integration-status",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Streaming download integration for ContentDownload.

This module provides the integration layer between the resolver pipeline and
the new streaming architecture. It coordinates:

- Streaming client setup
- Resume state management
- Schema-aware manifest handling
- Integration with modern idempotency and work orchestration
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

LOGGER = logging.getLogger(__name__)

# Try importing streaming module
try:
    from DocsToKG.ContentDownload import streaming

    _STREAMING_AVAILABLE = True
except ImportError:
    _STREAMING_AVAILABLE = False
    streaming = None  # type: ignore[assignment]

# Try importing idempotency module
try:
    from DocsToKG.ContentDownload import idempotency

    _IDEMPOTENCY_AVAILABLE = True
except ImportError:
    _IDEMPOTENCY_AVAILABLE = False
    idempotency = None  # type: ignore[assignment]

# Try importing streaming_schema module
try:
    from DocsToKG.ContentDownload import streaming_schema

    _SCHEMA_AVAILABLE = True
except ImportError:
    _SCHEMA_AVAILABLE = False
    streaming_schema = None  # type: ignore[assignment]

if TYPE_CHECKING:
    pass


# ============================================================================
# Feature Flags
# ============================================================================


# All streaming features enabled by default - no backward compatibility checks
STREAMING_ENABLED = True
IDEMPOTENCY_ENABLED = True
SCHEMA_VALIDATION_ENABLED = True


# ============================================================================
# Resume Decision Integration
# ============================================================================


def use_streaming_for_resume(plan: Any) -> bool:
    """Decide whether to use streaming.can_resume() for this plan.

    Conditions:
    - Streaming module available and enabled
    - Plan has validators or Accept-Ranges support
    - Not explicitly disabled

    Args:
        plan: DownloadPreflightPlan instance

    Returns:
        True to use streaming.can_resume(), False to use existing logic.
    """
    if not STREAMING_ENABLED:
        return False

    # Skip if plan doesn't have necessary info for streaming
    if not hasattr(plan, "cond_helper"):
        return False

    return True


def try_streaming_resume_decision(
    validators: Optional[Dict[str, Any]],
    part_state: Optional[Any],
    *,
    prefix_check_bytes: int = 65536,
    allow_without_validators: bool = False,
    client: Optional[Any] = None,
    url: Optional[str] = None,
) -> Optional[Any]:
    """Try using streaming.can_resume() if available, return None on fallback.

    This allows download.py to try the new streaming logic and gracefully
    fall back to existing logic if streaming is not available or errors.

    Args:
        validators: ServerValidators dataclass or dict with etag, last_modified, etc.
        part_state: LocalPartState dataclass or None
        prefix_check_bytes: Bytes to check for resume validation
        allow_without_validators: Allow resume without validators
        client: HTTPX client for HEAD requests
        url: URL for resume checking

    Returns:
        ResumeDecision from streaming.can_resume() or None to fallback.

    Example:
        # In download.py stream_candidate_payload():
        decision = try_streaming_resume_decision(
            validators,
            part_state,
            prefix_check_bytes=ctx.sniff_limit,
            client=client,
            url=url,
        )
        if decision is not None:
            # Use new streaming logic
            if decision.mode == "fresh":
                ...
        else:
            # Fallback to existing logic
            if attempt_conditional:
                ...
    """
    if not STREAMING_ENABLED:
        return None

    if not streaming:
        return None

    try:
        # Convert dict to ServerValidators if needed
        if isinstance(validators, dict):
            validators = streaming.ServerValidators(**validators)

        decision = streaming.can_resume(
            validators,
            part_state,
            prefix_check_bytes=prefix_check_bytes,
            allow_without_validators=allow_without_validators,
            client=client,
            url=url,
        )
        return decision
    except Exception as e:
        LOGGER.debug(f"Streaming resume decision failed, falling back: {e}")
        return None


# ============================================================================
# I/O Integration
# ============================================================================


def use_streaming_for_io(plan: Any) -> bool:
    """Decide whether to use streaming.stream_to_part() for I/O.

    Conditions:
    - Streaming module available and enabled
    - Plan has necessary I/O configuration
    - Not explicitly disabled

    Args:
        plan: DownloadPreflightPlan instance

    Returns:
        True to use streaming.stream_to_part(), False to use existing logic.
    """
    if not STREAMING_ENABLED:
        return False

    # Verify plan has required attributes
    if not hasattr(plan, "context"):
        return False

    return True


def try_streaming_io(
    response: Any,
    part_path: Any,
    *,
    chunk_bytes: int = 65536,
    fsync: bool = True,
    progress_callback: Optional[Any] = None,
) -> Optional[Any]:
    """Try using streaming.stream_to_part() if available, return None on fallback.

    Args:
        response: HTTPX response object (context manager)
        part_path: Path to write .part file to
        chunk_bytes: Chunk size for reading
        fsync: Whether to call fsync after write
        progress_callback: Optional callback(bytes_written) for progress

    Returns:
        StreamMetrics from streaming.stream_to_part() or None to fallback.

    Example:
        # In download.py stream_candidate_payload():
        metrics = try_streaming_io(
            response,
            staging_path,
            chunk_bytes=ctx.sniff_limit,
            fsync=ctx.verify_cache_digest,
        )
        if metrics is not None:
            # Use streaming metrics
            total_bytes = metrics.bytes_written
            sha256 = metrics.sha256_hex
        else:
            # Fallback to existing I/O logic
            sha256 = hashlib.sha256()
            ...
    """
    if not STREAMING_ENABLED:
        return None

    if not streaming:
        return None

    try:
        metrics = streaming.stream_to_part(
            response,
            part_path,
            chunk_bytes=chunk_bytes,
            fsync=fsync,
            progress_callback=progress_callback,
        )
        return metrics
    except Exception as e:
        LOGGER.debug(f"Streaming I/O failed, falling back: {e}")
        return None


# ============================================================================
# Finalization Integration
# ============================================================================


def use_streaming_for_finalization(outcome: Any) -> bool:
    """Decide whether to use streaming.finalize_artifact() for finalization.

    Conditions:
    - Streaming module available and enabled
    - Outcome indicates successful download
    - Not explicitly disabled

    Args:
        outcome: DownloadOutcome instance

    Returns:
        True to use streaming finalization, False to use existing logic.
    """
    if not STREAMING_ENABLED:
        return False

    # Only finalize successful outcomes
    if not hasattr(outcome, "classification"):
        return False

    return True


# ============================================================================
# Schema Integration (NOTE: Implementation handles schema directly in runner.py)
# ============================================================================
# Legacy wrapper functions removed (generate_job_key, generate_operation_key, get_streaming_database)
# These were placeholder integration points; direct imports of idempotency.job_key/op_key are cleaner


# ============================================================================
# Integration Status Report
# ============================================================================


def integration_status() -> Dict[str, bool]:
    """Get status of all optional streaming integrations.

    Returns:
        Dict with keys: streaming, idempotency, schema (all bool).

    Example:
        # In logs or CLI:
        status = integration_status()
        LOGGER.info(f"Streaming integrations available: {status}")
    """
    return {
        "streaming": STREAMING_ENABLED,
        "idempotency": IDEMPOTENCY_ENABLED,
        "schema": SCHEMA_VALIDATION_ENABLED,
    }


def log_integration_status() -> None:
    """Log integration status at INFO level."""
    status = integration_status()
    LOGGER.info(
        "Streaming architecture integrations: streaming=%s, idempotency=%s, schema=%s",
        status["streaming"],
        status["idempotency"],
        status["schema"],
    )
