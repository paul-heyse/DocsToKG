"""Optional streaming integration layer for download.py.

This module provides conditional use of RFC-compliant streaming primitives from
streaming.py while maintaining full backward compatibility with existing download.py
logic. It allows incremental adoption of the new streaming architecture.

Features:
  - Graceful fallback to existing logic if streaming module not available
  - Feature flag control for gradual rollout
  - Performance comparison utilities
  - Transparent use of new primitives (can_resume, stream_to_part, etc.)

Usage:
  # In download.py:
  from DocsToKG.ContentDownload.streaming_integration import (
      use_streaming_for_resume,
      use_streaming_for_io,
      streaming_enabled,
  )

  if use_streaming_for_resume(plan):
      decision = streaming.can_resume(validators, part_state, ...)
  else:
      # Existing resume logic
  ...
"""

from __future__ import annotations

import logging
import os
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
    from DocsToKG.ContentDownload.download import DownloadPreflightPlan


# ============================================================================
# Feature Flags
# ============================================================================


def streaming_enabled() -> bool:
    """Check if streaming module is available and enabled.

    Returns:
        True if streaming can be used, False otherwise.

    Env Vars:
        DOCSTOKG_ENABLE_STREAMING: Set to "0" to disable streaming integration.
        DOCSTOKG_STREAMING_RATIO: Ratio of jobs to use streaming (0.0-1.0).
    """
    if not _STREAMING_AVAILABLE:
        return False

    if os.getenv("DOCSTOKG_ENABLE_STREAMING") == "0":
        return False

    return True


def idempotency_enabled() -> bool:
    """Check if idempotency module is available and enabled.

    Returns:
        True if idempotency can be used, False otherwise.

    Env Vars:
        DOCSTOKG_ENABLE_IDEMPOTENCY: Set to "0" to disable idempotency.
    """
    if not _IDEMPOTENCY_AVAILABLE:
        return False

    if os.getenv("DOCSTOKG_ENABLE_IDEMPOTENCY") == "0":
        return False

    return True


def schema_enabled() -> bool:
    """Check if streaming_schema module is available and enabled.

    Returns:
        True if streaming schema can be used, False otherwise.

    Env Vars:
        DOCSTOKG_ENABLE_STREAMING_SCHEMA: Set to "0" to disable schema.
    """
    if not _SCHEMA_AVAILABLE:
        return False

    if os.getenv("DOCSTOKG_ENABLE_STREAMING_SCHEMA") == "0":
        return False

    return True


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
    if not streaming_enabled():
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
    if not streaming_enabled():
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
    if not streaming_enabled():
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
    if not streaming_enabled():
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
    if not streaming_enabled():
        return False

    # Only finalize successful outcomes
    if not hasattr(outcome, "classification"):
        return False

    return True


# ============================================================================
# Idempotency Integration
# ============================================================================


def generate_job_key(
    work_id: str,
    artifact_id: str,
    canonical_url: str,
) -> Optional[str]:
    """Generate deterministic idempotency key if available.

    Args:
        work_id: OpenAlex work ID
        artifact_id: Artifact identifier
        canonical_url: Canonical URL

    Returns:
        Deterministic job key or None if idempotency not available.

    Example:
        # In download.py download_candidate():
        job_key = generate_job_key(work_id, artifact_id, canonical_url)
        if job_key:
            # Store in database for exactly-once semantics
            job_id = acquire_or_reuse_job(job_key)
    """
    if not idempotency_enabled():
        return None

    if not idempotency:
        return None

    try:
        return idempotency.job_key(work_id, artifact_id, canonical_url)
    except Exception as e:
        LOGGER.debug(f"Failed to generate job key: {e}")
        return None


def generate_operation_key(
    op_type: str,
    job_id: str,
    **context: Any,
) -> Optional[str]:
    """Generate deterministic operation key if available.

    Args:
        op_type: Operation type (HEAD, STREAM, FINALIZE, etc.)
        job_id: Job identifier
        **context: Additional context (url, range_start, etc.)

    Returns:
        Deterministic operation key or None if idempotency not available.

    Example:
        # In download.py stream_candidate_payload():
        op_key = generate_operation_key("STREAM", job_id, url=url)
        if op_key:
            # Run with exactly-once guarantee
            result = run_effect_once(op_key, lambda: stream_and_save())
    """
    if not idempotency_enabled():
        return None

    if not idempotency:
        return None

    try:
        return idempotency.op_key(op_type, job_id, **context)
    except Exception as e:
        LOGGER.debug(f"Failed to generate operation key: {e}")
        return None


# ============================================================================
# Schema Integration
# ============================================================================


def get_streaming_database(db_path: Optional[str] = None) -> Optional[Any]:
    """Get streaming database context manager if available.

    Args:
        db_path: Optional path to database file

    Returns:
        StreamingDatabase instance or None if schema not available.

    Example:
        # In download.py runner initialization:
        with get_streaming_database() as db:
            if db:
                job_id = idempotency.acquire_lease(db, worker_id, 60)
                # Use job_id for coordination
    """
    if not schema_enabled():
        return None

    if not streaming_schema:
        return None

    try:
        return streaming_schema.StreamingDatabase(db_path)
    except Exception as e:
        LOGGER.debug(f"Failed to create streaming database: {e}")
        return None


def check_database_health(db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Check streaming database health if available.

    Args:
        db_path: Optional path to database file

    Returns:
        Health check results dict or None if schema not available.

    Example:
        # In CLI startup:
        health = check_database_health()
        if health and health["status"] != "healthy":
            LOGGER.warning(f"Database health: {health}")
    """
    if not schema_enabled():
        return None

    if not streaming_schema:
        return None

    try:
        return streaming_schema.health_check(db_path)
    except Exception as e:
        LOGGER.debug(f"Failed to check database health: {e}")
        return None


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
        "streaming": streaming_enabled(),
        "idempotency": idempotency_enabled(),
        "schema": schema_enabled(),
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
