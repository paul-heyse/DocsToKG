# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.idempotency_telemetry",
#   "purpose": "Telemetry event emission for idempotency system",
#   "sections": [
#     {
#       "id": "emit-event",
#       "name": "emit_event",
#       "anchor": "function-emit-event",
#       "kind": "function"
#     },
#     {
#       "id": "emit-job-planned",
#       "name": "emit_job_planned",
#       "anchor": "function-emit-job-planned",
#       "kind": "function"
#     },
#     {
#       "id": "emit-job-leased",
#       "name": "emit_job_leased",
#       "anchor": "function-emit-job-leased",
#       "kind": "function"
#     },
#     {
#       "id": "emit-job-state-changed",
#       "name": "emit_job_state_changed",
#       "anchor": "function-emit-job-state-changed",
#       "kind": "function"
#     },
#     {
#       "id": "emit-lease-renewed",
#       "name": "emit_lease_renewed",
#       "anchor": "function-emit-lease-renewed",
#       "kind": "function"
#     },
#     {
#       "id": "emit-lease-released",
#       "name": "emit_lease_released",
#       "anchor": "function-emit-lease-released",
#       "kind": "function"
#     },
#     {
#       "id": "emit-operation-started",
#       "name": "emit_operation_started",
#       "anchor": "function-emit-operation-started",
#       "kind": "function"
#     },
#     {
#       "id": "emit-operation-completed",
#       "name": "emit_operation_completed",
#       "anchor": "function-emit-operation-completed",
#       "kind": "function"
#     },
#     {
#       "id": "emit-crash-recovery",
#       "name": "emit_crash_recovery",
#       "anchor": "function-emit-crash-recovery",
#       "kind": "function"
#     },
#     {
#       "id": "emit-idempotency-replay",
#       "name": "emit_idempotency_replay",
#       "anchor": "function-emit-idempotency-replay",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Telemetry event emission for idempotency system.

Emits structured events for job lifecycle, leasing, state transitions, and crash
recovery to enable SLO monitoring and observability.

Event Types:
  - job_planned: Job created, awaiting lease
  - job_leased: Worker claims exclusive access
  - job_state_changed: State transition (PLANNED→LEASED→...)
  - lease_renewed: TTL extended for long operations
  - lease_released: Lease cleared for re-claiming
  - operation_started: Effect execution begins
  - operation_completed: Effect execution finishes
  - crash_recovery_event: Stale lease/ops cleanup
  - idempotency_replay: Cached result returned (no re-execution)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional

LOGGER = logging.getLogger(__name__)

# === EVENT TYPE CONSTANTS ===

EVENT_JOB_PLANNED = "job_planned"
EVENT_JOB_LEASED = "job_leased"
EVENT_JOB_STATE_CHANGED = "job_state_changed"
EVENT_LEASE_RENEWED = "lease_renewed"
EVENT_LEASE_RELEASED = "lease_released"
EVENT_OPERATION_STARTED = "operation_started"
EVENT_OPERATION_COMPLETED = "operation_completed"
EVENT_CRASH_RECOVERY = "crash_recovery_event"
EVENT_IDEMPOTENCY_REPLAY = "idempotency_replay"


# === TELEMETRY SINK ===


def emit_event(event_type: str, payload: Dict[str, Any]) -> None:
    """Emit structured telemetry event.

    Parameters
    ----------
    event_type : str
        Event type (one of EVENT_* constants)
    payload : dict
        Event payload (will be JSON serialized)

    Notes
    -----
    Events are emitted as DEBUG-level log messages for non-blocking
    propagation to telemetry sinks.
    """
    payload_json = json.dumps(payload, default=str)
    LOGGER.debug(f"TELEMETRY|{event_type}|{payload_json}")


# === JOB LIFECYCLE EVENTS ===


def emit_job_planned(
    job_id: str,
    work_id: str,
    artifact_id: str,
    canonical_url: str,
    idempotency_key: str,
) -> None:
    """Emit job_planned event.

    Parameters
    ----------
    job_id : str
        Unique job identifier (UUID)
    work_id : str
        Work identifier
    artifact_id : str
        Artifact identifier
    canonical_url : str
        Canonical URL for the artifact
    idempotency_key : str
        SHA256 idempotency key
    """
    emit_event(
        EVENT_JOB_PLANNED,
        {
            "job_id": job_id,
            "work_id": work_id,
            "artifact_id": artifact_id,
            "canonical_url": canonical_url,
            "idempotency_key": idempotency_key,
            "timestamp": time.time(),
        },
    )


def emit_job_leased(job_id: str, owner: str, ttl_sec: int) -> None:
    """Emit job_leased event.

    Parameters
    ----------
    job_id : str
        Job identifier
    owner : str
        Worker/process identifier
    ttl_sec : int
        Lease time-to-live in seconds
    """
    emit_event(
        EVENT_JOB_LEASED,
        {
            "job_id": job_id,
            "owner": owner,
            "ttl_sec": ttl_sec,
            "timestamp": time.time(),
        },
    )


def emit_job_state_changed(
    job_id: str,
    from_state: str,
    to_state: str,
    reason: Optional[str] = None,
) -> None:
    """Emit job_state_changed event.

    Parameters
    ----------
    job_id : str
        Job identifier
    from_state : str
        Previous state
    to_state : str
        New state
    reason : str, optional
        Reason for transition (e.g., 'crash_recovery')
    """
    emit_event(
        EVENT_JOB_STATE_CHANGED,
        {
            "job_id": job_id,
            "from_state": from_state,
            "to_state": to_state,
            "reason": reason,
            "timestamp": time.time(),
        },
    )


# === LEASING EVENTS ===


def emit_lease_renewed(job_id: str, owner: str, new_ttl_sec: int) -> None:
    """Emit lease_renewed event.

    Parameters
    ----------
    job_id : str
        Job identifier
    owner : str
        Worker/process identifier
    new_ttl_sec : int
        New TTL in seconds
    """
    emit_event(
        EVENT_LEASE_RENEWED,
        {
            "job_id": job_id,
            "owner": owner,
            "new_ttl_sec": new_ttl_sec,
            "timestamp": time.time(),
        },
    )


def emit_lease_released(job_id: str, owner: str) -> None:
    """Emit lease_released event.

    Parameters
    ----------
    job_id : str
        Job identifier
    owner : str
        Worker/process identifier
    """
    emit_event(
        EVENT_LEASE_RELEASED,
        {
            "job_id": job_id,
            "owner": owner,
            "timestamp": time.time(),
        },
    )


# === OPERATION EVENTS ===


def emit_operation_started(
    job_id: str,
    op_key: str,
    op_type: str,
) -> None:
    """Emit operation_started event.

    Parameters
    ----------
    job_id : str
        Job identifier
    op_key : str
        Operation idempotency key (SHA256)
    op_type : str
        Operation type (HEAD, STREAM, FINALIZE, INDEX, DEDUPE)
    """
    emit_event(
        EVENT_OPERATION_STARTED,
        {
            "job_id": job_id,
            "op_key": op_key,
            "op_type": op_type,
            "timestamp": time.time(),
        },
    )


def emit_operation_completed(
    job_id: str,
    op_key: str,
    op_type: str,
    result_code: str,
    elapsed_ms: int,
) -> None:
    """Emit operation_completed event.

    Parameters
    ----------
    job_id : str
        Job identifier
    op_key : str
        Operation idempotency key (SHA256)
    op_type : str
        Operation type
    result_code : str
        Result code (OK, RETRYABLE, NON_RETRYABLE, ERROR)
    elapsed_ms : int
        Execution time in milliseconds
    """
    emit_event(
        EVENT_OPERATION_COMPLETED,
        {
            "job_id": job_id,
            "op_key": op_key,
            "op_type": op_type,
            "result_code": result_code,
            "elapsed_ms": elapsed_ms,
            "timestamp": time.time(),
        },
    )


# === RECOVERY EVENTS ===


def emit_crash_recovery(
    recovered_leases: int,
    abandoned_ops: int,
) -> None:
    """Emit crash_recovery_event.

    Parameters
    ----------
    recovered_leases : int
        Number of stale leases recovered
    abandoned_ops : int
        Number of abandoned operations marked
    """
    emit_event(
        EVENT_CRASH_RECOVERY,
        {
            "recovered_leases": recovered_leases,
            "abandoned_ops": abandoned_ops,
            "timestamp": time.time(),
        },
    )


def emit_idempotency_replay(
    job_id: str,
    op_key: str,
    op_type: str,
    reused_from_time: float,
) -> None:
    """Emit idempotency_replay event (operation result cached, not re-executed).

    Parameters
    ----------
    job_id : str
        Job identifier
    op_key : str
        Operation idempotency key
    op_type : str
        Operation type
    reused_from_time : float
        Timestamp of original execution
    """
    emit_event(
        EVENT_IDEMPOTENCY_REPLAY,
        {
            "job_id": job_id,
            "op_key": op_key,
            "op_type": op_type,
            "reused_from_time": reused_from_time,
            "timestamp": time.time(),
        },
    )
