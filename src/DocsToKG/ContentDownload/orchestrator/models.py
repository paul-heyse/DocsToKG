# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.orchestrator.models",
#   "purpose": "Job state enums, result types, and coordination dataclasses",
#   "sections": [
#     {"id": "jobstate", "name": "JobState", "anchor": "#class-jobstate", "kind": "enum"},
#     {"id": "jobresult", "name": "JobResult", "anchor": "#class-jobresult", "kind": "dataclass"}
#   ]
# }
# === /NAVMAP ===

"""Job state models and coordination types.

Defines stable enums and dataclasses for work queue coordination, job leasing,
and result tracking. Integrates with SQLite schema in queue.py.

**State Machine (Jobs):**

    QUEUED
      ↓ (lease) → set worker_id, lease_expires_at
      ↓
    IN_PROGRESS
      ↓ (ack done/skipped/error)
      ├→ DONE
      ├→ SKIPPED
      └→ ERROR

If lease_until < now while IN_PROGRESS → can re-lease (crash recovery).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional


class JobState(str, Enum):
    """Job lifecycle states.

    - QUEUED: Ready to run
    - IN_PROGRESS: Leased by worker
    - DONE: Success
    - SKIPPED: Non-error terminal (robots, 304, etc.)
    - ERROR: Failed after max_job_attempts
    """

    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass(frozen=True)
class JobResult:
    """Result of a job execution.

    Attributes:
        job_id: Row ID in jobs table
        artifact_id: Unique identifier (e.g., doi:10.1234/example)
        state: Terminal JobState
        outcome: Short description (e.g., "success", "rate_limited")
        attempts: Total attempts made
        last_error: Last error message (truncated, if any)
        resolver_hint: Optional resolver name that succeeded
    """

    job_id: int
    artifact_id: str
    state: JobState
    outcome: str
    attempts: int
    last_error: Optional[str] = None
    resolver_hint: Optional[str] = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def is_terminal(self) -> bool:
        """Check if job is in terminal state."""
        return self.state in (JobState.DONE, JobState.SKIPPED, JobState.ERROR)

    def is_success(self) -> bool:
        """Check if job succeeded."""
        return self.state == JobState.DONE
