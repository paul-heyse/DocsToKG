# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.orchestrator",
#   "purpose": "Work queue orchestration with bounded concurrency, crash recovery, and fairness",
#   "sections": [
#     {"id": "workqueue", "name": "WorkQueue", "anchor": "#class-workqueue", "kind": "class"},
#     {"id": "orchestrator", "name": "Orchestrator", "anchor": "#class-orchestrator", "kind": "class"},
#     {"id": "keyedlimiter", "name": "KeyedLimiter", "anchor": "#class-keyedlimiter", "kind": "class"},
#     {"id": "worker", "name": "Worker", "anchor": "#class-worker", "kind": "class"}
#   ]
# }
# === /NAVMAP ===

"""Work Queue Orchestration for ContentDownload.

This package provides a **persistent work queue**, **bounded-concurrency orchestrator**,
and **graceful resume** without changing core pipeline, caching, or telemetry contracts.

**Key Components:**

- `queue.WorkQueue`: SQLite-backed, idempotent enqueue, lease/ack/fail/retry
- `scheduler.Orchestrator`: Dispatcher loop, heartbeat, bounded worker pool
- `limits.KeyedLimiter`: Per-resolver and per-host concurrency fairness
- `workers.Worker`: Thread wrapper around pipeline.process()
- `models`: JobState enums and dataclasses

**Design Principles:**

- **Idempotence**: artifact_id unique index prevents duplicates
- **Crash-safety**: Leasing with TTL enables recovery on worker crash
- **Politeness**: Keyed semaphores + rate limits = host fairness
- **Observability**: OTel metrics for queue depth and throughput
- **Simplicity**: SQLite on single host; easy to scale to Postgres later

**Example Usage:**

```python
from DocsToKG.ContentDownload.orchestrator import WorkQueue, Orchestrator
from DocsToKG.ContentDownload.orchestrator.config import OrchestratorConfig

# Create queue and orchestrator
queue = WorkQueue("state/workqueue.sqlite", wal_mode=True)
config = OrchestratorConfig(max_workers=8, max_per_host=4)
orch = Orchestrator(config, queue, pipeline, telemetry=None)

# Start workers
orch.start()

# Enqueue artifacts
queue.enqueue("doi:10.1234/example", {"doi": "10.1234/example"})

# Monitor
stats = queue.stats()  # {'queued': 100, 'in_progress': 8, 'done': 15, ...}
```

**References:**

- Architecture & Flow: See ContentDownload AGENTS.md § Work Orchestration
- PR #8 Plan: ContentDownload Work Orchestrator & Bounded Concurrency
- Integration: `cli_orchestrator.py` for CLI commands
"""

from __future__ import annotations

from .limits import KeyedLimiter, host_key
from .models import JobState, JobResult
from .queue import WorkQueue
from .workers import Worker

__all__ = [
    "KeyedLimiter",
    "host_key",
    "JobState",
    "JobResult",
    "WorkQueue",
    "Worker",
]
