# Data Model & Idempotency Implementation

**Version**: 1.0
**Status**: Production-Ready
**Last Updated**: October 21, 2025
**Tests**: 22/22 passing ✅

---

## Overview

This document describes the Data Model & Idempotency system (Optimization 8) for ContentDownload. It provides:

- **Exactly-once guarantees** for artifact downloads, finalization, indexing, and deduplication
- **Crash recovery** through state reconciliation and anomaly detection
- **Multi-worker coordination** via SQLite-backed leasing
- **Deterministic job planning** with idempotency keys
- **State machine enforcement** to prevent invalid transitions

---

## Key Concepts

### Idempotency Keys

All operations are identified by deterministic SHA256-based keys:

- **Job Key**: Computed from `(work_id, artifact_id, canonical_url)` - identifies a unique download task
- **Operation Key**: Includes job key + operation type + contextual fields (URL, range, hash)

Keys are deterministic: same inputs always produce identical keys across restarts, retries, and multiple workers.

### Job State Machine

```
PLANNED
  └─(lease)→ LEASED
      ├─(head done)→ HEAD_DONE
      │   └─(resume ok)→ RESUME_OK
      │       └─(stream)→ STREAMING
      │           └─(finalize)→ FINALIZED
      │               ├─(index)→ INDEXED
      │               └─(dedupe)→ DEDUPED
      └─(error)→ FAILED

(found duplicate) → SKIPPED_DUPLICATE
```

State transitions are **strictly monotonic**: only forward progress allowed.

### Leasing Mechanism

- One worker per job at a time (atomic via SQLite UPDATE)
- Configurable TTL (default 120 seconds)
- Automatic expiry for crashed workers
- Renewal and release support

### Cross-Process Coordination

SQLite cooldown store shares:

- Job leases (for worker coordination)
- Operation results (for idempotency)
- Crash recovery metadata

---

## Modules

### 1. `schema_migration.py`

Database schema for artifact tracking:

```python
from DocsToKG.ContentDownload.schema_migration import apply_migration
import sqlite3

conn = sqlite3.connect("manifest.sqlite3")
apply_migration(conn)  # Creates artifact_jobs, artifact_ops, _meta tables
```

**Tables**:

- `artifact_jobs`: Job planning, state, leasing
- `artifact_ops`: Operation ledger (exactly-once tracking)
- `_meta`: Schema versioning

### 2. `idempotency.py`

Deterministic key generation:

```python
from DocsToKG.ContentDownload.idempotency import job_key, op_key

# Job key (one per artifact)
jkey = job_key(work_id="w-123", artifact_id="a-456", canonical_url="https://...")

# Operation keys (one per side-effect)
head_key = op_key("HEAD", jkey, url="https://...")
stream_key = op_key("STREAM", jkey, url="https://...", range_start=0)
finalize_key = op_key("FINALIZE", jkey, sha256="abc123")
```

### 3. `job_planning.py`

Idempotent job creation:

```python
from DocsToKG.ContentDownload.job_planning import plan_job_if_absent

job_id = plan_job_if_absent(
    conn,
    work_id="w-123",
    artifact_id="a-456",
    canonical_url="https://..."
)
# Replanning same artifact returns same job_id
```

### 4. `job_leasing.py`

Multi-worker coordination:

```python
from DocsToKG.ContentDownload.job_leasing import (
    lease_next_job, renew_lease, release_lease
)

# Claim a job
job = lease_next_job(conn, owner="worker-1", ttl_s=120)
if job:
    try:
        # Do work...
        renew_lease(conn, job_id=job["job_id"], owner="worker-1")
    finally:
        release_lease(conn, job_id=job["job_id"], owner="worker-1")
```

### 5. `job_state.py`

State machine with enforcement:

```python
from DocsToKG.ContentDownload.job_state import advance_state, get_current_state

# Valid transition
advance_state(conn, job_id=jid, to_state="HEAD_DONE", allowed_from=("LEASED",))

# Invalid transition → RuntimeError
advance_state(conn, job_id=jid, to_state="STREAMING", allowed_from=("PLANNED",))
```

### 6. `job_effects.py`

Exactly-once operation logging:

```python
from DocsToKG.ContentDownload.job_effects import run_effect, get_effect_result

# First call: executes effect_fn(), stores result
result = run_effect(
    conn,
    job_id=jid,
    kind="HEAD",
    opkey=op_key("HEAD", jid, url="https://..."),
    effect_fn=lambda: {"code": "OK", "status": 200}
)

# Second call with same opkey: returns cached result (no re-execution)
result2 = run_effect(conn, job_id=jid, kind="HEAD", opkey=..., effect_fn=...)
assert result == result2  # Same without calling effect_fn again
```

### 7. `job_reconciler.py`

Crash recovery and cleanup:

```python
from DocsToKG.ContentDownload.job_reconciler import (
    reconcile_jobs, cleanup_stale_leases, cleanup_stale_ops
)

# On startup or periodically:
reconcile_jobs(conn, staging_root=Path("runs/content/.staging"))
cleanup_stale_leases(conn)
cleanup_stale_ops(conn)
```

---

## Typical Usage Pattern

```python
import sqlite3
from pathlib import Path
from DocsToKG.ContentDownload.schema_migration import apply_migration
from DocsToKG.ContentDownload.idempotency import job_key, op_key
from DocsToKG.ContentDownload.job_planning import plan_job_if_absent
from DocsToKG.ContentDownload.job_leasing import lease_next_job, renew_lease, release_lease
from DocsToKG.ContentDownload.job_state import advance_state
from DocsToKG.ContentDownload.job_effects import run_effect
from DocsToKG.ContentDownload.job_reconciler import cleanup_stale_leases

# Setup
conn = sqlite3.connect("manifest.sqlite3")
conn.row_factory = sqlite3.Row
apply_migration(conn)

# Worker startup: heal any crash debris
cleanup_stale_leases(conn)

# Plan a job
job_id = plan_job_if_absent(
    conn,
    work_id="work-123",
    artifact_id="arxiv-2024-001",
    canonical_url="https://api.arxiv.org/pdf/2024.00001v1"
)

# Main loop: claim, work, release
while True:
    job = lease_next_job(conn, owner="worker-1", ttl_s=120)
    if not job:
        break

    job_id = job["job_id"]
    url = job["canonical_url"]

    try:
        # HEAD request
        advance_state(conn, job_id=job_id, to_state="HEAD_DONE", allowed_from=("LEASED",))
        head_key = op_key("HEAD", job_id, url=url)
        head_result = run_effect(
            conn,
            job_id=job_id,
            kind="HEAD",
            opkey=head_key,
            effect_fn=lambda: fetch_head(url)
        )

        # Stream/finalize
        advance_state(
            conn, job_id=job_id, to_state="STREAMING",
            allowed_from=("HEAD_DONE", "RESUME_OK")
        )
        stream_key = op_key("STREAM", job_id, url=url, range_start=0)
        stream_result = run_effect(
            conn,
            job_id=job_id,
            kind="STREAM",
            opkey=stream_key,
            effect_fn=lambda: stream_to_file(url)
        )

        # Finalize
        advance_state(
            conn, job_id=job_id, to_state="FINALIZED",
            allowed_from=("STREAMING",)
        )
        finalize_key = op_key("FINALIZE", job_id, sha256=stream_result["sha256"])
        finalize_result = run_effect(
            conn,
            job_id=job_id,
            kind="FINALIZE",
            opkey=finalize_key,
            effect_fn=lambda: finalize_artifact()
        )

        advance_state(
            conn, job_id=job_id, to_state="INDEXED",
            allowed_from=("FINALIZED",)
        )

    except Exception as e:
        advance_state(conn, job_id=job_id, to_state="FAILED", allowed_from=("LEASED", "HEAD_DONE", "STREAMING"))
        raise
    finally:
        release_lease(conn, job_id=job_id, owner="worker-1")
```

---

## Safety Guarantees

### Exactly-Once Downloads

- Job planning is idempotent (UNIQUE constraint on idempotency key)
- Each operation effect recorded once (UNIQUE constraint on op_key)
- Replayed operations return cached result without side-effects

### Crash Recovery

- Stale leases auto-expire (TTL-based)
- Orphaned `.part` files cleaned up (age-based)
- Abandoned operations marked for visibility
- DB↔FS state healed on startup

### State Machine Integrity

- Invalid transitions raise `RuntimeError` immediately
- No cycles or backtracking allowed
- All paths eventually reach terminal state (FINALIZED, FAILED, etc.)

---

## Configuration & Tuning

### Lease TTL

Default: 120 seconds. Adjust based on:

- Expected operation duration
- Acceptable re-work window on crash
- Number of concurrent workers

### Reconciliation Intervals

Run on:

- Worker startup
- Hourly (periodic background task)
- After detecting cascading failures

### Stale File Threshold

Default: 3600 seconds (1 hour). Clean `.part` files older than this.

---

## Testing

22 comprehensive tests covering:

- ✅ Idempotency key determinism
- ✅ Job planning and replanning
- ✅ Atomic leasing and multi-worker coordination
- ✅ Lease renewal and release
- ✅ Lease expiry handling
- ✅ State machine enforcement
- ✅ Invalid transition rejection
- ✅ Full state progressions
- ✅ Exactly-once effect execution
- ✅ Cached result replay
- ✅ Crash recovery and reconciliation
- ✅ Stale lease cleanup
- ✅ Abandoned operation detection
- ✅ Stale file cleanup

Run: `pytest tests/content_download/test_idempotency.py -v`

---

## Integration Points

### Phase 9: Download Pipeline Integration

When integrating with `download.py` and `runner.py`:

1. Apply migration on runner startup
2. Plan jobs during manifest ingestion
3. Claim/work/release in download loop
4. Handle state transitions at each major operation
5. Use exactly-once effects for file I/O

### Feature Gates

Suggested rollout with feature flags:

```python
ENABLE_IDEMPOTENCY = os.getenv("DOCSTOKG_ENABLE_IDEMPOTENCY", "false").lower() == "true"

if ENABLE_IDEMPOTENCY:
    apply_migration(conn)
    cleanup_stale_leases(conn)
```

---

## Post-Migration Checklist

- [ ] Schema migration successful (check `_meta.schema_version` == 3)
- [ ] All modules importable without errors
- [ ] Tests passing (22/22)
- [ ] No stale leases or operations on startup
- [ ] Job planning works idempotently
- [ ] State transitions enforced correctly
- [ ] Effects are replayed with cached results
- [ ] Crash recovery tested manually

---

## Next Steps

- **Phase 9**: Integrate into download.py / runner.py with feature gates
- **Phase 10**: Update AGENTS.md with usage patterns and operational runbooks
- **Future**: Add Redis support for distributed deployments
