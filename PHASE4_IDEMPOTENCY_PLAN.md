# Phase 4: Idempotency Integration & Crash Recovery

**Status:** IN PROGRESS 🔄

**Estimated Duration:** 10 days

**Target Completion:** TBD

---

## Executive Summary

Phase 4 integrates exactly-once semantics and crash recovery into the download pipeline. By leveraging deterministic idempotency keys and a state machine, we ensure:

- **Zero Duplicate Processing**: Same artifact never downloaded twice, even across crashes
- **Automatic Crash Recovery**: Stale jobs detected and recovered on startup
- **Multi-Worker Safety**: Safe concurrent processing without race conditions
- **Operation Ledger**: Complete audit trail of all operations

---

## Architecture Overview

### Idempotency Model

```
Job Idempotency:
  [work_id, artifact_id, canonical_url] → job_key (SHA-256)
                                         → UNIQUE constraint
                                         → INSERT OR IGNORE

Operation Idempotency:
  [kind, job_id, context...] → op_key (SHA-256)
                             → INSERT OR IGNORE
                             → Ledger table (artifact_ops)

State Machine:
  PLANNED → LEASED → HEAD_DONE → RESUME_OK → STREAMING 
         → FINALIZED → INDEXED → DEDUPED → SUCCESS
         └─ FAILED, SKIPPED_DUPLICATE (terminal states)
```

### Integration Points

#### 1. **process_one_work() - Job Initialization**
```python
# Generate job key
job_key_val = job_key(artifact.work_id, artifact.id, canonical_url)

# Check if already completed
state = check_job_state(job_key_val)
if state == "SUCCESS":
    return {"skipped": True}  # Already done

# Acquire lease (PID-based)
acquire_lease(job_key_val, os.getpid())
```

#### 2. **stream_candidate_payload() - Operation Logging**
```python
# Log HEAD request
op_key_head = op_key("HEAD", job_key_val, url=url)
run_effect("HEAD", op_key_head, lambda: head_request(...))

# Log STREAM operation
op_key_stream = op_key("STREAM", job_key_val, url=url, range_start=0)
run_effect("STREAM", op_key_stream, lambda: stream_to_file(...))

# Advance state
advance_state(job_key_val, "STREAMING")
```

#### 3. **finalize_candidate_download() - Finalization**
```python
# Log finalization
op_key_finalize = op_key("FINALIZE", job_key_val, path=dest_path)
run_effect("FINALIZE", op_key_finalize, lambda: atomic_finalize(...))

# Advance to final state
advance_state(job_key_val, "FINALIZED")

# Release lease
release_lease(job_key_val)
```

#### 4. **Startup - Crash Recovery**
```python
# On app initialization
reconcile_stale_leases(timeout_seconds=3600)  # 1 hour default
# Marks stale leases as FAILED for retry
```

---

## Implementation Phases

### Phase 4.1: Core Integration (2 days)

**Deliverables:**
- Import idempotency functions into download.py
- Add job_key generation to process_one_work()
- Add op_key generation to stream_candidate_payload()
- Add op_key generation to finalize_candidate_download()

**Expected LOC:** ~60

**Key Functions:**
```python
from DocsToKG.ContentDownload.idempotency import (
    job_key,
    op_key,
    acquire_lease,
    release_lease,
    advance_state,
    run_effect,
    reconcile_stale_leases,
)
```

### Phase 4.2: Lease Management (2 days)

**Deliverables:**
- Implement acquire_lease() calls on job start
- Implement renew_lease() during streaming
- Implement release_lease() on completion
- Handle lease timeout scenarios

**Expected LOC:** ~40

**Key Logic:**
```python
# On job start
pid = os.getpid()
acquire_lease(job_key_val, pid)

# Periodically during streaming
renew_lease(job_key_val)

# On completion
release_lease(job_key_val)
```

### Phase 4.3: State Machine (2 days)

**Deliverables:**
- Call advance_state() at each milestone
- Validate state transitions
- Handle conflicts (e.g., state already advanced)

**Expected LOC:** ~30

**State Transitions:**
```
PLANNED → LEASED → HEAD_DONE → RESUME_OK → STREAMING 
       → FINALIZED → INDEXED → DEDUPED → SUCCESS
```

### Phase 4.4: Crash Recovery (2 days)

**Deliverables:**
- Call reconcile_stale_leases() on startup
- Implement recovery for abandoned jobs
- Add logging for recovery events

**Expected LOC:** ~20

**Recovery Logic:**
```python
# On app startup
stale_jobs = reconcile_stale_leases(timeout_seconds=3600)
for job_id in stale_jobs:
    LOGGER.info(f"Recovered stale job {job_id}, marking for retry")
```

### Phase 4.5: Testing & Verification (2 days)

**Deliverables:**
- Unit tests for key generation (determinism, collisions)
- Integration tests for state machine
- Crash recovery tests (multi-worker, timeout scenarios)
- End-to-end tests for exactly-once semantics

**Expected Test Count:** 30+ new tests

---

## Integration Checklist

### Pre-Integration
- [ ] Review idempotency.py API
- [ ] Understand job_key() and op_key() generation
- [ ] Plan database schema additions (if needed)
- [ ] Design state machine validation

### Core Integration
- [ ] Import idempotency functions
- [ ] Add job_key to process_one_work()
- [ ] Add op_key to stream_candidate_payload()
- [ ] Add op_key to finalize_candidate_download()

### Lease Management
- [ ] Implement acquire_lease() on start
- [ ] Implement renew_lease() during streaming
- [ ] Implement release_lease() on completion
- [ ] Handle timeout scenarios

### State Machine
- [ ] Add advance_state() calls
- [ ] Validate transitions
- [ ] Handle conflicts

### Crash Recovery
- [ ] Call reconcile_stale_leases() on startup
- [ ] Implement recovery logic
- [ ] Add telemetry for recovery

### Testing
- [ ] Key generation tests (determinism)
- [ ] State machine tests
- [ ] Lease management tests
- [ ] Crash recovery tests
- [ ] Multi-worker safety tests
- [ ] End-to-end tests

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| State conflicts | Low | High | Test all transitions |
| Lease timeout issues | Medium | Medium | Add comprehensive logging |
| Database migration | Medium | High | Version schema carefully |
| Performance impact | Low | Medium | Benchmark key generation |
| Multi-worker race conditions | Low | High | Thorough concurrency testing |

---

## Success Criteria

✅ **Functionality**
- All job keys generated deterministically
- All operations logged with op_keys
- State machine transitions valid
- Leases managed correctly
- Crash recovery works

✅ **Quality**
- 30+ new tests, all passing
- 0 type errors
- 0 linting errors
- Backward compatible

✅ **Performance**
- Key generation < 1ms
- Lease operations < 10ms
- No significant performance degradation

✅ **Reliability**
- Multi-worker safety verified
- Crash recovery tested
- State machine validated

---

## Next Steps

1. **Day 1-2:** Phase 4.1 - Core Integration
2. **Day 3-4:** Phase 4.2 - Lease Management
3. **Day 5-6:** Phase 4.3 - State Machine
4. **Day 7-8:** Phase 4.4 - Crash Recovery
5. **Day 9-10:** Phase 4.5 - Testing & Verification

---

## Cumulative Status

| Phase | Status | LOC | Tests | Cumulative |
|-------|--------|-----|-------|-----------|
| 1-3 | ✅ Complete | 205 | 24 | 205 LOC, 24 tests |
| 4 (Current) | IN PROGRESS | ~200 est. | ~30 est. | ~405 LOC, ~54 tests |
| 5+ | Pending | - | - | - |

---

## Key Metrics to Track

- Job key generation time (target: < 1ms)
- Lease acquisition time (target: < 10ms)
- State transition time (target: < 5ms)
- Crash recovery time (target: < 1s per 100 jobs)
- Test coverage (target: 100%)
- Backward compatibility (target: 100%)

