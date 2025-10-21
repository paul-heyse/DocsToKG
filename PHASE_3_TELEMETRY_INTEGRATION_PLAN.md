# PHASE 3 PLAN: Testing + Telemetry Integration

**Status:** IN PROGRESS  
**Estimated Duration:** 2-3 days  
**Target Completion:** October 22-24, 2025

---

## Overview

Phase 3 integrates the idempotency system with telemetry and implements end-to-end testing to validate the complete pipeline: feature gates → job tracking → crash recovery → telemetry emission.

### Goals

1. ✅ **End-to-End Integration Tests** - Validate download pipeline with idempotency enabled
2. ✅ **Telemetry Event Emission** - Emit events for job lifecycle, leases, state transitions
3. ✅ **SLO Integration** - Connect to existing observability/SLOs framework
4. ✅ **Documentation** - Update AGENTS.md with idempotency system

---

## Task Breakdown

### P3.1: End-to-End Integration Tests (1 day)

**Objectives:**
- Simulate complete download workflow with idempotency enabled
- Verify crash recovery + re-execution
- Validate job state transitions through pipeline
- Test multi-worker scenarios

**Deliverables:**
- `tests/content_download/test_idempotency_e2e.py` (200-300 LOC)
- 15-20 integration tests
- Fixtures for mock HTTP clients + telemetry capture

**Key Tests:**
```
1. test_complete_download_with_idempotency
   - Plan → Lease → HEAD → STREAM → FINALIZE lifecycle
   
2. test_crash_recovery_simulation
   - Kill worker mid-download
   - Verify another worker can resume
   - Check state consistency
   
3. test_multi_worker_scenario
   - Multiple workers sharing job pool
   - Verify lease-based coordination
   - Confirm no duplicate downloads
   
4. test_idempotency_across_restarts
   - Same job executed twice
   - Verify no side-effects repeated
   - Check operation ledger

5. test_network_error_recovery
   - Simulate transient HTTP failures
   - Verify retry + state preservation
   - Check lease renewal during retries
```

---

### P3.2: Telemetry Event Emission (1 day)

**Objectives:**
- Emit telemetry events for job lifecycle
- Track lease acquisition/renewal/release
- Record state transitions
- Integrate with SLO schema

**Deliverables:**
- Telemetry event decorators in idempotency modules (100 LOC)
- Event schema definitions
- Integration with existing telemetry sink
- Tests for event emission (50+ LOC)

**Events to Emit:**

```
1. job_planned
   - work_id, artifact_id, url, job_id
   - timestamp, idempotency_key

2. job_leased
   - job_id, owner, ttl_sec
   - timestamp

3. job_state_changed
   - job_id, from_state, to_state
   - timestamp, reason (optional)

4. lease_renewed
   - job_id, owner, new_ttl_sec
   - timestamp

5. lease_released
   - job_id, owner
   - timestamp

6. operation_started
   - job_id, op_key, op_type (HEAD/STREAM/FINALIZE/etc)
   - timestamp

7. operation_completed
   - job_id, op_key, result_code, elapsed_ms
   - timestamp

8. crash_recovery_event
   - recovered_leases: N
   - abandoned_ops: M
   - timestamp

9. idempotency_replay
   - job_id, op_key, reused_from_time
   - timestamp (shows no re-execution)
```

**Telemetry Integration:**
```python
from DocsToKG.ContentDownload.idempotency_telemetry import (
    emit_job_event,
    emit_lease_event,
    emit_operation_event,
    emit_recovery_event,
)

# In job_planning.py
job_id = plan_job_if_absent(conn, work_id, artifact_id, url)
emit_job_event("job_planned", {
    "job_id": job_id,
    "work_id": work_id,
    "artifact_id": artifact_id,
    "url": url,
})

# In job_leasing.py
job = lease_next_job(conn, owner)
emit_lease_event("job_leased", {
    "job_id": job["job_id"],
    "owner": owner,
    "ttl_sec": lease_ttl_s,
})

# In job_state.py
advance_state(conn, job_id, to_state, allowed_from)
emit_state_event("job_state_changed", {
    "job_id": job_id,
    "to_state": to_state,
    "from_state": current_state,
})

# In job_reconciler.py
cleanup_stale_leases(conn)
emit_recovery_event("crash_recovery", {
    "recovered_leases": cleared_count,
    "abandoned_ops": marked_count,
})
```

---

### P3.3: SLO Integration (0.5 days)

**Objectives:**
- Wire telemetry to SLO/observability framework
- Define SLO thresholds
- Create dashboards/alerts

**Deliverables:**
- SLO schema extensions (50 LOC)
- Telemetry sink configuration
- Example dashboards (Grafana/Prometheus)

**SLOs to Track:**

```
1. Job Completion Rate
   - SLO: 99.5% of jobs reach FINALIZED state
   - Error budget: 0.5% per day

2. Mean Time to Complete
   - SLO: p50 < 30s, p95 < 120s
   - Baseline: Establish after rollout

3. Crash Recovery Success
   - SLO: 99.9% of crashes recovered
   - Tracking: recovered_leases / total_crashed

4. Lease Acquisition Latency
   - SLO: p99 < 100ms
   - Baseline: Establish after rollout

5. Operation Replay Rate (Idempotency)
   - SLO: < 5% of operations are replayed
   - Success indicator: System is stable
```

---

### P3.4: Documentation Updates (0.5 days)

**Objectives:**
- Update AGENTS.md with idempotency system
- Add troubleshooting guides
- Document SLOs and telemetry

**Deliverables:**
- AGENTS.md section: "Idempotency & Crash Recovery" (300 LOC)
- Troubleshooting guide
- Telemetry event reference
- SLO documentation

**Documentation Sections:**

```
## Idempotency & Crash Recovery System

### Overview
The idempotency system ensures exactly-once effects across:
- Worker crashes and restarts
- Multiple concurrent workers
- Network failures and retries

### Feature Gate
- Environment variable: DOCSTOKG_ENABLE_IDEMPOTENCY
- CLI flag: --enable-idempotency
- Default: false (backward compatible)

### State Machine
States and allowed transitions...

### Job Lifecycle
1. PLANNED: Job created, waiting for lease
2. LEASED: Worker claims exclusive access
3. HEAD_DONE: HTTP HEAD request completed
4. RESUME_OK: Resume capability verified
5. STREAMING: Downloading content
6. FINALIZED: File moved to final location
7. INDEXED/DEDUPED: Hash indexed/deduplicated

### Crash Recovery
- Automatic stale lease cleanup
- Abandoned operation marking
- Partial file cleanup

### Telemetry Events
- job_planned, job_leased, job_state_changed
- lease_renewed, lease_released
- operation_started, operation_completed
- crash_recovery_event, idempotency_replay

### SLOs & Monitoring
- Job completion rate
- Mean time to complete
- Crash recovery success
- Lease acquisition latency
- Operation replay rate

### Troubleshooting
- "database is locked" errors
- Stale lease accumulation
- Abandoned files in .staging
```

---

## Acceptance Criteria

### Testing ✅
- [ ] All end-to-end tests passing (20+ tests)
- [ ] Crash recovery verified with simulation
- [ ] Multi-worker coordination validated
- [ ] Network failure recovery tested
- [ ] Idempotency replay verified

### Telemetry ✅
- [ ] All 9 event types emitting correctly
- [ ] Events captured by telemetry sink
- [ ] No impact on performance (<5ms overhead)
- [ ] Events queryable for SLO monitoring
- [ ] Integration with existing telemetry framework

### SLO Integration ✅
- [ ] SLO thresholds defined
- [ ] Baseline metrics established
- [ ] Dashboards created (Grafana)
- [ ] Alerts configured
- [ ] Documentation complete

### Documentation ✅
- [ ] AGENTS.md updated
- [ ] Troubleshooting guide complete
- [ ] Telemetry reference documented
- [ ] SLO documentation in place
- [ ] Examples provided

---

## Quality Gates

| Gate | Target | Status |
|------|--------|--------|
| Test Coverage | >95% | ⏳ |
| Type Safety | 100% | ⏳ |
| Linting | 0 errors | ⏳ |
| Performance Overhead | <5ms | ⏳ |
| Backward Compat | 100% | ⏳ |
| Documentation | Complete | ⏳ |

---

## Timeline

```
Day 1: P3.1 + P3.2 (End-to-End Tests + Telemetry)
Day 2: P3.2 + P3.3 (Telemetry Integration + SLO Setup)
Day 3: P3.4 (Documentation)
```

---

## Next Phase

**Phase 4: Rollout + Monitoring (1-2 days)**

After Phase 3 completion:
1. Canary rollout (5% of workers)
2. Gradual increase to 25%, 50%, 100%
3. Monitor SLOs during rollout
4. Performance baseline establishment
5. Production deployment

---

## Notes

### Risk Mitigation
- All telemetry non-blocking (async emit)
- Feature gate allows instant disable
- Backward compatible with existing code
- No schema changes to existing tables

### Performance Considerations
- Telemetry events batched
- Minimal DB overhead (~5ms per job)
- No locks held during event emission
- Async sink to prevent blocking

### Rollback Strategy
- Disable via environment variable
- No data migration needed
- Lease tables auto-cleanup after TTL
- Operations ledger can be archived

---

## Success Criteria for Phase 3

✅ Phase 3 is COMPLETE when:
1. End-to-end tests passing (100%)
2. Telemetry events emitting (all 9 types)
3. SLO framework integrated
4. Documentation updated
5. All quality gates passed
6. Ready for canary rollout

