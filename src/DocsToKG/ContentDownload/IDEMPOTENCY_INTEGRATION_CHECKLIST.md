# Idempotency System - Production Integration Checklist

**Document Version**: 1.0
**Status**: READY FOR PHASE 9 INTEGRATION
**Target**: Phases 9-10 (2-3 weeks estimated)

---

## Executive Summary

The data model & idempotency system (7 modules, 1200+ LOC) is **100% complete** and **100% tested** (22/22 passing). However, it is **NOT YET PROPAGATED** into the production download pipeline.

This checklist covers what's needed to move from "component-ready" to "production-deployed".

**Current Status**: ✅ Modules complete, ❌ Integration incomplete

---

## PHASE 9: INTEGRATION (2 weeks, HIGH PRIORITY)

### 9.1 Feature Gate Implementation

**Status**: ❌ NOT STARTED

```python
# Location: src/DocsToKG/ContentDownload/runner.py (line ~50)
# Add at top of module:

import os
ENABLE_IDEMPOTENCY = os.getenv("DOCSTOKG_ENABLE_IDEMPOTENCY", "false").lower() == "true"

# In DownloadRun.__init__():
if ENABLE_IDEMPOTENCY:
    from DocsToKG.ContentDownload.schema_migration import apply_migration
    from DocsToKG.ContentDownload.job_reconciler import cleanup_stale_leases, cleanup_stale_ops

    apply_migration(self.telemetry_sink.conn)
    cleanup_stale_leases(self.telemetry_sink.conn)
    cleanup_stale_ops(self.telemetry_sink.conn)
```

**Checklist**:

- [ ] Add `ENABLE_IDEMPOTENCY` env var check
- [ ] Call `apply_migration()` on runner startup
- [ ] Call `cleanup_stale_leases()` on startup
- [ ] Call `cleanup_stale_ops()` on startup
- [ ] Document in AGENTS.md: "Set DOCSTOKG_ENABLE_IDEMPOTENCY=true to enable"
- [ ] Test with feature disabled (backward compatibility)
- [ ] Test with feature enabled (new code path)

---

### 9.2 Runner Startup Integration

**Status**: ❌ NOT STARTED

```python
# Location: src/DocsToKG/ContentDownload/runner.py
# In DownloadRun.run() method, after setup_sinks() and before pipeline execution:

if ENABLE_IDEMPOTENCY:
    self.logger.info("Idempotency system enabled - initializing job tracking")
    # Cleanup will already be done in __init__
```

**Checklist**:

- [ ] Verify migration runs on first runner startup
- [ ] Verify cleanup runs before worker loop starts
- [ ] Check logs for migration status message
- [ ] Verify _meta.schema_version = 3 in telemetry DB
- [ ] Add test: runner with idempotency enabled starts cleanly

---

### 9.3 Download Loop Integration

**Status**: ❌ NOT STARTED

**Location**: `src/DocsToKG/ContentDownload/download.py::process_one_work()`

```python
# At top of process_one_work():
if ENABLE_IDEMPOTENCY:
    from DocsToKG.ContentDownload.job_planning import plan_job_if_absent
    from DocsToKG.ContentDownload.idempotency import job_key, op_key

    # Plan the job
    job_id = plan_job_if_absent(
        telemetry_sink.conn,
        work_id=work.id,
        artifact_id=artifact_id,
        canonical_url=work.canonical_url
    )
    logger.debug(f"Job planned: {job_id}")

# During HEAD request:
if ENABLE_IDEMPOTENCY:
    from DocsToKG.ContentDownload.job_effects import run_effect
    from DocsToKG.ContentDownload.job_state import advance_state

    advance_state(
        telemetry_sink.conn,
        job_id=job_id,
        to_state="HEAD_DONE",
        allowed_from=("LEASED",)
    )
    head_key = op_key("HEAD", job_id, url=work.canonical_url)
    head_result = run_effect(
        telemetry_sink.conn,
        job_id=job_id,
        kind="HEAD",
        opkey=head_key,
        effect_fn=lambda: do_head_request()
    )

# Similar for STREAM, FINALIZE, INDEX, DEDUPE operations
```

**Checklist**:

- [ ] Import job_planning, idempotency, job_state, job_effects at top
- [ ] Call plan_job_if_absent() before starting work
- [ ] Advance state before each major operation
- [ ] Wrap each operation with run_effect()
- [ ] Handle state transition errors (RuntimeError)
- [ ] Add try/finally to release lease on exit
- [ ] Test: single download with idempotency
- [ ] Test: repeated download (should reuse job, return cached results)
- [ ] Test: crash during download (should recover)

---

### 9.4 Pipeline Integration

**Status**: ❌ NOT STARTED

**Location**: `src/DocsToKG/ContentDownload/pipeline.py`

```python
# When yielding candidates for download:
# (Optional) Queue jobs in advance if needed:

if ENABLE_IDEMPOTENCY:
    from DocsToKG.ContentDownload.job_planning import plan_job_if_absent

    # Could pre-plan all jobs during manifest phase
    # for candidate in candidates:
    #     plan_job_if_absent(conn, work_id=..., artifact_id=..., url=...)
```

**Note**: Pipeline integration is *optional* - jobs are planned in download loop

**Checklist**:

- [ ] Verify pipeline doesn't conflict with job planning
- [ ] Document that job planning happens in download loop (not pipeline)
- [ ] Test pipeline with idempotency enabled

---

### 9.5 Configuration

**Status**: ❌ NOT STARTED

**Options**:

1. Add to existing `config/breakers.yaml`
2. Create new `config/idempotency.yaml`

**Recommended**: Add to `config/breakers.yaml` (grouped with reliability settings)

```yaml
# src/DocsToKG/ContentDownload/config/breakers.yaml (append at end)

idempotency:
  enabled: false                           # Set via DOCSTOKG_ENABLE_IDEMPOTENCY
  lease_ttl_s: 120                        # Worker lease time-to-live
  stale_file_age_s: 3600                  # Clean .part files older than this
  reconciliation_interval_s: 300           # Run cleanup every N seconds (optional)
  db_path: null                           # Auto-set to telemetry DB (leave null)
```

**Checklist**:

- [ ] Add idempotency section to breakers.yaml or new file
- [ ] Document all settings
- [ ] Add env var overrides (DOCSTOKG_IDEMPOTENCY_*)
- [ ] Load config in runner.py
- [ ] Validate configuration on startup

---

## PHASE 10: OPERATIONS (1 week, HIGH PRIORITY)

### 10.1 CLI Status Commands

**Status**: ❌ NOT STARTED

**Location**: `src/DocsToKG/ContentDownload/cli_breakers.py` (extend) or new `cli_jobs.py`

```python
# Add new command: docstokg job status

def cmd_job_status(args):
    """Show job statistics and status."""
    conn = get_telemetry_conn()

    # Query artifact_jobs for state distribution
    rows = conn.execute("""
        SELECT state, COUNT(*) as count FROM artifact_jobs
        WHERE created_at > ? GROUP BY state
    """, (time.time() - args.window_s,)).fetchall()

    # Print summary
    for state, count in rows:
        print(f"  {state:20s}: {count:6d}")

def cmd_job_leases(args):
    """Show currently leased jobs."""
    conn = get_telemetry_conn()

    rows = conn.execute("""
        SELECT job_id, lease_owner, lease_until FROM artifact_jobs
        WHERE lease_owner IS NOT NULL
        ORDER BY lease_until DESC
    """).fetchall()

    now = time.time()
    for job_id, owner, until in rows:
        remaining_s = max(0, until - now)
        print(f"  {owner:20s} → {job_id[:8]}... ({remaining_s:3.0f}s)")
```

**Checklist**:

- [ ] Create `job status` subcommand
- [ ] Create `job leases` subcommand
- [ ] Create `job stats` subcommand (summary)
- [ ] Add `--window-s` flag to limit query window
- [ ] Add to main CLI parser
- [ ] Test all commands with real data
- [ ] Add examples to AGENTS.md

---

### 10.2 AGENTS.md Documentation

**Status**: ❌ NOT STARTED

**Location**: `src/DocsToKG/ContentDownload/AGENTS.md`

Add new section after "Circuit Breaker Operations":

```markdown
## Idempotency & Job Management

### Overview

The idempotency system ensures exactly-once execution of downloads, finalization,
indexing, and deduplication across crashes and retries. It tracks artifact jobs
through a state machine and stores operation results for replay.

### Configuration

```yaml
idempotency:
  enabled: false                # Set DOCSTOKG_ENABLE_IDEMPOTENCY=true to enable
  lease_ttl_s: 120             # Worker lease time-to-live
  stale_file_age_s: 3600       # Clean .part files older than 1 hour
```

### CLI Commands

# Show job statistics

python -m DocsToKG.ContentDownload.cli job status

# Show currently leased jobs

python -m DocsToKG.ContentDownload.cli job leases

# Clear stale leases (manual recovery)

python -m DocsToKG.ContentDownload.cli job cleanup

### Operational Playbooks

[Include similar format to "Scenario 1: Host Repeatedly Opens" from Circuit Breaker]

### Best Practices

1. Enable for production runs (DOCSTOKG_ENABLE_IDEMPOTENCY=true)
2. Monitor job status metrics
3. Run periodic cleanup (automatic on startup)
4. Check for abandoned operations if timeouts occur

```

**Checklist**:
- [ ] Add "Idempotency & Job Management" section to AGENTS.md
- [ ] Document configuration options
- [ ] Add CLI command examples
- [ ] Add 3-5 operational playbooks (e.g., "Job stuck in LEASED state")
- [ ] Add best practices
- [ ] Add troubleshooting section
- [ ] Link to DATA_MODEL_IDEMPOTENCY.md

---

### 10.3 Operational Runbooks

**Status**: ❌ NOT STARTED

**Scenarios to document** (add to AGENTS.md):

1. **Scenario 1: Jobs Stuck in LEASED State**
   - Symptom: `job status` shows LEASED jobs from hours ago
   - Cause: Worker crashed without releasing
   - Resolution: Run `job cleanup` or wait for TTL expiry

2. **Scenario 2: Duplicated Downloads**
   - Symptom: Same artifact downloaded twice
   - Cause: New job planned before replanning existing (should not happen)
   - Resolution: Check idempotency key generation

3. **Scenario 3: Stale .part Files Accumulating**
   - Symptom: Many `*.part` files in staging after crashes
   - Cause: Normal - reconciler cleans them on startup
   - Resolution: Manual cleanup: `find .staging -name "*.part" -mtime +1 -delete`

4. **Scenario 4: High Job Failure Rate**
   - Symptom: Many jobs in FAILED state
   - Cause: Network issues, disk full, permissions
   - Resolution: Check logs, fix underlying issue, jobs auto-retry

5. **Scenario 5: Single Machine, High Worker Count (50+)**
   - Symptom: SQLite "database is locked" errors
   - Cause: Too many concurrent writers
   - Resolution: Reduce workers, enable connection pooling (Phase 1b)

**Checklist**:
- [ ] Document all 5 scenarios
- [ ] Add diagnostic queries for each
- [ ] Add resolution steps
- [ ] Add prevention tips
- [ ] Test each scenario manually

---

## PHASE 1b: SCALE (2 weeks, MEDIUM PRIORITY)

### 1b.1 Load Testing

**Status**: ❌ NOT STARTED

```python
# tests/content_download/test_idempotency_scale.py

def test_10k_concurrent_jobs():
    """Stress test with 10K jobs."""
    # Create 10K jobs
    # Lease them all
    # Verify no conflicts
    # Measure performance
```

**Checklist**:

- [ ] Create `test_idempotency_scale.py`
- [ ] Test 1K concurrent jobs
- [ ] Test 10K concurrent jobs
- [ ] Measure lease time, query time
- [ ] Identify bottlenecks
- [ ] Measure database file size growth
- [ ] Test database vacuum/cleanup

---

### 1b.2 Connection Pooling

**Status**: ❌ NOT STARTED

```python
# src/DocsToKG/ContentDownload/job_db.py (new)

class ConnectionPool:
    """Thread-safe SQLite connection pool."""
    def __init__(self, db_path, pool_size=5):
        self.db_path = db_path
        self.pool_size = pool_size
        self.connections = []
        # Initialize pool
```

**Checklist**:

- [ ] Implement thread-safe connection pool
- [ ] Wire into runner.py
- [ ] Load test with pool
- [ ] Measure improvement in contention

---

### 1b.3 Redis Support

**Status**: ❌ NOT STARTED

**Note**: See `redis_cooldown_store.py` from Circuit Breaker phase (Phase 1b there)

Can reuse same pattern for job leasing across machines.

**Checklist**:

- [ ] Implement RedisCooldownStore-based job leasing
- [ ] Support distributed workers
- [ ] Test multi-machine scenario
- [ ] Document deployment

---

## Critical Path (Minimum for Production)

To get idempotency into PRODUCTION with minimum effort:

**Week 1 (PHASE 9):**

1. Feature gate in runner.py (2 hours)
2. Download loop integration (4 hours)
3. Configuration (1 hour)
4. Testing (3 hours)

**Week 2 (PHASE 10):**

1. CLI status command (2 hours)
2. AGENTS.md documentation (2 hours)
3. Operational runbooks (2 hours)
4. Manual testing & fixes (2 hours)

**Total: ~20 hours (~1 week if focused)**

---

## Testing Checklist

### Unit Tests

- [x] 22/22 unit tests passing

### Integration Tests (NEW - to create)

- [ ] Test download.py with idempotency enabled
- [ ] Test download.py with idempotency disabled (backward compat)
- [ ] Test crash recovery scenario
- [ ] Test multi-worker scenario
- [ ] Test feature gate enable/disable

### Manual Testing

- [ ] Single download with idempotency enabled
- [ ] Repeated download (verify cached result)
- [ ] Simulate crash mid-download (restart and verify recovery)
- [ ] Verify `.part` files cleaned up
- [ ] Verify stale leases expire
- [ ] Monitor SQLite database size growth
- [ ] Test with 10+ workers
- [ ] Verify no race conditions

---

## Deployment Checklist

- [ ] Feature gate defaults to OFF (backward compatible)
- [ ] DOCSTOKG_ENABLE_IDEMPOTENCY env var documented
- [ ] Configuration section added to breakers.yaml
- [ ] Migration handles existing databases gracefully
- [ ] Cleanup runs automatically on startup
- [ ] CLI commands work with/without idempotency
- [ ] Tests pass with idempotency ON and OFF
- [ ] Documentation complete in AGENTS.md
- [ ] Operational runbooks reviewed by ops team
- [ ] Load testing completed (at least 1K concurrent jobs)
- [ ] Monitoring queries ready for dashboards

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Database lock on high concurrency | Medium | High | Load test, connection pooling (Phase 1b) |
| Network FS + SQLite issues | Low | Critical | Warn in docs, use local DB only |
| Feature gate bugs | Low | Medium | Test both ON and OFF paths |
| Configuration errors | Low | Medium | Validate config on startup |
| Backward compatibility breaks | Very low | High | Keep idempotency disabled by default |

---

## Success Criteria

✅ Phase 9 Integration:

- [ ] Feature gate working (enable/disable)
- [ ] Download loop using jobs + state machine
- [ ] All integration tests passing
- [ ] No errors with large job counts (1K+)

✅ Phase 10 Operations:

- [ ] CLI commands useful for ops
- [ ] AGENTS.md comprehensive
- [ ] Operational runbooks tested
- [ ] All manual tests passing

✅ Production Deployment:

- [ ] Idempotency disabled by default (safe)
- [ ] Can be enabled with single env var
- [ ] Zero breaking changes
- [ ] Monitoring and observability ready

---

## Timeline Estimate

| Phase | Duration | Effort | Priority |
|-------|----------|--------|----------|
| Phase 9 (Integration) | 1 week | 20 hours | HIGH |
| Phase 10 (Operations) | 4 days | 10 hours | HIGH |
| Phase 1b (Scale) | 2 weeks | 30 hours | MEDIUM |
| **Total to Production** | **9 days** | **30 hours** | **HIGH** |
| **Total with Scale** | **4 weeks** | **60 hours** | - |

---

## Success: FULL PRODUCTION DEPLOYMENT

Once all items in this checklist are completed:

✅ Idempotency system fully integrated
✅ Jobs tracked through state machine
✅ Exactly-once guarantees for all downloads
✅ Crash recovery automatic
✅ Multi-worker coordination safe
✅ Production-scale ready (1K+ workers)
✅ Operational visibility complete
✅ Zero breaking changes

The download pipeline will be **crash-resilient**, **idempotent**, and **production-ready**.
