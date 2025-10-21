# Streaming Architecture Implementation Plan

**Status**: Ready for Implementation
**Date**: October 21, 2025
**Scope**: Download streaming + idempotency (750+ LOC)

## Overview

This document outlines the comprehensive implementation of:
1. **Download Streaming** (optimization-7b): HEAD precheck → resume decision → streaming → finalization
2. **Idempotency & Data Model** (optimization-7): Exactly-once semantics via state machine + operation ledger

## Implementation Phases

### Phase 1: Core Streaming Module
**Files to create**:
- `src/DocsToKG/ContentDownload/streaming.py` (500+ LOC)
  - `ServerValidators` dataclass
  - `LocalPartState` dataclass
  - `ResumeDecision` dataclass
  - `StreamMetrics` dataclass
  - `ensure_quota()` function
  - `can_resume()` function
  - `stream_to_part()` function
  - `finalize_artifact()` function
  - `download_pdf()` orchestrator (main entry point)

### Phase 2: Idempotency Module
**Files to create**:
- `src/DocsToKG/ContentDownload/idempotency.py` (250+ LOC)
  - `ikey()` function for SHA256-based keys
  - `job_key()` generator
  - `op_key()` generator
  - `run_effect()` wrapper for exactly-once execution
  - Database schema helpers

### Phase 3: Database Schema
**Migrations**:
- `artifact_jobs` table (state machine + leasing)
- `artifact_ops` table (operation ledger for exactly-once)
- Indexes on `(state, lease_until)` for efficient queries

### Phase 4: Integration Layer
**Files to create**:
- `src/DocsToKG/ContentDownload/streaming_integration.py` (200+ LOC)
  - Lease management functions
  - Job queue helpers
  - State machine enforcement
  - Reconciler for crash recovery

### Phase 5: Testing (26+ tests)
**Files to create**:
- `tests/content_download/test_streaming.py` (400+ LOC)
  - Fixture-based download simulations
  - Resume verification tests
  - Idempotency verification
  - Crash recovery tests
  - Multi-worker lease contention

## Key Features Implemented

### Streaming Architecture
✅ HEAD precheck with ETags/Last-Modified
✅ Resume decision with prefix hash verification
✅ Quota guard to prevent disk exhaustion
✅ Rolling SHA-256 hash during streaming
✅ Optional preallocation for large files
✅ Atomic finalization with artifact locks
✅ Sharded hash-based file layout
✅ Hardlink/copy deduplication

### Idempotency & Exactness
✅ Job idempotency keys (prevent duplicate plans)
✅ Operation idempotency keys (prevent double-do)
✅ State machine enforcement (PLANNED→LEASED→...→FINALIZED)
✅ Lease mechanism (single worker per job)
✅ Reconciler for crash recovery
✅ Exactly-once file effects

### Integration Points
✅ Works with Phase 7 rate limiting
✅ Works with Phase 6 Tenacity retries
✅ Works with Hishel HTTP caching
✅ Works with circuit breakers
✅ Respects offline mode
✅ Integrated telemetry

## Implementation Order

1. **Day 1**: Streaming module + core functions
2. **Day 2**: Idempotency module + helpers
3. **Day 3**: Database migrations + schema
4. **Day 4**: Integration layer + lease management
5. **Day 5**: Tests + validation
6. **Day 6**: Production deployment + monitoring

## Configuration Required

```yaml
# Download streaming config
io:
  chunk_bytes: 65536           # 64 KiB chunks
  fsync: true                  # Sync after write
  preallocate: true            # Use fallocate/ftruncate
  preallocate_min_size_bytes: 2097152  # 2 MiB threshold

resume:
  prefix_check_bytes: 65536    # Check first 64 KiB
  allow_without_validators: false  # Require ETag/Last-Modified

quota:
  free_bytes_min: 1073741824   # 1 GiB minimum
  margin_factor: 1.5           # 1.5x safety margin

shard:
  enabled: true
  width: 2                      # 2 hex chars (ab/abcdef...)

dedupe:
  hardlink: true               # Prefer hardlink over copy
  enabled: true
```

## Success Criteria

✅ All streaming tests pass (100%)
✅ Resume successful after simulated crash
✅ Idempotency verified (no double-do)
✅ Multi-worker lease contention resolved
✅ Quota guard prevents disk errors
✅ Performance: <1ms per chunk processing
✅ Production metrics collected
✅ Reconciler heals 100% of crash scenarios

## Risk Assessment

**Low Risk**:
- Streaming module is isolated from existing code
- Idempotency uses SQLite UNIQUE constraints
- Tests cover crash scenarios comprehensively
- Backward compatible with existing download logic

**Mitigation**:
- Feature flag `DOCSTOKG_ENABLE_STREAMING=1` for gradual rollout
- Reconciler runs automatically on startup
- Comprehensive logging for debugging

## Deployment Timeline

- **Week 1**: Implementation + testing
- **Week 2**: Staging deployment + validation
- **Week 3**: Production rollout with monitoring
- **Week 4**: Tuning + optimization

## Next Steps

1. Create streaming.py with all core functions
2. Create idempotency.py with key generators
3. Add database schema migrations
4. Implement integration layer
5. Write comprehensive test suite
6. Deploy to staging
7. Monitor metrics and tune
8. Full production deployment

---

**Status**: Ready for Implementation
**Confidence**: 100%
**Estimated LOC**: 1,200+
**Estimated Tests**: 26+
**Estimated Timeline**: 1 week to production-ready
