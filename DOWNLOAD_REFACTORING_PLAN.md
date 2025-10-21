# Download.py Refactoring Plan - Streaming Integration

**Status**: Phase 1 Implementation Starting
**Target**: Incremental adoption of RFC-compliant streaming primitives
**Risk Level**: LOW (Phase 1) → HIGH (Phase 4)

---

## Strategic Overview

This document outlines the incremental refactoring of `download.py` to optionally use the new streaming architecture from `streaming_integration.py`. The refactoring is designed to be:

1. **Non-Breaking**: All changes are additive or conditional
2. **Reversible**: Feature flags allow instant rollback
3. **Measurable**: Each phase has clear success metrics
4. **Production-Ready**: Each phase can be deployed independently

---

## Phase 1: Resume Decision Integration (LOW RISK - Recommended NOW)

### Scope

Integrate `try_streaming_resume_decision()` into `stream_candidate_payload()` to provide RFC 7232-compliant conditional request handling.

### Changes

**File**: `src/DocsToKG/ContentDownload/download.py`

**Location**: Top of file (imports section)

```python
# Add import
from DocsToKG.ContentDownload.streaming_integration import (
    try_streaming_resume_decision,
    streaming_enabled,
)
```

**Location**: In `stream_candidate_payload()`, around line 626 (resume decision point)

Before (manual resume logic):

```python
if attempt_conditional:
    headers.update(cond_helper.build_headers())
```

After (with optional streaming integration):

```python
# TRY: Use RFC-compliant streaming resume logic if available
if streaming_enabled() and use_streaming_for_resume(plan):
    resume_decision = try_streaming_resume_decision(
        validators={
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
        },
        part_state=None,  # Would be actual part state if resuming
        prefix_check_bytes=sniff_limit,
        client=client,
        url=url,
    )

    if resume_decision is not None:
        # Log decision for observability
        LOGGER.debug(
            "streaming_resume_decision",
            extra={
                "extra_fields": {
                    "mode": resume_decision.mode,
                    "reason": resume_decision.reason,
                    "url": url,
                }
            },
        )
    else:
        # Fallback to existing logic
        if attempt_conditional:
            headers.update(cond_helper.build_headers())
else:
    # Fallback to existing logic
    if attempt_conditional:
        headers.update(cond_helper.build_headers())
```

### Success Metrics

- ✅ Resume decision log appears for > 50% of conditional requests
- ✅ No degradation in cache hit rate
- ✅ No 304 Not Modified responses become errors
- ✅ Error logs show zero failed fallbacks

### Testing

```bash
# Run with streaming enabled
export DOCSTOKG_ENABLE_STREAMING=1
pytest tests/content_download/test_download_execution.py -v -k "conditional"

# Run with streaming disabled (fallback)
export DOCSTOKG_ENABLE_STREAMING=0
pytest tests/content_download/test_download_execution.py -v -k "conditional"

# Compare results
```

### Rollback

```bash
export DOCSTOKG_ENABLE_STREAMING=0
# Existing logic automatically used
```

---

## Phase 2: I/O Integration (MEDIUM RISK - 1-2 Weeks Later)

### Scope

Integrate `try_streaming_io()` to replace manual streaming/hashing with RFC 7233 partial content handling.

### Key Changes

- Replace manual `hashlib.sha256()` with streaming metrics
- Wrap response iteration with `try_streaming_io()`
- Use `StreamMetrics` for performance tracking
- Log I/O metrics for observability

### Location Changes

- Around line 963-1000 (manual hasher initialization)
- Streaming loop (lines 989-1015)

### Success Metrics

- Performance improvement > 5%
- SHA-256 mismatches: 0%
- Memory usage reduction > 10%

---

## Phase 3: Finalization Integration (LOW RISK - 1 Week Later)

### Scope

Integrate `use_streaming_for_finalization()` for atomic file operations.

### Changes

- Check `use_streaming_for_finalization(outcome)` before file promotion
- Use `streaming.finalize_artifact()` when available
- Add telemetry for finalization success/failure

### Location

- Around line 1380+ (`finalize_candidate_download()`)

### Success Metrics

- Zero partial files found on crash
- Atomic promotion success: 100%

---

## Phase 4: Idempotency Integration (HIGH RISK - 2-4 Weeks Later)

### Scope

Integrate `generate_job_key()` and `generate_operation_key()` for exactly-once semantics.

### Changes

- Generate deterministic job keys for each download
- Store job state in `StreamingDatabase`
- Track operations for crash recovery
- Coordinate lease management across workers

### Location

- Around line 2422+ (`download_candidate()`)

### Success Metrics

- Duplicate downloads: 0%
- Worker coordination: 100%
- Crash recovery: automatic

---

## Phase 5: Full Production Deployment (LOW RISK - 1 Week Later)

### Scope

Enable all features by default and remove fallback paths (optional).

### Changes

- Set `DOCSTOKG_ENABLE_STREAMING=1` by default
- Document as production standard
- Optional: Remove fallback code after 1-2 weeks of stability

### Success Metrics

- All metrics green across all phases
- No rollback required

---

## Implementation Timeline

| Phase | Start | Duration | Risk | Status |
|-------|-------|----------|------|--------|
| 1 | NOW | 1 day | Low | 🟡 Implementation |
| 2 | +1 week | 2 days | Med | ⚪ Planned |
| 3 | +1.5 weeks | 1 day | Low | ⚪ Planned |
| 4 | +2.5 weeks | 3 days | High | ⚪ Planned |
| 5 | +3.5 weeks | 1 day | Low | ⚪ Planned |

**Total Timeline**: 5-6 weeks to full production deployment

---

## Code Change Summary

### Phase 1 Code Changes (Minimal)

```
Files Modified: 1
  - src/DocsToKG/ContentDownload/download.py

Lines Added: ~20
  - 2 imports
  - 18 conditional logic lines

Lines Modified: ~5
  - Conditional resume decision handling

Net Change: ~25 lines (0.6% of file)

Backward Compatibility: 100% (feature flag toggles)
```

### All Phases Combined

```
Files Modified: 1 (download.py)
Lines Added: ~100-150
Lines Modified: ~30-50
Net Change: ~150-200 lines (4-5% of file)

Backward Compatibility: 100%
```

---

## Feature Flag Control

### Environment Variables

```bash
# Phase 1
export DOCSTOKG_ENABLE_STREAMING=1|0

# Phase 4
export DOCSTOKG_ENABLE_IDEMPOTENCY=1|0
export DOCSTOKG_ENABLE_STREAMING_SCHEMA=1|0

# All phases
export DOCSTOKG_ENABLE_STREAMING_INTEGRATION=1|0
```

### CLI Flag (Future)

```bash
python -m DocsToKG.ContentDownload.cli --enable-streaming-integration
python -m DocsToKG.ContentDownload.cli --streaming-ratio 0.5  # Canary: 50% of jobs
```

---

## Testing Strategy

### Unit Tests (Already Complete)

- ✅ 24 tests in `test_streaming_integration.py`
- ✅ 26 tests in `test_streaming.py`
- ✅ 17 tests in `test_streaming_schema.py`

### Integration Tests (To Add)

- `test_download_with_streaming_enabled.py` - Phase 1
- `test_download_with_streaming_io.py` - Phase 2
- `test_download_idempotency.py` - Phase 4

### End-to-End Tests

```bash
# Phase 1 E2E
export DOCSTOKG_ENABLE_STREAMING=1
python -m DocsToKG.ContentDownload.cli --max 500 --out runs/phase1_enabled
export DOCSTOKG_ENABLE_STREAMING=0
python -m DocsToKG.ContentDownload.cli --max 500 --out runs/phase1_disabled
# Compare metrics

# Performance regression testing
pytest --benchmark-only tests/content_download/test_download_performance.py
```

---

## Monitoring & Observability

### Metrics to Track (Phase 1)

```python
# In manifest.metrics.json
{
    "streaming": {
        "resume_decisions_made": <int>,
        "resume_decisions_fallback": <int>,
        "resume_decision_errors": <int>,
        "average_decision_latency_ms": <float>,
    }
}
```

### Logs to Monitor

```bash
# Filter for streaming decisions
grep "streaming_resume_decision" logs/download.log

# Filter for fallbacks
grep "falling back" logs/download.log

# Filter for errors
grep -i "error.*streaming" logs/download.log
```

### Health Check CLI

```bash
python -c "
from DocsToKG.ContentDownload.streaming_integration import integration_status
import json
print(json.dumps(integration_status(), indent=2))
"
```

---

## Rollback Procedures

### Instant Rollback (Any Phase)

```bash
# Disable streaming
export DOCSTOKG_ENABLE_STREAMING=0

# Existing logic automatically used
python -m DocsToKG.ContentDownload.cli --resume-from manifest.jsonl
```

### Data Safety

- ✅ All changes are non-destructive
- ✅ Manifests remain compatible
- ✅ No database schema changes
- ✅ Resume from old manifests always works

---

## Success Criteria

### Phase 1

- [ ] Resume decision integration working
- [ ] No 304 Not Modified errors
- [ ] Cache hit rate maintained
- [ ] Zero fallback errors
- [ ] Telemetry showing > 50% adoption

### Phase 2

- [ ] I/O integration working
- [ ] Performance improved > 5%
- [ ] SHA-256 match: 100%
- [ ] Memory usage down > 10%

### Phase 3

- [ ] Finalization atomic
- [ ] Zero partial files
- [ ] Promotion success: 100%

### Phase 4

- [ ] Duplicate downloads: 0%
- [ ] Worker coordination: 100%
- [ ] Crash recovery: automatic

### Phase 5

- [ ] All metrics green
- [ ] Production ready
- [ ] Documentation updated

---

## Next Steps

### Immediate (TODAY)

1. ✅ Create this plan document
2. ⏳ Implement Phase 1 code changes
3. ⏳ Add integration tests
4. ⏳ Deploy to staging
5. ⏳ Monitor metrics

### This Week

6. Analyze Phase 1 metrics
7. Plan Phase 2 if metrics green
8. Document lessons learned

### This Month

9-12. Execute Phases 2-5
13. Production deployment
14. Celebrate! 🎉

---

## References

- Integration Layer: `STREAMING_INTEGRATION_GUIDE.md`
- Architecture Review: `STREAMING_ARCHITECTURE_REVIEW.md`
- Implementation Guide: `STREAMING_INTEGRATION_SUMMARY.md`

---

**Status**: Ready for Phase 1 Implementation
**Owner**: [Your Team]
**Last Updated**: October 21, 2025
