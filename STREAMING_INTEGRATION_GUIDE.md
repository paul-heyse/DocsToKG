# Streaming Integration Guide

**Version**: 1.0
**Date**: October 21, 2025
**Status**: ✅ Production-Ready

---

## Overview

The `streaming_integration` module provides an optional, backward-compatible integration layer that allows `download.py` to incrementally adopt RFC-compliant streaming primitives from `streaming.py`, `idempotency.py`, and `streaming_schema.py`.

### Key Features

- ✅ **Zero-Breaking Changes**: Existing code continues to work without modification
- ✅ **Graceful Fallback**: All functions return `None` or `False` if modules unavailable
- ✅ **Feature Flags**: Control adoption via environment variables
- ✅ **Transparent Integration**: Minimal code changes required in download.py
- ✅ **Performance**: No overhead when features disabled

---

## Architecture

```
download.py (existing orchestration layer)
    ↓
streaming_integration.py (optional adapter)
    ↓
streaming.py (RFC 7232/7233 primitives) ← NEW
idempotency.py (deterministic keys) ← NEW
streaming_schema.py (database layer) ← NEW
```

### Integration Points

| Function | Replaces | Benefit |
|----------|----------|---------|
| `use_streaming_for_resume()` | Manual resume checks | RFC-compliant decision logic |
| `try_streaming_resume_decision()` | `stream_candidate_payload()` resume logic | Deterministic, tested |
| `use_streaming_for_io()` | Manual I/O | Rolling SHA-256, resume support |
| `try_streaming_io()` | Manual streaming/hashing | RFC 7233 partial content |
| `use_streaming_for_finalization()` | Manual finalization | Atomic, content-addressed |
| `generate_job_key()` | Manual tracking | Exactly-once semantics |
| `generate_operation_key()` | Manual operation tracking | Crash recovery |

---

## Usage Patterns

### Pattern 1: Resume Decision Integration

**Before** (download.py, existing):

```python
def stream_candidate_payload(plan: DownloadPreflightPlan) -> DownloadStreamResult:
    ...
    while True:
        headers = dict(base_headers)
        if attempt_conditional:
            headers.update(cond_helper.build_headers())  # Manual ETag/Last-Modified

        response = request_with_retries(client, "GET", url, ...)

        if response.status_code == 304:
            # Manual 304 handling
            cached = cond_helper.interpret_response(response)
            ...
        elif response.status_code == 206:
            # Manual 206 handling
            ...
```

**After** (download.py, enhanced):

```python
from DocsToKG.ContentDownload.streaming_integration import (
    try_streaming_resume_decision,
)

def stream_candidate_payload(plan: DownloadPreflightPlan) -> DownloadStreamResult:
    ...
    # TRY: Use new streaming primitives if available
    decision = try_streaming_resume_decision(
        validators={"etag": etag_from_response, "last_modified": lm_from_response},
        part_state=existing_part_state,  # Or None for fresh download
        prefix_check_bytes=ctx.sniff_limit,
        client=client,
        url=url,
    )

    if decision is not None:
        # USE: New logic
        if decision.mode == "fresh":
            # Download fresh copy
            response = request_with_retries(client, "GET", url, ...)
        elif decision.mode == "resume":
            # Resume from byte offset
            headers["Range"] = f"bytes={decision.resume_byte_offset}-"
            response = request_with_retries(client, "GET", url, headers=headers, ...)
        elif decision.mode == "cached":
            # Reuse cached file
            return DownloadStreamResult(outcome=decision.outcome)
    else:
        # FALLBACK: Existing logic
        if attempt_conditional:
            headers.update(cond_helper.build_headers())
        response = request_with_retries(client, "GET", url, ...)
```

### Pattern 2: I/O Integration

**Before** (download.py, existing):

```python
def stream_candidate_payload(plan: DownloadPreflightPlan) -> DownloadStreamResult:
    ...
    sha256 = hashlib.sha256()
    bytes_written = 0
    with response:
        for chunk in response.iter_bytes(chunk_size=8192):
            if not chunk:
                break
            sha256.update(chunk)
            staging_path.write_bytes(chunk, append=True)
            bytes_written += len(chunk)

            if progress_callback:
                progress_callback(bytes_written)
```

**After** (download.py, enhanced):

```python
from DocsToKG.ContentDownload.streaming_integration import (
    try_streaming_io,
)

def stream_candidate_payload(plan: DownloadPreflightPlan) -> DownloadStreamResult:
    ...
    # TRY: Use new streaming primitives if available
    metrics = try_streaming_io(
        response,
        staging_path,
        chunk_bytes=ctx.sniff_limit,
        fsync=ctx.verify_cache_digest,
        progress_callback=progress_callback,
    )

    if metrics is not None:
        # USE: New metrics
        bytes_written = metrics.bytes_written
        sha256_hex = metrics.sha256_hex
    else:
        # FALLBACK: Existing logic
        sha256 = hashlib.sha256()
        bytes_written = 0
        for chunk in response.iter_bytes(chunk_size=8192):
            if not chunk:
                break
            sha256.update(chunk)
            staging_path.write_bytes(chunk, append=True)
            bytes_written += len(chunk)
```

### Pattern 3: Idempotency Integration

**Before** (no idempotency):

```python
def download_candidate(artifact: WorkArtifact, ...):
    # If crash occurs mid-download, entire job retried
    # May result in duplicate downloads
    result = stream_candidate_payload(plan)
    finalize_candidate_download(result)
```

**After** (with idempotency):

```python
from DocsToKG.ContentDownload.streaming_integration import (
    generate_job_key,
    generate_operation_key,
)

def download_candidate(artifact: WorkArtifact, ...):
    # Generate deterministic keys
    job_key = generate_job_key(
        work_id=artifact.work_id,
        artifact_id=artifact.artifact_id,
        canonical_url=canonical_url,
    )

    if job_key:
        # Store job_key in database for exactly-once semantics
        with get_streaming_database() as db:
            job_id = idempotency.acquire_or_reuse_job(db, job_key, worker_id)

            # Stream with operation tracking
            op_key = generate_operation_key("STREAM", job_id, url=url)
            result = idempotency.run_effect(db, op_key,
                lambda: stream_candidate_payload(plan))

            # Finalize with operation tracking
            op_key = generate_operation_key("FINALIZE", job_id)
            idempotency.run_effect(db, op_key,
                lambda: finalize_candidate_download(result))
    else:
        # Fallback to existing logic
        result = stream_candidate_payload(plan)
        finalize_candidate_download(result)
```

### Pattern 4: Schema Integration

**Before** (no persistent state):

```python
def runner_main():
    # Resume only from in-memory manifests
    run = DownloadRun(config)
    run.run()  # If crash, all resume state lost
```

**After** (with persistent state):

```python
from DocsToKG.ContentDownload.streaming_integration import (
    get_streaming_database,
    check_database_health,
)

def runner_main():
    # Check database health
    health = check_database_health()
    if health and health["status"] != "healthy":
        LOGGER.warning(f"Database needs repair: {health}")
        # Auto-repair if necessary

    # Use persistent state
    with get_streaming_database() as db:
        if db:
            # Use db for job coordination, recovery, etc.
            run = DownloadRun(config, db=db)
            run.run()
        else:
            # Fallback to existing logic
            run = DownloadRun(config)
            run.run()
```

---

## Feature Flags

### Environment Variables

```bash
# Disable all streaming features
export DOCSTOKG_ENABLE_STREAMING=0

# Disable idempotency
export DOCSTOKG_ENABLE_IDEMPOTENCY=0

# Disable schema
export DOCSTOKG_ENABLE_STREAMING_SCHEMA=0
```

### Programmatic Control

```python
from DocsToKG.ContentDownload.streaming_integration import (
    streaming_enabled,
    idempotency_enabled,
    schema_enabled,
    integration_status,
    log_integration_status,
)

# Check individual features
if streaming_enabled():
    print("Streaming available")

if idempotency_enabled():
    print("Idempotency available")

# Get all statuses
status = integration_status()
print(f"Features available: {status}")

# Log status
log_integration_status()
```

---

## Incremental Adoption Strategy

### Phase 1: Resume Decisions (Low Risk, High Value)

**Time**: ~1 day
**Risk**: Low - New logic only used when available
**Value**: RFC-compliant resume decisions, fewer duplicate downloads

1. Add `try_streaming_resume_decision()` calls to `stream_candidate_payload()`
2. Keep existing logic as fallback
3. Monitor error logs for exceptions
4. Compare cache hit rates

### Phase 2: I/O Integration (Medium Risk, Medium Value)

**Time**: ~1-2 days
**Risk**: Medium - Affects all downloads
**Value**: Atomic streaming, better performance metrics

1. Add `try_streaming_io()` calls to `stream_candidate_payload()`
2. Keep existing I/O as fallback
3. Compare performance metrics (latency, memory usage)
4. Monitor for SHA-256 mismatches

### Phase 3: Finalization (Low Risk, Low Value)

**Time**: ~0.5 day
**Risk**: Low - Only on successful downloads
**Value**: Atomic file operations, less chance of partial files

1. Add `use_streaming_for_finalization()` checks
2. Use `streaming.finalize_artifact()` when available
3. Monitor for file corruption issues

### Phase 4: Idempotency (High Risk, High Value)

**Time**: ~3-5 days
**Risk**: High - Requires database coordination
**Value**: Exactly-once download semantics, crash recovery

1. Set up `streaming_schema.StreamingDatabase`
2. Integrate `generate_job_key()` / `generate_operation_key()`
3. Coordinate job leases across workers
4. Test crash recovery scenarios

### Phase 5: Full Integration (Production Ready)

**Time**: After phases 1-4 complete
**Risk**: Low - All features individually tested
**Value**: Complete streaming architecture in production

1. Enable all features by default
2. Monitor production metrics
3. Gradually disable fallback logic
4. Document as production standard

---

## Rollback Strategy

If issues arise at any point:

```python
# Option 1: Disable feature via environment variable
export DOCSTOKG_ENABLE_STREAMING=0
python -m DocsToKG.ContentDownload.cli --resume-from manifest.jsonl

# Option 2: Re-run with old code (existing download.py logic always available)
# No code changes required, just disable the feature flag

# Option 3: Migrate back to old resume manifests
# All existing JSONL manifests continue to work
python -m DocsToKG.ContentDownload.cli --warm-manifest-cache manifest.jsonl
```

---

## Testing Strategy

### Unit Tests

```python
# Test the integration layer itself
pytest tests/content_download/test_streaming_integration.py -v
```

### Integration Tests

```python
# Test download.py with streaming enabled
export DOCSTOKG_ENABLE_STREAMING=1
pytest tests/content_download/test_download_execution.py -v

# Test download.py with streaming disabled (fallback)
export DOCSTOKG_ENABLE_STREAMING=0
pytest tests/content_download/test_download_execution.py -v

# Compare results - should be identical
```

### Performance Tests

```python
# Benchmark new vs old I/O
./.venv/bin/pytest tests/content_download/test_streaming.py::test_streaming_performance -v

# Benchmark new vs old resume decisions
# (create comparative benchmark)
```

### End-to-End Tests

```python
# Test full pipeline with streaming
export DOCSTOKG_ENABLE_STREAMING=1
python -m DocsToKG.ContentDownload.cli \
    --topic "machine learning" \
    --year-start 2024 \
    --year-end 2024 \
    --max 100 \
    --out runs/streaming_test \
    --workers 4

# Compare against baseline
export DOCSTOKG_ENABLE_STREAMING=0
python -m DocsToKG.ContentDownload.cli \
    --topic "machine learning" \
    --year-start 2024 \
    --year-end 2024 \
    --max 100 \
    --out runs/fallback_test \
    --workers 4

# Compare metrics
jq '.statistics' runs/streaming_test/manifest.metrics.json
jq '.statistics' runs/fallback_test/manifest.metrics.json
```

---

## Monitoring and Observability

### Logs

```bash
# Check integration status
grep "Streaming architecture integrations" logs/download.log

# Check resume decision usage
grep "streaming_resume_decision" logs/download.log

# Check fallbacks
grep "falling back" logs/download.log
```

### Metrics

```python
# In manifest.metrics.json
{
    "streaming": {
        "resume_decisions_made": 1234,
        "io_operations": 5678,
        "fallbacks_to_existing": 12,
        "average_latency_ms": 45.6
    }
}
```

### CLI

```bash
# Check streaming availability
python -m DocsToKG.ContentDownload.cli --streaming-status

# Output:
# Streaming: enabled
# Idempotency: enabled
# Schema: enabled
```

---

## FAQ

### Q: Do I need to modify download.py?

**A**: Not at all! The integration is completely optional. Existing code continues to work unchanged. You can optionally add try/except blocks around streaming calls to use new primitives.

### Q: What if streaming module is not available?

**A**: All functions gracefully return `None` or `False`. Existing logic continues to work.

### Q: Can I enable features selectively?

**A**: Yes! Use environment variables:

```bash
export DOCSTOKG_ENABLE_STREAMING=1
export DOCSTOKG_ENABLE_IDEMPOTENCY=0  # Still using old resume tracking
export DOCSTOKG_ENABLE_STREAMING_SCHEMA=0  # Still using in-memory state
```

### Q: How do I measure impact?

**A**: Compare metrics before/after:

```bash
# Before
export DOCSTOKG_ENABLE_STREAMING=0
python -m DocsToKG.ContentDownload.cli --max 1000 --out runs/before

# After
export DOCSTOKG_ENABLE_STREAMING=1
python -m DocsToKG.ContentDownload.cli --max 1000 --out runs/after

# Compare
diff runs/before/manifest.metrics.json runs/after/manifest.metrics.json
```

### Q: Is there performance overhead?

**A**: No. When features are disabled, the integration layer adds zero overhead. When features are enabled, there's negligible overhead from the adapter layer.

### Q: Can I run this in production?

**A**: Yes! The integration layer is production-ready. However, it's recommended to:

1. Start with phase 1 (resume decisions only)
2. Monitor for 1-2 weeks
3. Roll out remaining phases gradually
4. Keep rollback mechanism ready

---

## References

- `streaming_integration.py`: Integration adapter (330+ LOC)
- `streaming.py`: RFC 7232/7233 primitives (670 LOC)
- `idempotency.py`: Deterministic keys (481 LOC)
- `streaming_schema.py`: Database layer (432 LOC)
- `test_streaming_integration.py`: Comprehensive tests (400+ LOC)

---

## Support

For issues or questions:

1. Check logs: `grep "streaming" logs/download.log`
2. Verify features enabled: `python -c "from DocsToKG.ContentDownload.streaming_integration import integration_status; print(integration_status())"`
3. Test with fallback: `export DOCSTOKG_ENABLE_STREAMING=0 && pytest ...`
4. Report issue with feature status and logs
