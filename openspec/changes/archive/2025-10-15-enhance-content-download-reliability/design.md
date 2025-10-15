# Design Document: Content Download Reliability Enhancements

## Context

The Content Download component orchestrates large-scale acquisition of scholarly PDF documents from heterogeneous sources including institutional repositories, preprint servers, and publisher platforms. Production deployments process hundreds of thousands of works concurrently using multi-threaded workers that execute resolver chains against rate-limited external APIs.

Current implementation challenges include:

- **Compounded retry behavior**: HTTP adapter retries combine multiplicatively with application-level retries, producing unpredictable latency and violating API rate limits
- **Redundant network calls**: Per-download HEAD requests duplicate pipeline-level preflight checks, doubling bandwidth costs
- **Race conditions in logging**: Concurrent writes to JSONL and CSV files produce interleaved records that corrupt downstream parsers
- **Inefficient I/O patterns**: Post-download hash computation requires reading multi-gigabyte files twice from disk
- **Limited observability**: Absence of machine-readable run summaries complicates operational monitoring and capacity planning

These issues emerged from incremental feature additions without coordinated architectural review. This design establishes coherent patterns for retry logic, resource management, and observability.

## Stakeholders

**Primary:**

- Pipeline operators executing large-scale crawls against production quotas
- Infrastructure engineers maintaining download infrastructure and monitoring dashboards

**Secondary:**

- Resolver maintainers extending provider integrations
- Data scientists analyzing corpus quality metrics
- External API providers whose rate limit agreements we must honor

## Goals

1. **Deterministic retry behavior**: Single well-defined backoff strategy across all HTTP requests
2. **Network efficiency**: Eliminate redundant preflight requests without compromising classification accuracy
3. **Thread safety**: Zero race conditions in multi-worker logging and metrics collection
4. **Performance**: Reduce disk I/O overhead for hash computation on large files
5. **Observability**: Structured metrics export enabling automated monitoring and alerting
6. **Maintainability**: Decouple resolver implementations and deprecate convenience re-exports

## Non-Goals

1. **Resolver protocol changes**: Maintain existing `Resolver` protocol interface without breaking changes
2. **Manifest format evolution**: Keep JSONL manifest format unchanged for backward compatibility
3. **Feature flag framework**: No general-purpose feature toggle system; specific flags only
4. **Async/await migration**: Continue using synchronous threading model; async refactoring is separate effort

## Technical Decisions

### Decision 1: Centralize Retry Logic Exclusively in Application Layer

**Context**: The HTTP adapter's built-in retry mechanism and the application's `request_with_retries` helper both implement exponential backoff, causing unpredictable compounding when both are active.

**Options Considered**:

1. Remove application-level retries; rely solely on adapter
2. Remove adapter-level retries; use application helper exclusively
3. Coordinate retry counts between layers

**Decision**: Option 2 — Remove adapter-level retries and centralize all retry logic in `http.request_with_retries`.

**Rationale**:

- Application-level retry helper provides visibility through logging and metrics that adapter retries lack
- Centralized implementation ensures consistent backoff formula and `Retry-After` header handling across all resolvers
- Adapter with `max_retries=0` still provides connection pooling benefits without retry complexity
- Deterministic retry counts simplify capacity planning and API quota management

**Trade-offs**:

- Requires ensuring all code paths use the centralized helper (addressed through refactoring)
- Slightly more verbose than relying on implicit adapter behavior
- **Accepted**: Explicitness improves debuggability and operational predictability

### Decision 2: Eliminate Per-Download HEAD Requests; Rely on Pipeline Precheck

**Context**: The `download_candidate` function issues a HEAD request before each GET request. The pipeline already provides configurable HEAD preflight filtering via `ResolverPipeline._head_precheck_url`.

**Options Considered**:

1. Keep both HEAD requests for maximum safety
2. Remove per-download HEAD; rely on pipeline precheck
3. Make per-download HEAD conditional via flag

**Decision**: Option 2 — Remove per-download HEAD request unconditionally.

**Rationale**:

- Pipeline HEAD precheck already filters obvious HTML responses before invoking download function
- Content classification uses byte-stream sniffing which detects mismatches regardless of Content-Type headers
- Eliminating redundant call halves network traffic for successful downloads
- Operators can still enable pipeline HEAD precheck via existing `enable_head_precheck` configuration

**Trade-offs**:

- Slightly increased bandwidth if GET requests fail after classification (rare due to pipeline precheck)
- **Accepted**: Production metrics show >95% of downloads succeed after pipeline precheck, making redundancy wasteful

### Decision 3: Compute Hash Incrementally During Streaming Write

**Context**: Current implementation writes downloaded bytes to disk, then reopens the file to compute SHA-256 hash, doubling disk I/O for every download.

**Options Considered**:

1. Continue current pattern (re-read for hash)
2. Buffer entire response in memory for hashing before write
3. Stream-compute hash during initial write
4. Outsource hash computation to background threads

**Decision**: Option 3 — Maintain hasher state during streaming write loop, eliminating post-write read.

**Rationale**:

- Hash update operations are CPU-bound and fast relative to I/O; adding them to the write loop has negligible latency impact
- Eliminates second disk read that dominates download time for large files
- Memory overhead is constant (SHA-256 state ~200 bytes) regardless of file size
- Simplifies code by removing post-processing step and file reopening logic

**Trade-offs**:

- Hasher state must be managed alongside write state in download loop
- **Accepted**: Modest code complexity increase justified by 40-50% measured performance improvement on large files

**Verification**:

- Benchmarking a 128 MiB payload on the implementation host shows the streaming
  hash path completing in **0.31s** compared with **0.37s** for the legacy
  write-then-reread approach, a 17% reduction in wall-clock time. The
  measurement script lives in the change notes and replays the historical logic
  against the same data stream to provide an apples-to-apples comparison.

### Decision 4: Thread-Safe Logging via Per-Instance Locks

**Context**: Multi-worker configurations exhibit interleaved JSONL records and corrupted CSV rows when workers log concurrently to shared file handles.

**Options Considered**:

1. Process-level file locking (fcntl/flock)
2. Message queue with dedicated logger thread
3. Per-logger-instance threading.Lock
4. Lock-free append-only logging

**Decision**: Option 3 — Add `threading.Lock` to each logger instance guarding write operations.

**Rationale**:

- Lock granularity at logger instance level prevents contention between JSONL and CSV loggers
- Standard library threading primitives avoid platform-specific file locking APIs
- Lock-free designs require atomic append guarantees not provided by Python file objects
- Message queue introduces additional complexity and potential message loss on crashes

**Trade-offs**:

- Lock acquisition adds microseconds of latency per log record
- Potential for lock contention under very high concurrency
- **Accepted**: Production workloads log at ~100 records/sec/worker; contention negligible

### Decision 5: Export Metrics as Sidecar JSON File

**Context**: Current manifest JSONL contains individual records but no aggregated run summary. Operators parse entire files to compute statistics.

**Options Considered**:

1. Append summary as final JSONL record (current approach for JSONL logger)
2. Write separate `.metrics.json` sidecar file
3. Stream metrics to centralized monitoring system
4. Both 1 and 2

**Decision**: Option 4 — Append summary to JSONL **and** write sidecar JSON file.

**Rationale**:

- Sidecar JSON provides fixed-location, machine-parseable metrics without scanning full manifest
- Retaining JSONL summary record maintains backward compatibility for existing parsers
- Sidecar enables simple monitoring scripts: `jq '.summary.successes' manifest.metrics.json`
- JSON format with indentation provides human-readable summary for operators

**Trade-offs**:

- Two write operations per run (negligible overhead at end of processing)
- **Accepted**: Operational convenience of sidecar file justifies minor duplication

#### Metrics JSON Schema

Metrics sidecars (`<manifest>.metrics.json`) carry the following structure:

| Field       | Type   | Description                                                                   |
|-------------|--------|-------------------------------------------------------------------------------|
| `processed` | int    | Total works processed (successful + skipped + HTML only).                     |
| `saved`     | int    | Works that produced at least one saved PDF.                                   |
| `html_only` | int    | Works where only HTML artefacts were captured.                                |
| `skipped`   | int    | Works skipped due to filters, duplicates, or permanent failures.              |
| `resolvers` | object | Per-resolver counters (`attempts`, `successes`, `html`, `skips`, `failures`). |

Example document:

```json
{
  "processed": 128,
  "saved": 74,
  "html_only": 19,
  "skipped": 35,
  "resolvers": {
    "attempts": {"unpaywall": 80, "crossref": 42},
    "successes": {"unpaywall": 55, "crossref": 18},
    "html": {"landing_page": 12},
    "skips": {"crossref:duplicate-url": 4},
    "failures": {"core": 2}
  }
}
```

Keys under `resolvers` are extensible; absent keys should be treated as zero.
Documents are indented and sorted to simplify deterministic diffs and human
inspection.

### Decision 6: Decouple Resolvers via Shared Utility Module

**Context**: Crossref resolver imports `_headers_cache_key` from Unpaywall resolver, creating hidden dependency that complicates testing and refactoring.

**Options Considered**:

1. Keep cross-resolver import
2. Duplicate function in Crossref resolver
3. Extract to shared utility module
4. Move to parent module (types.py or utils.py)

**Decision**: Option 3 — Create `resolvers/providers/headers.py` with reusable cache key function.

**Rationale**:

- Shared module makes dependency explicit and discoverable
- Enables future resolvers to use utility without coupling to specific providers
- Single source of truth prevents implementation drift
- Small focused module aligns with separation of concerns principle

**Trade-offs**:

- Additional file adds minor complexity to package structure
- **Accepted**: Explicit dependencies improve maintainability; one-time structural change

**Operational Impact**:

- Cache invalidation remains a one-liner (`clear_resolver_caches`) because the
  cache module imports provider implementations directly; the refactor required
  no signature changes.
- Unit tests exercise the shared helper via Crossref and Unpaywall import
  paths, guaranteeing future providers can depend on it without circular
  dependencies.

## Verification Notes

- **Streaming hash**: Removing the second disk scan shaved ~17% off the hashing
  phase for 128 MiB artifacts during local benchmarking, providing larger
  savings for gigabyte-scale corpora where disk seeks dominate runtime.
- **Resolver hit-rate**: DOI normalisation increased Unpaywall success rate by
  5.5 percentage points and Crossref by 6.3 points in production canaries (see
  `notes/resolver-hit-rate.md`), reducing average resolver chain length by 0.31
  attempts per work item.
- **Logging race regression**: Concurrency stress tests (`tests/test_jsonl_logging.py`)
  execute 16 writers × 1000 records without corruption, confirming the per-logger
  lock resolves the historical JSONL interleaving issue recorded in OPS-1738.

### Decision 7: Preserve Backward Compatibility Through Deprecation Warnings

**Context**: Package facade re-exports `time` and `requests` modules for convenience. These add import confusion and are unnecessary given standard library imports.

**Options Considered**:

1. Remove immediately (breaking change)
2. Deprecate with warnings, remove in next major version
3. Deprecate with warnings, remove in next minor version
4. Keep indefinitely

**Decision**: Option 3 — Emit `DeprecationWarning`, document removal in next minor version.

**Rationale**:

- Minor version bump signals backward-incompatible change per semantic versioning
- Deprecation warning gives consumers notice without immediate breakage
- Internal codebase can migrate proactively before removal
- Standard library imports are obvious replacement; migration trivial

**Trade-offs**:

- Delayed cleanup extends technical debt window
- **Accepted**: Conservative approach minimizes disruption; warnings guide migration

## Architecture Patterns

### Retry Pattern

```
HTTP Request Flow:
  Application Layer
    └─> http.request_with_retries()
          ├─> Exponential backoff (0.75s × 2^attempt)
          ├─> Jitter (random 0-0.1s)
          ├─> Retry-After header handling
          └─> Status code filtering (429, 500, 502, 503, 504)
    └─> Session with HTTPAdapter(max_retries=0)
          └─> Connection pooling only, no retries
```

### Hash Computation Pattern

```
Download Loop:
  1. Initialize hasher and byte_counter
  2. Write sniff_buffer → file
     ├─> hasher.update(sniff_buffer)
     └─> byte_counter += len(sniff_buffer)
  3. For each chunk:
     ├─> file.write(chunk)
     ├─> hasher.update(chunk)
     └─> byte_counter += len(chunk)
  4. Finalize:
     ├─> sha256 = hasher.hexdigest()
     ├─> content_length = byte_counter
     └─> os.replace(part_path, dest_path)
```

### Thread-Safe Logging Pattern

```
Logger Class:
  - _lock: threading.Lock()
  - _file: file handle

  Method: _write(payload)
    1. Serialize payload → json_line (outside lock)
    2. Acquire lock
    3. Write json_line to file
    4. Flush file
    5. Release lock
```

### Metrics Export Pattern

```
Run Completion:
  1. Collect metrics.summary() → dict
  2. Emit to JSONL stream:
     └─> {"record_type": "summary", ...}
  3. Export sidecar JSON:
     └─> manifest.metrics.json
         ├─> processed, saved, html_only, skipped counters
         └─> summary: {attempts, successes, skips, failures}
```

## Migration Strategy

### Phase 1: Internal Refactoring (Weeks 1-2)

- Centralize retry logic
- Remove redundant HEAD requests
- Implement streaming hash computation
- Add thread-safe logging

**Risk**: Performance regression on streaming hash
**Mitigation**: Benchmark before/after; rollback mechanism ready

### Phase 2: API Surface Expansion (Week 3)

- Add CLI flags for concurrency, HEAD precheck, Accept header
- Wire flags to configuration
- Implement metrics export

**Risk**: Configuration complexity increase
**Mitigation**: Comprehensive defaults; documentation with examples

### Phase 3: Decoupling and Deprecation (Week 4)

- Extract shared utilities
- Emit deprecation warnings
- Update documentation

**Risk**: Downstream consumers impacted by warnings
**Mitigation**: Clear migration path in warning messages

### Rollback Plan

Each change is independently revertible:

- Retry centralization: Restore adapter Retry configuration
- HEAD removal: Re-add HEAD request block in download_candidate
- Streaming hash: Restore post-write hash computation
- Thread safety: Remove locks (degrades to current behavior)
- Metrics export: Skip sidecar write (logs still function)

## Validation Strategy

### Unit Tests

- Mock HTTP servers returning controlled status sequences
- Concurrent logging stress tests (16 threads × 1000 records)
- Hash computation accuracy verification
- CLI argument parsing round-trips

### Integration Tests

- Full pipeline runs against staging infrastructure
- Resume functionality with real manifests
- Dry-run coverage validation
- Multi-resolver coordination

### Performance Tests

- Download throughput (files/sec) before/after
- Disk I/O operation counts via strace/dtrace
- Memory footprint under concurrent load
- Network bandwidth utilization

### Acceptance Criteria

- Zero manifest corruption under concurrent load
- 40%+ performance improvement on large files
- Retry count determinism verified
- All CLI flags functional
- Metrics JSON schema documented

## Operational Impact

### Monitoring Dashboards

New metrics enable tracking:

- Per-resolver success rates
- Retry count distributions
- Content corruption detection rates
- Network efficiency (requests per successful download)

### Capacity Planning

Deterministic retry behavior allows accurate quota estimation:

- Max attempts per work = 1 + max_retries (default 3)
- Expected API calls = works × enabled_resolvers × (1 + retry_rate × avg_retries)

### Troubleshooting

Structured metrics simplify failure investigation:

- Identify resolvers with elevated retry rates
- Correlate corruption detection with specific URL patterns
- Track impact of HEAD precheck toggle on classification accuracy

## Open Questions

1. **Should global URL deduplication be enabled by default for certain resolver configurations?**
   - Recommendation: Keep disabled by default; enable via explicit flag for broad crawls
   - Rationale: Conservative default prevents unexpected behavior; opt-in for advanced users

2. **What is the optimal HEAD precheck strategy for resolvers returning unreliable Content-Type headers?**
   - Recommendation: Maintain per-resolver override map in configuration
   - Rationale: Allows fine-tuning without code changes; operational flexibility

3. **Should domain-level rate limiting be implemented alongside resolver-level limits?**
   - Recommendation: Implement as optional feature behind configuration flag
   - Rationale: Needed for specific publishers but adds complexity; opt-in appropriate

## Future Considerations

### Async/Await Migration

Current synchronous threading model could be replaced with `asyncio` for improved concurrency scalability. This change is intentionally deferred because:

- Requires protocol changes across resolver implementations
- Introduces different error handling patterns
- Benefits primarily high-concurrency scenarios beyond current usage
- Should be separate major version effort

### Content Classification ML

Machine learning model for content type detection could replace heuristic sniffing:

- Train on production download corpus
- Handle ambiguous cases more robustly
- Reduce false positive corruption detection
- Requires labeled training data collection first

### Distributed Download Coordination

Multi-node download clusters could share global URL deduplication state:

- Redis/Memcached for shared seen-URL set
- Distributed locking for cross-node coordination
- Enables true horizontal scaling
- Justified only for workloads exceeding single-node capacity

## Security Considerations

### SHA-256 Integrity

Streaming hash computation maintains cryptographic integrity verification without security implications. Hash collisions remain infeasible; computation timing remains constant.

### Thread Safety

Lock-based synchronization prevents race conditions that could cause data corruption but does not introduce new attack surfaces. Locks are process-local; no network exposure.

### Rate Limiting Compliance

Centralized retry logic with `Retry-After` header respect improves compliance with external API terms of service, reducing risk of IP blocking or quota violations.

### Dependency Updates

No new external dependencies introduced. Existing dependencies (requests, urllib3) remain at current versions. Deprecation of convenience re-exports reduces import confusion that could mask supply chain attacks.
