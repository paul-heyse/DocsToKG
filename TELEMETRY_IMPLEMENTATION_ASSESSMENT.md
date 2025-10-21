# ContentDownload Telemetry Implementation Assessment

**Date**: 2025-10-21  
**Status**: ~85% Complete - Ready for Gap Closure  
**Task**: Implement remaining scope per specification documents

---

## Executive Summary

The telemetry system for ContentDownload is substantially implemented with strong foundational infrastructure:

**COMPLETE (100%)**:
- Core telemetry sink architecture (JSONL, CSV, SQLite, Multi-sink)
- Attempt record serialization and manifest entry handling
- Resume helpers (JsonlResumeLookup, SqliteResumeLookup)
- Telemetry helpers (emit_http_event, emit_rate_event, emit_fallback_attempt)
- RunTelemetry façade with metrics aggregation
- CLI telemetry commands (summary, query, export, parquet)
- SLO schema and computation (6 SLOs, metrics tracking)
- Telemetry records types (TelemetryAttemptRecord)
- Prometheus exporter integration
- Parquet export functionality

**GAPS (Remaining 15%)**:
1. **HTTP Session Bootstrap** - Shared HTTPX session with polite headers
2. **Per-Resolver HTTP Client** - Rate limit + retry wrapper per resolver
3. **Bootstrap Integration** (`run_from_config`) - Coordinate all layers
4. **Attempt CSV Format** - Stable schema with specific token set
5. **Manifest JSONL Format** - Finalization and integration
6. **Pipeline Integration** - Wiring telemetry into execution stages
7. **Test Coverage** - End-to-end smoke tests for bootstrap/CSV/manifest
8. **CLI Artifact Ingestion** - Optional `--input` file support

---

## Current Implementation Status by Component

### 1. Telemetry Foundation (✅ COMPLETE)

**Files**:
- `telemetry.py` - Core sinks, RunTelemetry, ManifestEntry
- `telemetry_records/records.py` - TelemetryAttemptRecord
- `telemetry_helpers.py` - emit_http_event, emit_rate_event, emit_fallback_attempt
- `cli_telemetry.py` - CLI commands
- `cli_telemetry_summary.py` - SLO evaluation CLI

**Status**: ✅ 100% implemented
- Sink implementations (JsonlSink, CsvSink, SqliteSink, MultiSink)
- Manifest serialization with config_hash
- Resume lookup mechanisms
- Metrics aggregation

### 2. Download Execution (✅ MOSTLY COMPLETE)

**Files**:
- `download_execution.py` - Three-stage pipeline (prepare, stream, finalize)
- `api/types.py` - Canonical types (DownloadPlan, DownloadStreamResult, DownloadOutcome)
- `api/exceptions.py` - SkipDownload, DownloadError

**Status**: ✅ 85% implemented
- Core stages implemented
- Telemetry emission points defined via `_emit()` helper
- Need: Full integration with AttemptRecord emission

### 3. Resolver Architecture (✅ PARTIALLY COMPLETE)

**Files**:
- `resolvers/base.py` - Resolver protocol
- `api/__init__.py` - Public API exports

**Status**: ✅ 70% implemented
- Protocol defined for all resolvers
- Need: Per-resolver HTTP client injection

### 4. HTTP & Rate Limiting (❌ GAPS)

**Missing**:
- Shared HTTPX session factory with polite headers
- Per-resolver HTTP client wrapper (rate limit + retry)
- Rate limit token bucket implementation
- Request/response logging for telemetry

**Scope**:
```python
# Need to implement:
1. get_http_session(config.http) -> httpx.Client
   - Sets User-Agent + mailto headers
   - Configures connection pooling
   - TLS/proxy settings

2. PerResolverHttpClient(session, resolver_config)
   - Wraps Session
   - Applies rate limits (TokenBucket)
   - Implements retry/backoff with Retry-After
   - Emits telemetry for rate limits and retries
```

### 5. Bootstrap Integration (❌ CRITICAL GAPS)

**Missing**: `bootstrap.run_from_config(cfg)` function

This is the **glue** that orchestrates:
```python
def run_from_config(cfg) -> None:
    1. Build telemetry sinks from cfg.telemetry + run_id
    2. Build shared HTTPX session from cfg.http
    3. Materialize resolvers from cfg.resolvers (order matters)
    4. For each resolver: create PerResolverHttpClient(session, resolver_config)
    5. Create ResolverPipeline with:
       - ordered resolvers
       - client map (resolver_name → HttpClient)
       - telemetry, run_id
       - policy knobs (robots, download)
    6. If artifact iterator provided: iterate and record manifest
```

**Scope**: ~300 LOC

### 6. CSV & Manifest Formats (⚠️ PARTIAL)

**Current**:
- Manifest JSONL exists but needs verification against spec
- CSV format defined but not integrated

**Spec Requirements**:
```
CSV Header:
  ts,run_id,resolver,url,verb,status,http_status,content_type,elapsed_ms,bytes_written,content_length_hdr,reason

Manifest JSONL:
  { "ts","run_id","artifact_id","resolver","url","outcome","ok","reason","path","content_type","bytes","html_paths","config_hash","dry_run" }

Stable Tokens:
  status: http-head, http-get, http-200, http-304, robots-fetch, robots-disallowed, retry, size-mismatch
  reason: ok, not-modified, robots, retry-after, backoff, conn-error, policy-type, policy-size, size-mismatch, timeout, tls-error, download-error
  outcome: success, skip, error
  classification: (same as outcome)
```

### 7. Pipeline Integration (❌ GAPS)

**Missing**: Wiring telemetry into download execution

**Need**:
- `prepare_candidate_download()` - emit prepare-stage attempts
- `stream_candidate_payload()` - emit HEAD/GET/retry attempts + CSV rows
- `finalize_candidate_download()` - emit final outcome + manifest row
- Exception handling (SkipDownload → skip, DownloadError → error)

### 8. End-to-End Tests (❌ MISSING)

**Missing**:
- Bootstrap smoke test (end-to-end with fake resolver/HTTP)
- Retry/backoff + rate limit test (429 → sleep → 200)
- CSV/Manifest verification test

---

## Implementation Roadmap

### Phase 1: HTTP Session & Per-Resolver Client (2-3 hours)

**Files to create**:
- `http_session.py` - Factory for shared HTTPX session
- `resolver_http_client.py` - PerResolverHttpClient wrapper

**Files to modify**:
- `pipeline.py` - Accept client map, select per-resolver client

### Phase 2: Bootstrap Integration (2-3 hours)

**Files to create**:
- `bootstrap.py` - `run_from_config()` orchestrator

**Files to modify**:
- `cli.py` - Call bootstrap.run_from_config instead of legacy paths
- `runner.py` - Delegate to bootstrap

### Phase 3: Pipeline Integration (1-2 hours)

**Files to modify**:
- `download_execution.py` - Emit attempts at each stage
- `download.py` - Use download_execution functions with telemetry
- `pipeline.py` - Catch exceptions, emit outcomes

### Phase 4: CSV & Manifest Verification (1 hour)

**Files to modify**:
- `telemetry.py` - Verify CsvSink schema
- `telemetry.py` - Verify ManifestEntry serialization

### Phase 5: Tests (2-3 hours)

**Files to create**:
- `tests/content_download/test_bootstrap_e2e.py`
- `tests/content_download/test_telemetry_csv_manifest.py`
- `tests/content_download/test_http_retry_backoff.py`

---

## Stable Token Vocabularies

### Status Tokens
```python
status: Literal[
    "http-head",          # HEAD request sent
    "http-get",           # GET response received (handshake)
    "http-200",           # Stream complete, 200 OK
    "http-304",           # Not Modified
    "robots-fetch",       # Fetched robots.txt
    "robots-disallowed",  # Policy decision (skip)
    "retry",              # Retry attempt (sleeping)
    "size-mismatch",      # Content-Length mismatch
    "download-error",     # General download error
]
```

### Reason Tokens
```python
reason: Literal[
    "ok",                 # Success
    "not-modified",       # 304 Not Modified
    "robots",             # Blocked by robots.txt
    "retry-after",        # Slept due to Retry-After
    "backoff",            # Slept due to exponential backoff
    "conn-error",         # Connection/TLS error
    "policy-type",        # Content-type policy violation
    "policy-size",        # Size policy violation
    "size-mismatch",      # Content-Length mismatch
    "timeout",            # Request timeout
    "tls-error",          # TLS error
    "download-error",     # Generic download error
]
```

### Outcome Tokens (Manifest)
```python
outcome: Literal["success", "skip", "error"]
```

---

## Quality Gates

1. **Type Safety**: 100% mypy compliance
2. **Linting**: 0 ruff violations
3. **Test Coverage**: ≥95% for new code
4. **Backward Compatibility**: Zero breaking changes
5. **Production Readiness**: Feature-gated (disabled by default)

---

## Dependencies

- **No new packages** - Uses existing httpx, tenacity, etc.
- **Integrates with**: 
  - `api/types.py` (canonical types)
  - `telemetry.py` (sinks)
  - `telemetry_records/records.py` (TelemetryAttemptRecord)
  - `resolvers/` (resolver registry)

---

## Next Actions

1. Implement HTTP session factory + per-resolver client
2. Create bootstrap.run_from_config orchestrator
3. Wire telemetry into download_execution stages
4. Write end-to-end tests
5. Verify CSV/Manifest formats match spec
