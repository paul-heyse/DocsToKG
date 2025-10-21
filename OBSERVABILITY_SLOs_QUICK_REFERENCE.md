# Observability & SLOs — Quick Reference (Phase Implementation)

**Status:** 85% complete | 4–6 days remaining | 1 engineer required

---

## What's Already Done ✅ (85%)

| Component | Status | File | Coverage |
|-----------|--------|------|----------|
| Telemetry Schema (DDL) | ✅ | `telemetry_schema.sql` | 6 tables, optimized indexes |
| SQLite Sinks | ✅ | `telemetry.py` | Thread-safe, multi-table |
| Prometheus Exporter | ✅ | `telemetry_prom_exporter.py` | 8 metrics, low cardinality |
| Parquet Export | ✅ | `telemetry_export_parquet.py` | DuckDB-based, ZSTD compression |
| SLO Evaluation CLI | ✅ | `cli_telemetry_summary.py` | 8 SLIs, pass/fail exit code |
| Helper Utilities | ✅ | `telemetry_helpers.py` | emit_http/rate/breaker/fallback |

---

## What Remains ⚠️ (15%)

### Phase 1: HTTP Events (Days 1–1.5)

**File:** `src/DocsToKG/ContentDownload/networking.py`

```python
# In request_with_retries() after response received:
from DocsToKG.ContentDownload.telemetry_helpers import emit_http_event
from DocsToKG.ContentDownload.urls import canonical_for_index
import hashlib

url_canonical = canonical_for_index(url)
url_hash = hashlib.sha256(url_canonical.encode()).hexdigest()[:16]

emit_http_event(
    telemetry=telemetry,
    run_id=run_id,
    host=response.url.host,
    role="metadata",  # or "landing", "artifact"
    method="GET",
    status=response.status_code,
    url_hash=url_hash,
    from_cache=1 if from_hishel else 0,
    revalidated=1 if status==304 else 0,
    stale=0,  # query Hishel cache controller
    retry_count=retry_attempt - 1,
    retry_after_s=int(float(response.headers.get("Retry-After", 0))),
    rate_delay_ms=rate_wait_ms,
    breaker_state=breaker.current_state,  # "closed", "open", "half_open"
    breaker_recorded="success" if status<400 else "failure",
    elapsed_ms=int((time.time() - start_time) * 1000),
    error=None if status<500 else response.__class__.__name__,
)
```

**Checklist:**

- [ ] Add imports + helper call to `request_with_retries()`
- [ ] Extract Hishel cache metadata (from_cache, revalidated, stale)
- [ ] Hash URL (not raw URL)
- [ ] Write test: `tests/content_download/test_telemetry_http.py`

---

### Phase 2: Rate & Breaker Events (Days 1.5–3)

**Files:** `ratelimit/manager.py`, `networking_breaker_listener.py`, `pipeline.py`

#### 2A: Rate Limiter Events

```python
# In ratelimit/manager.py, on limiter.acquire():
emit_rate_event(
    telemetry=telemetry,
    run_id=run_id,
    host=host,
    role="metadata",  # or "landing", "artifact"
    action="acquire",  # or "block", "head_skip"
    delay_ms=int(waited * 1000),
    max_delay_ms=max_delay_ms,
)
```

#### 2B: Breaker Listener

```python
# In networking_breaker_listener.py:
class NetworkingBreakerListener:
    def on_state_change(self, breaker, old_state, new_state, reset_timeout_s):
        event = {
            "run_id": self.run_id,
            "ts": time.time(),
            "host": breaker.name,  # hostname
            "scope": "host",  # or "resolver"
            "old_state": old_state,
            "new_state": new_state,
            "reset_timeout_s": reset_timeout_s,
        }
        self.telemetry.log_breaker_transition(event)
```

**Checklist:**

- [ ] Wire `emit_rate_event()` in acquire/block/head_skip paths
- [ ] Complete `NetworkingBreakerListener` state change handler
- [ ] Register listener: `breaker.notify(listener)` in `pipeline.py`
- [ ] Write tests: `test_telemetry_rate.py`, `test_telemetry_breaker.py`

---

### Phase 3: Fallback & Wayback (Days 3–4)

**Files:** `fallback/orchestrator.py`, `telemetry_wayback_sqlite.py`

```python
# In fallback/orchestrator.py on adapter completion:
fallback_event = {
    "run_id": run_id,
    "ts": time.time(),
    "work_id": work.id,
    "artifact_id": artifact.id,
    "tier": "direct_oa",
    "source": "unpaywall_pdf",  # adapter name
    "host": adapter.host,
    "outcome": "success",  # or "timeout", "retryable", "nonretryable", "error"
    "reason": "http_200",  # short code
    "status": 200,
    "elapsed_ms": int((time.time() - start_time) * 1000),
}
telemetry.log_attempt(fallback_event)  # or new method: log_fallback_attempt
```

**Checklist:**

- [ ] Emit fallback events in `orchestrator.py` per adapter
- [ ] Bridge Wayback events → fallback_attempts table (use source='wayback_*')
- [ ] Verify TTFP calculations work
- [ ] Write test: `test_fallback_telemetry.py`, `test_wayback_telemetry_integration.py`

---

### Phase 4: CLI & Docs (Days 4–5)

```bash
# Test end-to-end flow:
./.venv/bin/python -m DocsToKG.ContentDownload.cli \
  --topic "test" --max 10 --dry-run \
  --out /tmp/obs_test

# Get run_id
RUN_ID=$(jq -r '.run_id' /tmp/obs_test/manifest.summary.json)

# Evaluate SLOs
./.venv/bin/python -m DocsToKG.ContentDownload.cli_telemetry_summary \
  --db /tmp/obs_test/manifest.sqlite3 \
  --run $RUN_ID

# Should show ✅ or ❌ for each SLI
```

**Checklist:**

- [ ] Update `AGENTS.md` with "Observability & SLOs" section
- [ ] Wire CLI subcommand: `cli telemetry summary --db X --run Y`
- [ ] Document SLI definitions + runbooks
- [ ] End-to-end smoke test passes
- [ ] All docs updated + examples included

---

## Test Commands

```bash
# Unit tests
./.venv/bin/pytest tests/content_download/test_telemetry_*.py -v --cov

# Integration test (full run)
./.venv/bin/python -m DocsToKG.ContentDownload.cli \
  --topic "vision" --year-start 2024 --year-end 2024 --max 20 \
  --out runs/telemetry_test

# Check database populated
sqlite3 runs/telemetry_test/manifest.sqlite3 << 'SQL'
SELECT COUNT(*) as http_count FROM http_events;
SELECT COUNT(*) as rate_count FROM rate_events;
SELECT COUNT(*) as breaker_count FROM breaker_transitions;
SELECT COUNT(*) as fallback_count FROM fallback_attempts;
SQL

# Run SLO check
./.venv/bin/python -m DocsToKG.ContentDownload.cli_telemetry_summary \
  --db runs/telemetry_test/manifest.sqlite3 \
  --run <run_id_from_summary>

# Export to Parquet
./.venv/bin/python -m DocsToKG.ContentDownload.telemetry_export_parquet \
  --sqlite runs/telemetry_test/manifest.sqlite3 \
  --out runs/telemetry_test/parquet

# Run Prometheus exporter
./.venv/bin/python -m DocsToKG.ContentDownload.telemetry_prom_exporter \
  --db runs/telemetry_test/manifest.sqlite3 \
  --port 9108 &

# Query metrics
curl -s http://localhost:9108/metrics | grep docstokg_run_yield_pct
```

---

## SLO Thresholds (Configurable)

```python
SLO = {
    "yield_pct_min": 85.0,           # ≥85% artifacts successful
    "ttfp_p50_ms_max": 3000,         # p50 time-to-first-PDF ≤3s
    "ttfp_p95_ms_max": 20000,        # p95 ≤20s
    "cache_hit_pct_min": 60.0,       # ≥60% metadata cache hits
    "rate_delay_p95_ms_max": 250,    # p95 limiter delay ≤250ms
    "http429_pct_max": 2.0,          # ≤2% 429 ratio
    "corruption_max": 0,             # zero corrupted files
}
```

---

## Files to Modify (Summary)

| File | Changes | LOC |
|------|---------|-----|
| `networking.py` | Add `emit_http_event()` call | +30 |
| `ratelimit/manager.py` | Add `emit_rate_event()` calls | +20 |
| `networking_breaker_listener.py` | Complete listener impl | +50 |
| `pipeline.py` | Register listener + pass telemetry | +5 |
| `fallback/orchestrator.py` | Emit fallback events | +30 |
| `telemetry_wayback_sqlite.py` | Bridge to fallback_attempts | +40 |
| `cli_telemetry_summary.py` | Already complete ✅ | 0 |
| `telemetry.py` | Already complete ✅ | 0 |
| Tests (new) | Unit + integration tests | +300 |

**Total Effort:** ~470 LOC + tests | 4–6 days

---

## Definition of Done Checklist

- [ ] All 5 layers emitting telemetry (HTTP, Rate, Breaker, Fallback, Wayback)
- [ ] >90% test coverage on telemetry modules
- [ ] All SLI queries verified on real data
- [ ] End-to-end smoke test: full run → summary → SLO check ✅
- [ ] Prometheus metrics exposed + queryable
- [ ] Parquet export working
- [ ] AGENTS.md updated with ops guide
- [ ] Zero linting errors: `ruff check src/DocsToKG/ContentDownload/telemetry*.py`
- [ ] Zero type errors: `mypy src/DocsToKG/ContentDownload/telemetry*.py`
- [ ] All tests passing: `pytest tests/content_download/test_telemetry*.py -v`
- [ ] Production readiness: no TODOs/FIXMEs

---

## Resources

- **Main Plan:** `OBSERVABILITY_SLOs_VALIDATION_AND_PLAN.md`
- **Schema:** `telemetry_schema.sql`
- **Existing Code:**
  - `telemetry.py` (sinks)
  - `telemetry_helpers.py` (emit functions)
  - `cli_telemetry_summary.py` (SLO CLI)
  - `telemetry_prom_exporter.py` (Prometheus)
  - `telemetry_export_parquet.py` (Parquet)

---

**Status:** Ready for Phase 1 handoff 🚀
