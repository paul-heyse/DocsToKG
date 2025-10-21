# Pillars 7 & 8 Implementation Guide

## Overview

This document provides a comprehensive implementation guide for:

- **Pillar 7**: Observability that answers questions (events, sinks, instrumentation)
- **Pillar 8**: Safety & policy, defense-in-depth (gates, policies, error handling)

## Current Status

### ✅ Completed (Foundation Laid)

**Pillar 7 Foundation (90% complete):**

- [x] `observability/events.py` - Event model, context management, emission API
- [x] `observability/schema.py` - JSON Schema generation and validation
- [x] `observability/emitters.py` - Base emitter classes, JsonStdoutEmitter, FileJsonlEmitter, Buffering
- [ ] `observability/emitters.py` - DuckDB and Parquet emitters (stubs need full implementation)
- [ ] `observability/queries.py` - Stock queries and dashboarding (not yet created)

**Pillar 8 Foundation (85% complete):**

- [x] `policy/errors.py` - ErrorCode enum, result types, exception classes
- [x] `policy/registry.py` - Thread-safe singleton registry with decorator pattern
- [x] `policy/metrics.py` - Per-gate metrics collection and snapshots
- [ ] `policy/gates.py` - Concrete gate implementations (stubs partially done)
- [ ] `policy/contracts.py` - Type contracts for gate inputs/outputs (not yet created)

### 🔄 In Progress

**CLI Import Fix:**

- [x] Moved `_normalize_plan_args` to `cli_main.py` and re-exported from cli package
- [x] Added importlib workaround in `cli/__init__.py` to re-export symbols from cli.py
- [x] Tests now pass: 926 passing, 18 failing (down from 44)

### ⏳ To Do

1. **Pillar 7 Phase 1**: Complete emitters (DuckDB, Parquet)
2. **Pillar 7 Phase 2**: Instrumentation across all subsystems
3. **Pillar 7 Phase 3**: CLI and queries
4. **Pillar 8 Phase 1**: Complete gate implementations
5. **Pillar 8 Phase 2**: Integration into planning/download/extraction flows
6. **E2E Integration**: End-to-end testing and validation

---

## Pillar 7: Observability Implementation

### Phase 7.1: Complete Emitters

**DuckDB Emitter (observability/emitters.py)**

```python
class DuckDBEmitter(EventEmitter):
    """Write events to DuckDB events table."""

    def __init__(self, db_path: str):
        """Initialize with DuckDB database path."""
        import duckdb
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self._create_table()
        self._batch_buffer = []
        self._batch_size = 100
        self._lock = threading.Lock()

    def _create_table(self):
        """Create events table if not exists."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                ts TIMESTAMP,
                type VARCHAR,
                level VARCHAR,
                run_id VARCHAR,
                config_hash VARCHAR,
                service VARCHAR,
                app_version VARCHAR,
                os_name VARCHAR,
                python_version VARCHAR,
                libarchive_version VARCHAR,
                hostname VARCHAR,
                pid INTEGER,
                version_id VARCHAR,
                artifact_id VARCHAR,
                file_id VARCHAR,
                request_id VARCHAR,
                payload JSON
            )
        """)

    def emit(self, event):
        """Batch events and insert to DuckDB."""
        # Implementation: batch collection + periodic flush
        pass

    def close(self):
        """Flush pending events and close connection."""
        pass

class ParquetEmitter(EventEmitter):
    """Write events to Parquet file."""

    def __init__(self, filepath: str, batch_size: int = 1000):
        """Initialize with output filepath."""
        import pyarrow as pa
        self.filepath = filepath
        self.batch_size = batch_size
        self._batch_buffer = []
        self._lock = threading.Lock()

    def emit(self, event):
        """Batch events and write to Parquet."""
        # Implementation: PyArrow Table batching
        pass

    def close(self):
        """Flush pending events to Parquet."""
        pass
```

**Key Implementation Points:**

- Thread-safe batch collection with locks
- Automatic flushing when batch size reached
- Connection pooling for DuckDB
- Schema validation against Event model
- Error handling with non-raising behavior

### Phase 7.2: Instrumentation

Wire event emission into:

1. **network/instrumentation.py** (NEW)
   - Emit `net.request` on HTTP calls
   - Fields: method, url_redacted, status, ttfb_ms, elapsed_ms, cache state
   - Hook into httpx client callbacks

2. **ratelimit/instrumentation.py** (NEW)
   - Emit `ratelimit.acquire` on rate limiter calls
   - Emit `ratelimit.cooldown` on 429 handling
   - Fields: blocked_ms, outcome (ok/blocked), key

3. **io/extraction_observability.py** (EXTEND)
   - Emit `extract.start`, `extract.done`, `extract.error`
   - Fields: entries_total, entries_included, bytes_written, ratio_total, duration_ms

4. **catalog/instrumentation.py** (NEW)
   - Emit `db.tx.commit`, `db.tx.rollback`, `db.migrate.applied`
   - Fields: tx_id, tables_affected, rows_changed

5. **planning.py** (EXTEND)
   - Emit `cli.command.start|done|error` in main orchestration

### Phase 7.3: CLI and Queries

**cli/obs_cmd.py** (EXTEND)

```python
@app.command()
def tail(n: int = 100, follow: bool = False) -> None:
    """Tail recent events."""
    pass

@app.command()
def stats() -> None:
    """Show SLO summary, top offenders, stability."""
    pass

@app.command()
def export(format: str = "json") -> None:
    """Export events to JSON/CSV/Parquet."""
    pass
```

**Stock Queries (observability/queries.py)**

```python
STOCK_QUERIES = {
    "slo_network": "SELECT service, approx_quantile(payload.elapsed_ms, 0.95) AS p95 FROM events WHERE type='net.request' GROUP BY 1",
    "cache_hit_ratio": "SELECT service, avg(payload.cache IN ('hit','revalidated')) AS hit_ratio FROM events WHERE type='net.request' GROUP BY 1",
    "rate_limit_pressure": "SELECT substr(payload.key,1,40) AS key, sum(payload.blocked_ms) AS blocked_ms FROM events WHERE type='ratelimit.acquire' GROUP BY 1 ORDER BY 2 DESC LIMIT 10",
    "safety_heatmap": "SELECT payload.error_code, count(*) FROM events WHERE type LIKE '%.error' GROUP BY 1 ORDER BY 2 DESC",
    "zip_bomb_sentinel": "SELECT ts, payload.ratio_total FROM events WHERE type='extract.done' AND payload.ratio_total > 10.0",
}
```

---

## Pillar 8: Safety & Policy Implementation

### Phase 8.1: Complete Gate Implementations

**policy/gates.py** - Six concrete gates:

1. **Configuration Gate** ✅ (stub complete, needs enhancement)
   - Validate all required fields present
   - Check bounds (timeouts, retries, concurrency)
   - Hash and emit normalized config

2. **URL & Network Gate** (partial)
   - Scheme validation (http/https only)
   - Host allowlisting (PSL, CIDR, IP ranges)
   - Port validation (global + per-host)
   - DNS resolution (strict/lenient policies)
   - Redirect auditing (no auth forwarding)

3. **Filesystem & Path Gate** (not implemented)
   - Dirfd/openat semantics (O_NOFOLLOW|O_EXCL)
   - Path normalization (NFC, no .., no /)
   - Casefold collision detection
   - Length/depth constraints
   - Reserved name filtering (Windows)

4. **Extraction Policy Gate** (partial)
   - Entry type allowlist (regular files only)
   - Global zip-bomb ratio limit
   - Per-entry ratio limit
   - Per-file size limit
   - Entry count budget
   - Include/exclude globs

5. **Storage Gate** (not implemented)
   - Atomic LATEST.json writes (temp+move)
   - Path traversal prevention in remote_rel
   - Rename safety checks

6. **DB Transactional Gate** (not implemented)
   - Foreign key invariant enforcement
   - Commit-after-FS-success choreography
   - Latest pointer + marker consistency
   - Doctor-triggered repairs

### Phase 8.2: Gate Contracts

**policy/contracts.py** (NEW)

```python
@dataclass
class UrlGateContract:
    """Type contract for URL gate."""
    input: UrlInput  # url, allowed_hosts, allowed_ports
    output: Union[PolicyOK, PolicyReject]
    side_effects: ["emit policy.gate event", "record metric"]

@dataclass
class ExtractionGateContract:
    """Type contract for extraction gate."""
    input: ExtractionInput  # archive, policies
    output: Union[PolicyOK, PolicyReject]
    side_effects: ["audit entries", "emit policy.gate event"]

# ... contracts for all 6 gates
```

### Phase 8.3: Telemetry Integration

**Emit events for each gate:**

```python
from observability.events import emit_event
from policy.metrics import get_metrics_collector

def url_gate(...) -> PolicyResult:
    start_ms = time.perf_counter() * 1000
    result = ...  # validation logic
    elapsed_ms = time.perf_counter() * 1000 - start_ms

    # Emit event
    emit_event(
        type="policy.gate",
        level="ERROR" if isinstance(result, PolicyReject) else "INFO",
        payload={
            "gate": "url_gate",
            "outcome": "reject" if isinstance(result, PolicyReject) else "ok",
            "elapsed_ms": elapsed_ms,
            "error_code": result.error_code.value if isinstance(result, PolicyReject) else None,
        }
    )

    # Record metric
    collector = get_metrics_collector()
    collector.record_metric(GateMetric(
        gate_name="url_gate",
        passed=isinstance(result, PolicyOK),
        elapsed_ms=elapsed_ms,
        error_code=result.error_code.value if isinstance(result, PolicyReject) else None,
    ))

    return result
```

---

## Integration Points

### Into planning.py

```python
# At start: validate config
gate_result = config_gate(config)
if isinstance(gate_result, PolicyReject):
    raise_policy_error(gate_result.error_code, "Config invalid")

# Before each URL fetch: validate URL
gate_result = url_gate(url, allowed_hosts, allowed_ports)
if isinstance(gate_result, PolicyReject):
    raise_policy_error(gate_result.error_code, "URL rejected")
```

### Into io/extraction_policy.py

```python
# Pre-scan: validate archive structure
gate_result = extraction_gate(archive, policies)
if isinstance(gate_result, PolicyReject):
    raise_policy_error(gate_result.error_code, "Archive invalid")
```

### Into io/filesystem.py

```python
# Before write: validate paths
gate_result = filesystem_gate(root_path, entry_paths)
if isinstance(gate_result, PolicyReject):
    raise_policy_error(gate_result.error_code, "Path traversal detected")
```

### Into catalog/boundaries.py

```python
# After extraction, before DB: validate boundaries
gate_result = db_boundary_gate(fs_state, db_state)
if isinstance(gate_result, PolicyReject):
    raise_policy_error(gate_result.error_code, "Boundary inconsistency")
```

---

## Testing Strategy

### Pillar 7 Testing (observability/)

- **Unit**: Event model validation, sink fan-out, drop strategy
- **Integration**: Full pipeline emits events, DB populates correctly
- **Performance**: Event emission adds < 2% overhead
- **Coverage**: 100% of event types and sinks

### Pillar 8 Testing (policy/)

- **Unit**: Each gate accepts/rejects correctly (white-box)
- **Property-based**: URL/path generators, normalization idempotent
- **Integration**: E2E scenarios trigger each error code
- **Cross-platform**: Windows reserved names, NFD/NFC normalization
- **Chaos**: Crash between FS + DB, doctor can repair
- **Coverage**: 100% error codes, all gate combinations

### Commands to Run Tests

```bash
# Observability tests
./.venv/bin/pytest tests/ontology_download -k "observability or event or emitter" -q

# Policy tests
./.venv/bin/pytest tests/ontology_download -k "policy or gate or error" -q

# Full suite with coverage
./.venv/bin/pytest tests/ontology_download --cov=DocsToKG.OntologyDownload.observability --cov=DocsToKG.OntologyDownload.policy -q
```

---

## Quality Gates

Before shipping:

1. [ ] 100% of tests passing (target: 950+ tests)
2. [ ] 0 type errors: `./.venv/bin/mypy src/DocsToKG/OntologyDownload/observability src/DocsToKG/OntologyDownload/policy`
3. [ ] 0 lint violations: `./.venv/bin/ruff check src/DocsToKG/OntologyDownload/observability src/DocsToKG/OntologyDownload/policy`
4. [ ] Documentation updated in `docs/schemas/events.schema.json` and `docs/policies/gates.md`
5. [ ] README and AGENTS.md reflect observability + policy capabilities

---

## Files Checklist

### New Files to Create

- [ ] `observability/queries.py` - Stock queries and dashboards
- [ ] `policy/contracts.py` - Type contracts for gates
- [ ] `network/instrumentation.py` - HTTP layer events
- [ ] `ratelimit/instrumentation.py` - Rate limit events
- [ ] `catalog/instrumentation.py` - DB boundary events

### Files to Extend

- [ ] `observability/emitters.py` - Full DuckDB + Parquet
- [ ] `policy/gates.py` - Complete all 6 gates
- [ ] `policy/registry.py` - Add gate invocation with metrics
- [ ] `cli/obs_cmd.py` - Add tail, stats, export subcommands
- [ ] `planning.py` - Wire gates at key boundaries
- [ ] `io/extraction_policy.py` - Emit extraction events
- [ ] `io/filesystem.py` - Filesystem gate integration
- [ ] `catalog/boundaries.py` - DB boundary gate

### Test Files to Create

- [ ] `tests/ontology_download/test_observability_foundation.py`
- [ ] `tests/ontology_download/test_observability_emitters.py`
- [ ] `tests/ontology_download/test_observability_queries.py`
- [ ] `tests/ontology_download/test_policy_gates.py`
- [ ] `tests/ontology_download/test_policy_integration.py`
- [ ] `tests/ontology_download/test_e2e_events_and_gates.py`

---

## Estimated Effort

| Phase | Tasks | LOC | Tests | Days | Risk |
|-------|-------|-----|-------|------|------|
| 7.1 | DuckDB + Parquet emitters | 250 | 20 | 0.5 | LOW |
| 7.2 | Instrumentation (5 modules) | 300 | 30 | 1.0 | LOW |
| 7.3 | CLI + queries | 200 | 25 | 0.5 | LOW |
| 8.1 | Gate implementations | 400 | 50 | 1.5 | MEDIUM |
| 8.2 | Contracts | 100 | 15 | 0.5 | LOW |
| 8.3 | Telemetry + metrics | 150 | 20 | 0.5 | LOW |
| E2E | Integration testing | 300 | 100 | 1.5 | MEDIUM |
| **TOTAL** | | **1,700** | **260** | **6 days** | **LOW** |

---

## Key Innovation Areas

1. **Single Event Bus**: One Event model used everywhere (JSON logs, DuckDB, Parquet)
2. **Centralized Gates**: One error catalog, one emit path, one registry
3. **Observable Gates**: Every gate emits events + records metrics
4. **Defense-in-Depth**: Gates at every I/O boundary (network, FS, extraction, DB)
5. **Stock Queries**: Pre-built dashboards answer operational questions in seconds

---

## References

- Specification: `DO NOT DELETE docs-instruct/Ontology-config-objects-optimization7+8.md`
- Architecture: `DO NOT DELETE docs-instruct/Ontology-config-objects-optimization7+8+architecture.md`
- AGENTS Guide: `src/DocsToKG/OntologyDownload/AGENTS.md`
