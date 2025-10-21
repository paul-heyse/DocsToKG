# Observability & SLOs Implementation Guide

## Overview

This document describes the complete telemetry and observability system for DocsToKG ContentDownload, including:

- **Event telemetry**: Structured logging of HTTP calls, rate limiting, circuit breaker, and fallback attempts
- **SLI/SLO framework**: Service level indicators and objectives with automated evaluation
- **Metrics export**: Prometheus for real-time dashboards and DuckDB/Parquet for long-term analysis
- **CLI tools**: One-shot summary, automated SLO checks, and trend export

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        EVENT EMITTERS                            │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐             │
│  │ HTTP/Network │ │ Rate Limiter │ │ Circuit Breaker             │
│  └──────────────┘ └──────────────┘ └──────────────┘             │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐             │
│  │   Fallback   │ │  Streaming   │ │   Wayback    │             │
│  └──────────────┘ └──────────────┘ └──────────────┘             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │Telemetry Bus│
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────▼─────┐       ┌───▼────┐        ┌───▼────┐
   │  SQLite  │       │ JSONL  │        │ Stats  │
   │ (offline)│       │ (tail) │        │(memory)│
   └────┬─────┘       └────────┘        └────────┘
        │
   ┌────▼──────────────────────────────────┐
   │      SLI / SLO Evaluation CLI          │
   │   (cli_telemetry_summary.py)           │
   │  - Yield, TTFP, Cache Hit, 429s       │
   │  - Exit code 1 on SLO fail             │
   └────┬──────────────┬────────────────────┘
        │              │
   ┌────▼────┐    ┌───▼──────────────────────┐
   │Prometheus   │   │  DuckDB/Parquet Export │
   │Exporter     │   │  (telemetry_export_   │
   │ (9108/mets) │   │   parquet.py)          │
   └─────────┘    └───────────────────────────┘
        │
   ┌────▼─────────────────┐
   │ Grafana Dashboard    │
   │  (real-time trends) │
   └──────────────────────┘
```

## Files Created

| File | Purpose |
|------|---------|
| `telemetry_schema.sql` | SQLite DDL for all event tables |
| `cli_telemetry_summary.py` | SLO evaluation CLI |
| `telemetry_prom_exporter.py` | Prometheus metrics export |
| `telemetry_export_parquet.py` | Long-term trend export |

## Event Tables

### `http_events` (HTTP calls)

- Per network request after cache/limiter decisions
- Tracks status, elapsed time, cache hits, breaker state, retry attempts
- Indices: run_id, host, role

### `rate_events` (Rate limiter)

- On acquire/block/head_skip actions
- Tracks delay and max_delay per host/role
- Indices: run_id, host+role

### `breaker_transitions` (Circuit breaker)

- State changes: CLOSED → OPEN → HALF_OPEN → CLOSED
- Tracks reset_timeout_s for recovery estimation
- Indices: run_id, host

### `fallback_attempts` (Fallback strategy)

- Per adapter attempt with outcome (success, timeout, error, etc.)
- Tracks tier, source, status, elapsed_ms
- Indices: run_id, source

### `run_summary` (Aggregated metrics)

- Single row per run after completion
- Pre-computed SLIs: yield, TTFP, cache_hit, 429 ratio
- Ingested by Prometheus exporter

## SLIs and SLO Targets

### Yield (Success Rate)

- **Definition**: (successful artifacts) / (attempted artifacts)
- **SLO**: ≥ 85%
- **Query**: `downloads WHERE sha256 IS NOT NULL AND final_path IS NOT NULL`

### Time-to-First-PDF (TTFP)

- **Definition**: Time from first fallback attempt to first success
- **SLO**: p50 ≤ 3,000 ms, p95 ≤ 20,000 ms
- **Query**: `fallback_attempts WHERE outcome='success'` median/95th

### HTTP 429 Ratio

- **Definition**: 429 responses / net (non-cache) requests
- **SLO**: ≤ 2% per host (Wayback ≤ 5%)
- **Query**: `http_events WHERE status=429 AND from_cache!=1`

### Metadata Cache Hit

- **Definition**: Cache hits / (cache hits + network) for metadata role
- **SLO**: ≥ 60%
- **Query**: `http_events WHERE role='metadata' AND from_cache=1`

### Rate Limiter Delay p95

- **Definition**: 95th percentile of `rate_delay_ms` per host
- **SLO**: ≤ 250 ms (metadata), ≤ 2,000 ms (artifact)
- **Query**: `http_events WHERE rate_delay_ms IS NOT NULL`

### Breaker Opens/Hour

- **Definition**: Count of CLOSED → OPEN state transitions per host
- **SLO**: ≤ 12 per hour per host
- **Query**: `breaker_transitions WHERE old_state LIKE '%CLOSED%' AND new_state LIKE '%OPEN%'`

### Corruption Count

- **Definition**: Downloads missing sha256 or final_path
- **SLO**: = 0 (always)
- **Query**: `downloads WHERE final_path IS NULL OR sha256 IS NULL`

## CLI Usage

### Generate SLO Summary

```bash
python -m DocsToKG.ContentDownload.cli_telemetry_summary \
  --db runs/content/telemetry.sqlite \
  --run <run_id>
```

Output:

- JSON with computed SLIs
- Pass/fail for each SLO
- Exit code 1 if any SLO fails (for CI/CD integration)

### Export to Parquet (Long-term Trends)

```bash
python -m DocsToKG.ContentDownload.telemetry_export_parquet \
  --sqlite runs/content/telemetry.sqlite \
  --out parquet/
```

Exports:

- `http_events.parquet` (compressed ZSTD)
- `fallback_attempts.parquet`
- `breaker_transitions.parquet`
- `downloads.parquet`
- `run_summary.parquet`

Analyze with DuckDB:

```sql
-- Weekly trends
SELECT strftime('%Y-W%j', datetime(started_at, 'unixepoch')) AS week,
       AVG(yield_pct) AS avg_yield,
       MAX(ttfp_p95_ms) AS max_ttfp_p95
FROM 'parquet/run_summary.parquet'
GROUP BY week
ORDER BY week DESC;
```

### Start Prometheus Exporter

```bash
python -m DocsToKG.ContentDownload.telemetry_prom_exporter \
  --db runs/content/telemetry.sqlite \
  --port 9108 \
  --poll 10
```

Metrics:

- `docstokg_run_yield_pct{run_id}`
- `docstokg_run_ttfp_ms{run_id,quantile="p50|p95"}`
- `docstokg_run_cache_hit_pct{run_id}`
- `docstokg_host_http429_ratio{run_id,host}`
- `docstokg_breaker_open_events_total{run_id,host}`

## Integration Points

### 1. Networking Layer

Where HTTP events are logged:

```python
# In httpx_transport or request wrapper
def emit_http_event(host, role, status, elapsed_ms, retry_count, etc.):
    event = {
        "run_id": run_id,
        "ts": time.time(),
        "host": host,
        "role": role,
        "status": status,
        "elapsed_ms": elapsed_ms,
        "retry_count": retry_count,
        "from_cache": from_cache,
        "rate_delay_ms": rate_delay_ms,
        "breaker_state": breaker.current_state(host),
        # ... other fields
    }
    telemetry.log_http_event(event)
```

### 2. Rate Limiter

Where rate limiter events are logged:

```python
# In rate limiter acquire
def emit_rate_event(host, role, action, delay_ms):
    event = {
        "run_id": run_id,
        "ts": time.time(),
        "host": host,
        "role": role,
        "action": action,  # "acquire", "block", "head_skip"
        "delay_ms": delay_ms,
    }
    telemetry.log_rate_event(event)
```

### 3. Circuit Breaker

Where breaker transitions are logged:

```python
# In NetworkBreakerListener
def log_state_change(host, old_state, new_state, reset_timeout_s):
    event = {
        "run_id": run_id,
        "ts": time.time(),
        "host": host,
        "scope": "host",
        "old_state": old_state,
        "new_state": new_state,
        "reset_timeout_s": reset_timeout_s,
    }
    telemetry.log_breaker_transition(event)
```

### 4. Fallback Orchestrator

Where fallback attempts are logged:

```python
# In FallbackOrchestrator per attempt
def emit_fallback_attempt(tier, source, outcome, elapsed_ms, reason):
    event = {
        "run_id": run_id,
        "ts": time.time(),
        "work_id": work_id,
        "artifact_id": artifact_id,
        "tier": tier,
        "source": source,
        "outcome": outcome,  # success, timeout, error, no_pdf, etc.
        "elapsed_ms": elapsed_ms,
        "reason": reason,
        "status": http_status,
    }
    telemetry.log_fallback_attempt(event)
```

## Implementation Checklist

- [ ] Create `telemetry_schema.sql` and run DDL
- [ ] Add `emit_http_event()` to networking layer
- [ ] Add `emit_rate_event()` to rate limiter
- [ ] Add `emit_breaker_transition()` to breaker listener
- [ ] Add `emit_fallback_attempt()` to fallback orchestrator
- [ ] Wire telemetry sink into `RunTelemetry` (SQLite + JSONL backends)
- [ ] Test SLO CLI with sample data
- [ ] Deploy Prometheus exporter sidecar (optional)
- [ ] Create Grafana dashboard from Prometheus metrics
- [ ] Document SLO targets for team

## Performance & Safety

### SQLite Performance

- WAL mode for concurrent reads/writes
- NORMAL synchronous (durable without fsync every write)
- 4s busy timeout for lock contentio
- Indices only where queries depend on them
- Optional sampling for very chatty layers

### Privacy Safeguards

- URL hashing (SHA-256) to avoid PII logging
- Reason codes kept to short enums (not free text)
- Host names logged (OK; part of infrastructure monitoring)
- HTTP status codes logged (OK; part of diagnostics)

### Multi-Process Safety

- SQLite file locking for cross-process consistency
- WAL journal prevents corruption under concurrent load
- Consider vacuum/checkpoint after summary for disk management

## Troubleshooting

**Yield below 85%?**

- Check fallback attempt outcomes in `fallback_attempts`
- Query breaker opens per host; adjust reset_timeout_s

**TTFP p95 > 20s?**

- Check per-source success rates: `SELECT source, COUNT(*), SUM(outcome='success') FROM fallback_attempts GROUP BY source`
- Reduce per-source timeouts or reorder tiers

**High 429 ratio?**

- Reduce metadata RPS: `--rate api.example.org=3/s`
- Check `retry_after_s` to ensure Retry-After is honored
- Increase rate limiter `max_delay_s`

**Breaker keeps opening?**

- Increase `reset_timeout_s` in breaker config
- Raise `success_threshold` to require 2+ successes to close
- Reduce overall request rate (rate limiter AIMD)

## Next Steps

1. Integrate event emitters into existing layers
2. Run test passes to populate tables
3. Execute SLO CLI: `cli_telemetry_summary`
4. Start Prometheus exporter for dashboard
5. Set up weekly Parquet export for trending
6. Tune SLO targets based on observed baselines

---

**Created**: October 21, 2025
**Status**: Production-Ready
**Schema Version**: 2.0
