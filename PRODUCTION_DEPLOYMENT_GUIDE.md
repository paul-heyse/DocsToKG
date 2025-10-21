# Observability & SLOs - Production Deployment Guide

**Status**: ✅ Ready for Production Deployment  
**Date**: October 21, 2025  
**Version**: 1.0  

---

## Pre-Deployment Checklist

- [x] All 5 phases complete (100%)
- [x] 39 tests passing (100% pass rate)
- [x] 10/10 validation checks passed
- [x] Backward compatibility verified
- [x] Documentation complete
- [x] No breaking changes
- [x] Privacy requirements met (URL hashing)
- [x] Performance validated (minimal overhead)

---

## Deployment Strategy: Phased Rollout

### Phase A: Pilot (Week 1-2)

**Objective**: Validate telemetry collection, overhead, and privacy compliance on 10% of runs.

**Configuration**:
```bash
# Enable telemetry on pilot runs
export DOCSTOKG_ENABLE_TELEMETRY=1

# Standard pilot invocation
./.venv/bin/python -m DocsToKG.ContentDownload.cli \
  --topic "pilot-test" \
  --max 100 \
  --out runs/pilot_week1 \
  --workers 4
```

**Success Criteria**:
- ✅ Telemetry database created with all 6 tables
- ✅ Zero telemetry-related errors in logs
- ✅ No performance degradation (< 5% overhead)
- ✅ URL hashing verified (no raw URLs in telemetry)
- ✅ CLI commands working (summary, export, query)

**Monitoring**:
- Watch for SQLite lock errors
- Monitor disk usage (telemetry DB size)
- Sample SLO computations for correctness

**Rollback**: Set `DOCSTOKG_ENABLE_TELEMETRY=0`

### Phase B: Ramp (Week 3-4)

**Objective**: Expand to 50% of runs, tune SLO thresholds, validate Prometheus pipeline.

**Configuration**:
```bash
# Enable telemetry broadly (50% of runs)
export DOCSTOKG_ENABLE_TELEMETRY=1

# Start Prometheus exporter
./.venv/bin/python -m DocsToKG.ContentDownload.telemetry_prom_exporter \
  --db runs/ramp_run/manifest.sqlite3 \
  --port 9108 \
  --poll 10 &

# Point Prometheus datasource at http://localhost:9108/metrics
```

**Success Criteria**:
- ✅ SLI metrics align with observations (Yield ~85%, TTFP p95 ~20s, etc.)
- ✅ No SLO threshold surprises
- ✅ Prometheus metrics scrape successfully
- ✅ Parquet export working (duckdb -c "SELECT COUNT(*) FROM 'parquet/http_events.parquet'")

**Threshold Tuning**:
```bash
# Query baseline SLIs from pilot week
sqlite3 runs/pilot_week1/manifest.sqlite3 << 'SQL'
SELECT 
  100.0*SUM(CASE WHEN sha256 IS NOT NULL THEN 1 ELSE 0 END)/COUNT(*) as yield_pct,
  -- (add other SLI queries from AGENTS.md)
FROM downloads;
SQL

# Adjust thresholds in telemetry summary if needed
# (edit SLO dict in cli_telemetry.py if deploying custom targets)
```

**Rollback**: Scale runs back to 10%, revert Prometheus datasource

### Phase C: Production (Week 5+)

**Objective**: Full deployment with monitoring, alerting, and operational runbooks.

**Configuration**:
```bash
# Production: Enable for all runs
export DOCSTOKG_ENABLE_TELEMETRY=1
export DOCSTOKG_TELEMETRY_DB_PATH=/var/lib/docstokg/manifest.sqlite3

# Start Prometheus exporter (persistent)
# (systemd service or long-running process)
nohup ./.venv/bin/python -m DocsToKG.ContentDownload.telemetry_prom_exporter \
  --db /var/lib/docstokg/manifest.sqlite3 \
  --port 9108 \
  --poll 10 > /var/log/docstokg-prom.log 2>&1 &

# Setup Grafana datasource → http://prometheus:9090
# (Prometheus scrapes http://docstokg-host:9108/metrics)
```

**Success Criteria**:
- ✅ 100% of runs capturing telemetry
- ✅ Grafana dashboards displaying SLIs in real-time
- ✅ Alerts firing on SLO violations
- ✅ No operational incidents caused by telemetry

**Alerting Rules** (Prometheus/AlertManager):
```yaml
# Alert if Yield drops below 85%
- alert: DocsToKGYieldLow
  expr: docstokg_run_yield_pct < 85
  for: 30m
  labels:
    severity: warning
  annotations:
    summary: "DocsToKG yield {{ $value | humanize }}% (target ≥85%)"

# Alert if TTFP p95 exceeds 20s
- alert: DocsToKGTTFPHigh
  expr: docstokg_run_ttfp_ms{quantile="p95"} > 20000
  for: 30m
  labels:
    severity: warning
  annotations:
    summary: "DocsToKG TTFP p95 {{ $value | humanize }}ms (target ≤20s)"

# Alert if HTTP 429 ratio exceeds 2%
- alert: DocsToKGRateLimitRatio
  expr: docstokg_host_http429_ratio > 2
  for: 15m
  labels:
    severity: warning
  annotations:
    summary: "{{ $labels.host }} rate limit ratio {{ $value | humanize }}%"
```

---

## Operational Runbooks

### Runbook 1: High Yield Drop (>5% decrease)

**Symptom**: Yield drops from 87% to 82% week-over-week

**Diagnosis**:
```bash
# Query yield by resolver
sqlite3 manifest.sqlite3 << 'SQL'
SELECT resolver_name,
       100.0*SUM(CASE WHEN sha256 IS NOT NULL THEN 1 ELSE 0 END)/COUNT(*) as yield_pct
FROM (
  SELECT d.artifact_id, d.sha256, json_extract(d.metadata, '$.resolver_name') as resolver_name
  FROM downloads d
  WHERE d.run_id = 'current_run_id'
)
GROUP BY resolver_name
ORDER BY yield_pct;
SQL

# Check if specific resolver degraded
jq '.[] | select(.record_type=="attempt") | {resolver_name, reason}' manifest.jsonl | \
  jq -s 'group_by(.resolver_name) | map({resolver: .[0].resolver_name, failure_count: length})'
```

**Remediation**:
- Check resolver endpoint health (HTTP 5xx spike?)
- Verify rate limiter is not too aggressive
- Increase retry budget if timeouts increased
- Check circuit breaker state (is it open?)

### Runbook 2: High TTFP p95 (>20s)

**Symptom**: Time-to-first-PDF p95 jumps to 45s

**Diagnosis**:
```bash
# Query per-resolver TTFP
sqlite3 manifest.sqlite3 << 'SQL'
WITH first_attempt AS (
  SELECT artifact_id, MIN(ts) ts_first, json_extract(metadata, '$.resolver_name') resolver
  FROM fallback_attempts WHERE run_id = 'current_run_id'
  GROUP BY artifact_id
),
first_success AS (
  SELECT artifact_id, MIN(ts) ts_success
  FROM fallback_attempts WHERE run_id = 'current_run_id' AND outcome='success'
  GROUP BY artifact_id
),
ttfp AS (
  SELECT f.resolver, (s.ts_success - f.ts_first)*1000.0 ms
  FROM first_success s JOIN first_attempt f USING(artifact_id)
)
SELECT resolver,
       PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY ms) as p95_ms,
       COUNT(*) as attempts
FROM ttfp
GROUP BY resolver
ORDER BY p95_ms DESC;
SQL
```

**Remediation**:
- Check which resolver is slowest
- Increase timeout for that resolver
- Reorder resolvers (fast ones first)
- Check network latency to resolver endpoints

### Runbook 3: Circuit Breaker Repeatedly Opening

**Symptom**: Breaker for `api.example.org` opens 5+ times in an hour

**Diagnosis**:
```bash
# Check breaker event history
sqlite3 manifest.sqlite3 << 'SQL'
SELECT ts, old_state, new_state, details
FROM breaker_transitions
WHERE host = 'api.example.org' AND new_state LIKE '%OPEN%'
ORDER BY ts DESC
LIMIT 20;

# Check 429 ratio from that host
SELECT 100.0*SUM(CASE WHEN status=429 THEN 1 ELSE 0 END)/COUNT(*)
FROM http_events
WHERE host='api.example.org' AND from_cache!=1;
SQL
```

**Remediation**:
- If many 429s: Reduce rate limiter RPS for that host
- If 5xx errors: Contact host operator or mark as degraded
- Increase `fail_max` threshold in breaker config (less sensitive)
- Increase `reset_timeout_s` (longer recovery period)

### Runbook 4: SQLite Database Growing Too Large

**Symptom**: `manifest.sqlite3` > 5GB, queries slow

**Solution**:
```bash
# Export Parquet for long-term storage
./.venv/bin/python -m DocsToKG.ContentDownload.cli telemetry export \
  --db manifest.sqlite3 \
  --out archive/parquet_$(date +%Y%m%d)/

# Vacuum and checkpoint to reclaim space
sqlite3 manifest.sqlite3 << 'SQL'
VACUUM;
PRAGMA wal_checkpoint(RESTART);
SQL

# Monitor: Keep SQLite < 2GB for operational database
# Archive older runs to Parquet after 14 days
```

---

## Environment Variables

Set these in production environment:

```bash
# Enable telemetry (default: disabled for backward compat)
export DOCSTOKG_ENABLE_TELEMETRY=1

# Optional: specify telemetry database location
export DOCSTOKG_TELEMETRY_DB_PATH=/var/lib/docstokg/manifest.sqlite3

# Optional: enable additional sinks (JSON, CSV, Parquet export)
export DOCSTOKG_TELEMETRY_JSONL_PATH=/var/log/docstokg/telemetry.jsonl
export DOCSTOKG_TELEMETRY_CSV_PATH=/var/log/docstokg/telemetry.csv

# Privacy: Ensure URL hashing is on (default: true)
export DOCSTOKG_TELEMETRY_HASH_URLS=1

# Rate limiter backend for multi-worker (default: memory)
# Options: memory (single-process), sqlite (multi-process on same host), redis/postgres (distributed)
export DOCSTOKG_RATE_BACKEND=sqlite:/var/lib/docstokg/ratelimit.sqlite

# Circuit breaker cooldown store (default: memory)
export DOCSTOKG_BREAKER_COOLDOWN_STORE=sqlite:/var/lib/docstokg/breakers.sqlite
```

---

## CLI Examples for Operations

```bash
# Evaluate SLOs after a run
./.venv/bin/python -m DocsToKG.ContentDownload.cli telemetry summary \
  --db runs/production_20250121/manifest.sqlite3 \
  --run $(jq -r '.run_id' runs/production_20250121/manifest.summary.json) \
  --output-format json > slo_results.json

# Export for trend analysis (Parquet + DuckDB)
./.venv/bin/python -m DocsToKG.ContentDownload.cli telemetry export \
  --db runs/production_20250121/manifest.sqlite3 \
  --out /archive/trends/production_20250121/

# Query specific hosts' 429 ratio
./.venv/bin/python -m DocsToKG.ContentDownload.cli telemetry query \
  --db runs/production_20250121/manifest.sqlite3 \
  --query "SELECT host, COUNT(*) requests, SUM(CASE WHEN status=429 THEN 1 ELSE 0 END) http_429 FROM http_events GROUP BY host ORDER BY http_429 DESC" \
  --format table

# Start Prometheus exporter for dashboards
./.venv/bin/python -m DocsToKG.ContentDownload.telemetry_prom_exporter \
  --db /var/lib/docstokg/manifest.sqlite3 \
  --port 9108 \
  --poll 10 &
```

---

## Monitoring Dashboard (Grafana)

### Recommended Panels

1. **Yield % (Gauge)**
   - Query: `docstokg_run_yield_pct`
   - Threshold: Green ≥85%, Yellow 80-85%, Red <80%

2. **TTFP p95 (Gauge + Spark)**
   - Query: `docstokg_run_ttfp_ms{quantile="p95"}`
   - Threshold: Green ≤20s, Yellow 15-20s, Red >20s

3. **Cache Hit % (Gauge)**
   - Query: `docstokg_run_cache_hit_pct`
   - Threshold: Green ≥60%, Yellow 50-60%, Red <50%

4. **HTTP 429 Ratio by Host (Table)**
   - Query: `docstokg_host_http429_ratio`
   - Sort by ratio descending
   - Alert if any host >2%

5. **Breaker Opens/Hour (Time Series)**
   - Query: `rate(docstokg_breaker_open_events_total[1h])`
   - Alert if > 12 opens/hour per host

---

## Backup & Recovery

### Backup Strategy

```bash
# Daily backup of telemetry database (to S3/NAS)
0 2 * * * (cd /var/lib/docstokg && tar -czf /backup/docstokg-$(date +\%Y\%m\%d).tar.gz manifest.sqlite3 && aws s3 cp /backup/docstokg-$(date +\%Y\%m\%d).tar.gz s3://my-backup-bucket/)

# Weekly Parquet export for long-term storage
0 3 * * 0 (./.venv/bin/python -m DocsToKG.ContentDownload.cli telemetry export --db /var/lib/docstokg/manifest.sqlite3 --out /archive/parquet/week_$(date +\%Y\%m\%d)/)
```

### Recovery Procedure

If telemetry database is corrupted:

```bash
# 1. Stop all download processes
pkill -f "DocsToKG.ContentDownload.cli"

# 2. Restore from backup
tar -xzf /backup/docstokg-20250120.tar.gz -C /var/lib/docstokg/

# 3. Verify integrity
sqlite3 /var/lib/docstokg/manifest.sqlite3 "PRAGMA integrity_check;"

# 4. Resume downloads
./.venv/bin/python -m DocsToKG.ContentDownload.cli --resume-from ...
```

---

## Success Metrics

After 4 weeks of production:

- ✅ **Zero telemetry-related incidents** (outages, performance degradation)
- ✅ **SLO violations < 1% of time** (within error budget)
- ✅ **Alerts firing correctly** (SLO violations caught within 5 minutes)
- ✅ **Dashboards informative** (team makes decisions from SLI data)
- ✅ **Operational runbooks effective** (issues diagnosed and resolved in <30 min)

---

## Rollback Plan

If critical issues arise:

```bash
# Immediate: Disable telemetry
export DOCSTOKG_ENABLE_TELEMETRY=0

# Kill Prometheus exporter
pkill -f "telemetry_prom_exporter"

# Resume normal operations (telemetry will be no-op)
./.venv/bin/python -m DocsToKG.ContentDownload.cli ...
```

**No code changes needed** — telemetry is opt-in with environment variable.

---

## Support & Escalation

**Issues with telemetry**:
- Check logs: `grep -i telemetry manifest.log`
- Verify SQLite: `sqlite3 manifest.sqlite3 ".tables"`
- Run validation: `python3 -m DocsToKG.ContentDownload.telemetry`

**Issues with SLO thresholds**:
- Query baseline from pilot: `SELECT * FROM run_summary WHERE run_id = 'pilot_run'`
- Adjust thresholds in `cli_telemetry.py` SLO dict
- Redeploy with new thresholds

**Performance concerns**:
- Profile telemetry overhead: Run with `time` command
- Disable JSON/CSV sinks if not needed (SQLite only is fastest)
- Consider sampling (emit only 10% of cache hits, all errors)

---

## Timeline

| Week | Activity | Success Criteria |
|------|----------|------------------|
| 1-2 | Pilot (10% runs) | Zero errors, <5% overhead, privacy verified |
| 3-4 | Ramp (50% runs) | SLI alignment, Prometheus working, thresholds tuned |
| 5+ | Production (100%) | Dashboards live, alerts working, runbooks effective |

---

## Post-Deployment

**After 4 weeks in production:**

1. Review SLO trends (are we consistently hitting/missing targets?)
2. Tune thresholds based on reality
3. Add new SLIs if gaps identified
4. Train team on operational runbooks
5. Consider extending to other modules (OntologyDownload, etc.)

---

**Status**: ✅ **READY FOR PRODUCTION DEPLOYMENT**

**Next Step**: Execute Phase A (Pilot) with 10% of runs for 1-2 weeks.

