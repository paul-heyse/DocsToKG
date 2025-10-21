Absolutely — here are the **two helper pieces** I promised, ready to paste into your repo:

1. a tiny **SQLite → Prometheus exporter** you can run as a sidecar and point Grafana at, and
2. a **DuckDB-based Parquet exporter** (both CLI and one-liner SQL) for easy long-term trend analysis.

I’ve kept them “production-practical”: low cardinality, explicit labels, safe polling, and no blocking queries.

---

## 1) SQLite → Prometheus exporter (Grafana-ready)

> File: `src/DocsToKG/ContentDownload/telemetry_prom_exporter.py`

* Exposes `/metrics` on a configurable port
* Polls the run database every N seconds
* Publishes a small set of **stable, low-cardinality** metrics:

  * `docstokg_run_yield_pct{run_id}`
  * `docstokg_run_ttfp_ms{run_id,quantile="p50|p95"}`
  * `docstokg_run_cache_hit_pct{run_id}`
  * `docstokg_run_rate_delay_p95_ms{run_id,role="metadata|artifact"}`
  * `docstokg_host_http429_ratio{run_id,host}`
  * `docstokg_breaker_open_events_total{run_id,host}` (counter)
  * `docstokg_run_dedupe_saved_mb{run_id}`
  * `docstokg_run_corruption_count{run_id}`

It uses `run_summary` if present; otherwise it computes on the fly with the same SQL from the plan. For per-host 429s it queries `http_events`.

```python
from __future__ import annotations
import argparse, sqlite3, threading, time
from typing import Optional, Dict
from prometheus_client import start_http_server, Gauge, Counter

# -------- Prom metrics (static registry) --------
RUN_YIELD_PCT         = Gauge("docstokg_run_yield_pct", "Yield percent for a run", ["run_id"])
RUN_TTFP_MS           = Gauge("docstokg_run_ttfp_ms", "Time-to-first-PDF (ms) quantiles", ["run_id", "quantile"])
RUN_CACHE_HIT_PCT     = Gauge("docstokg_run_cache_hit_pct", "Metadata cache hit percent", ["run_id"])
RUN_RATE_P95_MS       = Gauge("docstokg_run_rate_delay_p95_ms", "Rate delay p95 (ms)", ["run_id","role"])
HOST_HTTP429_RATIO    = Gauge("docstokg_host_http429_ratio", "HTTP 429 ratio (%) by host", ["run_id","host"])
BREAKER_OPENS_TOTAL   = Counter("docstokg_breaker_open_events_total", "Breaker state changes to OPEN", ["run_id","host"])
RUN_DEDUPE_SAVED_MB   = Gauge("docstokg_run_dedupe_saved_mb", "Dedupe savings in MB for a run", ["run_id"])
RUN_CORRUPTION_COUNT  = Gauge("docstokg_run_corruption_count", "Rows missing final/hash (should be 0)", ["run_id"])

def _fetchone(cx, sql, args=()):
    row = cx.execute(sql, args).fetchone()
    return row[0] if row and row[0] is not None else None

def _latest_run_id(cx) -> Optional[str]:
    # Prefer explicit run_summary ordering; fallback to max ts in http_events
    row = cx.execute("SELECT run_id FROM run_summary ORDER BY finished_at DESC LIMIT 1").fetchone()
    if row: return row[0]
    row = cx.execute("SELECT run_id FROM http_events ORDER BY ts DESC LIMIT 1").fetchone()
    return row[0] if row else None

def _compute_run_summary(cx, run_id: str) -> Dict[str, float]:
    out: Dict[str,float] = {}

    # Yield
    tot, ok = cx.execute("""
      SELECT COUNT(*), SUM(CASE WHEN sha256 IS NOT NULL AND final_path IS NOT NULL THEN 1 ELSE 0 END)
      FROM downloads WHERE run_id=?""", (run_id,)).fetchone()
    tot = tot or 0; ok = ok or 0
    out["yield_pct"] = 100.0 * ok / max(1, tot)

    # TTFP p50/p95 from fallback_attempts
    row = cx.execute("""
    WITH s AS (SELECT artifact_id, MIN(ts) ts_s FROM fallback_attempts WHERE run_id=? AND outcome='success' GROUP BY 1),
         f AS (SELECT artifact_id, MIN(ts) ts_f FROM fallback_attempts WHERE run_id=? GROUP BY 1),
         d AS (SELECT (s.ts_s - f.ts_f)*1000.0 ms FROM s JOIN f USING(artifact_id))
    SELECT
      (SELECT ms FROM d ORDER BY ms LIMIT 1 OFFSET (SELECT CAST(COUNT(*)*0.50 AS INT) FROM d)) AS p50,
      (SELECT ms FROM d ORDER BY ms LIMIT 1 OFFSET (SELECT CAST(COUNT(*)*0.95 AS INT) FROM d)) AS p95
    """, (run_id, run_id)).fetchone() or (0,0)
    out["ttfp_p50_ms"], out["ttfp_p95_ms"] = int(row[0] or 0), int(row[1] or 0)

    # Cache hit (metadata)
    out["cache_hit_pct"] = _fetchone(cx, """
      SELECT 100.0*SUM(CASE WHEN from_cache=1 THEN 1 ELSE 0 END)/COUNT(*)
      FROM http_events WHERE run_id=? AND role='metadata'""", (run_id,)) or 0.0

    # Rate delay p95 (metadata; artifact optional if you add those rows)
    out["rate_p95_ms_meta"] = _fetchone(cx, """
      WITH x AS (SELECT rate_delay_ms FROM http_events WHERE run_id=? AND role='metadata' AND rate_delay_ms IS NOT NULL)
      SELECT (SELECT rate_delay_ms FROM x ORDER BY rate_delay_ms LIMIT 1
              OFFSET (SELECT CAST(COUNT(*)*0.95 AS INT) FROM x))""", (run_id,)) or 0

    # Dedupe saved MB
    out["saved_mb"] = _fetchone(cx, """
      SELECT SUM(CASE dedupe_action WHEN 'hardlink' THEN content_length
                                    WHEN 'copy'     THEN content_length
                                    WHEN 'skipped'  THEN content_length
                                    ELSE 0 END)/(1024.0*1024.0)
      FROM downloads WHERE run_id=? AND content_length IS NOT NULL""", (run_id,)) or 0.0

    # Corruption count
    out["corruption"] = _fetchone(cx,
      "SELECT COUNT(*) FROM downloads WHERE run_id=? AND (final_path IS NULL OR sha256 IS NULL)", (run_id,)) or 0
    return out

def _emit_host_429(cx, run_id: str):
    cur = cx.execute("""
        SELECT host,
               100.0*SUM(CASE WHEN status=429 THEN 1 ELSE 0 END)
               / NULLIF(SUM(CASE WHEN from_cache!=1 THEN 1 ELSE 0 END),0) AS pct
        FROM http_events WHERE run_id=? GROUP BY host""", (run_id,))
    for host, pct in cur.fetchall():
        HOST_HTTP429_RATIO.labels(run_id=run_id, host=host).set(float(pct or 0.0))

def _emit_breaker_opens(cx, run_id: str, seen):
    cur = cx.execute("""
      SELECT host, COUNT(*) FROM breaker_transitions
      WHERE run_id=? AND old_state LIKE '%CLOSED%' AND new_state LIKE '%OPEN%'
      GROUP BY host""", (run_id,))
    for host, cnt in cur.fetchall():
        key = (run_id, host)
        prev = seen.get(key, 0)
        inc = int(cnt or 0) - prev
        if inc > 0:
            BREAKER_OPENS_TOTAL.labels(run_id=run_id, host=host).inc(inc)
            seen[key] = prev + inc

def main():
    ap = argparse.ArgumentParser("docstokg-prom-exporter")
    ap.add_argument("--db", required=True, help="Path to telemetry sqlite")
    ap.add_argument("--port", type=int, default=9108)
    ap.add_argument("--poll", type=float, default=10.0, help="poll interval (s)")
    ap.add_argument("--run-id", help="If omitted, exporter uses latest run")
    args = ap.parse_args()

    start_http_server(args.port)
    seen_breaker = {}
    while True:
        try:
            cx = sqlite3.connect(args.db)
            cx.row_factory = sqlite3.Row
            run_id = args.run_id or _latest_run_id(cx)
            if run_id:
                # summary (from table if filled; otherwise compute)
                row = cx.execute("SELECT * FROM run_summary WHERE run_id=?", (run_id,)).fetchone()
                if row:
                    RUN_YIELD_PCT.labels(run_id).set(float(row["yield_pct"] or 0))
                    RUN_TTFP_MS.labels(run_id, "p50").set(int(row["ttfp_p50_ms"] or 0))
                    RUN_TTFP_MS.labels(run_id, "p95").set(int(row["ttfp_p95_ms"] or 0))
                    RUN_CACHE_HIT_PCT.labels(run_id).set(float(row["cache_hit_pct"] or 0))
                    RUN_RATE_P95_MS.labels(run_id, "metadata").set(int(row["rate_delay_p95_ms"] or 0))
                    RUN_DEDUPE_SAVED_MB.labels(run_id).set(float(row["dedupe_saved_mb"] or 0))
                    RUN_CORRUPTION_COUNT.labels(run_id).set(int(row["corruption_count"] or 0))
                else:
                    s = _compute_run_summary(cx, run_id)
                    RUN_YIELD_PCT.labels(run_id).set(s["yield_pct"])
                    RUN_TTFP_MS.labels(run_id, "p50").set(s["ttfp_p50_ms"])
                    RUN_TTFP_MS.labels(run_id, "p95").set(s["ttfp_p95_ms"])
                    RUN_CACHE_HIT_PCT.labels(run_id).set(s["cache_hit_pct"])
                    RUN_RATE_P95_MS.labels(run_id, "metadata").set(s["rate_p95_ms_meta"])
                    RUN_DEDUPE_SAVED_MB.labels(run_id).set(s["saved_mb"])
                    RUN_CORRUPTION_COUNT.labels(run_id).set(s["corruption"])

                _emit_host_429(cx, run_id)
                _emit_breaker_opens(cx, run_id, seen_breaker)
            cx.close()
        except Exception as e:
            # Keep exporter up; surface errors via logs
            print(f"[prom-exporter] poll error: {e}", flush=True)
        time.sleep(args.poll)

if __name__ == "__main__":
    main()
```

### How to run

```bash
pip install prometheus_client
python -m DocsToKG.ContentDownload.telemetry_prom_exporter \
  --db /path/to/telemetry.sqlite \
  --port 9108 --poll 10
```

**Grafana panel queries** (prometheus datasource):

* Yield: `docstokg_run_yield_pct{run_id="$run"}`
* TTFP p95: `docstokg_run_ttfp_ms{run_id="$run",quantile="p95"}`
* Cache hit: `docstokg_run_cache_hit_pct{run_id="$run"}`
* Host 429 ratio (table): `docstokg_host_http429_ratio{run_id="$run"}`
* Rate delay p95: `docstokg_run_rate_delay_p95_ms{run_id="$run",role="metadata"}`
* Breaker opens (bar): `increase(docstokg_breaker_open_events_total{run_id="$run"}[1h])`

> Tip: Treat `run_id` as a **Grafana variable** so you can flick between runs without changing panels.

---

## 2) DuckDB Parquet exporter

Export your SQLite tables to compressed Parquet for long-term trending (fast to read with DuckDB/Polars; tiny on disk).

### 2A) One-liner (DuckDB CLI)

```bash
duckdb -c "
ATTACH 'telemetry.sqlite' AS t (TYPE sqlite);
COPY (SELECT * FROM t.http_events)        TO 'parquet/http_events.parquet'        (FORMAT PARQUET, COMPRESSION ZSTD);
COPY (SELECT * FROM t.fallback_attempts)  TO 'parquet/fallback_attempts.parquet'  (FORMAT PARQUET, COMPRESSION ZSTD);
COPY (SELECT * FROM t.breaker_transitions)TO 'parquet/breaker_transitions.parquet'(FORMAT PARQUET, COMPRESSION ZSTD);
COPY (SELECT * FROM t.downloads)          TO 'parquet/downloads.parquet'          (FORMAT PARQUET, COMPRESSION ZSTD);
COPY (SELECT * FROM t.run_summary)        TO 'parquet/run_summary.parquet'        (FORMAT PARQUET, COMPRESSION ZSTD);
DETACH t;
"
```

### 2B) Python CLI (safe paths + schema drift tolerant)

> File: `src/DocsToKG/ContentDownload/telemetry_export_parquet.py`

```python
from __future__ import annotations
import argparse, duckdb, os, pathlib

TABLES = ["http_events","fallback_attempts","breaker_transitions","downloads","run_summary"]

def main():
    ap = argparse.ArgumentParser("export-parquet")
    ap.add_argument("--sqlite", required=True, help="telemetry sqlite path")
    ap.add_argument("--out", required=True, help="output dir for parquet")
    args = ap.parse_args()

    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"ATTACH '{args.sqlite}' AS t (TYPE sqlite)")
    for tbl in TABLES:
        # Skip if table missing
        exists = con.execute(f"SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='{tbl}'").fetchone()[0]
        if not exists:
            print(f"[export] skip missing table: {tbl}")
            continue
        dst = out / f"{tbl}.parquet"
        con.execute(f"COPY (SELECT * FROM t.{tbl}) TO '{dst.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        print(f"[export] wrote {dst}")
    con.execute("DETACH t")
    con.close()

if __name__ == "__main__":
    main()
```

Run:

```bash
pip install duckdb
python -m DocsToKG.ContentDownload.telemetry_export_parquet \
  --sqlite /path/to/telemetry.sqlite \
  --out /path/to/parquet/
```

### Quick DuckDB queries on Parquet

```sql
-- from duckdb shell
SELECT host, AVG(rate_delay_ms) avg_delay
FROM 'parquet/http_events.parquet'
WHERE role='metadata'
GROUP BY host ORDER BY avg_delay DESC LIMIT 10;
```

---

## (Optional) OpenTelemetry span hooks (10-line drop-in)

If you already run an OTEL collector, you can wrap the **artifact download** orchestrator with a span:

```python
# quick-otel.py (drop-in idea; keep spans coarse)
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor

def init_tracing(service_name="docstokg"):
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))  # reads OTEL_EXPORTER_OTLP_* envs
    trace.set_tracer_provider(provider)
    return trace.get_tracer(service_name)

# usage in your runner
tracer = init_tracing()
with tracer.start_as_current_span("artifact.resolve", attributes={"work_id": work_id, "artifact_id": art_id}):
    res = orchestrator.resolve_pdf(context=..., adapters=...)
```

Big win with little effort: traces stitched across layers (fallback attempts, streaming) in Grafana Tempo/Jaeger.

---

## Practical tips (to keep things sane)

* **Cardinality**: avoid raw URLs; hash them; keep `host`, `role`, `source`, and `run_id` as the primary labels.
* **Polling**: exporter defaults to 10s; safe for SQLite. If your DB is large, consider reading only recent runs or `WHERE run_id=?`.
* **Retention**: export Parquet weekly; keep SQLite recent (e.g., last 14 days) and vacuum.
* **Dashboards**: start with yield, TTFP p95, cache hit, rate delay p95 (metadata), top 429 hosts, breaker opens/hour, dedupe saved MB.

If you want, I can add a minimal **Grafana dashboard JSON** (panels prewired to these metric names) next; otherwise, you can drag these PromQLs into panels and be productive in minutes.
