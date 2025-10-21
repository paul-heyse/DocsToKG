Absolutely — here’s a **repo-shaped, junior-dev/agent-ready implementation plan** for **10) Observability & SLOs** that fits your current stack (HTTPX+Hishel+Tenacity+pyrate-limiter+pybreaker+Wayback+idempotent jobs + streaming). It gives you:

* a minimal **telemetry architecture** (what to log, where),
* **SQLite+JSONL** sinks (fast, low-touch),
* event **schemas** and indices,
* **SLIs/SLOs** with runnable SQL,
* a one-shot **summary CLI** (with pass/fail),
* optional **export to Parquet**, **Prometheus/OpenTelemetry** adapters,
* **privacy/cardinality** guardrails and performance tips.

---

## 0) Goals (what “good” looks like)

* You can answer quickly: *“why did this artifact fail?”*, *“which host is noisy?”*, *“what’s our time-to-PDF p95?”*.
* Your **SLOs** are computed per run and over time; runs **fail** CI/ops checks when SLOs regress.
* Minimal overhead; no production lock-ups; safe to run on multi-process.

---

## 1) Telemetry architecture (one bus, many streams)

Keep it simple: **one telemetry emitter** that fans out to **SQLite** and **JSONL** sinks. Every layer posts small, structured events:

* **http_events** – one row per network call (after cache decision).
* **rate_events** – limiter acquisitions/blocks.
* **breaker_transitions** – state changes & failures (pybreaker listener).
* **fallback_attempts** – each resolver/adapter try (outcome, reason).
* **downloads** – you already added streaming fields (bytes/fsync/hash/resume).
* **wayback_* ** – from your Wayback telemetry module (already built).
* **run_summary** – aggregated summary (computed at end; 1 row per run).

> Implementation: reuse your `TelemetryWayback` pattern (envelope → sinks), add small emitters per layer.

---

## 2) Event schemas (SQLite DDL) — copy/paste

> File: `src/DocsToKG/ContentDownload/telemetry_schema.sql`

```sql
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA busy_timeout=4000;

-- 2.1 HTTP calls (after cache & limiter; one row per request)
CREATE TABLE IF NOT EXISTS http_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  ts REAL NOT NULL,                   -- wall clock (time.time())
  host TEXT NOT NULL,
  role TEXT NOT NULL,                 -- metadata|landing|artifact
  method TEXT NOT NULL,               -- GET|HEAD
  status INTEGER,                     -- NULL if exception
  url_hash TEXT NOT NULL,             -- sha256 of canonical URL (to avoid PII)
  from_cache INTEGER,                 -- 0/1/NULL
  revalidated INTEGER,                -- 0/1/NULL
  stale INTEGER,                      -- 0/1/NULL (Hishel SWrV)
  retry_count INTEGER,                -- Tenacity attempts-1
  retry_after_s INTEGER,              -- if honored
  rate_delay_ms INTEGER,              -- limiter wait
  breaker_state TEXT,                 -- closed|half_open|open
  breaker_recorded TEXT,              -- success|failure|none
  elapsed_ms INTEGER,                 -- end-to-end
  error TEXT                          -- short exception class if any
);
CREATE INDEX IF NOT EXISTS idx_http_run ON http_events(run_id);
CREATE INDEX IF NOT EXISTS idx_http_host ON http_events(host);
CREATE INDEX IF NOT EXISTS idx_http_role ON http_events(role);

-- 2.2 Rate limiter events
CREATE TABLE IF NOT EXISTS rate_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  ts REAL NOT NULL,
  host TEXT NOT NULL,
  role TEXT NOT NULL,
  action TEXT NOT NULL,               -- acquire|block|head_skip
  delay_ms INTEGER,
  max_delay_ms INTEGER
);
CREATE INDEX IF NOT EXISTS idx_rate_run ON rate_events(run_id);
CREATE INDEX IF NOT EXISTS idx_rate_host_role ON rate_events(host, role);

-- 2.3 Breaker transitions (listener)
CREATE TABLE IF NOT EXISTS breaker_transitions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  ts REAL NOT NULL,
  host TEXT NOT NULL,
  scope TEXT NOT NULL,                -- host|resolver
  old_state TEXT NOT NULL,
  new_state TEXT NOT NULL,
  reset_timeout_s INTEGER
);
CREATE INDEX IF NOT EXISTS idx_brk_run ON breaker_transitions(run_id);
CREATE INDEX IF NOT EXISTS idx_brk_host ON breaker_transitions(host);

-- 2.4 Fallback attempts (one row per adapter try)
CREATE TABLE IF NOT EXISTS fallback_attempts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  ts REAL NOT NULL,
  work_id TEXT,
  artifact_id TEXT,
  tier TEXT NOT NULL,
  source TEXT NOT NULL,               -- e.g., unpaywall_pdf
  host TEXT,
  outcome TEXT NOT NULL,              -- success|retryable|nonretryable|timeout|skipped|error|no_pdf
  reason TEXT,                        -- short code
  status INTEGER,
  elapsed_ms INTEGER
);
CREATE INDEX IF NOT EXISTS idx_fb_run ON fallback_attempts(run_id);
CREATE INDEX IF NOT EXISTS idx_fb_source ON fallback_attempts(source);

-- 2.5 Downloads table already has new columns (from migration #7). Ensure indices exist:
CREATE INDEX IF NOT EXISTS idx_dl_sha ON downloads(sha256);
CREATE INDEX IF NOT EXISTS idx_dl_final ON downloads(final_path);
CREATE INDEX IF NOT EXISTS idx_dl_dedupe ON downloads(dedupe_action);

-- 2.6 Run summary (single row per run after aggregation)
CREATE TABLE IF NOT EXISTS run_summary (
  run_id TEXT PRIMARY KEY,
  started_at REAL,
  finished_at REAL,
  artifacts_total INTEGER,
  artifacts_success INTEGER,
  yield_pct REAL,
  ttfp_p50_ms INTEGER,                -- time-to-first-pdf
  ttfp_p95_ms INTEGER,
  cache_hit_pct REAL,                 -- metadata cache hit rate
  rate_delay_p95_ms INTEGER,
  http_429_ratio REAL,
  breaker_open_events INTEGER,
  dedupe_saved_mb REAL,               -- bytes saved / (1024*1024)
  corruption_count INTEGER
);
```

> JSONL mirror: log the same fields per event in `telemetry/*.jsonl` for tailing and human review. Keep lines short.

---

## 3) Instrumentation (where to emit)

* **Networking hub** (HTTPX wrapper):

  * On every request: emit **http_events** with envelope +:

    * `host`, `role`, `method`, `status`, `elapsed_ms`, `retry_count`, `retry_after_s`, `from_cache`, `revalidated`, `stale`, `rate_delay_ms`, `breaker_state`, `breaker_recorded`, `error`.
  * *Note*: hash the canonical URL (`url_hash = sha256(canonical_url)`) – do **not** store raw URLs.

* **RateLimitedTransport**:

  * On acquire: `rate_events(action="acquire", delay_ms, max_delay_ms)`.
  * On bounded wait exceed: `rate_events(action="block")`.
  * On HEAD skip: `rate_events(action="head_skip")`.

* **Breaker listener**:

  * `breaker_transitions(state_change)`.

* **Fallback orchestrator**:

  * On each adapter attempt completion: `fallback_attempts(...)`.

* **Streaming**:

  * You already write to **downloads** (bytes/fsync/hash/resume). Keep it.

* **Wayback**:

  * You already emit `wayback_*` events.

All emitters should pass `{run_id, ts}` automatically (envelope).

---

## 4) SLIs and SLOs (definitions + SQL you can run)

### Core SLIs

1. **Yield** = successful artifacts / attempted artifacts.
2. **TTFP** (Time-to-first-PDF) p50/p95 = median/95th of time from **first attempt** on an artifact to **first successful fallback_attempt with outcome='success'**.
3. **HTTP 429 ratio** per host = 429 / net requests.
4. **Metadata cache hit rate** = hits / (hits + network) for `role='metadata'`.
5. **Rate delay p95** per host+role.
6. **Breaker open count** per host (state_change OPEN).
7. **Dedupe saved MB** = sum(bytes avoided via hardlink/copy/skipped).
8. **Corruption count** = number of rows where final sha256 re-hash mismatched or partial rename detected (should be 0).

### SLO targets (tune to your environment)

* Yield ≥ **85%** overall (or per corpus).
* TTFP p50 ≤ **3000 ms**, p95 ≤ **20,000 ms**.
* 429 ratio per host ≤ **2%** (Wayback ≤ 5%).
* Metadata cache hit ≥ **60%**.
* Rate delay p95 ≤ **250 ms** (metadata) / **2000 ms** (artifact).
* Breaker opens/hour per host ≤ **N** (start with 12).
* Corruption = **0**.

### Example SQL (copy/paste)

**Yield & attempts (downloads table)**

```sql
-- artifacts_total / artifacts_success
SELECT
  COUNT(*) AS artifacts_total,
  SUM(CASE WHEN sha256 IS NOT NULL AND final_path IS NOT NULL THEN 1 ELSE 0 END) AS artifacts_success,
  ROUND(100.0 * SUM(CASE WHEN sha256 IS NOT NULL AND final_path IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS yield_pct
FROM downloads
WHERE run_id = :run_id;
```

**TTFP p50/p95 (ms) from fallback attempts**

```sql
WITH first_success AS (
  SELECT artifact_id, MIN(ts) AS ts_success
  FROM fallback_attempts
  WHERE run_id = :run_id AND outcome='success'
  GROUP BY artifact_id
),
first_try AS (
  SELECT artifact_id, MIN(ts) AS ts_first
  FROM fallback_attempts
  WHERE run_id = :run_id
  GROUP BY artifact_id
),
ttfp AS (
  SELECT (s.ts_success - f.ts_first)*1000.0 AS ttfp_ms
  FROM first_success s JOIN first_try f USING(artifact_id)
)
SELECT
  (SELECT ttfp_ms FROM ttfp ORDER BY ttfp_ms LIMIT 1 OFFSET (SELECT COUNT(*)*0.50 FROM ttfp)) AS p50_ms,
  (SELECT ttfp_ms FROM ttfp ORDER BY ttfp_ms LIMIT 1 OFFSET (SELECT COUNT(*)*0.95 FROM ttfp)) AS p95_ms;
```

**HTTP 429 ratio per host (net only)**

```sql
SELECT host,
       ROUND(100.0 * SUM(CASE WHEN status=429 THEN 1 ELSE 0 END) / NULLIF(SUM(CASE WHEN from_cache IS NOT 1 THEN 1 ELSE 0 END),0), 2)
       AS pct_429
FROM http_events
WHERE run_id=:run_id
GROUP BY host
ORDER BY pct_429 DESC;
```

**Metadata cache hit rate**

```sql
SELECT ROUND(100.0 * SUM(CASE WHEN from_cache=1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS cache_hit_pct
FROM http_events
WHERE run_id=:run_id AND role='metadata';
```

**Rate delay p95**

```sql
WITH x AS (
  SELECT rate_delay_ms FROM http_events WHERE run_id=:run_id AND rate_delay_ms IS NOT NULL
)
SELECT
  (SELECT rate_delay_ms FROM x ORDER BY rate_delay_ms LIMIT 1 OFFSET (SELECT COUNT(*)*0.95 FROM x)) AS rate_delay_p95_ms;
```

**Breaker opens per host**

```sql
SELECT host, COUNT(*) AS opens
FROM breaker_transitions
WHERE run_id=:run_id AND old_state LIKE '%CLOSED%' AND new_state LIKE '%OPEN%'
GROUP BY host ORDER BY opens DESC;
```

**Dedupe savings (MB)**

```sql
SELECT ROUND(SUM(
  CASE dedupe_action
    WHEN 'hardlink' THEN content_length
    WHEN 'copy' THEN content_length
    WHEN 'skipped' THEN content_length
    ELSE 0
  END) / (1024.0*1024.0), 2) AS saved_mb
FROM downloads
WHERE run_id=:run_id AND content_length IS NOT NULL;
```

**Corruption (should be zero)**

```sql
SELECT COUNT(*) AS corruption_count
FROM downloads
WHERE run_id=:run_id AND (final_path IS NULL OR sha256 IS NULL);
-- If you implement a post-verify rehash, add mismatches here.
```

---

## 5) One-shot summary CLI (with SLO pass/fail)

> File: `src/DocsToKG/ContentDownload/cli_telemetry_summary.py`

```python
import argparse, sqlite3, sys, json, time

SLO = {
    "yield_pct_min": 85.0,
    "ttfp_p50_ms_max": 3000,
    "ttfp_p95_ms_max": 20000,
    "cache_hit_pct_min": 60.0,
    "rate_delay_p95_ms_max": 250,       # for metadata; set separate for artifact if you want
    "http429_pct_max": 2.0,
    "corruption_max": 0,
}

def load_one(conn, sql, params):
    cur = conn.execute(sql, params); row = cur.fetchone()
    return row[0] if row and row[0] is not None else None

def summarize(db_path: str, run_id: str) -> int:
    cx = sqlite3.connect(db_path); cx.row_factory = sqlite3.Row
    # Yield
    y = cx.execute("""
      SELECT COUNT(*) AS tot,
             SUM(CASE WHEN sha256 IS NOT NULL AND final_path IS NOT NULL THEN 1 ELSE 0 END) AS ok
      FROM downloads WHERE run_id=?""", (run_id,)).fetchone()
    yield_pct = 100.0 * (y["ok"] or 0) / max(1, (y["tot"] or 1))

    # TTFP
    ttfp = cx.execute("""
    WITH s AS (SELECT artifact_id, MIN(ts) ts_s FROM fallback_attempts WHERE run_id=? AND outcome='success' GROUP BY 1),
         f AS (SELECT artifact_id, MIN(ts) ts_f FROM fallback_attempts WHERE run_id=? GROUP BY 1),
         d AS (SELECT (s.ts_s - f.ts_f)*1000.0 ms FROM s JOIN f USING(artifact_id))
    SELECT
      (SELECT ms FROM d ORDER BY ms LIMIT 1 OFFSET (SELECT COUNT(*)*0.5 FROM d)) p50,
      (SELECT ms FROM d ORDER BY ms LIMIT 1 OFFSET (SELECT COUNT(*)*0.95 FROM d)) p95
    """, (run_id, run_id)).fetchone() or {"p50": None, "p95": None}

    # Cache hit
    cache_hit_pct = load_one(cx, """
      SELECT 100.0*SUM(CASE WHEN from_cache=1 THEN 1 ELSE 0 END)/COUNT(*)
      FROM http_events WHERE run_id=? AND role='metadata'""", (run_id,)) or 0.0

    # Rate delay p95 (metadata)
    rate_p95 = load_one(cx, """
      WITH x AS (SELECT rate_delay_ms FROM http_events WHERE run_id=? AND role='metadata' AND rate_delay_ms IS NOT NULL)
      SELECT (SELECT rate_delay_ms FROM x ORDER BY rate_delay_ms LIMIT 1 OFFSET (SELECT COUNT(*)*0.95 FROM x))""", (run_id,)) or 0

    # 429 overall
    http429 = load_one(cx, """
      SELECT 100.0*SUM(CASE WHEN status=429 THEN 1 ELSE 0 END)/NULLIF(SUM(CASE WHEN from_cache!=1 THEN 1 ELSE 0 END),0)
      FROM http_events WHERE run_id=?""", (run_id,)) or 0.0

    # Dedupe saved
    saved_mb = load_one(cx, """
      SELECT SUM(CASE dedupe_action WHEN 'hardlink' THEN content_length
                                    WHEN 'copy'     THEN content_length
                                    WHEN 'skipped'  THEN content_length
                                    ELSE 0 END)/(1024.0*1024.0)
      FROM downloads WHERE run_id=? AND content_length IS NOT NULL""", (run_id,)) or 0.0

    # Corruption
    corruption = load_one(cx, "SELECT COUNT(*) FROM downloads WHERE run_id=? AND (final_path IS NULL OR sha256 IS NULL)", (run_id,)) or 0

    summary = {
      "run_id": run_id,
      "yield_pct": round(yield_pct,2),
      "ttfp_p50_ms": int(ttfp["p50"] or 0),
      "ttfp_p95_ms": int(ttfp["p95"] or 0),
      "cache_hit_pct": round(cache_hit_pct,1),
      "rate_delay_p95_ms": int(rate_p95),
      "http_429_pct": round(http429,2),
      "dedupe_saved_mb": round(saved_mb,2),
      "corruption_count": int(corruption),
      "finished_at": time.time(),
    }
    print(json.dumps(summary, indent=2))

    # SLO evaluation → nonzero exit on fail
    fail = (
      summary["yield_pct"] < SLO["yield_pct_min"] or
      summary["ttfp_p50_ms"] > SLO["ttfp_p50_ms_max"] or
      summary["ttfp_p95_ms"] > SLO["ttfp_p95_ms_max"] or
      summary["cache_hit_pct"] < SLO["cache_hit_pct_min"] or
      summary["rate_delay_p95_ms"] > SLO["rate_delay_p95_ms_max"] or
      summary["http_429_pct"] > SLO["http429_pct_max"] or
      summary["corruption_count"] > SLO["corruption_max"]
    )
    return 1 if fail else 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser("telemetry-summary")
    ap.add_argument("--db", required=True)
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    sys.exit(summarize(args.db, args.run))
```

**Usage:** call at run end:

```
python -m DocsToKG.ContentDownload.cli_telemetry_summary --db telemetry/run.sqlite --run <run_id>
```

* It prints JSON summary and exits **1** if any SLO failed → you can fail a CI job or Ops check.

---

## 6) Optional exports & metrics backends

* **Parquet export**: at run end, use DuckDB/Polars/Pandas to export tables `http_events`, `downloads`, `fallback_attempts`, `breaker_transitions` to Parquet for long-term trend analysis.
* **Prometheus**: expose counters/histograms for:

  * `http_requests_total{host,role,status}`
  * `http_request_duration_seconds_bucket{host,role}`
  * `cache_hits_total`, `cache_revalidated_total`
  * `rate_acquire_total`, `rate_block_total`
  * `breaker_opens_total{host}`
* **OpenTelemetry**: instrument the orchestrator and streamer as spans; attach attributes (host, role, adapter, status) and events (retry, breaker open). Keep spans coarse (per artifact) to avoid cardinality blow-ups.

---

## 7) Privacy / cardinality / performance guardrails

* **No raw URLs**: store only `url_hash` in telemetry; store raw URLs in the manifest (already part of your DB, but consider hashing there too).
* **Cap “reason”** codes** to a short enum (document them).
* **Indexes**: only add where queries depend on them (provided above). Avoid indexing `url_hash` unless you specifically join on it.
* **Sampling**: For very chatty layers, allow sampling (e.g., only every Nth `http_events` with `status=200`). Keep errors and 429s unsampled.
* **WAL checkpoint** after summary to keep DB size in check.

---

## 8) Tuning: what to look at after first week

* Hosts with **high 429 ratio** → reduce limiter RPS (AIMD) or increase max_delay; honor Retry-After more aggressively.
* Low **cache hit** → verify Hishel controller & per-host TTL; look for `Vary` pollution.
* High **breaker opens** → widen `reset_timeout_s`, raise `success_threshold`, improve rate-limiter smoothing.
* High **TTFP p95** → enable hedging (closed-form) for those publishers, or reorder fallback weight.
* **Dedupe saved MB** low → ensure url→hash is populated on success; turn on hardlink.

---

### Definition of Done

* All layers emit to SQLite & JSONL without noticeable overhead.
* A single **CLI summary** prints SLIs and returns **nonzero** when SLOs fail.
* Teams can answer “what/why/where” for any artifact in seconds.
* Knobs for cache/limiter/breaker/fallback are tuned using **measured** regressions, not guesswork.

If you want, I can also bundle a tiny **Grafana-ready SQLite→Prometheus exporter** snippet and a `duckdb` one-liner script for Parquet exports, but the plan above gets you a complete, production-practical Observability & SLOs layer today.
