love it—here are two “pop”-level architecture overviews you can paste straight into your docs. Each pair shows the parts, the data/ID contracts, and the operational flow so the integration is unmistakable.

---

# C) **HTTP That Never Surprises You** + **Rate-Limit Correctness**

## C1. Component map (one client, one limiter, many call-sites)

```
                          ┌──────────────────────────────────────────────────┐
                          │             Traced Settings (Pydantic)           │
                          │  http, cache, retry, security, ratelimit, …      │
                          └───────────────┬──────────────────────────────────┘
                                          │ normalized submodels
                                          ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                  Network Subsystem                                     │
│                                                                                        │
│  ┌────────────────────────────┐      ┌────────────────────┐      ┌──────────────────┐  │
│  │  HTTP Policy (constants)  │◄────►│  HTTPX Client      │◄────►│  Hishel Cache    │  │
│  │  timeouts, pool, headers  │      │  (single, shared)  │      │  (RFC 9111)      │  │
│  └────────────────────────────┘      └────────────────────┘      └──────────────────┘  │
│             ▲                                       ▲                    ▲             │
│             │                                       │ hooks              │ cache meta  │
│  ┌────────────────────┐                   ┌────────────────────┐        │             │
│  │ Redirect Audit     │                   │ Instrumentation    │────────┘             │
│  │ validate per hop   │                   │ net.request event  │                      │
│  └────────────────────┘                   └────────────────────┘                      │
│                                                                                        │
│  ┌────────────────────────────┐                             ┌───────────────────────┐  │
│  │ URL Security Gate          │◄──────── target URL ───────►│ Tenacity Backoff      │  │
│  │ scheme/IDN/PSL/DNS/ports   │                             │ 429/5xx policy        │  │
│  └────────────────────────────┘                             └───────────────────────┘  │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                          ▲
                                          │
                                  acquires before I/O
                                          │
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                    Rate-Limit Subsystem                                │
│                                                                                        │
│  ┌───────────────────────┐      key="service:host"     ┌───────────────────────────┐   │
│  │ Settings→RateSpecs    │ ───────────────────────────► │ Limiter Registry         │   │
│  │ default & per-service │                              │  {key → Limiter+Bucket} │   │
│  └───────────────────────┘                              └───────────────────────────┘   │
│           ▲                       ┌───────────────────────┐     ┌──────────────────┐    │
│           │                       │ InMemoryBucket (SP)   │     │ SQLiteBucket(MP) │    │
│           └────────── configurable└───────────────────────┘     └──────────────────┘    │
│                                                                                        │
│     acquire(mode="block|fail", weight)  →  ratelimit.acquire event (blocked_ms, outcome)│
└────────────────────────────────────────────────────────────────────────────────────────┘
```

**Legend**
SP=single-process; MP=multi-process. Redirects are **off** globally; the “Redirect Audit” module validates each hop with the **URL Security Gate** before following.

---

## C2. Data/ID contracts (what binds it all together)

* **Request IDs**: every HTTP call gets a `request_id` (UUID) that is echoed in:

  * `net.request` event payload,
  * higher-level `download.fetch` event (so a file write correlates to its HTTP).
* **Limiter keys**: `"{service or '_'}:{host or 'default'}"` – all rate telemetry aggregates on this.
* **Security inputs**: the **normalized** tuple from settings
  `(exacts, suffixes, per_host_ports, ip_literals)` + `allowed_port_set()`; all punycoded/ lowercase.

---

## C3. Control-flow (single GET with redirect & 429)

```
call-site
    │
    ├─► limiter.acquire(key="ols:ebi.ac.uk", mode="block")
    │        └─ emits ratelimit.acquire (blocked_ms=0)
    │
    ├─► httpx.request(GET, url, stream=True)
    │        └─ hooks.on_request → start timer, UA, run_id
    │
    ├─ 302 Location: https://…/final
    │     ├─ Redirect Audit: validate_url_security(Location)  ✓
    │     └─ httpx.request(GET, final, stream=True)
    │           └─ on_response → cache=miss, ttfb_ms, elapsed_ms
    │
    ├─ 429 Retry-After: 2
    │     ├─ Tenacity sleeps(2s + jitter)
    │     └─ limiter.acquire(key, mode="block") → blocked_ms≈2000
    │
    └─ 200 OK → iter_bytes → file.tmp → fsync → rename
              └─ emit download.fetch (bytes, ms, request_id)
```

---

## C4. Invariants & guardrails

* **One** HTTPX client per process; **no** auto-redirects.
* Every hop passes the **URL Security Gate** (scheme/http→https upgrade, PSL allowlist, DNS/private-net, per-host ports).
* **Transport retries** only for **connect**; status-aware backoff lives in Tenacity.
* I/O is always **streaming**; body is never fully buffered.
* Every acquire yields **one** `ratelimit.acquire` event (blocked or failed).

---

## C5. “What to query on Monday morning?”

* Top 5 limiter keys by **blocked_ms** → which service/host needs quota tuning.
* Cache hit ratio (Hishel) and `net.request` p95 per service.
* Count of redirect hops denied by **URL Security Gate** (should be 0 for curated sources).
* 429 occurrences & average cooldown observed.

---

# D) **DuckDB as the Brain** + **Polars Without Loops**

## D1. System view (catalog owns the truth; Polars owns the heavy lifting)

```
                ┌──────────────────────────────────────────────────┐
                │             Filesystem (blobs)                   │
                │  …/<service>/<version>/data/**                   │
                │  …/<service>/<version>/.extract.audit.json       │
                │  …/LATEST.json                                   │
                └───────────────────────┬──────────────────────────┘
                                        │ boundary choreography
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                                  DuckDB Catalog (brain)                                  │
│                                                                                          │
│  Tables: versions, artifacts, extracted_files, validations, latest_pointer, (events)     │
│  Views : v_version_stats, v_latest_files, v_validation_failures, v_artifacts_status, …   │
│                                                                                          │
│  Boundaries (atomic):                                                                     │
│   • download  → upsert artifacts (after file rename)                                     │
│   • extract   → bulk insert extracted_files (after writes & hashes)                      │
│   • validate  → insert validations                                                       │
│   • set latest→ upsert latest_pointer + write LATEST.json                                │
│                                                                                          │
│  Migrations: 0001…000N (idempotent)  |  Doctor/Prune: staging_fs_listing + v_fs_orphans │
└───────────────────────────────────────────┬──────────────────────────────────────────────┘
                                            │ arrow/frames for analytics or inserts
                                            ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                                  Polars Analytics (no loops)                              │
│                                                                                           │
│  scan_parquet / scan_ndjson (audits/events)  +  DuckDB→Arrow→Polars for catalog slices    │
│                                                                                           │
│  Lazy pipelines:                                                                          │
│   • Latest Summary     → files/bytes by format; top-N largest files                       │
│   • Version Growth A→B → ADDED/REMOVED/MODIFIED/RENAMED + churn vs net bytes             │
│   • Validation Health  → FIXED/REGRESSED/NEW/REMOVED by validator                         │
│   • Hotspots           → top sources/patterns by bytes/failures                           │
│                                                                                           │
│  collect(streaming=True) where big; export table/json/parquet; CLI `report` subcommands   │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## D2. Contracts & IDs that keep it coherent

* **Version identity**: `version_id` (string) — binds FS tree, DuckDB rows, analytics slices.
* **Artifact identity**: `artifact_id=sha256(archive)` — de-dupes downloads.
* **File identity**: `file_id=sha256(file bytes)` — enables rename detection across versions.
* **Audit JSON**: deterministic schema (entries with `path_rel, size, sha256, mtime, scan_index`) written atomically per extraction.
* **Delta macros** (in DuckDB): `version_delta_files(_a,_b)`, `…_rename_aware`, `…_summary`, `…_formats`, `version_validation_delta`.

---

## D3. End-to-end flow (one version ingestion)

```
download archive
    │
    ├─ stream → file.tmp → rename
    │
    ├─ DB TX: upsert artifacts(version_id, fs_relpath, size, etag, status)
    │
    ├─ extract (libarchive):
    │     pre-scan gates → stream files → hash → data/** + audit.json.tmp → rename
    │
    ├─ DB TX: bulk insert extracted_files (Arrow/Polars appender)
    │
    ├─ run validators → results
    │
    └─ DB TX: insert validations; emit events
```

*(Every step emits events with `run_id` & `config_hash`; doctor/prune keep DB↔FS in lock-step.)*

---

## D4. Polars pipelines (how analytics stay fast & loop-free)

* **Start lazy**: `scan_ndjson(audit)`, `scan_parquet(events)`, or fetch a narrow **Arrow** table from DuckDB.
* **Project early, filter early**: keep only `(version_id, relpath, size, format, sha256, ts, validator, passed)` columns.
* **Join** on typed keys (`file_id`, `version_id`) and **groupby_agg** for rollups.
* **Streaming collect** for large audits; enforce dtypes (`Int64`, `Categorical`, UTC times) for memory efficiency.
* **Interop**:

  * DuckDB ➜ Arrow ➜ `pl.from_arrow()` for fast hand-offs.
  * Polars ➜ Arrow ➜ `duckdb.register('tmp', arrow)` ➜ `INSERT … SELECT` (no Python loops).

---

## D5. “Answers on demand” (CLI + SQL)

* `ontofetch delta summary A B` → bytes_total_a/b, files_added/removed/modified/renamed, churn.
* `ontofetch report latest --version V` → files & bytes by format + top-N heavy files.
* `ontofetch report validation --version V` → FIXED/REGRESSED by validator and by format.
* `SELECT * FROM v_fs_orphans` after a scan → safe GC plan.

---

## D6. Invariants & safety rails

* **No torn writes**: DB commit happens **after** FS success; LATEST set is TX + atomic JSON write.
* **Idempotence**: re-runs never duplicate rows/files; overwrite policy enforced at extract time.
* **Determinism**: audit JSON order (header or path_asc), stable `config_hash` embedded in events.
* **Doctor/Prune**: staging diff governs deletes; `--dry-run` default; batch deletion with clear telemetry.

---

## D7. Performance knobs (where they live)

* DuckDB: `PRAGMA threads`, bulk appender, `EXPLAIN ANALYZE` behind `--profile`.
* Polars: `collect(streaming=True)`, categoricals for low-card strings, predicate pushdown via `scan_*`.

---

### How C & D click together (the big picture)

```
Settings (strict, hashed, traced)
   │                        ▲
   │ run_id, config_hash    │ events (net/request, extract, db, delta)
   ▼                        │
[Limiter] → [HTTPX+Hishel] →│→ [Filesystem writes] → [Extraction] → [Audit JSON]
               ▲            │                          │
               │            └──────────────► [DuckDB Catalog] ◄───────────────┐
               │                                        │                     │
               │                                        └─► [Polars Reports]  │
               │                                               (CLI)          │
               └────────── URL Security Gate (every hop)                      │
```

* Every **network byte** is rate-capped, audited, and attributed.
* Every **file written** lands in a deterministic tree with an audit and a DB row.
* Every **question** (What changed? Where did it go wrong? How big is it?) has a **first-class query** or CLI report.

---

If you want, I can produce the final pair’s visuals too—**Observability that answers questions** + **Safety & policy**—so the whole platform reads like one cohesive system.
