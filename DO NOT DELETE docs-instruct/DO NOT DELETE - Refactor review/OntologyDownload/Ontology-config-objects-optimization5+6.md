Absolutely—here’s a **deep, code-free, agent-ready** plan for the next two pillars, tailored to your OntologyDownload stack and the libraries you’ve curated. I’ll assume your settings, HTTP/Hishel, and rate-limit layers are already in place.

---

# 5) Local catalog that scales — **DuckDB as the “brain”**

## 5.1 Objectives (what “done” means)

* One **authoritative, local** catalog (DuckDB) for versions, artifacts, extracted files, validations, events.
* **Transactional boundaries** that keep DB and filesystem **in lock-step** (no torn writes).
* **Fast bulk ingest**, **cheap deltas**, **instant rollups** (no Python loops).
* **Simple operations**: doctor (reconcile), prune (GC), backup/restore, and profiling.

## 5.2 Module layout

* `catalog/connection.py` — connection manager (writer, reader), PRAGMAs, locking.
* `catalog/migrations.py` — idempotent migration runner (applies `0001…000N`).
* `catalog/repo.py` — high-level “repo” façade (upsert/list/query APIs).
* `catalog/boundaries.py` — transaction wrappers for each pipeline boundary.
* `catalog/prune.py` — FS↔DB diff and GC routines (uses staging table/view).
* `catalog/doctor.py` — reconciliation and fixups.
* `catalog/profile.py` — EXPLAIN/ANALYZE helpers.
* `cli/db_cmd.py` — user commands (stats/list/latest/doctor/prune/backup).

## 5.3 Connection & concurrency

* **Single process writer** policy (your default).

  * **Writer**: `duckdb.connect(db_path, read_only=False)` in a guarded context.
  * **Readers**: `duckdb.connect(db_path, read_only=True)` (lazy on demand).
* **Writer lock**: hold a **filesystem lock file** (`.duck.lock`) for the life of the writer process (or per boundary, if preferred). Detect stale locks (PID dead) and recover safely.
* PRAGMAs on open:

  * `PRAGMA threads = settings.db.threads;`
  * Keep defaults for memory until profiling suggests otherwise.
* **Close discipline**: context manager ensures commits/rollbacks and connection close on every boundary.

## 5.4 Migration runner (idempotent)

* On start of any command that may write:

  1. Open writer connection under lock.
  2. Inspect `schema_version`; collect applied IDs.
  3. Apply pending files `000X_*.sql` **in order** inside a single TX.
  4. Log `migrations_applied=[…]` and proceed.
* Provide `ontofetch db migrate --dry-run` to preview.

## 5.5 Transactional boundaries (DB↔FS choreography)

* **Download boundary**

  1. Stream archive to temp → atomic rename to final path.
  2. TX: upsert `versions` and `artifacts` (status `fresh|cached`, size, etag, fs_relpath).
  3. Commit → emit `events` (extract.start/extract.done).
* **Extraction boundary**

  1. Pre-scan & policy gates (already defined in your extractor).
  2. Stream writes + hashes; build rows for `extracted_files`.
  3. TX: **bulk insert** files (appender/`INSERT FROM` Arrow/Polars) linked to `artifact_id`.
  4. Commit → write audit JSON (atomic) → `events`.
* **Validation boundary**

  1. Run validators → collect results.
  2. TX: insert `validations` rows; optionally maintain a summary view.
  3. Commit → `events`.
* **Set latest boundary**

  1. TX: upsert `latest_pointer`.
  2. Write `LATEST.json` to FS via temp + rename.
  3. If FS write fails → ROLLBACK; else COMMIT.

> Each boundary logs its **config_hash**, `run_id`, and counts/bytes to `events`.

## 5.6 Bulk ingest patterns (fast paths)

* Prefer **DuckDB Appender** or `INSERT INTO … SELECT * FROM read_parquet/read_json/arrow_table` when you already hold Arrow/Polars frames (see §6).
* Normalize batching: target **5–50k rows per chunk**; keep each TX tight (<1–2s).
* For upserts:

  * Use `INSERT OR REPLACE` for small tables (e.g., `latest_pointer`) or **two-phase** (`DELETE` matching keys → `INSERT`) for large tables.
  * Consider `MERGE INTO` where available for declarative upsert rules.

## 5.7 Performance & profiling

* Add `--profile` switch to DB commands:

  * Wrap queries in `EXPLAIN ANALYZE` and emit **JSON plan** (or text) to logs.
  * Surface **hot operators** (e.g., hash join vs. nested loop); recommend indexes or pre-computed relations if needed.
* “Materialized” helper tables (optional):

  * For very heavy views, provide a `REFRESH` task that writes results to a table with a TTL and metadata (so dashboards are instant).

## 5.8 Doctor & prune workflows

* **Doctor** — reconcile DB to FS and fix known drifts:

  * FS file exists but no DB row → prompt to (a) insert stub artifact/file or (b) delete stray file.
  * DB row exists but FS missing → prompt to (a) drop row or (b) mark as failed/missing.
  * Latest mismatch **DB vs LATEST.json** → fix to match desired source of truth.
* **Prune** — safe GC of orphans:

  * Populate `staging_fs_listing` with a fast FS scan.
  * Use `v_fs_orphans` to list victims; print counts/bytes; `--apply` deletes in small batches with progress and rollback safety.
  * Record pruned entries to `events` for audit.

## 5.9 Backup & restore

* `ontofetch db backup`:

  * Run `CHECKPOINT;` then fs-copy the DB (`.duckdb`) and `docs/schemas/…` to a timestamped folder.
  * Optionally compress and sign.
* Restore instructions documented (copy back; run migrate; doctor).

## 5.10 Testing & acceptance

* **Unit**: migrations apply in order; upsert semantics; latest pointer idempotent; views return expected results on fixtures.
* **Integration**: end-to-end boundaries maintain DB↔FS consistency under failure injection (simulate crash between FS write and DB commit and vice versa).
* **Performance**: ingest 50k files under a budget (set by CI class); `EXPLAIN` plan contains expected operators (hash join on keys).
* **Acceptance checklist**

  * [ ] Migrations runner idempotent; schema at head.
  * [ ] Boundaries guarantee no torn writes.
  * [ ] Bulk inserts use appender/Arrow/Polars; no Python row loops.
  * [ ] Doctor and prune produce correct, safe actions; `--dry-run` and `--apply` covered.
  * [ ] Profiling available behind `--profile`; plans emitted.

---

# 6) Data processing **without Python loops** — **Polars** as the analytics engine

## 6.1 Objectives

* Treat audits, events, and ad-hoc analysis as **columnar pipelines**, not Python for-loops.
* Use **LazyFrame** + **scan_*()** to push predicates and projections; **streaming** for large flows.
* Seamless **DuckDB ↔ Arrow/Polars** interop (zero-copy where possible).

## 6.2 Module layout

* `analytics/polars_pipelines.py` — reusable pipeline builders (lazy).
* `analytics/reports.py` — named reports (latest stats, format distribution, deltas by bytes).
* `analytics/io.py` — readers for audit JSON, Parquet event logs, CSV lists (scan, not read).
* `cli/analytics_cmd.py` — `report` subcommands (`latest`, `growth`, `validation`, `hotspots`), `--format` output.

## 6.3 Data sources

* **DuckDB tables**: fetch **narrow Arrow batches** for summaries; for massive joins, let DuckDB do the heavy SQL and export to Arrow once.
* **Audit JSON** (`.extract.audit.json`): read via `scan_ndjson` (schema inferred or provided) with **streaming**.
* **Events**: if you choose Parquet logs, use `scan_parquet` with predicate pushdown.

## 6.4 Pipeline conventions (Polars)

* Always start with **`scan_*()`** (lazy):

  * Apply **projections early** (`select`) to reduce width.
  * Use **`filter`** on `version_id`, `format`, `ts` windows to reduce height.
  * Prefer **joins** on typed keys (`version_id`, `file_id`) and **groupby_agg** for rollups.
* **Streaming**:

  * For long pipelines or large files, pass `collect(streaming=True)` where applicable.
* **Typed columns**:

  * Normalize `size_bytes` to `pl.Int64`, timestamps to `pl.Datetime(time_zone="UTC")`, enums to `pl.Categorical` for memory efficiency.
* **Window & asof**:

  * For time-based KPIs (growth per day/week), use `groupby_dynamic` or `join_asof`.

## 6.5 Core reports (ready-made)

1. **Latest Version Summary**
   Inputs: `extracted_files` + optional `validations`
   Outputs: files, bytes, by format, pass/fail counts, top 10 largest files, top 10 largest growth contributors.
2. **Version Growth (A→B)**
   Use your DuckDB **delta macros** or build a Polars delta: ADDED/REMOVED/MODIFIED/RENAMED; bytes delta per path and per format.
   Output: tables + aggregates + “churn” metric (modified bytes) separate from net growth.
3. **Validation Health**
   Join `validations` with files; produce rates per validator; list REGRESSED vs FIXED paths (optionally by format).
4. **Hotspots**
   Identify **sources** (service/host) or **file patterns** that dominate bytes or failure rates; power law charting data (top N contributors).

> Each report returns a **Polars DataFrame** with a consistent schema and a renderer function (`as_table`, `as_jsonl`, `as_parquet`).

## 6.6 Interop strategies (no copies)

* **DuckDB → Polars**:
  `duckdb_conn.execute(sql).arrow()` → `pl.from_arrow()`
  (Arrow mediates without materializing Python objects.)
* **Polars → DuckDB**:
  `df.to_arrow()` → `duckdb_conn.register('tmp', arrow_table)` → SQL over it → `INSERT INTO … SELECT * FROM tmp`
* Prefer **one boundary** per report; avoid ping-pong.

## 6.7 Performance posture

* **Lazy pipelines**; **predicate pushdown**; project only needed columns.
* Prefer **categoricals** for low-cardinality strings (formats, validators).
* Cache small dimension tables (e.g., version dict) in memory once per command invocation.
* Use **parallelism** (Polars rayon) — default on; limit via settings for reproducibility when needed.

## 6.8 CLI analytics (fast & friendly)

* `ontofetch report latest --version <v> --format table|json|parquet --out /path`
* `ontofetch report growth --a <v1> --b <v2> --by format|path --top 50`
* `ontofetch report validation --version <v> --by validator|format`
* **Examples** in help epilog (+ `--profile` to print pipeline plan).

## 6.9 Testing

* **Unit**: small fixtures for audits/events to verify groupby and joins; schema assertions (dtypes) so regressions are caught.
* **Property**: random mix of added/removed/modified/renamed sets; ensure growth math is correct and rename accounting does not double-count.
* **Performance**: streaming collections complete under budget on CI (set by runner class), especially for audit scans > 1M rows.

## 6.10 Acceptance checklist

* [ ] All analytics read with `scan_*()` pipelines (no eager `.read_*()` on large files).
* [ ] Reports produce consistent schemas and support `--format` renderers.
* [ ] Interop with DuckDB is Arrow-backed (no Python lists/loops).
* [ ] Growth/validation summaries reconcile with DB views for cross-checks.
* [ ] Profiling available (`--profile`) showing optimized plans.
* [ ] Tests: unit/property/perf pass; schema types enforced.

---

## Suggested PR sequence (low-risk increments)

**PR-D1 — Catalog service layer**
Connection manager, migrations runner, repo façade (upsert/list), unit tests.

**PR-D2 — Boundaries & bulk ingest**
Two-phase choreography; Arrow/Polars → DuckDB ingest; end-to-end fixtures.

**PR-D3 — Doctor & prune**
Staging listing, orphans view, CLI `doctor|prune`; dry-run/apply + tests.

**PR-D4 — Analytics (Polars)**
Pipelines for latest/growth/validation; CLI `report` with renderers; interop tests.

**PR-D5 — Profiling & docs**
`--profile` across DB and Polars; examples and budgets in docs; CI checks for drift and perf.

This gets you a **scalable, zero-drift, loop-free** analytics and catalog layer that feels instant locally and is simple to operate—perfect for your ontology workflows.
