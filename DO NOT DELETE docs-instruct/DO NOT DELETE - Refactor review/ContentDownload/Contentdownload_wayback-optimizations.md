Love it—here’s a **“go-beyond” pack** of practical enhancements for the Wayback SQLite sink and telemetry path. I grouped them by theme, wrote the exact behaviors to implement, and called out where they live (mostly in `telemetry_wayback_sqlite.py`, `telemetry_wayback.py`, and your `locks.py`).

---

# 1) Performance & throughput

**A. Batch/transaction control (big win)**

* Add a tunable `auto_commit_every` (you already have it) and expose it via env/CLI:

  * `WAYBACK_SQLITE_AUTOCOMMIT_EVERY=100` for heavy crawls; `=1` for debugging.
* Keep one connection per process, reuse a single cursor per thread.
* For very chatty streams (e.g., `wayback_discovery` and `wayback_candidates`), buffer events in memory and flush with `executemany()` every N rows.
* Add a **backpressure meter**: if `emit()` takes > X ms on average over last Y calls, log a WARN and suggest increasing the batch size.

**B. WAL tuning**

* You already set `journal_mode=WAL`. Also set:

  * `wal_autocheckpoint = 1000` pages (roughly 4–8 MB by default page size).
  * At end-of-run, call `PRAGMA wal_checkpoint(TRUNCATE)` to trim WAL size.
* Keep `synchronous=NORMAL` for speed; if you want maximum safety for post-run summaries, temporarily set `FULL` around the final commit.

**C. Page & cache**

* Page size `4096` is fine; if you profile large runs on fast NVMe, consider `8192` (but only before the first table is created).
* `cache_size = -64 * 1024` (64 MB) is a good default for read-heavy post-run analysis.

**D. Prepared statements**

* Maintain a tiny pool of compiled statements (one per table) if you see parse overhead in profiles. Python’s sqlite3 caches a bit internally, but explicit reuse can shave a few percent.

**E. mmap I/O (optional)**

* Add `PRAGMA mmap_size = 268435456` (256 MB) on 64-bit systems to reduce syscalls on read-heavy dashboards.

---

# 2) Concurrency & locking

**A. Lock scope**

* You already guard writes with `locks.sqlite_lock(db_path)`. Keep critical sections **short**: create cursor → execute → (optional) commit → release.
* Don’t hold the lock while serializing large JSON strings or computing aggregates—do that outside the lock.

**B. Busy timeouts**

* You set `busy_timeout_ms`. Add an internal retry loop around `sqlite3.OperationalError: database is locked` *only* for idempotent inserts, capped at 3 attempts with 10–50 ms jitter. (Your file lock should prevent these, but a belt-and-suspenders retry keeps things smooth.)

**C. Thread model**

* If emitters can run from multiple threads, ensure your lock helper uses `thread_local=False` and the connection is created with `check_same_thread=False` (you already do). One connection per process is still best.

---

# 3) Reliability & safety

**A. Crash-safe close**

* On `atexit` and SIGTERM, call `sink.close()`; after close, optionally run:

  * `PRAGMA optimize;` (SQLite will build/refresh stats)
  * `PRAGMA wal_checkpoint(RESTART);` to compact WAL.

**B. Failsafe dual-sink**

* Keep JSONL as a secondary sink in high-risk environments. If SQLite `emit()` raises repeatedly, fail open to JSONL (log one ERROR and continue). You can reconcile later.

**C. Dead-letter queue**

* Add a tiny “DLQ” JSONL file: if an event can’t be persisted after N retries, write the raw event line there with the exception string.

---

# 4) Schema & storage optimization

**A. Partial & covering indexes (query-driven)**

* Add composite/covering indexes to speed common queries:

  * `CREATE INDEX idx_attempts_run_result ON wayback_attempts(run_id, result);`
  * `CREATE INDEX idx_emits_run_mode ON wayback_emits(run_id, source_mode, memento_ts);`
  * `CREATE INDEX idx_discovery_stage_run ON wayback_discoveries(run_id, stage);`
* If you often filter for successes: a **partial index** (SQLite ≥3.8.0):
  `CREATE INDEX idx_attempts_success ON wayback_attempts(result) WHERE result LIKE 'emitted%';`

**B. Dimension tables (space saver, optional)**

* If DB size matters: move enums to small integer FKs (e.g., `result_id`). This can shrink size 20–40% at millions of rows. Keep a lookup table `{id, value}` for readability.

**C. Prune debug detail**

* Keep `details` short and sanitized. If you need longer context, store only a hash and put the full text in a separate `wayback_debug` table or sidecar JSONL.

**D. Retention policy**

* Implement simple TTL: `DELETE FROM wayback_* WHERE run_id IN (SELECT run_id FROM runs WHERE ended_at < date('now','-90 day'));`
* Or **partition per run**: one DB file per run (`telemetry/wayback-<run_id>.sqlite`) and a lightweight aggregator that reads multiple DBs for dashboards. This keeps files small and avoids long-term VACUUM needs.

**E. Vacuum strategy**

* After big deletes, run `VACUUM;` (offline) or `PRAGMA incremental_vacuum(2000);` if you use `auto_vacuum=incremental`. For most telemetry, a `VACUUM` weekly cron is enough.

---

# 5) Observability & KPIs (sink-level)

**A. Sink metrics**

* Track these in memory and emit a periodic summary row (or log):

  * `events_total`, `commits_total`, `avg_emit_ms`, `p95_emit_ms`, `db_locked_retries`, `dead_letters_total`.
* Threshold alerts: if `p95_emit_ms > 50ms` or `db_locked_retries > 0.1%`, print a rate-limited WARN with tuning hints (raise `auto_commit_every`, increase `busy_timeout_ms`, etc.).

**B. Event sampling**

* Add per-stream sampling toggles to cap cardinality in huge runs:

  * `WAYBACK_SAMPLE_CANDIDATES=10` → record at most 10 candidate rows per attempt.
  * `WAYBACK_SAMPLE_DISCOVERY=first,last` → only first and last CDX batches.

---

# 6) Query helpers (developer productivity)

Add a tiny helper module `telemetry_wayback_queries.py` with ready-to-use functions:

* `yield_by_path(run_id) -> {pdf_direct: n, html_parse: n}`
* `p95_selection_latency(run_id) -> int_ms` (via percentile approximation with window functions or a simple Python calc over fetched durations)
* `skip_reasons(run_id) -> [(reason, count)]`
* `cache_assist_rate(run_id) -> float` (share of discovery rows with `from_cache=1`)
* `rate_smoothing_p95(run_id, role) -> int_ms` (p95 `rate_delay_ms`)
* `backoff_mean(run_id) -> float` (mean `retry_after_s` on 429/503)

This makes dashboards trivial and avoids stray SQL in notebooks.

---

# 7) Export & analytics pipeline

**A. Parquet export (weekly)**

* Add `docstokg telemetry export --run <id> --out wayback.parquet`.
* Store flattened tables with dictionary encoding; DuckDB/Polars can slice this instantly.
* Nice for longer-term trending without keeping every SQLite file online.

**B. Roll-up table (fast dashboards)**

* Maintain a small per-run summary table `wayback_run_metrics`:
  `run_id, attempts, emits, yield_pct, p95_latency_ms, cache_hit_pct, non_pdf_rate, below_min_size_rate`.
* Populate it at end-of-run to power your CLI `summary` output without scanning detail tables.

---

# 8) Fault-injection & robustness tests

* **Disk full:** simulate `ENOSPC` by writing to a quota-limited tmpfs; ensure DLQ catches and the main process continues.
* **DB locked:** intentionally hold a write transaction in a second process; verify your lock + busy timeout behavior.
* **Power loss:** (hard to simulate) but you can SIGKILL during writes and verify WAL recovery + no corruption on restart.
* **Stress:** spawn N writers adding rows as fast as possible and report p95 `emit()` latency and final row counts.

---

# 9) Security & privacy

* If telemetry can contain sensitive URLs:

  * Mask query strings by policy when logging to `details`.
  * Consider `mode=0o600` on DB and lock files; keep the `telemetry/` dir outside world-writable locations.
  * If you ever need encryption, prefer **filesystem-level** encryption (LUKS) or swap to SQLCipher in a future iteration; do not roll a home-grown scheme.

---

# 10) Evolution & migrations

* You already write `_meta(wayback_schema_version)`. Add a tiny **migrator**:

  * On open, read version; if behind, run targeted `ALTER TABLE` steps, then update the version row.
  * Keep each migration idempotent and stored as a simple Python function.

* Add a **compatibility policy**: sinks should accept **unknown columns** in events (ignore them), and events missing optional fields should still insert (use NULLs).

---

# 11) Operational toggles (all via env/CLI)

* `WAYBACK_SQLITE_PATH` (explicit file path)
* `WAYBACK_SQLITE_AUTOCOMMIT_EVERY` (batch size)
* `WAYBACK_SQLITE_BUSY_TIMEOUT_MS`
* `WAYBACK_SQLITE_WAL_AUTOCHECKPOINT`
* `WAYBACK_SQLITE_PAGE_SIZE` (first-create only)
* `WAYBACK_SAMPLE_*` (per-stream sampling)
* `WAYBACK_DISABLE_HTML_PARSE_LOG` (if you want to quiet noisy runs)
* `WAYBACK_EXPORT_PARQUET=1` (enable export step at end-of-run)

---

## Rollout plan (quick and safe)

1. **Ship batching & WAL checkpoint** (fast wins, zero schema impact).
2. Add **sink metrics** and print a compact **end-of-run** summary from SQLite (yield, p95 latency).
3. Implement **retention** (per-run DBs or TTL deletes).
4. Add **Parquet export** and the **roll-up table** for instant dashboards.
5. Tackle **dimension tables** only if DB size becomes a problem.
6. Add **sampling** switches if you see cardinality creep.

---

### TL;DR

* Turn on **batching** + **WAL maintenance** for immediate speed.
* Keep locks tight; short transactions; retry lightly on `busy`.
* Add **roll-up metrics** and **export** to Parquet for long-term trends.
* Build a small set of **query helpers** so dashboards don’t reinvent SQL.
* Plan for **retention** and **migrations** early; it keeps the store lean and dependable.

If you’d like, I can draft the CLI subcommands for `telemetry export`, `telemetry vacuum`, and a `telemetry summary` that pulls from the roll-up table so your ops flow is one command away.
