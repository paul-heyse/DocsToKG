Awesome—here’s a **repo-shaped, junior-dev-friendly plan** to add **file-based locking** with `filelock` to `src/DocsToKG/ContentDownload`, sized for **maximum consolidation** and “production-first” safety. I’ll cover (1) the straight-ahead implementation you can ship fast, and (2) advanced options/nice-to-haves that future-proof the system.

---

# What problems we’re solving (and where the locks go)

**Goal:** prevent race conditions and partial writes across **processes** (and threads) on the same machine when multiple workers:

* append/rotate **JSONL** telemetry and manifests,
* write/rename **artifact files** (`.part` → final pdf),
* update **SQLite** telemetry/state,
* produce **run summaries / reports**.

We’ll centralize all locking in one tiny module so **no other file-writer does its own concurrency tricks**.

---

# High-level shape (after refactor)

* One module: `src/DocsToKG/ContentDownload/locks.py` exports **3–5 tiny helpers**:

  * `manifest_lock(manifest_path)` – for JSONL manifests & rotations
  * `telemetry_lock(path)` – for JSONL metrics (if separate)
  * `sqlite_lock(db_path)` – for SQLite writes/migrations
  * `artifact_lock(artifact_path)` – for `.part` streaming & final rename
  * `summary_lock(summary_path)` – for one-off summary files
* Writers use **`with ...:`** blocks around the minimal critical section; **no sleeps** or home-grown mutexes remain elsewhere.
* Lock files are **separate `.lock` files** (never lock the data file itself), with short **timeouts** and clear error messages.

---

# Step 1 — Add `locks.py` (single source of truth)

**Module responsibilities**

1. **Lock placement & naming**

   * Default **lock directory**: `<run_root>/locks/` (create it on startup).
   * Derive lock name from the **absolute target path**, e.g.:

     * `locks/manifest.<sha256(abs-path)>.lock`
     * `locks/db.<sha256(abs-path)>.lock`
     * `locks/artifact.<sha256(final-path)>.lock`
   * Reason: stable names; avoids path length & punctuation issues; keeps lock files out of rotating data folders.

2. **Lock types and defaults**

   * Use `FileLock` (hard OS lock) by default; it’s re-entrant and cross-platform.
   * Expose a single fallback **`SoftFileLock`** path via env/flag for odd filesystems (rare; see caveats below).
   * **Timeouts** (sensible defaults):

     * manifest/telemetry/summary: **2–5 s**
     * SQLite: **5–10 s**
     * artifact (streaming + rename): **10–30 s**
   * **Poll interval**: 50–100 ms (don’t spin too fast).
   * **Permissions**: default `mode=0o640` (or `0o600` if files are sensitive).
   * **Thread sharing**: if your writers can be called across **threads** within a process, set `thread_local=False` when you construct the lock; otherwise the default is thread-local (changed in recent versions).

3. **Context helpers**

   * Provide small helpers (no code shown here—this tells the agent what to implement):

     * `with manifest_lock(path): ...`
     * `with artifact_lock(path): ...`
     * `with sqlite_lock(db_path): ...`
     * etc.
       Each helper **creates** a `FileLock` with category-specific timeouts and returns a context manager.

4. **Observability**

   * Every helper should log:

     * lock file path,
     * **blocked wait duration** (ms),
     * acquisition result (acquired / timeout).
       This gives you p50/p95 lock waits on the run summary.

---

# Step 2 — Wrap the **minimal** critical sections

> Small critical sections = better throughput. Acquire **just** before touching the file; release **immediately** after the write/rename.

### 2A) JSONL **manifests** & **telemetry** (append & rotation)

* **Append**: open file in append mode **inside** `with manifest_lock(path):` / `with telemetry_lock(path):`. Write one complete JSON line, flush, and fsync if you already do that.
* **Rotation** (e.g., `manifest.0001.jsonl` → `manifest.0002.jsonl`):

  * Acquire the **same** lock covering the base manifest **and** rotation (use the base path as the lock key).
  * Perform rotation + index updates inside the single critical section to avoid interleaving two writers’ rotations.

**Delete:** any bespoke “busy/try again” or ad-hoc file-rename races in these writers.

### 2B) **SQLite** writes (state/telemetry)

* SQLite already serializes internally, but multiple processes tend to create “database is locked” bursts. Wrap **each transaction** or **batch write** inside `with sqlite_lock(db_path):`.
* Keep **WAL mode** and your normal pragmas; the lock just prevents noisy back-off/retry loops in Python.

**Delete:** local retry loops or sleeps around `sqlite3.OperationalError: database is locked`.

### 2C) **Artifact** writes (stream `.part` → atomic rename)

* **Stream** to `<dest>.part` under `with artifact_lock(final_path):` so only one process writes that artifact at a time. (Hishel + limiter already reduce duplication; this enforces local single writer.)
* On success, **atomic rename** `.part` → final path **within the same lock**.
* If a second process wants the same artifact:

  * It blocks until the first one finishes or timeout occurs; on timeout you can **bail** (treat as “someone else is doing it”) or retry later.

**Delete:** any ad-hoc “if file exists then sleep” loops around artifact creation; locking makes it deterministic.

### 2D) **Run summary** (one-off write/replace)

* Wrap the summary file write in `with summary_lock(summary_path):` to avoid clobbers at the end of a run when multiple workers might finalize concurrently.

---

# Step 3 — Wire it through the codebase (where to call the helpers)

* **`telemetry/*` sinks**:

  * For JSONL sinks, wrap `append_line()` and any `rotate()` in the proper lock.
  * For SQLite sinks (if you have one), wrap `execute_many()` and schema migrations.
* **`download/*` strategies**:

  * Wrap the `.part` streaming and final rename for each artifact with `artifact_lock(final_path)`.
* **`pipeline.py`** (if it directly performs manifest writes/rotations):

  * Protect each write or rotation with `manifest_lock(...)`.
* **`runner`/`cli` end-of-run**:

  * Wrap summary write in `summary_lock(...)`.

When done, **no other module** should open those files without holding the appropriate lock.

---

# Step 4 — Configuration & failure policy

* **Config surface** (env or CLI flags):

  * `DOCSTOKG_LOCK_DIR` (default: `<run_root>/locks`)
  * `DOCSTOKG_LOCK_TIMEOUT_{MANIFEST|TELEMETRY|SQLITE|ARTIFACT|SUMMARY}`
  * `DOCSTOKG_LOCK_THREAD_LOCAL` (`true|false`, default `false` for cross-thread reuse)
  * `DOCSTOKG_LOCK_SOFT=1` (switches to `SoftFileLock` if you must)
* **On timeout**:

  * **Manifest/telemetry/summary**: log **warning** and **retry later** (or escalate a clear error if the operation is non-optional).
  * **SQLite**: retry via your existing retry policy for DB writes, or fail loudly and allow the orchestrator to re-enqueue.
  * **Artifact**: prefer **bail out** (another process is already downloading) and mark the plan row as “work in progress elsewhere.”

`Timeout` exceptions include the lock file path—log it as the first breadcrumb for ops.

---

# Step 5 — Observability (add this now; it pays dividends)

* **Per lock category** counters:

  * `lock_acquire_total{type, outcome=acquired|timeout}`
  * `lock_wait_ms_sum` and **p95** in the run summary
* **Debug logging** (opt-in): on `DEBUG`, log “try → acquired in X ms → released” with the lock path and calling site (module:function short name).
* Raise `filelock` logger level to `INFO` if its internal DEBUG is too chatty.

---

# Step 6 — Tests (goldens you can write quickly)

* **Single-writer guarantee**: spawn two processes that both try to append to the same JSONL; assert the resulting file has **exactly** two complete lines—never interleaved/partial.
* **Rotation race**: concurrently rotate + write small lines; assert the rotation log and resulting files are valid.
* **Artifact contention**: start two downloads for the same URL; assert one blocks then exits cleanly when the file appears.
* **SQLite contention**: two processes inserting many rows; assert no “database is locked” escapes your lock layer.
* **Timeout path**: set timeouts to 50 ms and assert the correct **Timeout** is logged and surfaced.
* **Thread behavior**: if you run writers in threads, set `thread_local=False` and assert both threads share the same lock instance properly.

---

# Advanced approaches & “nice-to-haves”

These are optional—add them for a best-in-class implementation.

## 1) **Two-phase commit** for artifact + manifest row

* Wrap **(a)** `.part` → final rename and **(b)** manifest/registry update in **one** `artifact_lock(final_path)` critical section.
* If you also need to serialize JSONL **and** SQLite, define a **lock order** to avoid deadlocks:

  1. `artifact_lock(final_path)`
  2. `manifest_lock(manifest_path)`
  3. `sqlite_lock(db_path)`
* Never acquire in a different order elsewhere.

## 2) **Lock leasing & watchdog warnings**

* Capture **hold time** (ms) and log a **WARN** when any critical section exceeds a budget (e.g., 5 s for artifacts).
* Add an optional **“lease TTL”**: if a process holds a lock for >TTL, emit telemetry or trigger a soft cancel path (you still release normally at block exit).

## 3) **Non-blocking “try-once” helpers**

* For periodic tasks (e.g., “only one run builds the index per hour”), expose `try_manifest_lock(..., timeout=0)` and **skip** if locked—no queue buildup.

## 4) **Async variant** (future)

* If any of your writers become `asyncio`-based, mirror the helpers using `AsyncUnixFileLock`/`AsyncWindowsFileLock` and `async with await lock.acquire()`. Keep the same naming & policy.

## 5) **Distributed runners**

* If you eventually run multiple machines on a shared volume:

  * First, **validate** the filesystem’s flock semantics (NFS/SMB vary).
  * If it’s unreliable, **don’t** use hard locks across hosts; either:

    * switch to `SoftFileLock` **with** a robust stale-file cleanup policy, or
    * adopt a **distributed lock** (Redis/Postgres) for cross-host mutexes, keeping the same helper interface.

## 6) **Lock doctor CLI**

* Add `docstokg locks doctor` to:

  * list active/stale lock files (check mtime vs. process list),
  * clean up **soft** stale locks,
  * print the effective lock policy (paths, timeouts, thread_local).

## 7) **Security hardening**

* Default `mode=0o600` on lock files if your datasets are sensitive, and place `LOCK_DIR` in a restricted folder (not world-writable).

---

# Caveats (and how we address them)

* **Don’t lock the file you’re writing.** Lock a **separate** `.lock` file—always. This is the project’s guidance and prevents partial write interleaving.
* **Network FS**: flock support is inconsistent; test on your stack or use a service-backed lock for cross-host coordination.
* **GC pitfalls**: always keep a reference or use `with`/`async with`; don’t rely on a temporary object surviving during the critical section.

---

# Rollout checklist (one PR, clean diff)

1. Add `locks.py` with: lock dir bootstrap; hash-based naming; helpers; defaults; logging.
2. Wrap **JSONL append** and **rotation** with `manifest_lock` / `telemetry_lock`. Delete any local file “busy” logic.
3. Wrap **SQLite** write/migrate calls with `sqlite_lock`. Remove manual retry/sleep wrappers.
4. Wrap **artifact** `.part` streaming + rename with `artifact_lock`. Remove duplicate-work guards.
5. Wrap **summary** write with `summary_lock`.
6. Add metrics for acquires/timeouts & p95 lock waits.
7. Add tests listed above.
8. Add env/CLI config & docstrings so agents know the policy surface.

---

## TL;DR

* Centralize **all** file concurrency in `locks.py`.
* Lock **only** the tiny section that writes/renames.
* Prefer **hard** `FileLock`; keep `SoftFileLock` as a last-resort fallback.
* Instrument everything (acquire counts, wait p95, timeouts).
* Add **two-phase commit** + **lock ordering** where operations span artifact + manifest + SQLite.

Shout if you want me to draft the exact helper signatures and docstrings to drop into `locks.py`; the structure above is ready for an agent to implement and wire in one pass.
