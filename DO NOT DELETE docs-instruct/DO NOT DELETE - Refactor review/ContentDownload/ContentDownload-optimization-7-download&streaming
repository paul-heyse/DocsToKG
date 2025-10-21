Awesome — here’s a **repo-shaped, junior-dev-friendly** implementation plan for the **next major optimization: Data model & idempotency** (the item after download streaming). The goal is to make every “side-effect” (downloading, finalizing, indexing, deduping) **exactly-once** across crashes, retries, and multiple workers.

---

# 0) Outcomes (what “done” looks like)

* A small **state machine** for artifacts (PLANNED → … → FINALIZED → INDEXED/DEDUPED), enforced by DB constraints.
* **Idempotency keys** for HTTP effects and file effects, so retries never double-do work.
* **Uniqueness** at the right layers (per-artifact job, per-operation idempotency, per-hash canonical file).
* A **lease** mechanism to avoid two workers doing the same work concurrently.
* A **reconciler** that heals partial states after crashes (e.g., file renamed but manifest not updated).
* All of this is **auditable** (every step is recorded once) and **fast** (WAL, short transactions).

---

# 1) Schema (new tables + minimal changes)

Add these to your **manifest/telemetry DB** (same one you just migrated). Keep columns nullable where safe; enforce correctness with `UNIQUE` and small `CHECK`s instead of heavy triggers.

## 1.1 Artifact job ledger (one row per URL you intend to fetch)

```sql
CREATE TABLE IF NOT EXISTS artifact_jobs (
  job_id           TEXT PRIMARY KEY,                 -- uuid
  work_id          TEXT NOT NULL,
  artifact_id      TEXT NOT NULL,
  canonical_url    TEXT NOT NULL,
  state            TEXT NOT NULL DEFAULT 'PLANNED',  -- see §2
  lease_owner      TEXT,                             -- process/node id
  lease_until      REAL,                             -- wall time sec
  created_at       REAL NOT NULL,
  updated_at       REAL NOT NULL,
  -- determinism / dedupe
  idempotency_key  TEXT NOT NULL,                    -- see §3
  UNIQUE(work_id, artifact_id, canonical_url),
  UNIQUE(idempotency_key),
  CHECK (state IN ('PLANNED','LEASED','HEAD_DONE','RESUME_OK','STREAMING','FINALIZED','INDEXED','DEDUPED','FAILED','SKIPPED_DUPLICATE'))
);
CREATE INDEX IF NOT EXISTS idx_artifact_jobs_state ON artifact_jobs(state);
CREATE INDEX IF NOT EXISTS idx_artifact_jobs_lease ON artifact_jobs(lease_until);
```

## 1.2 Operation idempotency (effects ledger)

```sql
CREATE TABLE IF NOT EXISTS artifact_ops (
  op_key        TEXT PRIMARY KEY,       -- see §3.2
  job_id        TEXT NOT NULL,
  op_type       TEXT NOT NULL,          -- HEAD | STREAM | FINALIZE | INDEX | DEDUPE
  started_at    REAL NOT NULL,
  finished_at   REAL,                   -- null if in-flight (best-effort)
  result_code   TEXT,                   -- e.g., OK | RETRYABLE | NON_RETRYABLE
  result_json   TEXT,                   -- small payload (status, bytes, path, etc.)
  FOREIGN KEY(job_id) REFERENCES artifact_jobs(job_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_artifact_ops_job ON artifact_ops(job_id);
```

**Why:** If a worker crashes and the task restarts, the same **op_key** means “we already did this effect,” so the new worker can **no-op** or **continue** safely.

## 1.3 You already added (from streaming scope)

* File I/O columns on `downloads` (final_path, sha256, etc.)
* `artifact_hash_index` / `url_hash_map` in a separate DB (recommended)

No additional columns needed there for idempotency.

---

# 2) State machine (enforced by code + simple DB checks)

```
PLANNED
  └─(lease)→ LEASED
      └─(HEAD ok)→ HEAD_DONE
          └─(resume ok)→ RESUME_OK   (or skip to STREAMING if fresh)
              └─(stream)→ STREAMING
                  └─(atomic rename)→ FINALIZED
                      ├─(hash index put)→ INDEXED
                      └─(dedupe link)→ DEDUPED
(Any failure) → FAILED
(Found by hash before network) → SKIPPED_DUPLICATE
```

* Only allow **monotonic** forward transitions; you can enforce in code by `UPDATE … WHERE state IN (allowed_previous)` and checking `rowcount==1`.
* **Never** go backwards; if you must, mark as FAILED and re-queue a new job.

---

# 3) Idempotency keys (what they include & how to compute)

## 3.1 Job idempotency key (for `artifact_jobs`)

**Deterministic JSON → sha256 hex:**

```json
{
  "v": 1,
  "work_id": "...",
  "artifact_id": "...",
  "canonical_url": "...",
  "role": "artifact",                // constant here
  "headers_affecting_identity": {    // the minimal set
    "Accept": "application/pdf"
  }
}
```

* Serialize with **sorted keys**, no whitespace → sha256 → lowercase hex.
* Guarantees the **same job** row across retries/runs; `UNIQUE(idempotency_key)` prevents duplicates.

## 3.2 Operation idempotency key (`artifact_ops.op_key`)

One key **per effect**:

* `HEAD`: sha256 of `{"v":1,"kind":"HEAD","job_id":...,"url":...}`
* `STREAM`: sha256 of `{"v":1,"kind":"STREAM","job_id":...,"url":...,"range_start":N}`
* `FINALIZE`: sha256 of `{"v":1,"kind":"FINALIZE","job_id":...,"sha256":...,"part":...,"final":...}`
* `INDEX`: sha256 of `{"v":1,"kind":"INDEX","job_id":...,"sha256":...}`
* `DEDUPE`: sha256 of `{"v":1,"kind":"DEDUPE","job_id":...,"sha256":...,"target":...}`

**Rule:** Before performing an effect, attempt `INSERT` of the `op_key`. If it **already exists**, read `result_json` and **short-circuit** (exactly-once).

---

# 4) Leases (avoid concurrent workers on the same job)

SQLite-friendly pattern (single worker claims a job row):

```sql
-- Try to claim one available job
UPDATE artifact_jobs
   SET lease_owner = :owner, lease_until = :now + :ttl, state='LEASED', updated_at=:now
 WHERE job_id = (
   SELECT job_id FROM artifact_jobs
    WHERE state IN ('PLANNED','FAILED') AND (lease_until IS NULL OR lease_until < :now)
    ORDER BY created_at
    LIMIT 1
 )
   AND (lease_until IS NULL OR lease_until < :now);
```

* If `rowcount == 1`, you own it.
* **Renew** the lease periodically: `UPDATE … WHERE job_id=? AND lease_owner=:owner`.
* On exit, **release** by setting `lease_until = NULL, lease_owner = NULL` (best-effort).
* Workers that see `rowcount == 0` just try the next job.

---

# 5) Exactly-once finalize (two-phase commit)

When moving `*.part → final`:

1. Start a **short transaction**: `BEGIN IMMEDIATE;`
2. **Check** allowed transition: `UPDATE artifact_jobs SET state='FINALIZED', updated_at=now WHERE job_id=? AND state IN ('STREAMING');` (expect `rowcount==1`)
3. **Perform rename & index put** **inside the artifact file lock** (the rename itself isn’t in SQLite txn; that’s OK since we keep the lock and immediately persist DB).
4. Upsert `downloads` row with final metrics, and insert `artifact_ops` (`op_type='FINALIZE'`).
5. `COMMIT;`

**Crash windows:** If you rename but crash before DB update, your **reconciler** (§8) re-links DB and filesystem on next run.

---

# 6) Integration (where to call what)

* **Planner** creates `artifact_jobs` with the **job idempotency key** (use `INSERT OR IGNORE`, then `SELECT` to get `job_id`).
* **Runner** loop:

  * `lease_next_job()` → HEAD (`op_key:HEAD`) → update validators in `downloads`.
  * `can_resume()` → `op_key:STREAM` with `range_start` (resume/fresh).
  * `stream_to_part()` (returns metrics) → update `downloads` and `artifact_ops`.
  * `finalize_artifact()` → `op_key:FINALIZE` + `downloads` + `artifact_hash_index`.
  * `dedupe_link_or_copy()` when applicable → `op_key:DEDUPE`.
  * Transition `state` accordingly after each step, with allowed previous states check.

**Each effect** first tries to create its `op_key` row (INSERT). If present, read `result_json` and **reuse** (e.g., if a previous process already finalized, just mark the job complete).

---

# 7) Reconciler (startup health pass)

On process start (or hourly):

* **Filesystem → DB**: scan `root/.staging` for stale `.part` older than N hours → delete.
* **DB → Filesystem**: for rows `state='FINALIZED'` but `final_path` missing →

  * if hash known and `artifact_hash_index` maps to an existing file, **restore** by relinking and set `final_path`;
  * else mark `FAILED` with reason `final_missing`.
* **Ops ledger**: find `artifact_ops` with `finished_at IS NULL AND started_at < now-10m` → mark `result_code='ABANDONED'` for visibility (don’t retry at ops level; job state machine will decide).
* **Leases**: clear stale leases (`lease_until < now`) by setting owner to NULL.

Keep this pass **fast**: cap scan depth and batch updates.

---

# 8) Pseudocode snippets (for an agent to fill in)

## 8.1 Idempotency key helpers

```python
# src/DocsToKG/ContentDownload/idempotency.py
import hashlib, json

def ikey(obj: dict) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def job_key(work_id, artifact_id, canonical_url) -> str:
    return ikey({"v": 1, "kind": "JOB", "work_id": work_id, "artifact_id": artifact_id, "url": canonical_url})

def op_key(kind: str, job_id: str, **extras) -> str:
    d = {"v": 1, "kind": kind, "job_id": job_id}
    d.update(extras)            # e.g., {"url":..., "range_start":...}
    return ikey(d)
```

## 8.2 Effect wrapper (exactly-once)

```python
def run_effect(conn, op_key: str, job_id: str, kind: str, fn) -> dict:
    """
    Try to INSERT op row; if exists, return stored result_json.
    Otherwise run fn() (which performs the side effect), then store result_json and return it.
    """
    now = time.time()
    try:
        conn.execute("INSERT INTO artifact_ops(op_key, job_id, op_type, started_at) VALUES (?,?,?,?)",
                     (op_key, job_id, kind, now))
    except sqlite3.IntegrityError:
        row = conn.execute("SELECT result_json FROM artifact_ops WHERE op_key=?", (op_key,)).fetchone()
        return json.loads(row["result_json"]) if row and row["result_json"] else {}

    result = fn()                 # perform the side effect exactly once
    conn.execute("UPDATE artifact_ops SET finished_at=?, result_code=?, result_json=? WHERE op_key=?",
                 (time.time(), result.get("code","OK"), json.dumps(result)[:20000], op_key))
    return result
```

---

# 9) Concurrency & locking

* Keep SQLite transactions **short** and scoped to state updates and op ledger writes.
* File operations are protected by your **`artifact_lock(final_path)`** and **atomic `os.replace()`**.
* Use **`locks.sqlite_lock`** on the DB where you expect bursts (ops ledger, hash index) to avoid “database is locked” storms.

---

# 10) Tests (high-value)

1. **Duplicate plan insert** → single `artifact_jobs` row (UNIQUE key enforced).
2. **Double workers**: both try to lease; only one wins (`rowcount==1`).
3. **Crash after finalize**: rename succeeds, DB not updated → reconciler re-binds.
4. **Effect replay**: run same `op_key:FINALIZE` twice → second call returns previous `result_json`, no double rename.
5. **Out-of-order transition**: attempt `HEAD_DONE` while not `LEASED` → update `rowcount==0`; ensure code raises.
6. **Idempotent dedupe**: multiple runs create dedupe link; only one op row is inserted; others reuse it.

---

# 11) Rollout order (safe)

1. **Ship code** with feature-detection (reads `_meta.schema_version`) but writing to old schema (no-ops for new features).
2. Run the **schema migration** (columns added earlier + these new tables).
3. Flip a flag `DOCSTOKG_ENABLE_IDEMPOTENCY=1` to start writing to `artifact_jobs` + `artifact_ops`.
4. Enable **reconciler** in a canary.
5. After a few runs, enforce **allowed transitions** strictly (raise on bad `rowcount`).

---

# 12) Post-migration state (what improves, concretely)

* **At-most-once** file effects: finalization, index put, dedupe.
* **No duplicate work**: jobs are unique by `(work_id, artifact_id, canonical_url)` and leased by **one** worker at a time.
* **Replay-safe** ops: repeating an op with the same key returns the same result without doing the work again.
* **Crash robustness**: reconciler heals the last mile (file/DB mismatches); leases expire automatically.
* **Auditable**: you can answer “what happened to URL X?” and “which worker finalized hash H?” in one query.

---

If you want, I can follow up with: (a) the small `artifact_jobs`/`artifact_ops` **migration SQL** ready to paste (it mirrors the CREATE TABLE above in a single IMMEDIATE transaction), and (b) two tiny helper functions: `lease_next_job()` (SQL + Python wrapper) and `advance_state(job_id, to_state, allowed_from={...})` so your agents can just call these and not re-invent the patterns.
