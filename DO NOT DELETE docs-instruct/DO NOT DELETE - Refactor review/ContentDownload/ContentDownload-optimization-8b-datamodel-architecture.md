You got it — here are the **exact follow-ups for item 8 (Data model & idempotency)** you asked for:

* a **single-transaction SQL migration** for `artifact_jobs` + `artifact_ops`,
* minimal, production-oriented **Python helpers** for planning, leasing, advancing state, and exactly-once operation logging,
* a short **usage sketch** (how these glue into your downloader loop),
* and **post-migration considerations** (tuning, backfill, safety).

---

# A → B: SQLite migration (one short IMMEDIATE txn)

> Run this once against your manifest/telemetry DB (replace `downloads` if your main table has a different name). It’s idempotent.

```sql
BEGIN IMMEDIATE;

PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;
PRAGMA busy_timeout=4000;

-- Meta version gate
CREATE TABLE IF NOT EXISTS _meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
INSERT OR IGNORE INTO _meta(key, value) VALUES ('schema_version', '2'); -- bump later

-- Artifact jobs (one row per URL you intend to fetch)
CREATE TABLE IF NOT EXISTS artifact_jobs (
  job_id           TEXT PRIMARY KEY,
  work_id          TEXT NOT NULL,
  artifact_id      TEXT NOT NULL,
  canonical_url    TEXT NOT NULL,
  state            TEXT NOT NULL DEFAULT 'PLANNED',
  lease_owner      TEXT,
  lease_until      REAL,
  created_at       REAL NOT NULL,
  updated_at       REAL NOT NULL,
  idempotency_key  TEXT NOT NULL,
  UNIQUE(work_id, artifact_id, canonical_url),
  UNIQUE(idempotency_key),
  CHECK (state IN ('PLANNED','LEASED','HEAD_DONE','RESUME_OK','STREAMING','FINALIZED','INDEXED','DEDUPED','FAILED','SKIPPED_DUPLICATE'))
);
CREATE INDEX IF NOT EXISTS idx_artifact_jobs_state ON artifact_jobs(state);
CREATE INDEX IF NOT EXISTS idx_artifact_jobs_lease ON artifact_jobs(lease_until);

-- Exactly-once operation ledger (per side-effect)
CREATE TABLE IF NOT EXISTS artifact_ops (
  op_key        TEXT PRIMARY KEY,
  job_id        TEXT NOT NULL,
  op_type       TEXT NOT NULL,         -- HEAD | STREAM | FINALIZE | INDEX | DEDUPE
  started_at    REAL NOT NULL,
  finished_at   REAL,
  result_code   TEXT,
  result_json   TEXT,
  FOREIGN KEY(job_id) REFERENCES artifact_jobs(job_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_artifact_ops_job ON artifact_ops(job_id);

-- Optional: a tiny “runs” table if you want to tag jobs by run_id (keep lean)
-- CREATE TABLE IF NOT EXISTS runs(run_id TEXT PRIMARY KEY, started_at REAL, finished_at REAL);

UPDATE _meta SET value='3' WHERE key='schema_version';
COMMIT;
```

**Why this shape works**

* **Idempotency at two levels**: job (`idempotency_key`) and each **effect** (`op_key`).
* **Concurrency-safe leasing**: `lease_owner/lease_until` allow one worker at a time.
* **Monotonic state machine** enforced by code (see helpers below).

---

# Python helpers (drop-in snippets)

> Use these as your one true API for job planning/claiming/advancing and exactly-once logging.

## 1) Idempotency keys

```python
# idempotency.py
import hashlib, json

def _sha256_json(obj: dict) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def job_key(work_id: str, artifact_id: str, canonical_url: str) -> str:
    return _sha256_json({"v": 1, "kind": "JOB", "work": work_id, "art": artifact_id, "url": canonical_url})

def op_key(kind: str, job_id: str, **extras) -> str:
    d = {"v": 1, "kind": kind, "job": job_id}
    d.update(extras)
    return _sha256_json(d)
```

## 2) Plan job if absent (idempotent)

```python
# jobs.py
import sqlite3, time, uuid

def plan_job_if_absent(cx: sqlite3.Connection, *, work_id: str, artifact_id: str, canonical_url: str) -> str:
    ik = job_key(work_id, artifact_id, canonical_url)
    now = time.time()
    job_id = str(uuid.uuid4())
    try:
        cx.execute(
            """INSERT INTO artifact_jobs(job_id, work_id, artifact_id, canonical_url,
                                         state, created_at, updated_at, idempotency_key)
               VALUES (?, ?, ?, ?, 'PLANNED', ?, ?, ?)""",
            (job_id, work_id, artifact_id, canonical_url, now, now, ik),
        )
        return job_id
    except sqlite3.IntegrityError:
        row = cx.execute("SELECT job_id FROM artifact_jobs WHERE idempotency_key=?", (ik,)).fetchone()
        return row[0]
```

## 3) Leasing (claim/renew/release)

```python
import time

def lease_next_job(cx: sqlite3.Connection, *, owner: str, ttl_s: int = 120,
                   from_states=("PLANNED","FAILED")) -> dict | None:
    now = time.time()
    row = None
    # Claim 1 job atomically
    cx.execute(
        f"""
        UPDATE artifact_jobs
           SET lease_owner=?, lease_until=?, state='LEASED', updated_at=?
         WHERE job_id = (
           SELECT job_id FROM artifact_jobs
            WHERE state IN ({",".join("?"*len(from_states))})
              AND (lease_until IS NULL OR lease_until < ?)
            ORDER BY created_at
            LIMIT 1
         )
           AND (lease_until IS NULL OR lease_until < ?)
        """,
        (owner, now + ttl_s, now, *from_states, now, now),
    )
    if cx.total_changes == 1:
        row = cx.execute("SELECT * FROM artifact_jobs WHERE lease_owner=? AND lease_until>?",
                         (owner, now)).fetchone()
        return dict(row) if row else None
    return None

def renew_lease(cx: sqlite3.Connection, *, job_id: str, owner: str, ttl_s: int = 120) -> bool:
    now = time.time()
    cur = cx.execute(
        "UPDATE artifact_jobs SET lease_until=?, updated_at=? WHERE job_id=? AND lease_owner=?",
        (now + ttl_s, now, job_id, owner),
    )
    return cur.rowcount == 1

def release_lease(cx: sqlite3.Connection, *, job_id: str, owner: str) -> bool:
    cur = cx.execute(
        "UPDATE artifact_jobs SET lease_until=NULL, lease_owner=NULL, updated_at=? WHERE job_id=? AND lease_owner=?",
        (time.time(), job_id, owner),
    )
    return cur.rowcount == 1
```

## 4) State transitions (monotonic, checked)

```python
def advance_state(cx: sqlite3.Connection, *, job_id: str, to_state: str, allowed_from: tuple[str, ...]) -> None:
    now = time.time()
    qmarks = ",".join("?"*len(allowed_from))
    cur = cx.execute(
        f"UPDATE artifact_jobs SET state=?, updated_at=? WHERE job_id=? AND state IN ({qmarks})",
        (to_state, now, job_id, *allowed_from),
    )
    if cur.rowcount != 1:
        # capture current state for diagnostics
        row = cx.execute("SELECT state FROM artifact_jobs WHERE job_id=?", (job_id,)).fetchone()
        raise RuntimeError(f"state_transition_denied job={job_id} wants={to_state} from={allowed_from} have={row['state'] if row else 'missing'}")
```

## 5) Exactly-once op logging

```python
import json, time

def run_effect(cx: sqlite3.Connection, *, job_id: str, kind: str, opkey: str, effect_fn) -> dict:
    """
    effect_fn() performs the side effect and returns a dict (serializable).
    If opkey already exists, we return stored result_json instead (no re-run).
    """
    now = time.time()
    try:
        cx.execute(
            "INSERT INTO artifact_ops(op_key, job_id, op_type, started_at) VALUES (?,?,?,?)",
            (opkey, job_id, kind, now),
        )
    except sqlite3.IntegrityError:
        row = cx.execute("SELECT result_json FROM artifact_ops WHERE op_key=?", (opkey,)).fetchone()
        return json.loads(row["result_json"]) if (row and row["result_json"]) else {}

    result = effect_fn()  # do the thing exactly once
    cx.execute(
        "UPDATE artifact_ops SET finished_at=?, result_code=?, result_json=? WHERE op_key=?",
        (time.time(), result.get("code","OK"), json.dumps(result)[:20000], opkey),
    )
    return result
```

---

# Usage sketch (tying it together)

```python
def process_one(cx, owner_id, http, head_http, job: dict):
    job_id = job["job_id"]
    url = job["canonical_url"]

    # HEAD
    advance_state(cx, job_id=job_id, to_state="HEAD_DONE", allowed_from=("LEASED",))
    head_op = op_key("HEAD", job_id, url=url)
    def _do_head():
        r = head_http.head(url, follow_redirects=True, timeout=(5,10))
        return {"code":"OK", "status": r.status_code,
                "accept_ranges": r.headers.get("Accept-Ranges",""),
                "etag": r.headers.get("ETag"),
                "last_modified": r.headers.get("Last-Modified"),
                "content_length": int(r.headers.get("Content-Length") or 0)}
    head_res = run_effect(cx, job_id=job_id, kind="HEAD", opkey=head_op, effect_fn=_do_head)

    # Decide resume/fresh (your can_resume logic), then STREAM
    advance_state(cx, job_id=job_id, to_state="STREAMING", allowed_from=("HEAD_DONE","RESUME_OK"))
    stream_op = op_key("STREAM", job_id, url=url, range_start=head_res.get("resume_from", 0))
    def _do_stream():
        # call stream_to_part(...); return {"code":"OK", "bytes":..., "sha256":..., "elapsed_ms":...}
        ...
    stream_res = run_effect(cx, job_id=job_id, kind="STREAM", opkey=stream_op, effect_fn=_do_stream)

    # FINALIZE
    advance_state(cx, job_id=job_id, to_state="FINALIZED", allowed_from=("STREAMING",))
    fin_op = op_key("FINALIZE", job_id, sha256=stream_res["sha256"])
    def _do_finalize():
        # finalize_artifact(...); return {"code":"OK", "final_path": str(path), "size": size}
        ...
    fin_res = run_effect(cx, job_id=job_id, kind="FINALIZE", opkey=fin_op, effect_fn=_do_finalize)

    # INDEX, DEDUPE if applicable → advance_state to INDEXED/DEDUPED
    ...
```

**Crash safety:** If the process dies after finalize, re-running with the same `op_key("FINALIZE", ...)` simply returns the stored result — no double rename.

---

# Post-migration considerations (read before shipping)

1. **Lease TTL**: Start at **120s** and renew every **30–60s** during long streams. On timeout, other workers can claim the job.
2. **Failure policy**: On unrecoverable errors (bad `Content-Range`, quota), set `state='FAILED'` and **release** (or auto-retry with exponential backoff through your planner).
3. **Backfill (optional)**: if you want historical rows in `artifact_jobs`, generate them from `downloads` (one job per `{work_id, artifact_id, canonical_url}`), with `state='FINALIZED'` and no ops entries.
4. **Fast queries**: always filter jobs by `state` and order by `created_at` or `updated_at`. The provided indexes cover both.
5. **Ops visibility**: keep `artifact_ops` light (small `result_json`); when you need detailed payloads, store a small hash and keep the large blob in JSONL.
6. **Idempotency across runs**: because `job_key()` is deterministic, re-planning the same (work, artifact, url) in future runs returns the same `job_id` — the new run simply **observes** it’s already finalized/indexed and can skip.
7. **Purging**: jobs older than N days with `state IN ('FINALIZED','DEDUPED','SKIPPED_DUPLICATE')` can be archived to a CSV/Parquet and deleted to keep DB small.

---

## Minimal tests to add

* **Double plan** returns same `job_id`.
* **Two workers**: only one **leases**, the other gets `None`.
* **State guard**: advancing from `PLANNED` → `STREAMING` raises.
* **Exactly-once**: run `FINALIZE` twice with same `op_key` → second returns stored result.
* **Lease renewal**: renew succeeds only for the owner.
* **Recovery**: simulate crash after finalize; reconciler binds DB↔FS; new worker sees `FINALIZED` and skips.

---

If you want, I can also hand you a tiny **reconciler** snippet next (DB↔FS healing pass) and a **staging cleaner** for `.part` files, but with the migration + helpers above you can turn on the idempotent job runner immediately.
