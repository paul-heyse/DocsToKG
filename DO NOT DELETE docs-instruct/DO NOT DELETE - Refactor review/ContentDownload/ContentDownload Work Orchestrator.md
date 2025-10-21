# PR #8 — Work Orchestrator & Bounded Concurrency (Queue, Workers, Fairness, Resume)

> Paste this into `docs/pr8-orchestrator.md` (or your PR description).
> This PR introduces a **persistent work queue**, a **bounded-concurrency orchestrator**, and **graceful resume** — without changing the core pipeline, hishel caching, or telemetry contracts.

---

## Goals

1. Process large artifact sets **safely in parallel** with **global** and **per-resolver / per-host** limits.
2. Persist a **work queue** (SQLite by default) with **idempotent** enqueue, **crash-safe** in-progress recovery, and **retry** policy.
3. Keep **politeness**: respect per-resolver **rate limits** and **Retry-After** (already implemented), and prevent **burst concurrency** to a single host.
4. Provide a **CLI** for queue operations (enqueue/import, run/drain, stats, retry-failed, pause/resume).
5. Add **OTel metrics** & spans for scheduler/worker activity.

---

## New/updated file tree

```text
src/DocsToKG/ContentDownload/
  orchestrator/
    __init__.py
    models.py           # JobState enums; dataclasses for Job, Result
    queue.py            # SQLite-backed WorkQueue (enqueue, lease, ack, fail, retry)
    scheduler.py        # Orchestrator: worker pool, keyed semaphores, graceful shutdown
    workers.py          # Worker wrapper around pipeline.process()
    limits.py           # Per-resolver and per-host concurrency limiters (keyed semaphores)
  httpx/
    client.py           # MOD: make TokenBucket thread-safe (locks); optional sleep histogram
  bootstrap.py          # MOD: build orchestrator when --queue-run is invoked
  config/
    models.py           # MOD: add OrchestratorConfig and QueueConfig
  cli/
    app.py              # MOD: 'queue' subcommands (enqueue/import/run/stats/retry/pause/resume/drain)
  telemetry/
    otel.py             # MOD: add orchestrator metrics (queue depth, throughput)
tests/
  contentdownload/
    test_queue_basic.py         # enqueue/idempotence/lease/ack/fail/retry
    test_scheduler_limits.py    # per-resolver/host concurrency is respected
    test_scheduler_recovery.py  # crash recovery from in-progress → queued
    test_cli_queue.py           # smoke for CLI commands
```

---

## 1) Config additions

### `OrchestratorConfig` & `QueueConfig`

```python
# src/DocsToKG/ContentDownload/config/models.py
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Dict

class OrchestratorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_workers: int = 8                       # global concurrency
    max_per_resolver: Dict[str, int] = Field(default_factory=dict)  # e.g., {"unpaywall": 2}
    max_per_host: int = 4                      # default cap per host
    # Recovery / heartbeat
    lease_ttl_seconds: int = 600               # consider 'in_progress' stale after this
    heartbeat_seconds: int = 30                # workers update heartbeat to extend lease
    # Retries
    max_job_attempts: int = 3                  # pipeline-level failures
    retry_backoff_seconds: int = 60            # enqueue delay on fail (simple backoff)
    jitter_seconds: int = 15

class QueueConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: str = "sqlite"
    path: str = "state/workqueue.sqlite"
    wal_mode: bool = True

class ContentDownloadConfig(BaseModel):
    # ...
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    queue: QueueConfig = Field(default_factory=QueueConfig)
```

---

## 2) Queue schema & contract

### Job states

* `queued` → ready to run
* `in_progress` → leased by worker (with `worker_id`, `lease_expires_at`)
* `done` → success (store short outcome)
* `skipped` → non-error terminal (robots/not-modified)
* `error` → failed after `max_job_attempts` or explicitly failed

### SQLite schema

```sql
CREATE TABLE IF NOT EXISTS jobs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  artifact_id TEXT NOT NULL,            -- unique id or hash (e.g., doi:..., url:...)
  artifact_json TEXT NOT NULL,          -- JSON payload to reconstruct Artifact
  state TEXT NOT NULL,                  -- queued | in_progress | done | skipped | error
  attempts INTEGER NOT NULL DEFAULT 0,
  last_error TEXT,
  resolver_hint TEXT,                   -- optional (if known)
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  lease_expires_at TEXT,                -- when in_progress, else NULL
  worker_id TEXT                        -- current worker lease holder
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_artifact ON jobs(artifact_id);
CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state);
CREATE INDEX IF NOT EXISTS idx_jobs_updated ON jobs(updated_at);
```

### API skeleton

```python
# src/DocsToKG/ContentDownload/orchestrator/queue.py
from __future__ import annotations
import sqlite3, json, uuid
from datetime import datetime, timedelta
from typing import Optional, Iterable

class WorkQueue:
    def __init__(self, path: str, wal_mode: bool = True):
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA foreign_keys=ON")
        if wal_mode:
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self): ...
    def enqueue(self, artifact_id: str, artifact: dict, resolver_hint: Optional[str] = None) -> bool:
        """Idempotent: returns False if artifact_id already present."""
        ...
    def bulk_import(self, rows: Iterable[tuple[str, dict, Optional[str]]]) -> int: ...
    def lease(self, worker_id: str, limit: int, lease_ttl_sec: int) -> list[dict]:
        """Atomically move up to `limit` queued (or stale) jobs to in_progress for worker_id."""
        ...
    def heartbeat(self, worker_id: str) -> None: ...
    def ack(self, job_id: int, outcome: str, last_error: Optional[str] = None) -> None:
        """Outcome ∈ {'done','skipped','error'}."""
        ...
    def fail_and_retry(self, job_id: int, backoff_sec: int, max_attempts: int, last_error: str) -> None:
        """Increment attempts; if attempts < max_attempts, move to queued (delayed); else error."""
        ...
    def stats(self) -> dict: ...
```

* **Idempotence:** `artifact_id` unique index prevents duplicates.
* **Leasing:** `lease()` selects `queued` or **stale** `in_progress` (now - `lease_expires_at`) and assigns `worker_id` with a new `lease_expires_at`.
* **Heartbeat:** workers extend the lease (preventing premature recycle).
* **Retry:** on failure, increment `attempts`, and either requeue (with minimal backoff) or mark `error`.

---

## 3) Keyed concurrency limiters (fairness)

We enforce concurrency caps with **semaphores keyed** by `resolver_name` and **host** (derived from the *actual* download plan once known).

```python
# src/DocsToKG/ContentDownload/orchestrator/limits.py
from __future__ import annotations
import threading
from urllib.parse import urlsplit

class KeyedLimiter:
    def __init__(self, default_limit: int, per_key: dict[str,int] | None = None):
        self.default = max(1, default_limit)
        self.per_key = per_key or {}
        self._locks: dict[str, threading.Semaphore] = {}
        self._mutex = threading.Lock()

    def _sem_for(self, key: str) -> threading.Semaphore:
        with self._mutex:
            if key not in self._locks:
                limit = self.per_key.get(key, self.default)
                self._locks[key] = threading.Semaphore(limit)
            return self._locks[key]

    def acquire(self, key: str): self._sem_for(key).acquire()
    def release(self, key: str): self._sem_for(key).release()

def host_key(url: str) -> str:
    parts = urlsplit(url)
    host = parts.hostname or ""
    port = parts.port
    if (parts.scheme == "http" and port == 80) or (parts.scheme == "https" and port == 443):
        port = None
    return f"{host}{(':'+str(port)) if port else ''}"
```

* In a worker, **after resolving**, we know `plan.resolver_name` and `host_key(plan.url)` — we acquire both semaphores, run the download, then release.

---

## 4) Thread-safety in the client

Make `TokenBucket` thread-safe (we reuse one per resolver across workers):

```python
# src/DocsToKG/ContentDownload/httpx/client.py
import threading
class TokenBucket:
    def __init__(...):
        ...
        self._lock = threading.Lock()
    def _tick(self):
        now = time.monotonic()
        dt = now - self.t0; self.t0 = now
        self.tokens = min(self.capacity + self.burst, self.tokens + dt * self.refill)
    def consume(self, amount=1.0) -> float:
        with self._lock:
            self._tick()
            if self.tokens >= amount:
                self.tokens -= amount; return 0.0
            need = amount - self.tokens
            return need / self.refill
    def refund(self, amount=1.0):
        with self._lock:
            self._tick()
            self.tokens = min(self.capacity + self.burst, self.tokens + amount)
```

> Optional: emit a **rate-limit sleep histogram** (`contentdownload_rate_sleep_ms_bucket`) when `sleep_s > 0` to feed the Grafana panel.

---

## 5) Worker wrapper around pipeline

```python
# src/DocsToKG/ContentDownload/orchestrator/workers.py
from __future__ import annotations
import threading, json, time
from opentelemetry import trace
from DocsToKG.ContentDownload.orchestrator.limits import KeyedLimiter, host_key
from DocsToKG.ContentDownload.api.types import DownloadOutcome
from DocsToKG.ContentDownload.pipeline import ResolverPipeline

class Worker:
    def __init__(self, worker_id: str, queue, pipeline: ResolverPipeline,
                 resolver_limiter: KeyedLimiter, host_limiter: KeyedLimiter,
                 heartbeat_sec: int, max_job_attempts: int, retry_backoff: int, jitter: int):
        self.worker_id = worker_id
        self.queue = queue
        self.pipeline = pipeline
        self.rlim = resolver_limiter
        self.hlim = host_limiter
        self.heartbeat_sec = heartbeat_sec
        self.max_job_attempts = max_job_attempts
        self.retry_backoff = retry_backoff
        self.jitter = jitter
        self._stop = threading.Event()
        self._tracer = trace.get_tracer("DocsToKG-ContentDownload")

    def stop(self): self._stop.set()

    def run_one(self, job: dict):
        # Rehydrate artifact
        artifact = self._artifact_from_json(job["artifact_json"])

        # Span per job
        with self._tracer.start_as_current_span("job", attributes={"job.id": job["id"], "artifact.id": job["artifact_id"]}):
            try:
                # Use pipeline to resolve FIRST plan (to know keys). We let pipeline try resolvers in order,
                # but we only acquire semaphores around the actual streaming.
                # Hook: pipeline should call a callback before stream to allow semaphore acquisition.
                outcome: DownloadOutcome = self.pipeline.process(artifact, ctx=None)
                # Ack
                state = "done" if outcome.ok else ("skipped" if outcome.classification == "skip" else "error")
                self.queue.ack(job["id"], state, last_error=outcome.reason)
            except Exception as e:
                self.queue.fail_and_retry(job["id"], self._backoff(), self.max_job_attempts, last_error=str(e))

    def _backoff(self) -> int:
        import random
        return self.retry_backoff + random.randint(0, self.jitter)

    def _artifact_from_json(self, s: str):
        d = json.loads(s)
        # Your Artifact adapter here; for now return a SimpleNamespace/dict
        return d
```

> **Semaphore hooks:** Easiest approach is to wrap **download execution** with a small callback/hook from the pipeline to acquire/release semaphores around `stream_candidate_payload(...)`. (Alternatively, expose a `with_limits(resolver_name, url):` context manager passed into the pipeline.)

### Add limits around the streaming call (in `pipeline.process`)

```python
# src/DocsToKG/ContentDownload/pipeline.py (injectable limiter)
class ResolverPipeline:
    def __init__(..., resolver_limiter=None, host_limiter=None):
        self._rlim = resolver_limiter
        self._hlim = host_limiter

    def process(...):
        ...
        for plan in rres.plans:
            # Acquire keyed semaphores only for the streaming window
            hkey = host_key(plan.url) if self._hlim else None
            if self._rlim: self._rlim.acquire(plan.resolver_name)
            if self._hlim: self._hlim.acquire(hkey)
            try:
                # prepare → stream → finalize (unchanged)
                stream = stream_candidate_payload(...)
                outcome = finalize_candidate_download(...)
            finally:
                if self._hlim: self._hlim.release(hkey)
                if self._rlim: self._rlim.release(plan.resolver_name)
            ...
```

---

## 6) Scheduler (worker pool, leasing, heartbeat, shutdown)

```python
# src/DocsToKG/ContentDownload/orchestrator/scheduler.py
from __future__ import annotations
import threading, time, uuid
from queue import Queue, Empty
from typing import Optional
from DocsToKG.ContentDownload.orchestrator.queue import WorkQueue
from DocsToKG.ContentDownload.orchestrator.workers import Worker
from DocsToKG.ContentDownload.orchestrator.limits import KeyedLimiter
from opentelemetry import metrics

class Orchestrator:
    def __init__(self, cfg, queue: WorkQueue, pipeline, telemetry):
        self.cfg = cfg
        self.queue = queue
        self.pipeline = pipeline
        self.telemetry = telemetry
        self.worker_id = f"cdw-{uuid.uuid4()}"
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._jobs_q: "Queue[dict]" = Queue(maxsize=cfg.orchestrator.max_workers * 2)
        self._rlim = KeyedLimiter(default_limit=cfg.orchestrator.max_per_resolver or 1,
                                  per_key=cfg.orchestrator.max_per_resolver)
        self._hlim = KeyedLimiter(default_limit=cfg.orchestrator.max_per_host)
        # Metrics
        meter = metrics.get_meter("DocsToKG-ContentDownload")
        self.m_queue_depth = meter.create_up_down_counter("contentdownload_queue_depth", unit="1")
        self.m_throughput = meter.create_counter("contentdownload_jobs_completed_total", unit="1")

    def start(self):
        # Start worker threads
        for _ in range(self.cfg.orchestrator.max_workers):
            w = Worker(self.worker_id, self.queue, self.pipeline, self._rlim, self._hlim,
                       self.cfg.orchestrator.heartbeat_seconds,
                       self.cfg.orchestrator.max_job_attempts,
                       self.cfg.orchestrator.retry_backoff_seconds,
                       self.cfg.orchestrator.jitter_seconds)
            t = threading.Thread(target=self._worker_loop, args=(w,), daemon=True)
            t.start()
            self._threads.append(t)
        # Start dispatcher
        dt = threading.Thread(target=self._dispatcher_loop, daemon=True)
        dt.start()
        self._threads.append(dt)
        # Start heartbeat thread
        hb = threading.Thread(target=self._heartbeat_loop, daemon=True)
        hb.start()
        self._threads.append(hb)

    def stop(self):
        self._stop.set()
        for t in self._threads:
            t.join(timeout=5)

    def _dispatcher_loop(self):
        while not self._stop.is_set():
            # Lease up to backlog slots
            free_slots = self._jobs_q.maxsize - self._jobs_q.qsize()
            if free_slots > 0:
                jobs = self.queue.lease(self.worker_id, free_slots, self.cfg.orchestrator.lease_ttl_seconds)
                for j in jobs:
                    self._jobs_q.put(j)
            # Emit queue depth metric
            stats = self.queue.stats()
            self.m_queue_depth.add(stats.get("queued", 0) - stats.get("leased", 0))
            time.sleep(1.0)

    def _heartbeat_loop(self):
        while not self._stop.is_set():
            self.queue.heartbeat(self.worker_id)
            time.sleep(self.cfg.orchestrator.heartbeat_seconds)

    def _worker_loop(self, worker: Worker):
        while not self._stop.is_set():
            try:
                job = self._jobs_q.get(timeout=1.0)
            except Empty:
                continue
            try:
                worker.run_one(job)
                self.m_throughput.add(1)
            finally:
                self._jobs_q.task_done()
```

---

## 7) CLI — queue commands

```python
# src/DocsToKG/ContentDownload/cli/app.py
queue_app = typer.Typer(help="Persistent work queue")
app.add_typer(queue_app, name="queue")

@queue_app.command("enqueue")
def queue_enqueue(config: Optional[str] = typer.Option(None, "--config", "-c"),
                  input: Optional[str] = typer.Option(None, "--input", "-i",
                    help="NL-delimited artifact JSON (or '-' for stdin)")): ...

@queue_app.command("import")
def queue_import(config: Optional[str] = typer.Option(None, "--config", "-c"),
                 file: str = typer.Argument(..., help="JSONL file of artifacts")): ...

@queue_app.command("stats")
def queue_stats(config: Optional[str] = typer.Option(None, "--config", "-c")): ...

@queue_app.command("run")
def queue_run(config: Optional[str] = typer.Option(None, "--config", "-c"),
              drain: bool = typer.Option(False, "--drain", help="Exit when queue empty")):
    cfg = load_config(config)
    q = WorkQueue(cfg.queue.path, wal_mode=cfg.queue.wal_mode)
    # Build pipeline with bootstrap pieces (hishel/httpx/clients/resolvers/telemetry)
    pipeline = build_pipeline_from_config(cfg)  # small helper in bootstrap
    orch = Orchestrator(cfg, q, pipeline, telemetry=None)
    orch.start()
    try:
        if drain:
            while q.stats().get("queued",0) + q.stats().get("leased",0) > 0:
                time.sleep(2)
        else:
            while True: time.sleep(5)
    except KeyboardInterrupt:
        orch.stop()

@queue_app.command("retry-failed")
def queue_retry_failed(...): ...
@queue_app.command("pause")    # optional: implementation via a DB flag
def queue_pause(...): ...
@queue_app.command("resume")
def queue_resume(...): ...
```

> `build_pipeline_from_config(cfg)` is a thin bootstrap seam that returns a `ResolverPipeline` configured exactly as `run_from_config` does.

---

## 8) OTel metrics & spans for orchestrator

Add to `telemetry/otel.py` or within `scheduler.py`:

* **Metrics**

  * `contentdownload_queue_depth` (UpDownCounter)
  * `contentdownload_jobs_completed_total` (Counter)
  * `contentdownload_jobs_failed_total` (Counter)
  * `contentdownload_jobs_inflight` (UpDownCounter) — optional

* **Spans**

  * `orchestrator.dispatch` — scheduling loop (attributes: leased count)
  * `job` — per job (already in Worker)
  * When acquiring semaphores: add events (`resolver_concurrency_wait_ms`, `host_concurrency_wait_ms`) if you want visibility into concurrency waits.

---

## 9) Testing plan

1. **Queue basics**

   * `enqueue` twice → second returns False; one row present.
   * `lease(limit=2)` moves queued → in_progress with lease/worker_id set.
   * `ack(done)` transitions to `done`.
   * `fail_and_retry` increments `attempts` and requeues until max → `error`.

2. **Scheduler limits**

   * Use a fake pipeline where `stream` sleeps for 0.5s.
   * Set `max_workers=8`, `max_per_resolver=2`, `max_per_host=1`; enqueue 20 jobs with same resolver+host.
   * Assert parallelism never exceeds caps (by measuring concurrent entries).
   * Total wall time ≈ jobs / caps × duration (within tolerance).

3. **Recovery**

   * Lease a job, simulate crash (no ack).
   * After `lease_ttl_seconds`, `lease()` returns that job again.
   * Worker processes it to completion.

4. **CLI smoke**

   * `queue enqueue -i file.jsonl` → `queue stats` shows queued>0.
   * `queue run --drain` completes and `queue stats` shows 0 queued+leased, >0 done.

5. **OTel**

   * Use **in-memory exporters** to assert span/metric emission for dispatcher and jobs.

---

## 10) Acceptance checklist

* [ ] Queue is **idempotent**, **crash-safe**, and supports **retry** with backoff.
* [ ] Orchestrator enforces **global**, **per-resolver**, and **per-host** caps.
* [ ] TokenBucket is **thread-safe**; politeness preserved (rate-limit + Retry-After).
* [ ] Pipeline unchanged; semaphores guard only the **streaming** window.
* [ ] CLI supports **enqueue/import/run/stats/retry-failed** (pause/resume optional).
* [ ] OTel metrics/spans expose queue depth, throughput, and job spans.
* [ ] Tests green; behavior deterministic.

---

## 11) Design notes & guardrails

* **Keep scheduling simple**: Let workers resolve plans, then apply keyed semaphores around streaming. We avoid pre-resolution complexity in the scheduler.
* **Avoid URL labels in metrics**: Stick to `resolver` & `host` where necessary; don’t label by full URL.
* **SQLite WAL**: Enables concurrent readers/writers; good for a single host. If you need **multi-host** queueing, graduate to Postgres or Redis streams later.
* **Backpressure**: `Queue(maxsize=2×workers)` avoids over-leasing; the dispatcher leases only when there’s capacity.
* **Graceful shutdown**: `KeyboardInterrupt` sets a stop flag; workers finish current jobs; leases for incomplete jobs expire and recover next run.

---

## 12) Future extensions (enabled by this design)

* **Multi-process** or **multi-host** pool (switch queue backend; workers unchanged).
* **Priority queues** (add `priority` column; order by `priority DESC, created_at`).
* **Domain-aware fairness** (round-robin by resolver/host; today we rely on keyed semaphores).
* **Dynamic throttling** (adapt `max_per_host` when `retry-after` spikes).
* **Batching** (group artifacts by host/resolver to improve locality).

---

### Minimal “diff guide” (what changes where)

* New modules under `orchestrator/` for queue and scheduler.
* `httpx/client.py` TokenBucket adds a lock.
* `pipeline.process` acquires/releases keyed semaphores around **streaming**.
* `bootstrap.py` adds `build_pipeline_from_config(cfg)` and wires orchestrator in `queue run`.
* `cli/app.py` gains `queue` subcommands.

---

This PR gives you **throughput** without sacrificing **politeness** or **observability**. It’s deliberately conservative (SQLite single host, thread pool), but the boundaries make it easy to evolve to multi-host and more sophisticated scheduling later.
