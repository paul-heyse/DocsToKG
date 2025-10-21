Absolutely—here’s a **surgical, production-grade implementation plan** for the **next optimization: unify execution with a generic runner** (the “single runner” we’ve been designing). It’s written to be AI-agent ready: small commits, clear seams, invariants, and testable acceptance for each step—no code, only exact instructions.

---

# Scope & intent

**Goal**
Replace the three stage-specific loops (Doctags → Chunk → Embed) with **one generic runner** that handles discovery, resume/force, parallelism, timeouts, retries, error budgets, manifests, progress, and telemetry. Stages contribute only:

1. a **Plan** (which items to work on),
2. a **Worker** (how to process one item),
3. optional **Hooks** (stage-wide setup/teardown; light per-item taps).

**Non-goals (this PR)**

* No behavior change in the actual conversion/chunking/embedding logic beyond using the runner.
* No new orchestrator (Prefect) surface.
* No provider refactor (dense/sparse/lexical) beyond what’s necessary to call existing functions; the full providers layer lands in a separate PR.

---

# Target design (authoritative contracts)

## 1) Contracts (data-only)

* **`StagePlan`**

  * `stage_name: str` (e.g., "doctags", "chunk", "embed")
  * `items: Iterable[WorkItem]` (deterministic order)
  * `total_items: int` (for progress, budgeting)

* **`WorkItem`** (immutable)

  * `item_id: str` (stable relative id, e.g., relpath without extension)
  * `inputs: dict[str, str|Path]` (stage-specific paths; strings OK)
  * `outputs: dict[str, str|Path]` (declared expected artifacts)
  * `cfg_hash: str` (stage-relevant config hash)
  * `cost_hint: float` (pages, bytes, chunk count—used by SJF; default 1.0)
  * `meta: dict[str, Any]` (small hints only; optional)
  * `satisfies() -> bool` (resume predicate: “all declared outputs exist & are non-empty (and fingerprints match if enabled)”)

* **`StageOptions`** (shared knobs)

  * `policy: "io"|"cpu"|"gpu"` → ThreadPool | ProcessPool(spawn) | ThreadPool
  * `workers: int`
  * `schedule: "fifo"|"sjf"` (SJF uses `cost_hint`)
  * `per_item_timeout_s: float` (0 = off)
  * `retries: int` & `retry_backoff_s: float` (with jitter)
  * `error_budget: int` (0 = stop on first failure)
  * `max_queue: int` (submission backpressure)
  * `diagnostics_interval_s: float`
  * `fingerprinting: bool` (exact resume requires `input_sha256 ∧ cfg_hash`)

* **`StageHooks`** (all optional; must not throw)

  * `before_stage(ctx) -> None` (allocate shared resources)
  * `after_stage(outcome) -> None` (cleanup; write summaries)
  * `before_item(item, ctx) -> None` (cheap, e.g., annotate size)
  * `after_item(item, result_or_error, ctx) -> None` (enrich telemetry)

* **`StageOutcome`**

  * `scheduled, skipped, succeeded, failed, cancelled: int`
  * `wall_ms, queue_p50_ms, exec_p50_ms, exec_p95_ms, items_per_s`
  * `errors: list[StageErrorSummary]` (first & most frequent)

* **`StageError`**

  * `{stage, item_id, category, message, retryable, detail}`
  * `category ∈ {input, config, runtime, timeout, io, provider, unknown}`

> These are pure data shapes: **no imports of heavy libs** in this module.

---

# Step-by-step implementation (small commits)

## Commit A — Core runner module

**Create:** `core/executors.py` (or equivalent) with:

* Type definitions above.
* `run_stage(plan, worker, options, hooks) -> StageOutcome` that:

  1. **Builds queue** from `plan.items` (apply `resume` via `WorkItem.satisfies()` unless `force`).
  2. **Schedules** items on the right executor:

     * `policy="io"` / `"gpu"` → ThreadPool
     * `policy="cpu"` → ProcessPool with **spawn**; pass only paths & scalars (pickle-safe).
  3. **Timeout**: wrap future wait; if exceeded, raise `StageError{category="timeout", retryable=False}`.
  4. **Retries**: on `retryable=True`, resubmit with exponential backoff + jitter (bounded by `retries`).
  5. **Error budget**: when `failed > error_budget`, stop new submissions, optionally cancel inflight (default: let inflight complete).
  6. **Progress & diagnostics**: single-line status every `diagnostics_interval_s` with done/total, succ/skip/fail, ETA, items/s; optional verbose slowest items and top error categories.
  7. **Manifests**: call injected writer helpers (see Commit B) to append success/skip/failure rows (atomic).
  8. **Telemetry**: emit counters/histograms via the existing sink if present.
  9. Return `StageOutcome`.

**Invariants**

* Deterministic iteration order (no implicit shuffle).
* Skip semantics identical across stages.
* No heavy imports at module top.

**Acceptance**

* Unit: synthetic “worker” with knobs to force success/fail/timeout/retry; validate counts, p50/p95, error budget, and cancellation.

---

## Commit B — Manifest & telemetry integration

**Goal:** The runner calls **one** set of helpers for all stages; no bespoke writing in stage code.

* Inject a **lock-aware JSONL writer** (from the PR-2 work): context manager that acquires a FileLock on `*.jsonl.lock` and calls atomic append.
* Standardize three write paths in a small adapter module (`core/manifest_sink.py`):

  * `log_config(stage, snapshot)` → writes `__config__` row at stage start.
  * `log_success(stage, item_id, input_path, output_path(s), duration, extras)`
  * `log_skip(stage, item_id, input_path, output_path, reason, duration, extras)`
  * `log_failure(stage, item_id, input_path, output_path, error, duration, extras)`
* Runner invokes these uniformly.

**Acceptance**

* All three stage manifests show identical base fields; only extras differ by stage (e.g., chunk counts vs vector dims).

---

## Commit C — Progress & diagnostics aggregator

* Build a small `Progress` utility: holds counters, moving averages for items/s, queue depth; prints one-line `"[doctags] 234/1200 | succ 230 | skip 2 | fail 2 | 92.4 it/s | ETA 00:10:14"`.
* Allow verbose mode (per `-v`) to add slowest items and top error categories every few intervals.
* Ensure **no TTY interference** with logs (write via stderr or a dedicated logger).

**Acceptance**

* With a synthetic plan of 1k items, progress updates at set cadence; no log spam.

---

## Commit D — Stage 1 integration: **Doctags → runner**

**Plan builder**

* Scan input root for `*.pdf` / `*.html` (or use your existing iterator).
* Derive `rel_id` and build `item_id` (stable, relative).
* Compute `outputs` (Doctags path) and `cfg_hash` (doctags-relevant fields).
* Set `cost_hint`: pages for PDF; bytes for HTML fallback.

**Worker**

* Invoke existing per-file doctags conversion (no changes).
* Return **result payload** for manifests: blocks/tables/figures, pages, bytes_in/out.

**Hooks**

* `before_stage`: write `__config__` manifest row (engine, models, dirs).
* `after_stage`: optional: write small summary (counts).

**Runner defaults**

* `policy="io"`, `workers=stage default`, `schedule="fifo"`.

**Acceptance**

* CLI `doctags` uses runner; per-item manifests unchanged except consistent base fields.
* Progress and error budget behave identically to embed/chunk after later steps.

---

## Commit E — Stage 2 integration: **Chunk → runner**

**Plan builder**

* Scan Doctags output for `*.jsonl` (and later `*.parquet` if you adopt that fully).
* Map to chunk outputs (format from config); compute `cfg_hash`.
* `cost_hint`: number of doc blocks (or estimated token count if cheaply available; otherwise file size).

**Worker**

* Invoke existing chunker; return counts (chunks, tokens p50/p95, empty chunks).

**Runner defaults**

* `policy="cpu"` (ProcessPool, spawn), `workers=stage default`, `schedule="fifo"`.

**Acceptance**

* `chunk` CLI now uses runner; manifests contain chunk-specific extras; performance equal or better (no pool duplication).

---

## Commit F — Stage 3 integration: **Embed → runner**

**Plan builder**

* Scan Chunks output for `*.parquet` / `*.jsonl` chunks.
* For each file, decide enabled families (dense/sparse/lexical).
* Compute outputs (one per family, depending on vector format); compute `cfg_hash`.
* `cost_hint`: predicted number of chunks (can read file metadata/header cheaply).

**Hooks**

* `before_stage`: open any shared resources (e.g., current embedder objects, HTTP sessions), stash in a small stage context.

  > If you haven’t refactored to providers yet, this hook instantiates the existing model/server clients; you can migrate to providers in the dedicated PR later.
* `after_stage`: close resources; write a **corpus summary** row (you already do this in embedding) with totals (vectors by family, p95/99 nnz, norm distribution).

**Worker**

* Invoke existing embedding path for the families requested; return per-file counts (vectors per family, dims, batches, elapsed per family).

**Runner defaults**

* `policy="gpu"` (ThreadPool), `workers=stage default`, `schedule="fifo"`.
* If your embed logic already rate-limits internally, ensure `workers` × provider inflight == safe value (document it; we’ll auto-tune later).

**Acceptance**

* `embed` CLI uses runner; per-file and `__corpus__` manifests remain; outcomes mirror existing behavior with cleaner progress.

---

## Commit G — CLI wiring & options mapping

* Each stage command (Typer) builds `StageOptions` from the stage slice of the **effective config** (remember: **CLI > ENV > profile > defaults** once PR-7 lands; for now, parse flags and build options directly).
* Map shared knobs: `workers`, `policy`, `schedule`, `retries`, `retry_backoff_s`, `per_item_timeout_s`, `error_budget`, `max_queue`, `diagnostics_interval_s`, `fingerprinting`.

**Acceptance**

* `--retries`, `--timeout-s`, `--error-budget` now work identically across stages; help text is consistent.

---

## Commit H — Tests (runner + stages)

### Unit (runner semantics)

* **FIFO vs SJF**: with mixed `cost_hint`, SJF completes earlier (lower p95).
* **Timeout**: injected slow worker triggers `timeout` error; item counted failed; outcome correct.
* **Retries**: a worker that fails once then succeeds → total attempts tracked; final success up-counts.
* **Error budget**: budget=1; two failures → second item not submitted; `cancelled=true`.
* **Cancellation**: simulated SIGINT sets cancel flag; runner returns outcome with `cancelled=true`.

### Integration (per stage)

* Each stage run produces identical artifacts as before (bitwise/equality for JSONL; schema/row counts for Parquet).
* Manifests contain stage extras and consistent base fields (`stage, doc_id, status, duration_s, input_path, output_path, schema_version`).

---

## Commit I — Docs & migration note

* Update the developer docs:

  * “Writing a new stage” = implement plan+worker(+hooks), then call `run_stage`.
  * Runner semantics: timeouts, retries, error budget, resume/force.
  * How to set `cost_hint` (pages, bytes, chunk counts) per stage.

* CHANGELOG: “All DocParsing stages now use a single runner; behavior is unchanged but progress, timeouts, and retries are now uniform.”

---

# Optional gold-tier add-ons (guarded flags; can be follow-ups)

* **SJF on by default** for heterogeneous workloads (kept in this PR if trivial to enable; otherwise land later).
* **Adaptive concurrency**: slowly increase `workers` when p95 stable and failure rate low; decrease on spikes (off by default; expose `runner.adaptive`).
* **Attempts stream**: if you maintain a separate “attempts” JSONL, write `started` and `finished` entries around each item; useful for external monitoring.

---

# Acceptance (definition of “done”)

* `doctags`, `chunk`, and `embed` subcommands all call **the same runner**.
* Stage modules **no longer own** pools/progress/manifest logic.
* Timeouts/retries/error budget/resume behave identically across stages.
* Manifests share consistent base fields; stage-specific extras remain intact.
* CI: runner unit tests + stage integration tests pass; performance is on par or better (lower p95 due to SJF if enabled).

---

# Risks & mitigations

* **GPU/fork safety:** never use ProcessPool for embedding; default to ThreadPool; document that override is “dangerous” and block it unless an explicit `--i-know-what-im-doing` flag is set.
* **Pickle errors:** keep `WorkItem` small & immutable; pass only paths/scalars/functions defined at top level.
* **Partial file corruption:** all writes must be atomic (use existing PR-2 writer); never expose temp paths.
* **Behavior drift:** keep stage workers unchanged; only move loops. Compare outputs with a baseline corpus before merging.

---

# Rollout plan

1. Land the runner with **Doctags** only (env flag `DOCSTOKG_RUNNER=new` to enable).
2. Add **Chunk**; run the same corpus end-to-end; compare outputs.
3. Add **Embed**; verify per-file and `__corpus__` manifests; performance parity.
4. Flip default to the new runner (keep `DOCSTOKG_RUNNER=legacy` escape for one minor release).
5. Delete legacy loops after deprecation window.

---

# Quick “who-touches-what” checklist

* **New:** `core/executors.py` (contracts + runner)
* **New:** `core/manifest_sink.py` (3 helpers; uses existing atomic JSONL writer)
* **Doctags:** plan builder + worker adapter; remove local pool/loop
* **Chunk:** plan builder + worker adapter; remove local pool/loop
* **Embed:** plan builder + worker adapter; remove local pool/loop; stage hooks open/close shared resources
* **CLI:** map Typer options → `StageOptions`; pass to runner
* **Tests:** runner unit tests; stage integration; snapshot manifest rows (shape only)
* **Docs:** “authoring a stage,” runner semantics, migration note

---

Once this is in, the **providers** refactor (dense/sparse/lexical behind small interfaces) plugs straight into the **hooks + worker** seam and becomes a tiny follow-up PR—cleanly isolated from scheduling, resume, and telemetry.

Here’s a **single-source architecture guide** for the **Runner**—what it is, what it isn’t, and exactly how it connects to the rest of `DocParsing`. Treat this as the reference your team and agents can code to.

---

# 1) Purpose at a glance

**Runner = orchestration kernel** for a stage. It owns **scheduling, concurrency, retries, timeouts, resume/force, manifests, progress, telemetry**.
A stage (Doctags / Chunk / Embed) supplies only:

1. a **Plan** (what items exist),
2. a **Worker** (how to process one item),
3. optional **Hooks** (stage-wide setup/teardown & light per-item tap-ins).

Everything else is the Runner.

---

# 2) System map (connectivity)

```
                ┌──────────────────────────────────────────────────────────┐
                │                        Typer CLI                         │
                │     (profiles → ENV → CLI) builds App/Runner/Stage cfg   │
                └───────────────┬───────────────────────────┬──────────────┘
                                │                           │
                                ▼                           ▼
                    ┌───────────────────┐          ┌───────────────────┐
                    │  Stage Plan       │          │ Stage Hooks       │
                    │ (iter WorkItems)  │          │ before/after stage│
                    └─────────┬─────────┘          │ before/after item │
                              │                    └─────────┬─────────┘
                              │                              │(context)
                              ▼                              ▼
                     ┌───────────────────┐          ┌───────────────────┐
                     │      Runner       │──────────│   Worker(item)    │
                     │  (this proposal)  │   calls  │  stage-specific   │
                     └──┬─────────────┬──┘          └─────────┬─────────┘
                        │             │                        │
                        │             │                        │
   telemetry+manifests  │             │  outputs               │  libraries/storage
   (atomic JSONL, lock) │             │   (paths)              │  (Docling/Arrow/Parquet/Providers)
       ┌────────────────▼───┐     ┌───▼────────────────┐   ┌───▼────────────────────────┐
       │ Manifest Sink      │     │ Resume/Force test  │   │ Stage deps (Docling,       │
       │  (FileLock+append) │     │ Fingerprint, exist │   │ ParquetWriter, Providers)  │
       └────────────────────┘     └────────────────────┘   └────────────────────────────┘

```

* **Runner** uses an **Executor** (ThreadPool for IO/GPU; ProcessPool(spawn) for CPU).
* **Worker** uses stage dependencies only (Docling for Doctags, Chunker & Parquet writer for Chunks, Providers/Vector writers for Embed).
* **Manifest Sink** is injected (atomic JSONL writer behind `FileLock`), so Runner and Telemetry never roll their own locks.

---

# 3) Responsibilities split (RACI)

| Concern                                | Runner        | Stage Plan | Worker | Hooks | Manifest Sink | Providers/Storage |
| -------------------------------------- | ------------- | ---------: | -----: | ----: | ------------: | ----------------: |
| Item discovery                         |               |      **R** |        |       |               |                   |
| Scheduling (FIFO/SJF), queues          | **R**         |            |        |       |               |                   |
| Concurrency policy (io/cpu/gpu)        | **R**         |            |        |       |               |                   |
| Timeout, retries, error budget         | **R**         |            |        |       |               |                   |
| Resume/force & fingerprints            | **R**         |      **C** |        |       |               |                   |
| Progress/ETA                           | **R**         |            |        |       |               |                   |
| Manifests (success/skip/failure)       | **R** → **I** |            |  **C** |       |         **R** |                   |
| Telemetry (counters, spans)            | **R**         |            |        | **C** |               |                   |
| Stage-wide resources (e.g., providers) |               |            |        | **R** |               |             **I** |
| Stage logic (parse/chunk/embed)        |               |            |  **R** |       |               |             **I** |
| Atomic writes & locks                  |               |            |        |       |         **R** |             **I** |

R = Responsible, C = Consulted, I = Informed.

---

# 4) Core contracts (data shapes)

Keep these as **pure dataclasses/Pydantic models** (no heavy imports).

## 4.1 StagePlan

* `stage_name: Literal["doctags","chunk","embed"]`
* `items: Iterable[WorkItem]` (stable order; no randomness)
* `total_items: int`

## 4.2 WorkItem

* `item_id: str` (stable relative id)
* `inputs: dict[str, Path|str]`
* `outputs: dict[str, Path|str]`
* `cfg_hash: str` (hash of stage-relevant config)
* `cost_hint: float = 1.0` (pages, bytes, chunk count)
* `meta: dict[str, Any] = {}`
* `satisfies() -> bool` (resume: every declared output exists & non-empty; if fingerprinting → input_sha256==stored ∧ cfg_hash match)

## 4.3 StageOptions

* `policy: "io"|"cpu"|"gpu"` → ThreadPool | ProcessPool(spawn) | ThreadPool
* `workers: int`
* `schedule: "fifo"|"sjf"`
* `per_item_timeout_s: float` (0=off)
* `retries: int`, `retry_backoff_s: float` (+ jitter)
* `error_budget: int` (0=stop on first)
* `max_queue: int` (cap submissions)
* `diagnostics_interval_s: float`
* `fingerprinting: bool`

## 4.4 StageHooks

* `before_stage(ctx)` / `after_stage(outcome)`
* `before_item(item, ctx)` / `after_item(item, result|error, ctx)`

> Hooks **must not throw**; exceptions become warnings.

## 4.5 StageOutcome

* `scheduled, skipped, succeeded, failed, cancelled: int`
* `wall_ms, queue_p50_ms, exec_p50_ms, exec_p95_ms, items_per_s`
* `errors: list[StageErrorSummary]`

## 4.6 StageError

* `stage, item_id, category, message, retryable, detail`
* `category ∈ {input, config, runtime, timeout, io, provider, unknown}`

---

# 5) Control flow (sequence diagrams)

## 5.1 Success path with resume

```
CLI → Runner: build StageOptions from Settings
Runner → Plan: iterate WorkItems
Runner → WorkItem: satisfies()?  ──> yes → log_skip
                       │
                       └──> no → queue.submit(worker(item))
Executor → Worker: process
Worker → Manifest Sink: (handled by Runner) log_success(extras)
Runner → Progress: tick; update ETA, p50/p95
```

## 5.2 Failure with retries and error budget

```
Executor → Worker: raises ProviderError(retryable=True)
Runner: attempts < retries? yes → resubmit with backoff+jitter
... still fails ...
Runner: log_failure; failed++
Runner: failed > error_budget? if yes → stop new submissions; let inflight finish (or cancel, per policy)
Runner: outcome.cancelled=true
```

## 5.3 Timeout path

```
Runner: wait(future, timeout=per_item_timeout_s)
timeout → cancel future (best effort; threadpool cannot kill user code; mark failure with category="timeout")
log_failure; count failed; continue or cancel per budget
```

## 5.4 Embedding with providers (Hooks)

```
before_stage: open providers (dense/sparse/lexical); store in ctx
for each WorkItem:
  before_item: annotate expected chunk count
  worker: call providers.embed/encode/vector → writers
  after_item: attach provider tags/timings for telemetry
after_stage: summarize corpus stats, close providers
```

---

# 6) Scheduling & pools

* **policy → executor**

  * `"io"` (Doctags): ThreadPool (I/O bound; C-extensions often release GIL)
  * `"cpu"` (Chunk): ProcessPool with **spawn** (tokenization/splitting compute)
  * `"gpu"` (Embed): ThreadPool (GPU libs not fork-safe; providers internally batch/queue)
* **SJF** (shortest-job-first): sort by `cost_hint` ascending to reduce tail latency; keep FIFO fallback.
* **Backpressure**: `max_queue` caps queued submissions. For embedding, set provider inflight ≤ workers to avoid overloading GPU/HTTP.
* **Pickling rules (ProcessPool)**: `WorkItem` must be small & pickleable; pass **paths** and small scalars only.

---

# 7) Resume/force & fingerprints

* **Resume**: skip if `WorkItem.satisfies()` returns true.
* **Satisfies** base predicate: all declared outputs exist and are **non-empty**.
* **Fingerprint mode** (recommended): require both `input_sha256` and `cfg_hash` to match a sidecar `*.fp.json`. If mismatch → recompute.
* **Force**: ignore satisfies; recompute and **atomically replace** outputs.

---

# 8) Manifests & telemetry (uniform)

* **Uniform JSONL rows** (atomic append with `FileLock`):

  * **success**: `{stage, doc_id, status:"success", duration_s, input_path, output_path(s), schema_version, extras…}`
  * **skip**: `{…, status:"skip", reason, …}`
  * **failure**: `{…, status:"failure", error, …}`
* **Stage-specific extras**:

  * Doctags: blocks/tables/pages, parse engine/model
  * Chunk: chunks, tokens p50/p95, `chunks_format`, `row_group_count`, `parquet_bytes`
  * Embed: vectors per family, dims, `vector_format`, `avg_nnz`/norms, corpus summary `__corpus__`
* **Telemetry**:

  * Counters: items_total/succeeded/failed/skipped
  * Histograms: item_exec_ms, item_queue_ms
  * Gauges: active_workers, queue_depth; (optional) GPU mem
  * Traces: span per item; attributes: stage, policy, provider, model, batch size, device

---

# 9) Error taxonomy (mapping)

Map exceptions to `StageError.category` consistently:

| Exception source                              | Category             | Retryable                           |
| --------------------------------------------- | -------------------- | ----------------------------------- |
| Invalid input path/format                     | `input`              | no                                  |
| Misconfiguration (missing required URL/model) | `config`             | no                                  |
| HTTP 429/5xx in TEI                           | `provider`/`network` | yes (bounded)                       |
| GPU OOM (vLLM)                                | `runtime`            | no (unless provider can down-batch) |
| Tokenizer/model import failures               | `runtime`            | no                                  |
| Timeout                                       | `timeout`            | no                                  |

**Policy**: Only network/transient errors should be retried automatically.

---

# 10) Connectivity to other subsystems

## 10.1 Settings & profiles

* Root CLI callback builds effective config via **CLI > ENV > profile > defaults**.
* Produces `AppCfg`, `RunnerCfg`, and stage cfgs (`DocTagsCfg`, `ChunkCfg`, `EmbedCfg`) + `cfg_hash` per stage.
* Runner receives **StageOptions** derived from `RunnerCfg` + per-cmd overrides.

## 10.2 Providers (embedding)

* **Not required** for Runner, but Runner **expects** embedding to initialize once via `before_stage` and use a **thin provider API** (`embed`, `encode`, `vector`).
* When you land the providers PR, only **Hooks/Worker** change; Runner is unchanged.

## 10.3 Storage (writers/readers)

* Worker calls **Parquet/JSONL** writers; Runner is agnostic.
* Runner depends on **Manifest Sink** only (atomic JSONL).

## 10.4 Inspect & DatasetView

* Outside Runner; used by operators/QA. Runner **produces** Parquet datasets that `inspect` consumes.

## 10.5 Orchestrators (optional)

* Prefect/Airflow layer calls Runner exactly as CLI does. Runner’s semantics (timeouts, retries, budgets) remain **inside** Runner to guarantee parity.

---

# 11) State machine per item

```
PENDING
  ├─ (resume true) → SKIPPED
  └─ (enqueue) → QUEUED → RUNNING
        ├─ success → SUCCEEDED
        ├─ failure & retryable & attempts<retries → QUEUED
        ├─ failure (final) → FAILED
        └─ timeout → FAILED
[global]
  └─ failed > error_budget → CANCELLED (no new submissions; inflight drains or cancels by policy)
```

---

# 12) Performance & capacity playbook

* **Throughput**: start with `workers ≈ cores for cpu` or `≈ provider inflight capacity for gpu`; raise slowly while watching p95.
* **SJF**: turn on for heterogeneous corpora (mix of tiny/giant files).
* **Backpressure**: set `max_queue ≈ 2×workers` to keep memory bounded.
* **Batching (providers)**: use micro-batched provider calls; Runner should not micro-batch—providers know their token/VRAM limits better.
* **Row-group sizing (Parquet)**: keep chunk/dense ~16–64 MB groups to balance scan speed & memory locality.

---

# 13) Safety & determinism

* **Atomic writes** everywhere; never expose temp paths.
* **No heavy imports** in Runner module; keep stage logic & providers isolated.
* **Idempotence**: success manifests describe exact outputs; reruns with resume+fingerprints are no-ops.
* **Signal handling**: graceful cancel on SIGINT/SIGTERM—set cancel flag, stop submitting, optionally cancel inflight.

---

# 14) Authoring a new stage in <1 page

1. **Plan**: implement `StagePlan` that yields `WorkItem`s with `item_id`, `inputs`, `outputs`, `cfg_hash`, and `cost_hint`.
2. **Worker**: pure function `worker(item, ctx)` that reads `item.inputs`, writes `item.outputs` (atomic), and returns a small result dict for manifest extras.
3. **Hooks** (optional): open/close shared resources, and attach per-item metadata (e.g., precomputed file stats).
4. **CLI**: map Typer flags → StageOptions + StageCfg; call `run_stage(plan, worker, options, hooks)`.
5. **Tests**: unit test your worker; integration test: plan→runner→manifests.

You’re done—no pools, no progress code, no bespoke lock files.

---

# 15) Minimal acceptance for the Runner landing

* All stages call **one** runner; stage modules have no pool/progress/manifest code.
* Resume/force/timeout/retries/error budget work identically across stages.
* Manifests share base fields; stage extras remain.
* p50/p95 latency matches or beats legacy loops (SJF helps).
* The APIs above are stable and documented.

---

This architecture keeps the **runner clean and opinionated** while making stages **small and swappable**. It decouples your orchestration concerns from model/storage specifics, and it gives operators consistent control levers and observability—regardless of whether you’re running from CLI, Prefect, or unit tests.
