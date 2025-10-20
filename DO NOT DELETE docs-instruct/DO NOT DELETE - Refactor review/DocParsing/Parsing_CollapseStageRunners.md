Here’s a best-in-class, AI-agent-ready deep dive for **PR-5 — Collapse stage runners; remove duplicated concurrency/manifest code**. It spells out *exact* responsibilities, data contracts, invariants, tuning knobs, and optional “gold tier” optimizations so your team can implement confidently and systematically.

---

# High-level goal

Replace three bespoke loops (Doctags → Chunking → Embedding) with **one generic, stage-agnostic runner**. Each stage then contributes only:

1. a **Plan** (what to do),
2. a **Worker** (how to do 1 item),
3. optional **Hooks** (stage-specific setup/teardown & lightweight per-item tap-ins).

Everything else—parallelism, retries/timeouts, resume/force, error budgets, manifests, telemetry, progress—is **centralized**.

---

# Core architecture (exact contracts)

## `StagePlan`

* **Purpose:** Pure description of work, no side effects.
* **Fields (required):**

  * `stage_name`: `"doctags" | "chunk" | "embed"`.
  * `items: Iterable[WorkItem]` — stable order (determinism).
  * `total_items: int` — for progress & budgeting.
* **Invariants:**

  * Iteration is repeatable & pure (no lazy randomization).
  * Items reference **paths**, not loaded data, to keep IPC cheap.

## `WorkItem`

* **Purpose:** Immutable unit of work the runner schedules.
* **Fields (required):**

  * `item_id`: stable, file-path-derived identifier (e.g., relative path without extension).
  * `inputs: Dict[str, Path]` — e.g., `{"pdf": ..., "html": ...}` (stage-specific keys).
  * `outputs: Dict[str, Path]` — expected final artifacts (e.g., `{"doctags_jsonl": ...}`).
  * `cfg_hash: str` — SHA256 of the **stage-relevant** config (ensures resume correctness when config changes).
  * `cost_hint: float` — heuristic runtime cost (e.g., PDF pages; chunk count; bytes); used for smarter scheduling.
* **Optional:**

  * `meta: Dict[str, Any]` — small hints (mime, bytes, page_count).
  * `satisfies(): bool` — callable to check resume/force condition for this item (see below).

## `StageOptions`

* **Purpose:** Uniform knobs (all stages).
* **Fields:**

  * `policy: "io" | "cpu" | "gpu"` — maps to ThreadPool / ProcessPool(spawn) / ThreadPool.
  * `workers: int` — pool size.
  * `per_item_timeout_s: float` — `0` disables.
  * `retries: int` — bounded retry attempts for **retryable** errors.
  * `retry_backoff_s: float` — base backoff; consider jitter.
  * `error_budget: int` — fail-fast stop after N failures (0 = stop on first failure).
  * `max_queue: int` — cap pending submissions for backpressure.
  * `resume: bool`, `force: bool`.
  * `diagnostics_interval_s: float` — progress/throughput report cadence.
  * `seed: int` — for deterministic shuffling if enabled (defaults to no shuffle).
* **Derived (internal):**

  * `start_method="spawn"` when using ProcessPool (safety w/ C libs, tokenizers).

## `StageHooks` (all optional; must not throw)

* `before_stage(ctx) -> None` — allocate stage-wide resources (e.g., embedding providers).
* `after_stage(outcome) -> None` — close resources; flush telemetry.
* `before_item(item, ctx) -> None` — cheap side-effects (attach precomputed file stats).
* `after_item(item, result_or_error, ctx) -> None` — per-item enrichment (e.g., attach provider tags).

> **Rule:** Hooks may log/annotate, never alter scheduling, and exceptions in hooks are *downgraded to warnings*.

## `StageOutcome`

* `scheduled, skipped, succeeded, failed, cancelled: int`
* `wall_ms, queue_p50_ms, exec_p50_ms, exec_p95_ms, items_per_s`
* `errors: List[StageErrorSummary]` (first/most frequent)
* `manifest_path, telemetry_counters`: pointers & totals

## `StageError`

* `stage, item_id, category, message, retryable, detail`
* `category ∈ {input, config, runtime, timeout, io, provider, unknown}`

---

# Single entry point

## `run_stage(plan, worker, options, hooks) -> StageOutcome`

**What it does (and you must do nowhere else):**

1. **Discovery/Filtering:** Iterate `plan.items`, skip with `resume` logic via `WorkItem.satisfies()` unless `force`.
2. **Scheduling:** Submit `worker(item)` to the right executor:

   * `policy="io" | "gpu"` → **ThreadPool** (library C-extensions release GIL or GPU handles concurrency).
   * `policy="cpu"` → **ProcessPool** with **spawn** (pickle item only; pass file paths).
3. **Timeouts:** Enforce per-item deadlines (wrap future wait).
4. **Retries:** For `retryable=True`, resubmit with backoff ≤ `retries`.
5. **Error budget:** If `failed > error_budget`, set cancel flag, stop new submissions, drain/completion or cancel futures (configurable).
6. **Progress:** Print 1-line status (done/total, succ/skip/fail, ETA, items/s).
7. **Manifests:** Append **success/failure rows** via the PR-2 writer (FileLock + atomic append).
8. **Telemetry:** Emit standardized counters/histograms/traces.
9. **Outcome:** Return `StageOutcome` with stats & percentiles.

---

# Resume/force (precise semantics)

* **Resume**: An item is **skipped** iff *all* declared `outputs` exist, are **non-empty**, and (if present) the `fingerprint` file matches `cfg_hash + input_hash`.

  * If no fingerprint file is present, fallback to existence + non-empty size checks.
  * For embedding, “all outputs” means **each enabled vector family** (dense/sparse/lexical) has non-empty data.
* **Force**: Ignore resume checks; run worker and **atomically replace** outputs (PR-2).
* **Change detection** (recommended): Write a tiny sidecar `*.fp.json` per item: `{input_sha256, cfg_hash, created_at}`. On resume, skip only when both match.

**Gotcha:** Always compute `input_sha256` on the **bytes that the stage consumes**, not upstream source, to avoid subtle mismatches (e.g., doctags hashing PDF/HTML; chunking hashing doctags JSONL; embedding hashing chunk JSONL/Parquet).

---

# Worker contract per stage (no business logic in runner)

## Doctags worker

* **Input:** `WorkItem.inputs["pdf"|"html"]`
* **Output artifacts:** `outputs["doctags_jsonl"|"doctags_md" (optional)] + fingerprint`.
* **Result payload:** `{blocks, tables, figures, pages, bytes_in, bytes_out}`
* **Error mapping:** parse failures → `category="input"`; library errors → `runtime`; OOM/timeouts → `timeout`.

## Chunking worker

* **Input:** `inputs["doctags_jsonl"]`
* **Output artifacts:** `outputs["chunks_jsonl" or "chunks_parquet"] + fingerprint`.
* **Result payload:** `{chunks, tokens_p50, tokens_p95, empty_chunks}`
* **Error mapping:** malformed doctags → `input`; tokenizer errors → `runtime`; resource issues → `runtime/timeout`.

## Embedding worker

* **Input:** `inputs["chunks_jsonl|parquet"]`
* **Output artifacts:** up to three outputs `{dense, sparse, lexical} + fingerprint` depending on enabled families.
* **Result payload:** `{vectors_dense, vectors_sparse, vectors_lexical, dim_dense, batches_dense, elapsed_ms_dense, ...}`
* **Error mapping:** provider error taxonomy → mapped to `provider` (retryable depends on provider category); GPU OOM → `runtime` (retryable=False unless provider can transparently down-batch).

> **StageHooks**: For embedding, call `dense/sparse/lexical provider.open(cfg)` **once** in `before_stage`, stash in `ctx`, and `close()` in `after_stage`.

---

# Scheduling & concurrency (best-in-class)

## Executor policy (default per stage)

* Doctags → `io` (ThreadPool)
* Chunking → `cpu` (ProcessPool, spawn)
* Embedding → `gpu` (ThreadPool)

## Queue discipline

* **Submission cap:** `max_queue` prevents RAM blow-ups on very large plans.
* **Adaptive concurrency (optional gold tier):**

  * Track rolling p50 item time; if p50 falls, consider increasing `workers` (bounded).
  * If p95 rises or failure rate increases, decrease `workers`.
  * Stabilize with hysteresis to avoid oscillations.

## Job ordering (optional gold tier)

* **Shortest-job-first:** Sort items by `cost_hint` ascending for faster time-to-green and lower p95.
* **Stratified batching:** For embedding, group items by **chunk count buckets** so providers see steadier batch sizes.

## Backpressure & provider interplay

* If the embedding provider exposes `max_inflight_requests` / `queue_depth`, set it ≤ runner’s `workers` to avoid overload.
* If the provider returns a **backpressure signal** (e.g., queue full), the worker should **block briefly** or **retry with backoff**, not spin.

## ProcessPool rules of thumb (chunking)

* Use **spawn** (never fork) to avoid CUDA/Python C-ext weirdness.
* **Pass paths**, not large in-memory objects; load inside worker.
* Keep `WorkItem` pickle-friendly (no callables/dataframes/handles inside).

---

# Timeouts, retries, error budget (uniform semantics)

* **Timeout:** Wrap each future; on expiry mark `StageError(category="timeout", retryable=False)`. Do *not* kill the whole run—just the item—unless `cancel_on_timeout=true` (rare).
* **Retries:** Only if `error.retryable=True`. Strategy: `sleep = retry_backoff_s × 2^attempt + jitter`. Hard cap attempts via `retries`.
* **Error budget:** When `failed > error_budget`, raise the **cancel flag**:

  * **New submissions stop.**
  * In-flight items complete (default) or are cancelled if `cancel_inflight=true`.
  * **Outcome.cancelled = true** with reason “budget exceeded”.

---

# Manifests & telemetry (single, shared path)

## Manifests (PR-2 writer)

* **Success row**:
  `{stage, item_id, outputs: {...}, result: {...}, attempt, cfg_hash, input_sha256, timings: {queue_ms, exec_ms}, ts}`
* **Failure row**:
  `{stage, item_id, error: {category, message, retryable}, attempt, cfg_hash, input_sha256, timings, ts}`
* **Rotation (optional):** roll daily or per 100k rows: `.../manifests/{stage}-{YYYYMMDD}.jsonl`.

## Telemetry

* **Counters:** `stage_items_total`, `stage_items_succeeded`, `stage_items_failed`, `stage_items_skipped`.
* **Histograms:** `item_exec_ms`, `item_queue_ms`, `items_per_s` (computed).
* **Gauges:** `active_workers`, `pool_queue_depth`, **GPU mem** (embedding; optional).
* **Traces:** one span per item; child spans for provider calls (embedding).

---

# Plan builders (per stage; small, deterministic)

## DoctagsPlan

* Walk `input_root` for `*.pdf|*.html`.
* Compute `item_id = relpath_no_ext`, `cost_hint = pages_or_bytes`.
* Compute outputs (consistent, deterministic paths) and `cfg_hash` for doctags config.
* Optionally pre-compute `input_sha256` (fast streaming hash).

## ChunkPlan

* Walk doctags directory for `*.jsonl` (or doctags parquet if you support it).
* Cost hint = rows or bytes.
* Map outputs (`*.chunks.jsonl|parquet`).
* `cfg_hash` from chunker settings.

## EmbedPlan

* Walk chunks directory for `*.chunks.jsonl|parquet`.
* Cost hint = expected chunk count (read header/meta) or bytes.
* Outputs: choose per enabled family; `cfg_hash` from provider configs (dense/sparse/lexical).

> Plans must **not** touch the manifests; they are pure mappings.

---

# Progress & diagnostics

* One **single-line** progress updated every `diagnostics_interval_s` with:

  * `done/total`, `succ`, `skip`, `fail`, `ETA`, `items/s`, and **last minute’s** items/s.
* Optional **verbose** pane (when `-vv`): top 3 error categories and the slowest 3 items so far.
* “Dry run” mode: print plan summary (N items, predicted cost sum) without execution.

---

# Optional gold-tier optimizations

1. **SJF + bucketing** (noted above) to reduce tail latency.
2. **Speculative retry**: If a long-running item exceeds p95×X, trigger a parallel attempt on another worker and take first successful result (enable only for idempotent, read-only stages like embedding; throttle count).
3. **Read-ahead & double buffering**:

   * Stage-specific: pre-open PDFs (mmap) for doctags, pre-load chunk file headers for embedding.
   * Keep within memory budget; expose `read_ahead_files` knob.
4. **Adaptive worker pool**: slowly scale `workers` between `[min,max]` based on moving average throughput and memory/GPU pressure.
5. **Failure triage**: On repeated failures of the same `category` within a time window, pause submissions and surface a **single “actionable” banner** (e.g., “TEI rate-limited—consider lowering max inflight or switching backend”).
6. **Per-directory batching**: Group items by directory to improve FS locality (HDDs/NFS benefit); optional for SSDs.
7. **Provider-aware pacing**: Let embedding provider expose preferred micro-batch sizes; runner aggregates items accordingly, reducing tiny batches.

---

# Best practices & invariants checklist

* **Determinism:** Ordered plan; stable `item_id`; fixed seeds; pinned model/tokenizer revisions.
* **Isolation:** No heavy imports at runner top-level; stage workers import only what they need.
* **Pickle-safe:** Workers are **top-level** functions; `WorkItem` holds only primitives/paths.
* **Atomicity:** All writes go through PR-2 writer; partial files never visible.
* **Backwards-compat:** Same output shapes/paths; same CLI flags; same manifest schema keys (you may add fields, not remove/rename).
* **Uniform semantics:** Timeouts/retries/resume/force/error budget apply identically to all stages.
* **No shared global singletons:** Pass context via `ctx` (from root Typer callback) to runner hooks.

---

# Test plan (exhaustive)

## Unit — Runner

* **Resume/force parity:** Pre-create outputs → resume skips; force re-executes with replace.
* **Timeout path:** Simulated slow worker times out; manifest has failure row; outcome counts ok.
* **Retry path:** Failing-once worker then succeeds with `retries=1`; attempts counted; success manifest only once.
* **Error budget:** Two failing workers with budget=1 → outcome `cancelled=true`; second not scheduled (or cancelled).
* **Cancellation signal:** Inject SIGINT → graceful stop; outcome marks `cancelled=true`.
* **SJF ordering** (if enabled): large `cost_hint` items start later; check ordering of started times.

## Integration — Per stage

* Doctags/Chunk/Embed each run via **same runner**; outputs match baseline (bit-for-bit or FP tolerance).
* **Hooks**: Provider open/close invoked exactly once per stage; per-item hooks do not alter results or crash stage.
* **Large plan**: 10k items test: memory usage bounded; throughput acceptable; no deadlocks.

## Property tests (selective)

* **Idempotence**: Running resume twice produces 0 work with same manifests (aside from timestamps).
* **Parallel safety**: Two concurrent runs on **different** output roots do not interfere.

---

# Migration & rollout

1. **Commit A — Core runner + models**

   * Add `core/executors.py` (runner + contracts).
   * No stage wired yet; unit tests for runner.

2. **Commit B — Manifest/telemetry plumbing**

   * Integrate PR-2 writer; standardize success/failure payloads.
   * Add shared metrics/tracing.

3. **Commit C — Doctags on runner**

   * Replace doctags loop with plan + worker + simple hooks.

4. **Commit D — Chunking on runner**

   * Same pattern; ProcessPool(spawn); pass only paths.

5. **Commit E — Embedding on runner**

   * Use PR-4 providers in hooks; ensure GPU/thread safety.

6. **Commit F — Delete dead code**

   * Remove old per-stage loops, bespoke progress/manifests; keep import shims if needed for one release.

7. **Commit G — CI & docs**

   * Add stress tests & p95 thresholds (soft assertions); update “How to add a stage” docs with runner template.

**Feature flag rollout:** Allow `DOCSTOKG_RUNNER=legacy|new`; default to `new` after 1–2 internal cycles, then drop legacy.

---

# Reference “done” criteria

* All three subcommands call `run_stage(...)`.
* Resume/force/timeout/retry/error-budget **work identically across stages**.
* Per-stage code shrinks to: plan builder + worker (+ optional hooks).
* Manifests & telemetry unified; progress uniform; no bespoke loops left.
* Embedding providers created/closed exactly once per stage; no heavy ML imports in runner.

---

If you want, I can also produce a **field-accurate manifest specification** (key names, allowed types, and example rows per stage) and a **runner behaviors matrix** (what happens under every combination of `resume/force/retries/timeout/error_budget`) that your team can turn into tests 1:1.
