Here’s a surgical, narrative-only **implementation plan** for **PR-5: Collapse stage runners; remove duplicated concurrency/manifest code**. It’s written so an AI programming agent can execute it step-by-step—no guessing, no source edits outside the described seams, and **no code included**.

---

# Scope & intent

**Goals**

1. Replace the three near-duplicate stage loops (doctags → chunk → embed) with **one generic runner** that handles: discovery, filtering (resume/force), parallel execution, error budgeting, telemetry, and manifest writes.
2. Make the **“work per item”** function the only stage-specific code.
3. Centralize **timeouts, cancellation, logging, and progress**.
4. Keep all user-visible behavior (CLI flags, file layouts, manifests) unchanged.

**Non-goals**

* No change to file formats or schema fields (handled in PR-1/2).
* No change to CLI shape (already unified in PR-3) other than pointing subcommands to the new runner.

---

# Why this is needed (duplication & risk)

* Each stage currently re-implements some mix of: executor creation, per-file try/except, resume/force checks, progress reporting, and manifest writing. That causes drift (e.g., one stage might treat timeouts or partial outputs differently than another).
* The embed stage additionally mixes GPU/HTTP considerations (handled in PR-4). After PR-4, **all stages are “pure functions over inputs”** and ready to be run by a single runner.

---

# Target design (the “after” picture)

## A single generic runner (core/executors)

Introduce one **public** function:

* **`run_stage(plan, worker, options, hooks) -> StageOutcome`**

…and three tiny data contracts:

* **`StagePlan`**: iterable of immutable **`WorkItem`** objects (one item per input unit; e.g., a file).
* **`StageOptions`**: runtime knobs (degree of parallelism, policy = cpu|io|gpu, timeouts, resume/force, error budget, retry).
* **`StageHooks`** (optional): callouts for before/after stage and before/after item (for per-stage extras like setting provider context or enriching telemetry).

**Responsibilities of `run_stage`:**

* Build the queue of `WorkItem`s from the plan.
* Apply **resume/force** rules to skip items that are already satisfied.
* Schedule items on the correct executor (thread/process) based on policy.
* Enforce **per-item timeout**, a **global error budget**, and **cancellation** (e.g., stop after N failures).
* Capture standardized **success/failure events** and write **manifests** via the injected writer (from PR-2).
* Emit **progress** and **timing metrics**.
* Return a **StageOutcome** summary: counts (scheduled/skipped/succeeded/failed), durations, and selected percentiles (p50, p95 per item).

> **Invariant:** The runner is **stage-agnostic**. It knows nothing about PDFs or vectors. It receives a `worker(item, ctx)` callable and handles everything around it.

---

# Stage mapping (what becomes the “work”)

* **Doctags**: `worker(item) → DocTagsResult`

  * Input: raw doc path (PDF/HTML), output paths (doctags jsonl/md), conversion config.
  * Output counters: blocks/tables/figures extracted, bytes in/out, content hash.
* **Chunking**: `worker(item) → ChunkingResult`

  * Input: doctags path, chunk config (tokenizers, min/max tokens).
  * Output counters: chunks produced, tokens per chunk stats.
* **Embedding**: `worker(item) → EmbeddingResult`

  * Input: chunks path, provider configs (dense/sparse/lexical) from PR-4.
  * Output counters: vectors written per family, embedding dims, batches.

> Each `Result` is a **small Pydantic model** that the runner will serialize into success manifests. (You already have manifest helpers from PR-2; keep using them.)

---

# Concurrency policies (one knob for all stages)

Inside `StageOptions`, define a **policy** that the runner translates to the right executor:

* **`policy="io"`** → **ThreadPool** (ideal for doctags I/O & network).
* **`policy="cpu"`** → **ProcessPool** with **spawn** start method (for CPU-only heavy parse or tokenization).
* **`policy="gpu"`** → **ThreadPool** (GPU libraries are typically not fork-safe; the provider owns in-GPU parallelism).

**Other options (shared across stages):**

* **`workers`**: max workers (default from CLI flag you already have).
* **`per_item_timeout_s`**: hard timeout per item (0 = disabled).
* **`error_budget`**: max failures before early cancel (0 = stop on first error).
* **`retries`**: simple bounded retry with backoff for transient cases (applies to network-ish work; set to 0 for pure deterministic work).
* **`max_queue`**: bounded producer queue to avoid memory spikes on huge plans.
* **`diagnostics_interval_s`**: how often to emit progress/throughput snapshots.

---

# Resume/force contract (exact behavior)

* **Force**: always run the worker, even if all expected outputs exist. The runner still backs up/quarantines any pre-existing files the stage would overwrite (same logic you already use; keep using the atomic writer from PR-2).
* **Resume**: skip an item if **all** of its declared outputs already exist and are **non-empty** (and optionally match an expected suffix/pattern).

  * For doctags: JSONL (or chosen format) exists and has ≥1 row.
  * For chunking: chunk JSONL/Parquet exists and has ≥1 row.
  * For embedding: vector JSONL/Parquet exists and has ≥1 row **for each enabled family** (dense/sparse/lexical).
* **Changed inputs** (optional enhancement): If a **fingerprint** file exists (hash of input + config), recompute only when the fingerprint differs. If not present, resume uses existence/size checks only.

**Implementation detail:** The `WorkItem` can carry a `satisfies()` method (or a tiny predicate) so each stage declares its own “done” condition. The runner just calls it.

---

# Error taxonomy (one language across stages)

Standardize a small error model emitted by the runner:

* **`StageError`**: `{stage, item_id, category, message, retryable, detail}`

  * `category`: `input` (bad file), `config` (bad flags), `runtime` (exception), `timeout`, `io`, `provider` (from PR-4), `unknown`.
  * `retryable`: boolean (the runner uses it when `retries > 0`).
* **Failure paths**:

  * Write a **failure manifest row** with the error payload.
  * If `error_budget` exceeded: cancel remaining work and leave a **stage outcome** with a `cancelled=true` flag.

> The stage worker should raise *domain* exceptions (e.g., `ProviderError` or `DocConvertError`); the runner maps them to `StageError` categories and also preserves the original class in `detail`.

---

# Telemetry & manifests (one place, one shape)

* **Manifests**: Use the PR-2 **writer dependency** (FileLock + atomic append).

  * **Success row**: `{stage, item_id, outputs, counts, timings, cfg_hash, provider_tags?}`
  * **Failure row**: `{stage, item_id, error, timings, cfg_hash}`
* **Telemetry**: The runner emits **stage-level** metrics (items/s, active workers, p50/p95 item duration) and **per-item** spans/metrics:

  * `stage_name`, `policy`, `workers`, `batch_hint`
  * `time_submit_ms`, `time_start_ms`, `time_end_ms`, `time_queue_ms`, `time_exec_ms`
  * `status` = success|failure|skipped
* **Progress**: a single progress line with: done/total, succeeded, skipped, failed, ETA, and current throughput.

---

# Hooks (escape hatches without forking the runner)

`StageHooks` gives optional lifecycle events:

* `before_stage(ctx)` / `after_stage(outcome)`
* `before_item(item, ctx)` / `after_item(item, result|error, ctx)`

Use cases:

* **Embedding**: attach provider metadata (model id, dim) to the stage context once; reuse in success manifests.
* **Doctags**: attach file-size or mimetype precomputed during discovery to save work.
* **Chunking**: emit per-item token counts upstream to tune chunk sizes.

> Hooks must be **fast**, **side-effect-free** (aside from logging), and **exception-safe** (exceptions in hooks are converted to warnings and do not fail the item).

---

# Planning & discovery (small, deterministic)

Create (or keep) minimalist **plan builders** per stage that create `WorkItem`s:

* **DoctagsPlan**: scan input dirs for PDF/HTML; map to output doctags paths (consistent with current layout).
* **ChunkPlan**: scan doctags dir for doctags files; map to chunk outputs; skip partial temp files.
* **EmbedPlan**: scan chunks dir for chunk files; map to vector outputs (JSONL/Parquet) per family.

**Rule:** A `WorkItem` includes:

* `item_id` (stable; typically the relative path without extension)
* `input_paths` (1..n)
* `output_paths` (1..n)
* `cfg_hash` (hash of relevant config keys for reproducibility)
* `meta` (small dict: size hints, mime, provider tags)

Plans are **pure** (no I/O beyond filesystem walk) and **ordered** (stable order for reproducible scheduling).

---

# Timeouts, cancellation & retries (uniform semantics)

* **Per-item timeout**: If `per_item_timeout_s > 0`, wrap the worker execution and raise `StageError{category="timeout"}` on expiry. The runner cancels the item’s future and proceeds based on policy.
* **Global cancellation**: Maintain a shared flag set when:

  * SIGINT/SIGTERM received, or
  * `error_budget` exceeded, or
  * `cancel_on_first_failure=true` and a failure occurs.
* **Retries**: If `retries>0` and `StageError.retryable`, reschedule the item with exponential backoff (cap total tries). The manifest captures each attempt with `attempt_no`.

---

# Choosing the executor (correct by default)

* **Doctags**: `policy="io"` (ThreadPool).
* **Chunking**: `policy="cpu"` (ProcessPool with `spawn`).
* **Embedding**: `policy="gpu"` (ThreadPool); **providers** (PR-4) handle internal batching/queues.

> You can make `policy` overrideable via CLI or config, but keep these as **defaults**.

---

# Integration with CLI (PR-3)

* Subcommands `doctags`, `chunk`, `embed`, and `all` now call:

  * Plan builder → `StageOptions` builder → `run_stage(plan, worker, options, hooks)`
* Keep all current flags; only change **the call path** from per-stage loops to the runner.

---

# Testing strategy

## Unit (runner)

* **Happy path**: N items succeed; counts match; outcome p50/p95 computed.
* **Resume/force**: pre-create outputs; ensure resume skips; ensure force executes.
* **Timeout**: inject a slow worker; assert timeout error and manifest entry.
* **Error budget**: set budget=1; inject two failures; assert early cancel.
* **Retries**: make worker fail once retryable, then succeed; attempts recorded.
* **Cancellation**: simulate SIGINT; ensure clean shutdown and summary reported.

## Stage adapters

* **Doctags/Chunk/Embed worker**: tiny smoke tests verifying the worker contracts under the runner (no bespoke loops).
* **Hooks**: ensure exceptions in hooks don’t crash the stage; warnings logged.

## Performance sanity

* Measure **throughput** on a micro corpus; store p50/p95 baselines (not hard assertions, but logs to compare between commits).

---

# Migration plan (minimize blast radius)

**Commit A — Core runner scaffolding**

* Add `core/executors.py` with the runner, options, hooks, and outcome models.
* Add plan builders for doctags/chunk/embed (or adapt existing discoverers to emit `WorkItem`s).

**Commit B — Manifests & telemetry integration**

* Inject the PR-2 writer into the runner; wire success/failure manifest rows.
* Add standardized telemetry counters and progress.

**Commit C — Doctags refactor**

* Replace doctags’ local loop with the runner; add a `doctags_worker(item, ctx)` and a trivial `DoctagsPlan`.
* Keep CLI behavior identical.

**Commit D — Chunking refactor**

* Replace chunking’s loop with the runner; add `chunk_worker` and `ChunkPlan`.
* Honour min/max tokens; return chunk counts.

**Commit E — Embedding refactor**

* Replace embedding’s loop with the runner; add `embed_worker` using **providers** from PR-4 via a `StageHook.before_stage` that instantiates providers once and stores them in `ctx`, and `StageHook.after_stage` that calls `provider.close()`.
* Confirm JSONL/Parquet writers unchanged.

**Commit F — Delete dead code**

* Remove stage-local executors, progress code, and bespoke manifest logging paths.
* Ensure there are **no** remaining imports of old loops.

**Commit G — Tests & docs**

* Add runner unit tests; adapt stage tests to the new runner.
* Update internal docs: “Stage authoring guide” (how to write a `worker` and `plan`, how hooks work).

---

# Acceptance criteria (“done”)

* All three subcommands run through **the same runner**; no stage contains its own pool or bespoke manifest code.
* **Resume/force** work identically across stages and match previous behavior.
* **Timeouts, retries, error budget** are available to all stages with the same semantics.
* Manifests show **uniform** success/failure rows; telemetry includes the same fields across stages.
* **Embedding providers** are created once per stage via hooks and closed after; the runner does not import heavy model libraries.
* CI passes unit + integration tests; performance sanity checks do not regress materially.

---

# Risks & mitigations

* **GPU/fork safety**: Accidentally running embedding on a ProcessPool can crash GPU libs.
  ⇒ Default embedding `policy="gpu"` (ThreadPool), ignore any ProcessPool override unless explicitly forced with a “dangerous” flag.
* **Pickling limits**: ProcessPool needs pickleable `WorkItem`s and worker closures.
  ⇒ Keep `WorkItem` small & immutable; stage workers top-level functions only (no lambdas/closures).
* **Partial output corruption** on worker crash.
  ⇒ Continue using **atomic writes** and temporary files that promote only on success (already in PR-2).
* **Inconsistent resume semantics** between stages.
  ⇒ Stage-specific `satisfies()` predicate lives on `WorkItem` so semantics are declarative, reviewed, and unit-tested.

---

# Operational playbook

1. Land the runner behind an **env feature flag** (`DOCSTOKG_RUNNER=on`) for one release; keep legacy loops in the repo but disabled by default.
2. Dogfood on a small corpus; compare manifests and throughputs vs. baseline.
3. Flip the default to the new runner; keep the old one for one minor release with a deprecation warning.
4. Delete the old loops once telemetry shows comparable or better performance and no error-rate increase.

---

# Engineer’s checklist (quick)

* [ ] `core/executors.run_stage` exists; documented, pure, and stage-agnostic.
* [ ] `StagePlan/WorkItem/StageOptions/StageHooks/StageOutcome` defined and used.
* [ ] Doctags/Chunk/Embed now **only** provide a plan + worker (+ optional hooks).
* [ ] Resume/force implemented **once**; identical across stages.
* [ ] Timeouts, retries, error budget implemented **once**; identical across stages.
* [ ] Manifests and telemetry go through the **PR-2 writer** and standardized fields.
* [ ] Old stage loops deleted; no pools or bespoke progress code remain in stage modules.
* [ ] Tests cover runner semantics; stage integration passes.
* [ ] Docs updated (“How to add a new stage” using the runner template).

---

If helpful next, I can produce a **worker contract spec** (inputs/outputs per stage, exact manifest fields, suggested error categories) so your agent can patch each stage file with minimal diff.
