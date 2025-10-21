# DocParsing Runner Architecture Guide

**Last Updated**: October 21, 2025
**Audience**: Developers, operators, maintainers

---

## Table of Contents

1. [Purpose at a Glance](#purpose-at-a-glance)
2. [System Map](#system-map)
3. [Core Contracts](#core-contracts)
4. [Control Flow](#control-flow)
5. [Scheduling & Execution](#scheduling--execution)
6. [Resume & Fingerprints](#resume--fingerprints)
7. [Manifests & Telemetry](#manifests--telemetry)
8. [Error Handling](#error-handling)
9. [Safety & Determinism](#safety--determinism)
10. [Authoring a New Stage](#authoring-a-new-stage)
11. [Performance Tuning](#performance-tuning)
12. [Debugging & Troubleshooting](#debugging--troubleshooting)

---

## Purpose at a Glance

The **DocParsing Runner** is a unified orchestration kernel that handles scheduling, concurrency, retries, timeouts, resume/force semantics, manifests, and telemetry for all three DocParsing stages (DocTags, Chunk, Embed).

**Each stage supplies only**:

1. **Plan**: Which items to work on
2. **Worker**: How to process one item
3. **Hooks** (optional): Setup/teardown and per-item tapins

**The runner owns**:

- Concurrency (ThreadPool for I/O, ProcessPool for CPU)
- Retries with exponential backoff
- Timeouts per item
- Error budgets (stop after N failures)
- Resume/force semantics with fingerprinting
- Progress tracking and diagnostics
- Atomic manifest writing

---

## System Map

```
┌──────────────────────────────────────────────┐
│                 Typer CLI                     │
│    (profiles → ENV → CLI) builds cfg          │
└──────────────┬──────────────────────────────┘
               │
    ┌──────────▼──────────────┐
    │  Stage Plan & Hooks     │
    │ (discoverable items)    │
    └──────────┬──────────────┘
               │
    ┌──────────▼──────────────────────────────┐
    │         Run Stage Orchestrator           │
    │  (THIS MODULE)                           │
    │  ├─ Build queue (resume/force)           │
    │  ├─ Select executor (ThreadPool/ProcessPool) │
    │  ├─ Submit items with backpressure       │
    │  ├─ Handle timeouts & retries            │
    │  ├─ Enforce error budget                 │
    │  ├─ Write manifests (atomic)             │
    │  └─ Track progress & telemetry           │
    └──────────┬──────────────────────────────┘
               │
    ┌──────────┴──────────────────────────────┐
    │         Worker (user code)              │
    │  ├─ Process one item                    │
    │  ├─ Call stage libraries (Docling, etc) │
    │  └─ Return ItemOutcome                  │
    └─────────────────────────────────────────┘
```

---

## Core Contracts

All core contracts are **pure dataclasses** (no heavy imports, pickleable).

### StagePlan

Deterministic enumeration of work:

```python
@dataclass
class StagePlan:
    stage_name: str              # "doctags", "chunk", "embed"
    items: Sequence[WorkItem]    # Stable order
    total_items: int             # For budgeting
```

### WorkItem

Immutable description of a single unit of work:

```python
@dataclass(frozen=True)
class WorkItem:
    item_id: str                 # Stable relative ID (e.g., doc filename)
    inputs: Mapping[str, Path]   # Input artifacts
    outputs: Mapping[str, Path]  # Expected output paths
    cfg_hash: str                # Hash of stage-relevant config
    cost_hint: float             # Pages, bytes, chunk count (for SJF)
    metadata: Mapping[str, Any]  # Optional hints (small only)
    fingerprint: Optional[ItemFingerprint]  # Resume tracking
```

### StageOptions

Shared execution knobs:

```python
@dataclass
class StageOptions:
    policy: str                  # "io"|"cpu"|"gpu" → executor type
    workers: int                 # Max parallelism
    per_item_timeout_s: float    # Timeout per item (0=off)
    retries: int                 # Max retries on retryable errors
    retry_backoff_s: float       # Base backoff (exponential+jitter)
    error_budget: int            # Max failures before stop (0=unlimited)
    max_queue: int               # Backpressure limit (0=unlimited)
    resume: bool                 # Skip if outputs exist+fingerprint matches
    force: bool                  # Recompute regardless
    diagnostics_interval_s: float # Progress logging interval
    dry_run: bool                # Plan only, no execution
```

### ItemOutcome

Worker result wrapper:

```python
@dataclass
class ItemOutcome:
    status: str                  # "success"|"skip"|"failure"
    duration_s: float            # Execution time
    manifest: Mapping[str, Any]  # Stage-specific metadata
    result: Mapping[str, Any]    # Worker-defined result payload
    error: Optional[StageError]  # Error if status="failure"
```

### StageOutcome

Summary returned by `run_stage()`:

```python
@dataclass
class StageOutcome:
    scheduled: int               # Total items submitted
    skipped: int                 # Skipped via resume
    succeeded: int               # Successful completions
    failed: int                  # Failed attempts (after retries)
    cancelled: bool              # Cancelled due to error budget/signal
    wall_ms: float               # Total elapsed time
    queue_p50_ms: float          # Median queue wait time
    exec_p50_ms: float           # Median execution time
    exec_p95_ms: float           # 95th percentile execution time
    errors: Sequence[StageError] # Errors encountered
```

### StageHooks

Optional lifecycle callbacks:

```python
@dataclass
class StageHooks:
    before_stage: Optional[Callable[[StageContext], None]]
    after_stage: Optional[Callable[[StageOutcome, StageContext], None]]
    before_item: Optional[Callable[[WorkItem, StageContext], None]]
    after_item: Optional[Callable[[WorkItem, Union[ItemOutcome, StageError], StageContext], None]]
```

**Rules**:

- Hooks **must not throw** (exceptions logged as warnings)
- Hooks execute on **main thread** (safe for setup/teardown)
- After-item hooks execute **in order** for all completions

---

## Control Flow

### Success Path with Resume

```
CLI → runner: build StageOptions
runner → plan: iterate WorkItems
runner → WorkItem.satisfies(): check resume
         ├─ yes → log_skip, move to next
         └─ no → queue.submit(worker(item))

executor → worker: process
worker → log_success (via hooks)
runner → progress: tick, update ETA, p50/p95
```

### Failure with Retries & Budget

```
executor → worker: raises StageError(retryable=True)
runner: attempts < retries?
        ├─ yes → backoff, resubmit with jitter
        └─ no  → log_failure

runner: failed > error_budget?
        ├─ yes → set cancelled, stop new submissions
        └─ no  → continue
```

### Timeout & Cancellation

```
runner: wait(future, timeout=per_item_timeout_s)
        ├─ timeout → cancel future (best effort)
        └─ normal  → collect result

# Signal handling
SIGINT/SIGTERM → set cancelled flag, drain inflight, return with cancelled=True
```

---

## Scheduling & Execution

### Executor Selection

```python
if policy == "cpu":
    executor = ProcessPoolExecutor(max_workers=N, mp_context="spawn")
else:  # "io" or "gpu"
    executor = ThreadPoolExecutor(max_workers=N)
```

**Why spawn semantics**: GPU libraries (vLLM, FAISS) aren't fork-safe.

### Scheduling Policies

- **FIFO** (default): Process items in discovery order
- **SJF** (optional): Sort by `cost_hint` ascending to reduce tail latency

**Example**: 100 PDFs with sizes 1 MB to 100 MB

- FIFO: Last 100 MB item waits ~99 MB worth of work
- SJF: Last 100 MB item starts earlier, finishes sooner (p95 lower)

### Backpressure

`max_queue` caps queued submissions. For embedding with limited GPU:

```python
# GPU can handle 4 concurrent provider calls
max_queue = 4 * workers
```

---

## Resume & Fingerprints

### Resume Predicate

An item is skipped if **all** of:

1. `options.resume == True` and `options.force == False`
2. All declared outputs exist and are non-empty
3. Fingerprint on disk matches expected:
   - `input_sha256` matches
   - `cfg_hash` matches

### Fingerprints

Stored in `{output_path}.fp.json`:

```json
{
  "input_sha256": "abc123...",
  "cfg_hash": "cfg456..."
}
```

### Use Cases

- **Warm restart** (after crash): Resume skips successfully completed items
- **Config change**: Changed cfg_hash invalidates all fingeprints → recompute all
- **Input change** (e.g., new PDF): New input_sha256 → recompute for that PDF
- **Force flag**: Override resume, recompute everything

---

## Manifests & Telemetry

### Manifest Sink

Unified protocol for writing manifest entries (success/skip/failure):

```python
@protocol
class ManifestSink:
    def log_success(..., stage, item_id, input_path, output_paths, duration_s, schema_version, extras) -> None
    def log_skip(..., reason="resume-satisfied", extras) -> None
    def log_failure(..., error, extras) -> None
```

### Base Fields (Consistent Across Stages)

```json
{
  "stage": "chunk",
  "doc_id": "paper_123",
  "status": "success",
  "duration_s": 0.45,
  "input_path": "Data/DocTagsFiles/paper_123.jsonl",
  "output_path": "Data/ChunkedDocTagFiles/paper_123.parquet",
  "schema_version": "docparse/1.1.0",
  "input_hash": "sha256:abc...",
  "attempts": 1
}
```

### Stage-Specific Extras

**DocTags**:

```json
{
  "parse_engine": "docling-vlm",
  "model_name": "granite-docling-258M",
  "served_models": ["llama-2-7b"],
  "vllm_version": "0.3.0"
}
```

**Chunk**:

```json
{
  "chunk_count": 42,
  "tokens_p50": 256,
  "tokens_p95": 490,
  "chunks_format": "parquet"
}
```

**Embed**:

```json
{
  "vectors_dense": 42,
  "vectors_sparse": 42,
  "dim_dense": 2560,
  "avg_nnz_sparse": 87
}
```

---

## Error Handling

### Error Categories

| Category | Retryable | Example |
|---|---|---|
| `input` | No | Missing input file |
| `config` | No | Invalid URL configuration |
| `runtime` | No | GPU OOM |
| `timeout` | No | Item exceeded deadline |
| `network` | Yes | HTTP 429/5xx from provider |
| `transient` | Yes | Race condition in lock |
| `unknown` | No | Unclassified exception |

### Retry Policy

Only `retryable=True` errors are retried with exponential backoff:

```
delay = base * 2^(attempt-1) + jitter
```

**Example**: `retry_backoff_s=1.0`, `retries=3`

- Attempt 1: Immediate
- Attempt 2: 1.0 ± 0.25s
- Attempt 3: 2.0 ± 0.5s
- Attempt 4: 4.0 ± 1.0s → all retries exhausted, mark failed

---

## Safety & Determinism

### Atomic Writes

All manifest writes use `FileLock` for atomicity. No partial writes exposed.

### Deterministic Ordering

- Item discovery order preserved (no shuffling)
- Manifest rows appended in order
- p50/p95 computed consistently

### Signal Handling

`SIGINT`/`SIGTERM` sets cancel flag:

- No new submissions
- Inflight tasks allowed to complete (safe cleanup)
- Return with `cancelled=True`

### No Heavy Imports

`core/runner.py` depends only on `concurrent.futures`, `logging`, not on Docling, vLLM, or other heavy libraries. Stages import their deps.

---

## Authoring a New Stage

Create a stage that mirrors a DocParsing structure:

### 1. Plan Builder

```python
def build_my_plan(cfg: MyCfg) -> StagePlan:
    """Discover items and create WorkItems."""
    items = []
    for input_path in discover_inputs(cfg.input_dir):
        item_id = derive_id(input_path)
        output_path = derive_output_path(item_id, cfg.output_dir)

        item = WorkItem(
            item_id=item_id,
            inputs={"input": input_path},
            outputs={"output": output_path},
            cfg_hash=compute_cfg_hash(cfg),
            cost_hint=estimate_cost(input_path),
        )
        items.append(item)

    return StagePlan(stage_name="my_stage", items=items, total_items=len(items))
```

### 2. Worker

```python
def my_stage_worker(item: WorkItem) -> ItemOutcome:
    """Process one item, return ItemOutcome."""
    try:
        input_path = item.inputs["input"]
        output_path = item.outputs["output"]

        # Do actual work
        result = my_conversion_logic(input_path)

        return ItemOutcome(
            status="success",
            duration_s=result.duration,
            manifest={
                "output_size": result.bytes_written,
                "record_count": result.count,
            },
        )
    except Exception as exc:
        return ItemOutcome(
            status="failure",
            duration_s=0.0,
            error=StageError(
                stage="my_stage",
                item_id=item.item_id,
                category="runtime",
                message=str(exc),
                retryable=False,
            ),
        )
```

### 3. Hooks (Optional)

```python
def my_stage_hooks(cfg: MyCfg) -> StageHooks:
    def before_stage(ctx: StageContext) -> None:
        # Allocate shared resources (e.g., open model, HTTP pool)
        pass

    def after_stage(outcome: StageOutcome, ctx: StageContext) -> None:
        # Cleanup and summarize
        pass

    return StageHooks(
        before_stage=before_stage,
        after_stage=after_stage,
    )
```

### 4. CLI Wiring

```python
@app.command()
def my_stage(
    ctx: typer.Context,
    input_dir: Annotated[Path, typer.Option(...)] = None,
    # ... more options ...
    workers: Annotated[int, typer.Option(...)] = None,
    timeout_s: Annotated[float, typer.Option(...)] = None,
) -> None:
    app_ctx = ctx.obj
    cfg = create_config(app_ctx, input_dir, ...)
    plan = build_my_plan(cfg)

    options = StageOptions(
        policy="io",
        workers=workers or 4,
        per_item_timeout_s=timeout_s or 0.0,
    )
    hooks = my_stage_hooks(cfg)

    outcome = run_stage(plan, my_stage_worker, options, hooks)
    return 0 if outcome.failed == 0 else 1
```

---

## Performance Tuning

### Baseline Targets

- **DocTags PDF**: 5–10 docs/min (A100)
- **Chunk**: 10–20 docs/min (CPU)
- **Embed**: 5–8 docs/min (A100)

### Tuning Knobs

| Knob | Default | Guidance |
|---|---|---|
| `--workers` | 4 | Raise slowly while watching p95 |
| `--policy` | io | Change to cpu for compute-heavy stages |
| `--max-queue` | 0 | Set to 2×workers to bound memory |
| `--error-budget` | 0 | Raise to collect all errors before stop |
| `--diagnostics-interval-s` | 30 | Lower for frequent updates (verbose) |

### Profiling

```bash
# Profile a stage
python -m cProfile -m DocsToKG.DocParsing.core.cli chunk --limit 50

# Flame graph (with py-spy)
py-spy record -o profile.svg -- python -m DocsToKG.DocParsing.core.cli embed --limit 50
```

---

## Debugging & Troubleshooting

### Issue: Timeout Errors on Slow Items

**Check**:

```bash
# Inspect manifest for slow items
jq 'select(.duration_s > 10)' Data/Manifests/docparse.chunk.manifest.jsonl | head -5
```

**Fix**:

```bash
# Raise per-item timeout
docparse chunk --timeout-s 60
```

### Issue: Memory Spike During Embed

**Check**:

```bash
# Monitor max_queue
jq '.max_queue' Config
```

**Fix**:

```bash
# Limit queue backpressure
docparse embed --max-queue 8
```

### Issue: Intermittent Network Failures

**Check**:

```bash
# Count network errors
jq 'select(.category == "network")' Data/Manifests/docparse.embed.manifest.jsonl | wc -l
```

**Fix**:

```bash
# Enable retries for transient errors
docparse embed --retries 3 --retry-backoff-s 1.0
```

### Issue: Worker Process Hanging

**Check**:

```bash
# Look for infinite loops in worker logic
strace -p <pid>
```

**Fix**:

- Set `--timeout-s` to kill hung workers
- Review worker code for deadlocks

---

## References

- **Runner Source**: `src/DocsToKG/DocParsing/core/runner.py`
- **Manifest Sink**: `src/DocsToKG/DocParsing/core/manifest_sink.py`
- **Chunk Stage Example**: `src/DocsToKG/DocParsing/chunking/runtime.py`
- **Embed Stage Example**: `src/DocsToKG/DocParsing/embedding/runtime.py`
- **Tests**: `tests/docparsing/test_runner_semantics.py`
