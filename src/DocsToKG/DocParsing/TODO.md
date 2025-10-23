# DocParsing TODOs

## Usage

- Organize work by module below; keep items short and actionable.
- Use checkboxes; link PRs/issues inline when available.

- Large TODOs are expected; it's fine to leave them partially complete. Under the parent item, indent sub-bullets to track progress over time:
  - Resolved: <!-- note what has been completed so far -->
  - Remaining: <!-- note what still needs to be done -->

## Cross-cutting Workstreams

### Code Quality & Testing

- Items: <!-- add items here -->
- Tests: <!-- add items here -->
- Docs: <!-- add items here -->

### Observability & Telemetry

Files: `telemetry.py`, `logging.py`

- Items:
  - Instrumentation
    - [ ] Structured logging: ensure JSON logs with stable fields (stage, doc_id, status, duration, attempt, error_code).
    - [ ] Metrics: define counters/gauges for attempts (success/skip/failure), queue/exec latency (p50/p95/p99), throughput.
    - [ ] Tracing: add OpenTelemetry spans for stage/item; attributes: stage, doc_id, status, durations, error flags.
  - Tooling
    - [ ] Exporters: optional OTel stdout/OTLP exporters; Prometheus metrics endpoint (guarded by flag); Jaeger config recipe.
    - [ ] Log routing: support file JSONL + stdout concurrently; rotation guidance for JSONL files.
  - Implementation
    - [ ] Auto-instrument select libs (httpx) where relevant; correlate HTTP retries with item spans.
    - [ ] Central log setup in `logging.py` with mask/scrub of PII and paths when `DOCSTOKG_SCRUB_LOGS=1`.
    - [ ] Add `run_id` propagation to all telemetry calls in `StageTelemetry` and loggers.
    - [ ] Emit stage start/finish summary events with counts and wall time.
  - Best practices
    - [ ] Sampling: configurable trace sampling rates; disable high-cardinality labels by default.
    - [ ] Context: include correlation IDs, host, pid, policy, workers; avoid sensitive values.
  - Security & Compliance
    - [ ] Redaction policy: hash/omit doc paths; explicit allowlist for extras keys; GDPR notes for logs.
    - [ ] Transport security: recommend OTLP over TLS; document token/env handling.
  - Performance
    - [ ] Async/non-blocking handlers for logs/metrics; bounded queues with backpressure.
    - [ ] Measure telemetry overhead in profiling; target <3% CPU time.
  - Integration & Deployment
    - [ ] CI smoke: validate logs parse as JSON; metrics endpoint exposes expected series; traces exported in staging.
    - [ ] Dashboards: create suggested Prometheus/Grafana panels (success rate, p95 exec, backlog, error codes).

- Tests:
  - [ ] JSON logging: schema presence for base fields; no PII when scrubbing enabled.
  - [ ] Metrics: counters monotonic; latency histograms bucket correctly; throughput derived accurately.
  - [ ] Tracing: spans and attributes present; error spans on failures; sampling honored.
  - [ ] Run correlation: run_id present across attempts, manifests, provider events.
  - [ ] Overhead: telemetry on/off comparison within budget; async handlers do not block.

- Docs:
  - [ ] Telemetry guide: logging fields, metric names/units, trace attributes; exporter setup (OTLP, Prometheus, Jaeger).
  - [ ] Examples: code snippets for emitting events via `StageTelemetry`; HTTP client correlation.
  - [ ] Operations: dashboard cookbook; alert examples (error budget burn, skip spikes, p95 latency).

### Performance & Profiling

Files: `core/batching.py`, `core/concurrency.py`, `core/runner.py`

- Items:
  - Profiling tools
    - [ ] cProfile integration: add `--profile-cpu` CLI flag per stage; capture `.prof` output per run.
    - [ ] pstats rendering: helper to print top-N cumulative/self time; export TSV for CI artifacts.
    - [ ] line_profiler: optional dependency; annotate hotspots (runner dispatch, planning walkers) with `@profile` guards.
    - [ ] memory_profiler: optional `--profile-mem` sampling around `run_stage`; record peak RSS; guard overhead.
    - [ ] py-spy: recipe + script to record sampling profiles; generate SVG flamegraphs for long runs.
    - [ ] pyinstrument: provide one-shot profiler helper and CLI recipe; emit HTML report.
  - Visualization
    - [ ] SnakeViz: verify `.prof` compatibility and document `snakeviz` usage for interactive analysis.
    - [ ] Flame graphs: support `py-spy record --flame` and `gprof2dot` pipelines; store under `Data/Profiles/`.
  - Optimization strategies
    - [ ] Batching: ensure streaming generator path in `Batcher` for no-policy mode (no materialization); add fast path asserts.
    - [ ] Discovery: minimize `stat()` calls and directory sorting where not required; cache symlink resolutions.
    - [ ] Runner: reduce lock contention; minimize per-item logging; precompute hook lookups; cheap time sources.
    - [ ] IO: adopt buffered writes and atomic rename path; coalesce small writes.
    - [ ] Concurrency: evaluate thread vs process policies per stage; document guidance.
  - Best practices
    - [ ] Realistic workloads: fixture generator mirroring `Data/` scale tiers (S, M, L) for local benchmarking.
    - [ ] Hotspot focus: 80/20 rule checklist; only optimize top offenders from profiles.
    - [ ] Continuous monitoring: optional nightly profiling job storing artifacts; compare to baselines.
    - [ ] Regression budget: define acceptable deltas per stage; fail builds over threshold.

- Benchmarks:
  - [ ] Micro-benchmarks: `timeit` harness for tight loops (marker parsing, hashing, batching order).
  - [ ] pytest-benchmark: integrate for `runner`, `planning`, `discovery`; emit JSON and compare in CI.
  - [ ] Scenario runs: scripted timings for `doctags|chunk|embed` on S/M datasets; capture wall, p50/p95 exec, CPU, RSS.
  - [ ] Baseline matrix: track across Python versions (3.12/3.13), policies (io/cpu), worker counts, vector formats.
  - [ ] CI gating: warn on >10% regression, fail on >20%; store artifacts for inspection.

- Docs:
  - [ ] Profiling guide: cProfile + SnakeViz; py-spy flamegraphs; pyinstrument HTML.
  - [ ] Benchmarking guide: pytest-benchmark usage, baseline storage, interpreting regressions.
  - [ ] Optimization playbook: common hotspots and recommended fixes per module.
  - [ ] Performance ledger: changelog of major improvements with before/after metrics.

### Security & Policy Gates

Scope: config validation, file IO safety, network hygiene

- Items: <!-- add items here -->
- Tests: <!-- add items here -->
- Docs: <!-- add items here -->

### CI/CD & Releases

Scope: lint, type-check, tests, packaging, docs

- Items: <!-- add items here -->
- Checks: <!-- add items here -->
- Docs: <!-- add items here -->

### Documentation

Files: `README.md`, `AGENTS.md`, `CONFIGURATION.md`, `LibraryDocumentation/`

- Items: <!-- add items here -->
- Gaps: <!-- add items here -->
- Links: <!-- add items here -->

## Modules & Pipelines

### Core Orchestrator (`core/`)

Files: `planning.py`, `discovery.py`, `runner.py`, `manifest.py`, `manifest_sink.py`, `models.py`, `cli_utils.py`, `http.py`, `batching.py`, `concurrency.py`

- Items:
  - [ ] Runner: unify resume semantics with manifests. Decide contract: (a) rely on `ResumeController` everywhere, or (b) require worker-written `fingerprint` JSON for `_should_skip`. Document the invariant and enforce in all stages.
  - [ ] Runner: expose hook for error-budget exhaustion (e.g., `on_error_budget_exceeded`) and include `errors` snapshot.
  - [ ] Runner: propagate `cancelled=True` when budget or Ctrl-C aborts before all items complete; verify counts (`scheduled`, `succeeded`, `failed`, `skipped`).
  - [ ] Runner: extend metrics to include `queue_p95_ms`, `exec_p99_ms`, and total CPU time if available.
  - [ ] Runner: clarify `max_queue` behavior for single-threaded mode (no executor). Either ignore gracefully or emulate bounded queueing; document behavior.
  - [ ] Runner: deterministic jitter seeding. Gate `random.seed` with namespace to avoid global PRNG side effects across stages.
  - [ ] Runner: structured progress events for each diagnostics interval; include `throughput_docs_per_min`, `ETA`, and `pending`.
  - [ ] Runner: standardize item outcome statuses to enum-like constants; validate in `_handle_worker_payload`.
  - [ ] Runner: ensure `skip` reported by workers increments `skipped` without double-count in resume path; add guard.
  - [ ] Planning: make `PLAN_PREVIEW_LIMIT` configurable via env (`DOCSTOKG_PLAN_PREVIEW`) and CLI flag.
  - [ ] Planning: align default vector format with runtime/CLI env (`DOCSTOKG_EMBED_VECTOR_FORMAT`) instead of hard-coded `jsonl`.
  - [ ] Planning: add `--format parquet|jsonl` to `plan` subcommands and ensure mismatch notes explain remediation.
  - [ ] Planning: support `--output json` to emit machine-readable plan summaries (retain current pretty print). Wire to `display_plan`.
  - [ ] Planning: support `--limit` across doctags/chunk/embed plans to cap traversal work for previews.
  - [ ] Discovery: honor ignore patterns (e.g., `_tmp`, `_invalid`, dot-directories) via env and optional args.
  - [ ] Discovery: add case-insensitive suffix handling and guard for mixed extensions.
  - [ ] Discovery: expand `vector_artifact_name` to validate/normalize format; provide helpful error with known values.
  - [ ] Manifest: extend `should_skip_output` to consider vector format/dimension mismatches when `manifest_entry` provides hints.
  - [ ] Manifest: `ResumeController.can_skip_without_hash` should prefer `output_path` normalization (resolve symlinks) before existence check.
  - [ ] Manifest sink: add optional `attempts` parameter and persist attempts across retries for auditing.
  - [ ] Manifest sink: persist `vector_format` and other stage extras (e.g., qwen dim) in `extras` consistently; document keys.
  - [ ] Manifest sink: configurable FileLock timeout via env/constructor; emit structured error on lock timeout.
  - [x] Manifest sink: safe rotation/compaction utilities for large JSONL files (optional maintenance tool). Expose rotation via `JsonlManifestSink.rotate_if_needed` with optional dedupe snapshot for operations.
  - [ ] HTTP: expose `clone_with_headers` on shared session from `get_http_session` when base headers change per call; validate cookie/auth copying safety.
  - [ ] HTTP: allow per-request override of retry policy (total/backoff/status set) through kwargs or context.
  - [ ] HTTP: optionally honor `Retry-After` date header skew with bounded max wait.
  - [ ] Batching: add policy `balanced_length` to more evenly distribute long items across batches.
  - [ ] Batching: expose helper to compute lengths from metadata to avoid pre-materializing when not needed.
  - [ ] Concurrency: add atomic write pattern (write to temp file + fsync + rename) to `safe_write`; preserve current behavior under flag.
  - [ ] Concurrency: include `log_event` on lock acquisition and contention (with wait duration) for observability.
  - [ ] Concurrency: add option to keep `.lock` files on failure for forensics.

- Tests:
  - [ ] Runner: budget exhaustion aborts promptly; counts consistent; after_stage invoked once.
  - [ ] Runner: per-item timeouts cancel futures; retries respect backoff; jitter bounded; deterministic with seed.
  - [ ] Runner: resume skip via fingerprint doesn’t re-run; worker `skip` increments `skipped` exactly once.
  - [ ] Planning: vector format mismatch yields items in `process` bucket with explanatory note.
  - [ ] Planning: preview limit respected; env/CLI override works; JSON output matches schema.
  - [ ] Discovery: symlink de-dup works; mixed-case suffixes handled; ignore patterns applied.
  - [ ] Manifest: `should_skip_output` true only when status success/skip AND hashes match; path normalization honored.
  - [ ] Manifest sink: concurrent writers append atomically (fork/thread); lock timeout surfaces clean error.
  - [ ] HTTP: `Retry-After` seconds and date parsing; per-request overrides; base headers cloning.
  - [ ] Batching: fixed-size mode streams lazily; length policy ordering stable; buckets follow power-of-two.
  - [ ] Concurrency: `safe_write` atomic temp+rename survives crash simulation; lock contention metrics recorded.
  - [ ] Scheduler: cost-aware strategy executes short items first; no starvation under mixed lengths.
  - [ ] Dynamic concurrency: tuning reacts to latency/queue depth; respects min/max worker bounds.
  - [ ] OTel: spans emitted for stage/item with expected attributes; nested spans close on errors.
  - [ ] Security: redaction removes PII/paths; sandbox prevents writing outside data root; traversal attempts rejected.
  - [ ] Signals: `SIGTERM` triggers graceful cancellation with correct manifest entries and counts.
  - [ ] Pause/Resume: sentinel files pause progress and resume without dropping state; abort halts with `cancelled=True`.
  - [ ] Watcher: continuous plan detects new/removed inputs; debounces bursts; JSON output schema stable.
  - [ ] Lineage: manifest extras include expected parent/child pointers; downstream queries can reconstruct flows.
  - [ ] Windows/macOS: file locking and atomic rename behave as expected (skipped if platform not available in CI).

- Docs:
  - [ ] Define orchestrator invariants: idempotency, resume/force rules, manifest fields, fingerprint contract.
  - [ ] Runner lifecycle diagram with hooks (`before/after stage`, `before/after item`) and error-budget flow.
  - [ ] Planner CLI reference: flags, JSON schema, environment variables (preview limit, vector format default).
  - [ ] Manifest sink field dictionary, including common `extras` keys per stage (e.g., `vector_format`, `qwen_dim`).
  - [ ] Concurrency FAQ: locking, atomic writes, spawn semantics, free port reservations.
  - [ ] Plugin architecture: how to implement a custom `SchedulerStrategy`/`RetryPolicy`; stability guarantees.
  - [ ] Observability guide: OpenTelemetry spans + metrics; recommended exporters; field dictionary.
  - [ ] Security posture: redaction policy, sandboxing constraints, path normalization rules.
  - [ ] Operations: pause/resume/abort workflow using sentinels; continuous plan watcher usage and limitations.
  - [x] Operations: document manifest rotation workflow (`rotate_if_needed`) including rotation thresholds, compaction output, and lock safety expectations.
  - [ ] Versioning: manifest schema versioning, migration utilities, backward-compat policies.

### DocTags Stage (`doctags.py`)

- Items: <!-- add items here -->
- Tests: <!-- add items here -->
- Docs: <!-- add items here -->

### Chunking Stage (`chunking/`)

Files: `config.py`, `runtime.py`, `cli.py`

- Items: <!-- add items here -->
- Tests: <!-- add items here -->
- Docs: <!-- add items here -->

### Embedding Stage (`embedding/`)

Files: `config.py`, `runtime.py`, `cli.py`
Backends: `backends/dense/`, `backends/sparse/`, `backends/lexical/`, `backends/factory.py`, `backends/utils.py`, `backends/nulls.py`

- Items: <!-- add items here -->
- Backend-specific: <!-- add items here -->
- Tests: <!-- add items here -->
- Docs: <!-- add items here -->

### Storage & IO (`storage/`)

Files: `paths.py`, `writers.py`, `readers.py`, `chunks_writer.py`, `embedding_integration.py`, `parquet_schemas.py`, `dataset_view.py`

- Items: <!-- add items here -->
- Tests: <!-- add items here -->
- Docs: <!-- add items here -->

### Formats & Schemas (`formats/`, `schemas.py`)

Note: `schemas.py` is a deprecated shim; track removal window.
Files: `formats/markers.py`, `schemas.py`

- Items: <!-- add items here -->
- Migration/Deprecation: <!-- add items here -->
- Tests: <!-- add items here -->

### CLI Layer (`cli.py`, `cli_unified.py`, `cli_errors.py`)

- Items: <!-- add items here -->
- UX/Help Text: <!-- add items here -->
- Tests: <!-- add items here -->

### Configuration & Context

Files: `config.py`, `config_loaders.py`, `config_adapter.py`, `settings.py`, `env.py`, `app_context.py`, `context.py`, `profile_loader.py`

- Items: <!-- add items here -->
- Validation: <!-- add items here -->
- Tests: <!-- add items here -->
- Docs: <!-- add items here -->

### Interfaces & Contracts (`interfaces.py`)

- Items: <!-- add items here -->
- Tests: <!-- add items here -->
- Docs: <!-- add items here -->

### Token Profiles (`token_profiles.py`)

- Items: <!-- add items here -->
- Tests: <!-- add items here -->
- Docs: <!-- add items here -->

### IO & Utilities

Files: `io.py`, `app_context.py`, `context.py`

- Items: <!-- add items here -->
- Tests: <!-- add items here -->
- Docs: <!-- add items here -->

## Data & Manifests

Files: `core/manifest.py`, `core/manifest_sink.py`, `${DOCSTOKG_DATA_ROOT}/Data/Manifests/*`

- Items: <!-- add items here -->
- Integrity/Idempotency: <!-- add items here -->
- Ops/Recovery: <!-- add items here -->

## Backward Compatibility & Deprecations

Targets: `schemas.py` removal, config shims, CLI flags

- Items: <!-- add items here -->
- Migration Guides: <!-- add items here -->

## Operational Playbooks

Runbooks for common failure modes, retries, resume

- Items: <!-- add items here -->
- Docs: <!-- add items here -->
