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
    - [x] cProfile integration: `docparse perf run` emits `.pstats` and collapsed stacks for each stage (see docs/06-operations/docparsing-performance-monitoring.md).
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
    - [x] Continuous monitoring: `docparse perf run --baseline` enforces regression budgets with exit code 2 and documented cron recipe.
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
  - [x] Runner: propagate `cancelled=True` when budget or Ctrl-C aborts before all items complete; verify counts (`scheduled`, `succeeded`, `failed`, `skipped`).
  - [ ] Runner: extend metrics to include `queue_p95_ms`, `exec_p99_ms`, and total CPU time if available.
  - [ ] Runner: clarify `max_queue` behavior for single-threaded mode (no executor). Either ignore gracefully or emulate bounded queueing; document behavior.
  - [ ] Runner: deterministic jitter seeding. Gate `random.seed` with namespace to avoid global PRNG side effects across stages.
  - [ ] Runner: structured progress events for each diagnostics interval; include `throughput_docs_per_min`, `ETA`, and `pending`.
  - [ ] Runner: standardize item outcome statuses to enum-like constants; validate in `_handle_worker_payload`.
  - [x] Runner: ensure `skip` reported by workers increments `skipped` without double-count in resume path; add guard.
  - [ ] Planning: make `PLAN_PREVIEW_LIMIT` configurable via env (`DOCSTOKG_PLAN_PREVIEW`) and CLI flag.
  - [ ] Planning: align default vector format with runtime/CLI env (`DOCSTOKG_EMBED_VECTOR_FORMAT`) instead of hard-coded `jsonl`.
  - [ ] Planning: add `--format parquet|jsonl` to `plan` subcommands and ensure mismatch notes explain remediation.
  - [ ] Planning: support `--output json` to emit machine-readable plan summaries (retain current pretty print). Wire to `display_plan`.
  - [ ] Planning: support `--limit` across doctags/chunk/embed plans to cap traversal work for previews.
  - [ ] Discovery: honor ignore patterns (e.g., `_tmp`, `_invalid`, dot-directories) via env and optional args.
  - [ ] Discovery: add case-insensitive suffix handling and guard for mixed extensions.
  - [x] Discovery: expand `vector_artifact_name` to validate/normalize format; provide helpful error with known values.
  - [ ] Manifest: extend `should_skip_output` to consider vector format/dimension mismatches when `manifest_entry` provides hints.
  - [ ] Manifest: `ResumeController.can_skip_without_hash` should prefer `output_path` normalization (resolve symlinks) before existence check.
  - [ ] Manifest sink: add optional `attempts` parameter and persist attempts across retries for auditing.
  - [ ] Manifest sink: persist `vector_format` and other stage extras (e.g., qwen dim) in `extras` consistently; document keys.
  - [ ] Manifest sink: configurable FileLock timeout via env/constructor; emit structured error on lock timeout.
  - [x] Manifest sink: safe rotation/compaction utilities for large JSONL files (optional maintenance tool). Expose rotation via `JsonlManifestSink.rotate_if_needed` with optional dedupe snapshot for operations.
  - [x] HTTP: expose `clone_with_headers` on shared session from `get_http_session` when base headers change per call; validate cookie/auth copying safety.
  - [ ] HTTP: allow per-request override of retry policy (total/backoff/status set) through kwargs or context.
  - [ ] HTTP: optionally honor `Retry-After` date header skew with bounded max wait.
  - [ ] Batching: add policy `balanced_length` to more evenly distribute long items across batches.
  - [ ] Batching: expose helper to compute lengths from metadata to avoid pre-materializing when not needed.
  - [ ] Concurrency: add atomic write pattern (write to temp file + fsync + rename) to `safe_write`; preserve current behavior under flag.
  - [ ] Concurrency: include `log_event` on lock acquisition and contention (with wait duration) for observability.
  - [x] Concurrency: add option to keep `.lock` files on failure for forensics.
  - [x] Concurrency: include `log_event` on lock acquisition and contention (with wait duration) for observability.
    - Resolved: `_acquire_lock` records wait durations via monotonic timers and emits structured lock metadata when contention occurs.
    - Remaining: Feed contention timings into future telemetry/metrics surfaces once StageTelemetry aggregation lands.
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

- Status (pending commentary):
  - P1 focus
    - Runner diagnostics: add `throughput_docs_per_min`, ETA, and `queue_p95_ms`/`exec_p99_ms` in summaries.
    - Planner outputs: `--output json` + `--limit` and env override for preview; align default vector format.
    - Concurrency durability: add temp+rename atomic path to `safe_write`; emit lock contention duration logs.
    - Plugin hooks: formal `SchedulerStrategy` and `RetryPolicy` selection via options/env.
  - P2 focus
    - HTTP tuning: per-request retry overrides and bounded `Retry-After` wait.
    - Dynamic concurrency: adjust workers based on latency/backlog; respect min/max bounds.
    - Manifest sink: persist attempts count and standardize stage `extras` (e.g., vector metadata).
  - Owners: Core/DocParsing. Target: next minor release window.

### DocTags Stage (`doctags.py`)

- Items:
  - Functionality
    - [ ] PDF mode: harden vLLM bootstrap (health probe, timeout, retries) and surface clear errors when model not ready.
    - [ ] HTML mode: ensure `list_htmls/iter_htmls` parity with PDFs (sorted traversal, ignore patterns, symlink guards).
    - [ ] Auto mode: decide resolution strategy (prefer PDFs when both present?); document behavior.
    - [ ] Standardize DocTags schema version (`docparse/*`) and ensure manifest writes include version for both modes.
    - [ ] Expose tokenizer/profile knobs via CLI profiles; snapshot config into `__config__` manifest.
  - Integration
    - [ ] Use `ResumeController` consistently for skip decisions (hash + manifest) across pdf/html paths.
    - [ ] Ensure `run_stage` hooks are wired for per-item telemetry and error budgeting in both modes.
    - [ ] Align attempts/manifest sinks to `MANIFEST_STAGE` and `HTML_MANIFEST_STAGE` with consistent extras (parse_engine, model).
  - Error Handling
    - [ ] Convert docling exceptions and vLLM startup failures into `StageError` categories with actionable messages.
    - [ ] Skip malformed inputs with structured reasons; continue pipeline without aborting.
  - Performance
    - [ ] Batch-friendly worker sizing (GPU threads, port reuse) configurable by profile; avoid CPU oversubscription.
    - [ ] Reduce directory traversal overhead (reuse generators; avoid repeated stat calls).
  - Security & Compliance
    - [ ] Validate input paths reside under `${DOCSTOKG_DATA_ROOT}`; reject traversal; redact absolute paths in logs.
    - [ ] Sanitize HTML inputs to avoid script execution or unsafe includes during parsing.
  - Future Enhancements
    - [ ] Add support for additional formats (DOCX, TXT) via docling where feasible; guard with feature flags.
    - [ ] Explore ML-assisted tag refinement; plug-in hook for postprocessing.

- Tests:
  - [ ] vLLM bootstrap: unhealthy server yields clear failure, retries honored; metrics probe recorded.
  - [ ] Resume: unchanged inputs skip via manifest+hash; changed inputs reprocess; html/pdf parity.
  - [ ] Traversal: deterministic order; symlink dedupe; ignore patterns.
  - [ ] Worker: success/skip/failure outcomes produce correct manifest attempts and extras.
  - [ ] Error mapping: docling exceptions categorized; helpful messages bubbled to CLI.
  - [ ] Security: path sandbox enforced; HTML sanitization tested on malicious fixtures.
  - [ ] Performance: profile baseline for N PDFs/HTMLs; no >10% regression across releases.

- Docs:
  - [ ] DocTags schema reference (fields, versions, examples) and compatibility notes.
  - [ ] CLI usage for pdf/html/auto modes; profiles and environment variables.
  - [ ] Operational guidance: vLLM lifecycle, ports, GPU utilization, troubleshooting.

### Chunking Stage (`chunking/`)

Files: `config.py`, `runtime.py`, `cli.py`

- Items:
  - Implement chunking strategies
    - [ ] Fixed-size windows: calibrate `min_tokens`/`max_tokens` per tokenizer profile; guard against mid-sentence splits via hybrid coalescence.
    - [ ] Semantic-aware: tune `is_structural_boundary` markers (headings/captions) and `soft_barrier_margin` for coherence.
    - [ ] Hybrid: validate `coalesce_small_runs` behavior across diverse documents; ensure token ceilings respected.
  - Overlap & anchors
    - [ ] Optional overlap policy (N tokens) between consecutive chunks; expose via CLI; reflect in `ChunkRow` metadata.
    - [ ] Anchor injection: ensure unique `<<chunk:...>>` anchors; document stability and off-by-one guards.
  - Metadata & schema
    - [ ] Preserve refs/pages/image metadata; verify `ChunkRow` completeness and provenance population.
    - [ ] Maintain deterministic `uuid` per chunk (span hashing) across runs; document invariants.
  - Performance & scalability
    - [ ] Avoid repeated `stat()`/sorting during traversal; reuse lists and iterators.
    - [ ] Guard tokenizer initialisation once per worker; share across tasks.
    - [ ] Sharding correctness: stable shard mapping via `compute_stable_shard`; add fairness note.
  - Integration
    - [ ] Resume: unify fingerprint usage and `ResumeController` decisions; ensure `.fp.json` written for Parquet primary paths.
    - [ ] Format routing: Parquet default via `ParquetChunksWriter`; JSONL fallback path maintained; persist `chunks_format`.
    - [ ] Validate-only: ensure parquet path resolution via dataset layout; quarantine invalid artifacts.
  - Error handling & security
    - [ ] Map worker exceptions to `StageError` with actionable messages; continue pipeline.
    - [ ] Enforce data-root sandbox; redact absolute paths in logs.

- Tests:
  - [ ] Strategy: fixed/semantic/hybrid produce coherent chunks and respect token limits on varied fixtures.
  - [ ] Overlap: adjacent chunk boundaries include expected token overlap; metadata reflects overlap.
  - [ ] Metadata: `refs/pages/image` counters populated; `uuid` deterministic across runs.
  - [ ] Perf: single init of tokenizer per worker; traversal not quadratic; basic timing guard.
  - [ ] Sharding: stable assignment across runs; empty shard yields warning and exit 0.
  - [ ] Resume: skip when input hash unchanged; fingerprint path respected; force overrides.
  - [ ] Format: parquet write success path; parquet failure → JSONL fallback; `chunks_format` recorded.
  - [ ] Validate-only: quarantines bad artifacts; reports rows/row_groups; logs warnings.
  - [ ] Errors: exceptions categorized; manifests reflect failure with schema fields.
  - [ ] Security: sandbox prevents writes outside data root; redaction leaves no PII.

- Docs:
  - [ ] Strategy guide: choosing min/max tokens, barrier margin, markers; examples.
  - [ ] CLI reference and profiles; sharding usage and caveats.
  - [ ] Parquet dataset layout and JSONL fallback; validate-only workflow and quarantine.

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

- Items:
  - Data models & versions
    - [ ] Canonical Pydantic models for `ChunkRow` and `VectorRow` with strict types and defaults.
    - [ ] Centralize `CHUNK_SCHEMA_VERSION`/`VECTORS_SCHEMA_VERSION` and embed in Parquet footers + JSONL fields.
    - [ ] Footer metadata: include `created_by`, `cfg_hash`, `hash_alg`, `docling_version` consistently.
  - Validation
    - [ ] Strengthen `validate_chunk_row`/vector validation with clear error messages and fast-path option.
    - [ ] Parquet validation via `parquet_schemas.validate_parquet_file` for required columns/types.
  - Structural markers
    - [ ] Unify `load_structural_marker_profile` formats (JSON/YAML/TOML) and document precedence/merging rules.
    - [ ] Provide defaults and sanity checks (non-empty, printable markers) and warnings on invalid entries.
  - Hashing/IDs
    - [ ] Declare allowed hash algorithms; ensure stable `uuid` computation and record algorithm in manifests.

- Migration/Deprecation:
  - [ ] Announce `DocsToKG.DocParsing.schemas` removal (v0.3.0); emit deprecation warning from import shim.
  - [ ] Add temporary adapter in `formats.__init__` for legacy imports; search-and-replace guidance for users.
  - [ ] Release notes: breaking-window timeline, migration steps, code samples.

- Tests:
  - [ ] Golden fixtures for chunk/vector rows (JSONL + Parquet) validated against models and parquet schema.
  - [ ] Footer metadata presence and correctness; version bumps reflected in artifacts.
  - [ ] Marker loader accepts JSON/YAML/TOML and rejects invalid structures with actionable errors.
  - [ ] Backward-compat: older artifacts still validate when version policy allows.

- Docs:
  - [ ] Schema reference (fields, types, constraints, footers) with examples.
  - [ ] Marker profile format and merging policy; recommended headings/captions.
  - [ ] Versioning policy and migration guide; compatibility matrix.

### CLI Layer (`cli.py`, `cli_unified.py`, `cli_errors.py`)

- Items:
  - Command structure
    - [ ] Ensure unified commands: `doctags`, `chunk`, `embed`, `plan`, `manifest`, `token-profiles`, `all`.
    - [ ] Normalize shared flags (`--resume`, `--force`, `--data-root`, `--log-level`) across stages.
  - Argument parsing & outputs
    - [ ] Standardize exit codes (0 ok, 1 failure, 2 CLI validation error) via `cli_errors` helpers.
    - [ ] Add `--output json|table` for plan/manifest listings; JSON schema documented.
    - [ ] Add `--dry-run` to eligible commands; avoid side effects.
    - [ ] Add `--profile` consistency across stages; print applied defaults.
    - [ ] Add `--version` and shell completion artifacts.
  - Environment & precedence
    - [ ] Print effective configuration (`--show-config`) with masked secrets; show source (CLI/env/file/default).
    - [ ] Enforce precedence: CLI > env > config file > defaults; document per-stage overrides.

- UX/Help Text:
  - [ ] Curate help text with examples for common flows; group options logically; cross-link to docs.
  - [ ] Provide troubleshooting hints in error messages (e.g., missing dirs, dependency import failures).

- Tests:
  - [ ] Parse smoke for all commands; invalid args surface `cli_errors` with exit 2.
  - [ ] Help text generation stable and mentions core flags; examples compile.
  - [ ] `--output json` emits valid JSON; schema verified; table mode human-readable.
  - [ ] Precedence tests: flags override env which override file; `--show-config` masks secrets.
  - [ ] `--dry-run` executes without writes; exit codes correct.

### Configuration & Context

Files: `config.py`, `config_loaders.py`, `config_adapter.py`, `settings.py`, `env.py`, `app_context.py`, `context.py`, `profile_loader.py`

- Items:
  - Configuration management
    - [ ] Unify stage adapters (DoctagsCfg/ChunkerCfg/EmbedCfg) behind `ConfigurationAdapter` interface.
    - [ ] Support JSON/YAML/TOML via `config_loaders`; env var expansion; relative-path resolution from file.
    - [ ] Precedence policy: CLI > env > file > defaults; per-field source tracking for `to_manifest()`.
    - [ ] Mask sensitive values in snapshots/logs (tokens, URLs, absolute paths) with stable redaction.
  - Context propagation
    - [ ] `ParsingContext` carries run_id, data_root, dirs, resume/force, profile, workers; thread/process-safe.
    - [ ] `app_context` init helpers for one-time setup; avoid global mutable state leaks across forks.
    - [ ] Stable `cfg_hash` computation for manifests; include only semantically relevant fields.
  - Profiles & presets
    - [ ] `profile_loader` supports per-stage named presets; merging strategy and validation.
    - [ ] Provide builtin profiles (cpu-small, gpu-default, gpu-max, bert-compat) and allow user overrides.
  - Security & sandbox
    - [ ] Enforce data-root sandbox; normalize/resolve paths; reject traversal; document escape hatches.

- Validation:
  - [ ] Precedence unit tests (CLI/env/file/default) for all major fields; source tracking verified.
  - [ ] Path normalization and sandbox checks; missing file detection; helpful messages.
  - [ ] Invalid combinations (min>max, shard index range, unknown format) produce `ChunkingCLIValidationError`.
  - [ ] Snapshot manifests contain masked secrets and expected fields across stages.

- Tests:
  - [ ] Adapter roundtrip: from args/env/file → cfg → context → manifest snapshot.
  - [ ] Profile overlays apply correctly; defaults respected; error on unknown profile.
  - [ ] Concurrency safety: no cross-process leakage of context/config; worker init idempotent.

- Docs:
  - [ ] Configuration reference per stage with examples; precedence chart and env var mapping tables.
  - [ ] Profiles guide (when to use which); adding custom profiles.
  - [ ] Security guidance for sandboxing and secret redaction; troubleshooting config errors.

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
