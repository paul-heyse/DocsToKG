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

- Items: <!-- add items here -->
- Tests: <!-- add items here -->
- Docs: <!-- add items here -->

### Performance & Profiling

Files: `core/batching.py`, `core/concurrency.py`, `core/runner.py`

- Items: <!-- add items here -->
- Benchmarks: <!-- add items here -->
- Docs: <!-- add items here -->

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
  - [ ] Manifest sink: safe rotation/compaction utilities for large JSONL files (optional maintenance tool).
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

- Docs:
  - [ ] Define orchestrator invariants: idempotency, resume/force rules, manifest fields, fingerprint contract.
  - [ ] Runner lifecycle diagram with hooks (`before/after stage`, `before/after item`) and error-budget flow.
  - [ ] Planner CLI reference: flags, JSON schema, environment variables (preview limit, vector format default).
  - [ ] Manifest sink field dictionary, including common `extras` keys per stage (e.g., `vector_format`, `qwen_dim`).
  - [ ] Concurrency FAQ: locking, atomic writes, spawn semantics, free port reservations.

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
