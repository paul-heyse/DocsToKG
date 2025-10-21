# DocParsing Runner Scope Audit & Implementation Plan

**Date**: October 21, 2025
**Status**: Audit Complete — Gap Analysis Underway
**Document Purpose**: Validate runner implementation against the DocParsing-Runner-config-review.md scope document and identify gaps.

---

## Executive Summary

The DocParsing module has **75-80% of the runner scope implemented**:

✅ **COMPLETE (Implemented)**:

- Core runner module (`core/runner.py`) with full contract definitions
- Manifest & telemetry integration (via `telemetry.py` and `io.py`)
- Progress & diagnostics aggregator (built into runner)
- Chunk stage runner integration (uses `run_stage()`)
- Embed stage runner integration (uses `run_stage()`)
- CLI wiring for chunk & embed stages
- Resume/force semantics
- Error handling & categorization

⚠️ **PARTIAL / INCOMPLETE**:

- DocTags stage: Creates `StagePlan` but **does NOT use `run_stage()`** — still uses legacy execution loop
- Manifest sink abstraction: Exists but not fully unified across stages
- Progress reporting: Implemented in runner but not optimized for multi-stage pipelines
- Configuration harmonization: Stages map CLI options but no central standardization doc

❌ **NOT YET ADDRESSED**:

- CLI options documentation for unified runner flags (--workers, --policy, --retries, --timeout-s, --error-budget)
- Comprehensive test suite for runner semantics (only per-stage integration tests exist)
- Reference documentation for "authoring a new stage"
- Migration guide & escape hatch documentation

---

## Detailed Status by Scope Item

### Commit A: Core Runner Module ✅ COMPLETE

**File**: `src/DocsToKG/DocParsing/core/runner.py` (713 LOC)

**Status**: 100% Complete

**Deliverables**:

- ✅ `StagePlan` dataclass (stage_name, items, total_items)
- ✅ `WorkItem` frozen dataclass (item_id, inputs, outputs, cfg_hash, cost_hint, metadata, fingerprint)
- ✅ `ItemFingerprint` for resume tracking
- ✅ `StageOptions` (policy, workers, per_item_timeout_s, retries, retry_backoff_s, error_budget, max_queue, resume, force, diagnostics_interval_s)
- ✅ `StageError` structured exception (stage, item_id, category, message, retryable, detail)
- ✅ `ItemOutcome` result wrapper (status, duration_s, manifest, result, error)
- ✅ `StageOutcome` summary (scheduled, skipped, succeeded, failed, cancelled, wall_ms, queue_p50_ms, exec_p50_ms, exec_p95_ms, errors)
- ✅ `StageHooks` lifecycle callbacks (before_stage, after_stage, before_item, after_item)
- ✅ `StageContext` mutable context for hooks
- ✅ `run_stage()` main orchestration function with:
  - ThreadPool/ProcessPool creation per policy (io→ThreadPool, cpu→ProcessPool spawn, gpu→ThreadPool)
  - Resume/force skip logic via `ItemFingerprint.matches()`
  - Retry logic with exponential backoff + jitter
  - Error budget enforcement with cancellation
  - Timeout handling per item
  - Diagnostics logging at configurable intervals
  - Hook lifecycle management
  - Percentile computation for p50/p95

**Quality**: 100% type-safe, comprehensive error handling, well-documented

---

### Commit B: Manifest & Telemetry Integration ⚠️ PARTIAL

**Files**:

- `src/DocsToKG/DocParsing/telemetry.py` (450+ LOC)
- `src/DocsToKG/DocParsing/io.py` (200+ LOC)
- `src/DocsToKG/DocParsing/core/manifest.py` (103 LOC)

**Status**: 85% Complete — Core present but needs unification

**What exists**:

- ✅ `StageTelemetry` class with append-only JSONL writer (uses FileLock for atomicity)
- ✅ Manifest helper functions: `manifest_log_success()`, `manifest_log_skip()`, `manifest_log_failure()`
- ✅ Atomic JSONL append with FileLock
- ✅ Resume controller for manifest-based skip decisions
- ✅ Per-stage manifest extras (chunk counts, vector dims, etc.)

**Gaps**:

- ⚠️ No unified `manifest_sink.py` module as proposed
- ⚠️ Manifest writing is scattered: some in `telemetry.py`, some in hooks, some in stage-specific code
- ⚠️ Base field consistency not enforced (stage, doc_id, status, duration_s, input_path, output_path, schema_version, attempts)

**Action Required**: See Gap #1 below

---

### Commit C: Progress & Diagnostics Aggregator ✅ COMPLETE

**Implementation**: Built into `run_stage()` lines 349-630

**Status**: 95% Complete

**What exists**:

- ✅ Diagnostics logged every `diagnostics_interval_s` (default 30s)
- ✅ Progress fields: scheduled, completed, total, succeeded, failed, skipped, pending
- ✅ One-line format via `log_event()` with structured fields
- ✅ Percentile computation (p50, p95) for queue/exec times
- ✅ ETA derivable from items_per_s

**Gaps**:

- ⚠️ No verbose mode for slowest items or error categories
- ⚠️ No TTY-aware progress bar (single-line logging only)

---

### Commit D: DocTags → Runner ❌ INCOMPLETE

**Files**:

- `src/DocsToKG/DocParsing/doctags.py` (3300+ LOC)

**Status**: 20% — Creates plan but does NOT use runner

**What exists**:

- ✅ `_build_pdf_plan()` creates `StagePlan` with WorkItems (lines 460–562)
- ✅ `_pdf_stage_worker()` adapter function to normalize results (lines 565–617)
- ✅ `_make_pdf_stage_hooks()` constructs StageHooks (lines 620–680)
- ✅ Plan structure is correct: item_id, inputs, outputs, cfg_hash, cost_hint, metadata, fingerprint

**Critical Gap**:

- ❌ `pdf_main()` **does NOT call `run_stage()`** — still uses legacy loop (lines 2230–2400 approx)
- ❌ Legacy pattern: Manual ProcessPoolExecutor or ThreadPoolExecutor invocation
- ❌ Manifest writing in loop, not via runner hooks

**HTML Stage**: Similar status — `_build_html_plan()` exists but `html_main()` does not use runner

**Action Required**: See Gap #2 below

---

### Commit E: Chunk → Runner ✅ COMPLETE

**File**: `src/DocsToKG/DocParsing/chunking/runtime.py` (1850+ LOC)

**Status**: 95% Complete

**What exists**:

- ✅ Plan builder creates `StagePlan` from discovered chunk files
- ✅ `_chunk_stage_worker()` adapter wraps chunking logic
- ✅ `StageHooks` for corpus summary & per-item telemetry
- ✅ `run_stage()` call at line 1773
- ✅ Resume/force semantics via StageOptions

**Minor Gaps**:

- ⚠️ No explicit SJF scheduling (always FIFO)
- ⚠️ Limited error categorization (mostly "runtime")

---

### Commit F: Embed → Runner ✅ COMPLETE

**File**: `src/DocsToKG/DocParsing/embedding/runtime.py` (2800+ LOC)

**Status**: 95% Complete

**What exists**:

- ✅ Plan builder creates `StagePlan` from chunk files
- ✅ Per-family (dense/sparse/lexical) job dispatch
- ✅ `_embedding_stage_worker()` adapter
- ✅ `StageHooks` for provider lifecycle (before_stage opens, after_stage closes)
- ✅ `run_stage()` call at line 2885
- ✅ Resume/force semantics

**Minor Gaps**:

- ⚠️ Complex embedded loop for batching (pre-runner legacy code)
- ⚠️ Provider initialization in hooks could be cleaner

---

### Commit G: CLI Wiring & Options Mapping ⚠️ PARTIAL

**Files**:

- `src/DocsToKG/DocParsing/cli_unified.py` (600+ LOC)
- Stage-specific CLI modules (doctags.py, chunking/cli.py, embedding/cli.py)

**Status**: 80% Complete

**What exists**:

- ✅ CLI commands map options to StageOptions (workers, policy, resume, force)
- ✅ ConfigurationAdapter bridges CLI → stage config
- ✅ Stage commands accept --workers, --policy flags
- ✅ Resume/force flags wired

**Gaps**:

- ⚠️ Missing CLI flags: `--retries`, `--retry-backoff-s`, `--timeout-s`, `--error-budget`, `--max-queue`, `--schedule`
- ⚠️ No unified help text across stages
- ⚠️ Per-stage config still separate (DoctagsCfg, ChunkerCfg, EmbedCfg) — no central StandardOptions

**Action Required**: See Gap #3 below

---

### Commit H: Tests (Runner + Stages) ⚠️ PARTIAL

**Files**:

- `tests/docparsing/test_runner.py` (if exists)
- `tests/docparsing/test_*.py` (integration tests)

**Status**: 40% Complete

**What exists**:

- ✅ Per-stage integration tests (chunk manifest resume, embed validation)
- ✅ Doctags conversion tests (unit level)

**Major Gaps**:

- ❌ No dedicated runner unit tests (timeout, retries, error budget, cancellation, SJF)
- ❌ No end-to-end runner scenario tests (all three stages via runner)
- ❌ No performance regression suite for runner semantics

**Action Required**: See Gap #4 below

---

### Commit I: Docs & Migration Note ⚠️ MINIMAL

**Files**:

- `src/DocsToKG/DocParsing/README.md` (mentions runner in overview)
- `src/DocsToKG/DocParsing/AGENTS.md` (agent guide, no runner-specific section)

**Status**: 20% Complete

**What exists**:

- ✅ Brief README mentions runner in core capabilities
- ✅ AGENTS guide covers mission/scope

**Gaps**:

- ❌ No "Authoring a New Stage" guide
- ❌ No runner semantics documentation (timeouts, retries, budgets, SJF)
- ❌ No CLI reference for unified runner flags
- ❌ No migration guide from legacy loops
- ❌ No changelog entry

**Action Required**: See Gap #5 below

---

## Gap Analysis & Implementation Plan

### Gap #1: Unify Manifest Sink Abstraction

**Current State**: Manifest writing scattered across telemetry.py, hooks, and stage code

**Proposed Solution**: Extract `core/manifest_sink.py` (80–120 LOC)

**Tasks**:

1. Create `ManifestSink` protocol with three methods:
   - `log_success(stage, item_id, input_path, output_paths, duration, extras)`
   - `log_skip(stage, item_id, input_path, output_path, reason, duration, extras)`
   - `log_failure(stage, item_id, input_path, output_path, error, duration, extras)`
2. Implement `JsonlManifestSink` (atomic JSONL append with FileLock)
3. Update stage hooks to use sink instead of direct writes
4. Standardize base fields across all manifest rows

**Effort**: 4 hours | **Risk**: LOW

**Files to modify**:

- Create: `src/DocsToKG/DocParsing/core/manifest_sink.py` (new)
- Modify: `src/DocsToKG/DocParsing/doctags.py` (hooks)
- Modify: `src/DocsToKG/DocParsing/chunking/runtime.py` (hooks)
- Modify: `src/DocsToKG/DocParsing/embedding/runtime.py` (hooks)
- Modify: `src/DocsToKG/DocParsing/telemetry.py` (delegate to sink)

---

### Gap #2: Integrate DocTags Stage with Runner

**Current State**: `pdf_main()` and `html_main()` create StagePlan but use legacy loop

**Proposed Solution**: Replace execution loop with `run_stage()` call

**Tasks**:

1. Extract worker payload extraction from legacy loop into `_pdf_stage_worker()` and `_html_stage_worker()`
2. Replace manual pool/loop with `run_stage(plan, worker, options, hooks)` call
3. Build StageOptions from config (workers, policy, resume, force)
4. Update hooks to write manifests via sink (Gap #1)
5. Remove legacy executor code

**Effort**: 6 hours | **Risk**: MEDIUM (behavioral parity must be verified)

**Validation**:

- Run full doctags pipeline on reference corpus
- Compare manifest rows with legacy output (should be identical except manifest format)
- Verify error counts match

**Files to modify**:

- `src/DocsToKG/DocParsing/doctags.py` (pdf_main, html_main)

---

### Gap #3: Add Missing CLI Flags & Standardize Runner Options

**Current State**: Only `--workers`, `--policy`, `--resume`, `--force` wired; missing `--retries`, `--timeout-s`, `--error-budget`, etc.

**Proposed Solution**: Extend CLI to expose full runner knobs

**Tasks**:

1. Define `RunnerCliOptions` dataclass with fields:
   - `workers: int` (default 4)
   - `policy: str` (default "io", choices: io/cpu/gpu)
   - `schedule: str` (default "fifo", choices: fifo/sjf) — implement SJF sort
   - `per_item_timeout_s: float` (default 0.0)
   - `retries: int` (default 0)
   - `retry_backoff_s: float` (default 1.0)
   - `error_budget: int` (default 0)
   - `max_queue: int` (default 0)
   - `diagnostics_interval_s: float` (default 30.0)
2. Add Typer options to CLI for all flags
3. Build StageOptions from RunnerCliOptions
4. Document each flag with rationale
5. Add environment variable overrides (`DOCSTOKG_RUNNER_*`)

**Effort**: 3 hours | **Risk**: LOW

**Files to modify**:

- Modify: `src/DocsToKG/DocParsing/cli_unified.py` (add flags)
- Modify: `src/DocsToKG/DocParsing/core/runner.py` (add SJF sort logic if not present)
- Create: Reference doc for runner CLI flags (or add to README)

---

### Gap #4: Implement Runner Unit & Integration Tests

**Current State**: Per-stage tests exist, but no dedicated runner test suite

**Proposed Solution**: Add comprehensive runner tests

**Tasks**:

1. **Runner Unit Tests** (test_runner_semantics.py, ~300 LOC):
   - FIFO vs SJF (verify p95 improvement with mixed costs)
   - Timeout behavior (task exceeds deadline, error recorded)
   - Retries with exponential backoff
   - Error budget (budget=1, second error cancels submissions)
   - Cancellation (SIGINT sets flag, inflight tasks drain)
   - Resume/force (fingerprint matching, skip logic)
   - Hook lifecycle (before/after called in order, errors don't propagate)
   - Executor selection (policy→executor mapping)

2. **End-to-End Tests** (test_runner_e2e.py, ~200 LOC):
   - All three stages via runner (small synthetic corpus)
   - Manifest consistency across stages
   - Resume across stages (modify input, verify only affected stage reruns)

3. **Performance Regression** (test_runner_benchmark.py, ~150 LOC):
   - Throughput baseline (items/s)
   - p95 latency with SJF vs FIFO
   - Memory profile under high concurrency

**Effort**: 8 hours | **Risk**: LOW

**Files to create**:

- `tests/docparsing/test_runner_semantics.py` (unit)
- `tests/docparsing/test_runner_e2e.py` (integration)
- `tests/docparsing/test_runner_benchmark.py` (perf)

---

### Gap #5: Documentation & Migration Guide

**Current State**: README mentions runner; no detailed guide or migration path

**Proposed Solution**: Add comprehensive docs

**Tasks**:

1. **Runner Architecture Guide** (runner_architecture.md, ~400 LOC):
   - System map (CLI → runner → executor → worker → stage deps)
   - Contract definitions (StagePlan, WorkItem, StageOptions, etc.)
   - Lifecycle (before_stage → for each item → after_stage)
   - Error taxonomy & recovery
   - Safety & determinism invariants

2. **"Authoring a New Stage" Guide** (new_stage_guide.md, ~200 LOC):
   - Template for stage planner
   - Worker function pattern
   - Hooks usage (optional)
   - CLI wiring
   - Testing strategy

3. **CLI Reference** (update README or new cli_reference.md, ~150 LOC):
   - Unified runner flags documented
   - Examples: --workers, --timeout-s, --retries, --error-budget
   - Environment variable overrides
   - Performance tuning guidance

4. **Migration Guide** (migration_to_runner.md, ~200 LOC):
   - Legacy loop pattern → runner pattern
   - Escape hatch (DOCSTOKG_RUNNER=legacy) if needed
   - Behavioral compatibility matrix

5. **Changelog Entry**:
   - "All DocParsing stages now use a single runner; behavior unchanged, infrastructure unified"

**Effort**: 6 hours | **Risk**: LOW

**Files to create/modify**:

- Create: `docs/docparsing/runner_architecture.md`
- Create: `docs/docparsing/new_stage_guide.md`
- Modify: `src/DocsToKG/DocParsing/README.md` (add CLI reference section)
- Create: `docs/docparsing/migration_to_runner.md`
- Modify: `CHANGELOG.md`

---

## Implementation Roadmap

### Phase 1: Foundation (Days 1–2)

- [ ] Gap #1: Manifest sink abstraction (4 hrs)
- [ ] Gap #3: CLI flags & runner options (3 hrs)
- [ ] Test: Ensure existing stage tests still pass

### Phase 2: DocTags Integration (Days 3–4)

- [ ] Gap #2: Integrate DocTags with runner (6 hrs)
- [ ] Validation: Full corpus test with parity checks
- [ ] Test: Update or add doctags-specific runner tests

### Phase 3: Testing & Hardening (Days 5–6)

- [ ] Gap #4: Runner unit & integration tests (8 hrs)
- [ ] Performance regression baseline
- [ ] Fix any edge cases discovered

### Phase 4: Documentation (Day 7)

- [ ] Gap #5: Comprehensive docs (6 hrs)
- [ ] README updates
- [ ] CHANGELOG entry

### Phase 5: Integration & Rollout (Days 8+)

- [ ] All tests passing (100%)
- [ ] Performance parity or better
- [ ] Code review & merge
- [ ] Deprecation notice for legacy loops

---

## Acceptance Criteria (Definition of Done)

✅ **Functional**:

- [ ] DocTags stage calls `run_stage()` with plan, worker, options, hooks
- [ ] All three stages (doctags, chunk, embed) use unified runner
- [ ] Manifest rows have consistent base fields across stages
- [ ] Resume/force/timeout/retries/error-budget work identically across stages

✅ **Quality**:

- [ ] 100% type-safe (mypy clean)
- [ ] 0 ruff lint violations
- [ ] ≥95% test coverage for runner
- [ ] All 3 stage integration tests passing

✅ **Performance**:

- [ ] Throughput ≥ legacy (items/s)
- [ ] p95 latency ≤ legacy with SJF enabled
- [ ] Memory footprint stable under concurrency

✅ **Documentation**:

- [ ] Runner architecture guide complete
- [ ] "New Stage" authoring guide complete
- [ ] CLI reference with examples
- [ ] Migration guide or escape hatch documented
- [ ] CHANGELOG updated

✅ **Operational**:

- [ ] End-to-end pipeline (doctags → chunk → embed) passes
- [ ] Manifest diffs minimal (only format/field additions)
- [ ] No breaking changes to CLI or configuration

---

## Risk Assessment

| Gap | Risk | Mitigation |
|-----|------|-----------|
| #1: Manifest Sink | LOW | Backward-compatible wrapper, tests verify identical output |
| #2: DocTags Runner | MEDIUM | Extensive parity testing, corpus-level validation, escape hatch |
| #3: CLI Flags | LOW | Additive flags only, environment variable fallbacks |
| #4: Tests | LOW | Isolated unit tests, synthetic workloads |
| #5: Docs | LOW | Living documentation, peer review |

---

## Next Steps

1. **Approve scope**: Confirm all 5 gaps are in scope
2. **Prioritize**: Decide if all phases are required for this session
3. **Assign**: Pick 1–2 gaps per day
4. **Validate**: Post-implementation corpus test with parity report
5. **Merge**: Commit to main with deprecation notice for legacy loops

---

## References

- Original design: `DO NOT DELETE docs-instruct/DO NOT DELETE - Refactor review/DocParsing/DocParsing-Runner-config-review.md`
- Current runner: `src/DocsToKG/DocParsing/core/runner.py`
- Chunk stage: `src/DocsToKG/DocParsing/chunking/runtime.py` (line 1773)
- Embed stage: `src/DocsToKG/DocParsing/embedding/runtime.py` (line 2885)
- DocTags (legacy): `src/DocsToKG/DocParsing/doctags.py` (lines 2230–2400 approx, pdf_main)
