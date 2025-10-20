## Context
DocParsing currently implements file locking by creating `.lock` sentinels, polling PIDs, and evicting stale files through `core/concurrency.acquire_lock` and telemetry’s `_acquire_lock_for`. JSONL iteration depends on `_iter_jsonl_records`, a bespoke loop layered under `iter_jsonl` / `iter_jsonl_batches`. These primitives are duplicated, hard to verify, and drift from well-tested libraries (`filelock`, `jsonlines`). Recent documentation (`LibraryDocumentation/JSONL_standardization.md`) outlines the desired migration toward library-backed locking and streaming while retaining DocParsing’s hardened atomic append behavior.

## Goals / Non-Goals
- Goals: standardize on `filelock.FileLock`; replace `_iter_jsonl_records` with a library-backed adapter that preserves `skip_invalid`, `max_errors`, `start`, and `end`; inject a writer dependency into telemetry so locking and append logic live in one place; keep existing public signatures and the atomic guarantees provided by `jsonl_append_iter(..., atomic=True)`.
- Non-Goals: change manifest schemas, directory layout, CLI flags, or the existing `atomic_write` helper; introduce new telemetry fields; alter JSON content.

## Decisions
- Provide a thin helper (e.g., `file_lock(path: Path, timeout: float | None)`) that wraps `filelock.FileLock(path.with_suffix(".lock"))` to coordinate writers; remove PID polling and manual eviction logic.
- Build an internal adapter around `jsonlines` iteration that enforces DocParsing’s semantics (skip-invalid, error budget, slicing) and reuse it for both single-file and batched iterators; keep `jsonl_load` as a deprecation shim delegating to `iter_jsonl`.
- Pass a writer callable into `StageTelemetry` (default: lock + `jsonl_append_iter(..., atomic=True)`) so tests can inject fakes and the codebase has a single append implementation.
- Add the new dependencies to the DocParsing runtime extras in `pyproject.toml`, optionally introducing a `docparse-fastio` extra with `msgspec` for future acceleration.

## Risks / Trade-offs
- Library iteration may surface different exception types; mitigate by wrapping or re-raising with existing error messages so tests continue to pass.
- `FileLock` may block differently than the busy-wait loop; configure reasonable timeout defaults and provide actionable error messages.
- Adding dependencies increases the install surface; documenting them in README/AGENTS avoids surprise for operators.

## Migration Plan
1. Introduce the new dependencies and adapter helpers while keeping the old implementations available for reference.
2. Replace all lock acquisitions with the new helper, remove legacy lock utilities, and update tests.
3. Swap JSONL iterators to the adapter and delete `_iter_jsonl_records`; adjust any helper that relied on the old implementation.
4. Refactor telemetry to use the injected writer and remove `_acquire_lock_for`; expand tests to cover lock usage and custom writers.
5. Update documentation and add CI guards to prevent reintroduction of bespoke lock or JSON loops.

## Open Questions
- Do we want to expose a pluggable serialization hook (e.g., `msgspec`) through the new writer? Initial answer: no; keep JSON serialization with a seam for future enhancement.
