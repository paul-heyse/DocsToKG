## 1. Dependencies & Adapters
- [ ] 1.1 Add `filelock>=3.16.0` and `jsonlines>=4.0.0` to the DocParsing runtime extras in `pyproject.toml` (e.g., `[project.optional-dependencies]["docparse"]`); document the optional `msgspec` extra if we decide to expose a `docparse-fastio` profile.
- [ ] 1.2 Update dependency documentation (`src/DocsToKG/DocParsing/README.md` prerequisites + `openspec/AGENTS.md` prerequisites) to call out the new runtime requirements and optional fast-I/O extra.
- [ ] 1.3 Implement an internal adapter in `DocParsing/io.py` (e.g., `_jsonl_iter_adapter`) that wraps `jsonlines` iteration while enforcing `skip_invalid`, `max_errors`, `start`, and `end` semantics; keep it private (not exported via `__all__`).

## 2. Locking Consolidation
- [ ] 2.1 Replace `core/concurrency.acquire_lock` with a thin helper that yields a `FileLock(path.with_suffix(".lock"))` context manager; remove PID polling, jitter, and stale-eviction helpers.
- [ ] 2.2 Update all call sites (IO helpers, manifest writers, telemetry, stage runtimes, and related tests such as `tests/content_download/test_atomic_writes.py`) to use the new helper or direct `FileLock`; ensure no code manually creates `.lock` sentinels.
- [ ] 2.3 Adjust lock-related tests to assert `FileLock` semantics (lock file exists during hold, nested directories supported, timeout raises `TimeoutError` when configured).

## 3. JSONL Reader Rewrite
- [ ] 3.1 Delete `_iter_jsonl_records` and refactor `iter_jsonl` / `iter_jsonl_batches` (and re-exports in `DocsToKG.DocParsing.core`) to use the adapter; preserve parameters (`start`, `end`, `skip_invalid`, `max_errors`).
- [ ] 3.2 Keep `jsonl_load` as a deprecation shim that warns once and delegates to `iter_jsonl`; update tests (`tests/docparsing/test_docparsing_core.py`, `tests/hybrid_search/test_pipeline_jsonl_errors.py`, `tests/content_download/test_atomic_writes.py`) to cover the same error-budget and slicing scenarios under the new implementation.

## 4. Telemetry Writer Injection
- [ ] 4.1 Refactor `StageTelemetry` to accept a writer callable (default: `with file_lock(path): jsonl_append_iter(path, rows, atomic=True)`); remove `_acquire_lock_for` and direct lock manipulation.
- [ ] 4.2 Update telemetry tests (`tests/docparsing/test_chunk_validation_telemetry.py`, `tests/docparsing/test_doctags_overwrite.py`) to verify default locking, custom writer injection, and concurrent append behavior.

## 5. Documentation & Cleanup
- [ ] 5.1 Scrub docs (`docs/04-api/DocsToKG.DocParsing.telemetry.md`, `docs/04-api/DocsToKG.DocParsing.io.md`, DocParsing README/AGENTS) for references to `_acquire_lock_for`, `_iter_jsonl_records`, or custom `.lock` creation; document the new helper and library-backed iteration.
- [ ] 5.2 Add a CI/lint guard (e.g., `scripts/enforce_no_manual_docparse_locks.py`) that fails when code writes `.lock` sentinels directly or reintroduces `_iter_jsonl_records`; wire it into the existing lint/test harness.
- [ ] 5.3 Record the dependency and behavioral updates in `docs/06-operations/docparsing-changelog.md` and regenerate Sphinx/API docs to ensure no broken references remain.
