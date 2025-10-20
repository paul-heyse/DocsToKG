# Standardize DocParsing JSONL I/O and File Locks

## Why
DocParsing still relies on bespoke `.lock` sentinels, PID polling, and hand-written JSONL iterators. These reinventions drift from each other, complicate telemetry, and leave edge cases that mature libraries (`filelock`, `jsonlines`) already solve. Aligning on shared primitives reduces maintenance while preserving the hardened atomic append behavior the pipeline depends on.

## What Changes
- Adopt `filelock.FileLock` as the single locking primitive for manifests and telemetry, deleting custom lock helpers, busy-wait loops, and PID eviction logic.
- Replace `_iter_jsonl_records` with a library-backed adapter so `iter_jsonl` / `iter_jsonl_batches` retain their signatures (`skip_invalid`, `max_errors`, `start`, `end`) without bespoke parsing code.
- Refactor telemetry sinks to accept a writer callable (defaulting to `FileLock` + `jsonl_append_iter(..., atomic=True)`), eliminating `_acquire_lock_for` and consolidating append logic.
- Update tests and documentation to reflect the new locking and JSONL behavior, ensuring atomic writes and error budgeting remain unchanged.

## Impact
- **Affected specs:** docparsing.
- **Affected code:** `src/DocsToKG/DocParsing/io.py`, `telemetry.py`, `core/concurrency.py`, stage runtimes, associated tests, and dependency metadata (`pyproject.toml`, docs).
- **New dependencies:** add `filelock` and `jsonlines` to the DocParsing runtime extras (optionally expose `msgspec` in a fast-I/O extra for future optimization).
