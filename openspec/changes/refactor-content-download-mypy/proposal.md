## Why
- `pre-commit run mypy --all-files` currently fails with dozens of errors concentrated in the content download test suite (e.g. `tests/content_download/test_atomic_writes.py:198`, `tests/content_download/test_networking.py:2326`, `tests/content_download/test_runner_download_run.py:54`).
- The root causes include ad-hoc `ModuleType` shims baked directly into tests, mismatched helper signatures (tuples returning the wrong arity, enums passed as raw strings), duplicate fixture definitions, and missing third-party stub packages (`types-requests`, etc.).
- Without tightening these areas the repository-wide mypy hook remains red, blocking stronger typing guarantees for the download pipeline.

## What Changes
- Add a dedicated `tests/content_download/fakes/` package (with README + migration notes) that mirrors optional dependencies such as `pyalex` and reuses the doc parsing fake loader instead of inlined `ModuleType` definitions.
- Refactor each affected test module to use shared, typed helpers:
  * `test_atomic_writes.py`: return the correct tuple shapes, feed real tokenizer instances into `DummyHybridChunker`, and ensure `DownloadOutcome` uses `Classification` enums.
  * `test_networking.py`: collapse duplicate `_make_artifact` helpers, replace raw string classifications, supply typed telemetry loggers, and load `pyalex` fakes from the new package.
  * `test_download_strategy_helpers.py` / `test_network_unit.py`: fix generator / tuple annotations so helper functions satisfy MyPy expectations.
  * `test_runner_download_run.py`: broaden `_build_args` / `make_resolved_config` typing so `.update()` and downstream uses match `ResolvedConfig`.
- Update `RunTelemetry` to implement `__enter__` / `__exit__` and, if needed, provide a small typed wrapper for sinks used in tests so the class is instantiable under mypy.
- Add required stub packages (e.g. `types-requests`) to the dev dependency set and regenerate `requirements.txt` so third-party imports no longer appear untyped.

## Impact
- The content download tests will pass mypy without ignores, enabling the global pre-commit hook to run cleanly.
- Shared fake modules reduce duplication, making it easier to extend or adjust optional dependency behaviour in the future.
- Aligning test helpers with production interfaces lowers the risk of divergent assumptions between tests and real pipeline code.

## Open Questions
- Should the content download fakes live alongside the doc parsing fakes (e.g. shared package) or remain in a sibling namespace to keep concerns separated?
- Are additional third-party stub packages (e.g. `types-urllib3`) required once `types-requests` is installed, or are existing type hints sufficient?
