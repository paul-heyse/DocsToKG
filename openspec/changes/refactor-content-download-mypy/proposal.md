## Why
- `pre-commit run mypy --all-files` currently fails with dozens of errors concentrated in the content download test suite (e.g. `tests/content_download/test_atomic_writes.py:198`, `tests/content_download/test_networking.py:2326`, `tests/content_download/test_runner_download_run.py:54`).
- The root causes include ad-hoc `ModuleType` shims baked directly into tests, mismatched helper signatures (tuples returning the wrong arity, enums passed as raw strings), duplicate fixture definitions, and missing third-party stub packages (`types-requests`, etc.).
- Without tightening these areas the repository-wide mypy hook remains red, blocking stronger typing guarantees for the download pipeline.

- The implementation will land in phases to keep review scope manageable. **Phase 1** focuses exclusively on `tests/content_download/test_atomic_writes.py`, establishing the shared fakes framework and proving the approach. **Phase 2** targets `tests/content_download/test_networking.py`, which currently contributes the majority of content-download mypy failures due to duplicated helpers, raw string classifications, and inline pyalex stubs. Subsequent phases (tracked under the same change) will migrate the remaining suites once these two pillars are complete.
- Add a dedicated `tests/content_download/fakes/` package (with README + migration notes) that mirrors optional dependencies such as `pyalex` and reuses the doc parsing fake loader instead of inlined `ModuleType` definitions. Phase 1 will introduce only the modules required by the atomic writes suite (`docling_core`, `transformers`, `tqdm` shims can be re-exported from doc parsing stubs).
- Refactor `tests/content_download/test_atomic_writes.py` to rely on the shared fakes and typed helpers:
  * Replace inline `ModuleType` fabrication with calls into the new content download stub loader.
  * Update helper return types (`_download_with_session`, factory functions) so they align with `DownloadOutcome` signatures.
  * Ensure enum usage (`Classification`) replaces raw strings, and manifest log helpers return typed structures.
- After Phase 1, adjust the networking suite in Phase 2 by consolidating helper factories, introducing typed telemetry/attempt sink doubles, replacing raw strings with `Classification` enums, and loading optional dependency fakes through the shared stub installer. Remaining test modules (`test_download_strategy_helpers.py`, `test_network_unit.py`, `test_runner_download_run.py`) will follow in subsequent steps using the same template.
- Update `RunTelemetry` to implement `__enter__` / `__exit__` and, if needed, provide a small typed wrapper for sinks used in tests so the class is instantiable under mypy.
- Add required stub packages (e.g. `types-requests`) to the dev dependency set and regenerate `requirements.txt` so third-party imports no longer appear untyped.

## Impact
- The content download tests will pass mypy without ignores, enabling the global pre-commit hook to run cleanly.
- Shared fake modules reduce duplication, making it easier to extend or adjust optional dependency behaviour in the future.
- Aligning test helpers with production interfaces lowers the risk of divergent assumptions between tests and real pipeline code.

## Open Questions
- Should the content download fakes live alongside the doc parsing fakes (e.g. shared package) or remain in a sibling namespace to keep concerns separated?
- Are additional third-party stub packages (e.g. `types-urllib3`) required once `types-requests` is installed, or are existing type hints sufficient?
- Do we want to backfill broader documentation for the content download fakes once Phase 1 lands, or is a per-phase README update acceptable?
