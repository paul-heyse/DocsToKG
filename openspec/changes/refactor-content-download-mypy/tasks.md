## 1. Baseline the Current Failures
- [ ] 1.1 Run `pre-commit run mypy --files tests/content_download/*.py src/DocsToKG/ContentDownload/telemetry.py` and paste the output into `openspec/changes/refactor-content-download-mypy/mypy-baseline.md` for traceability.
- [ ] 1.2 Label each diagnostic with a theme (e.g. “bad return type”, “missing stub package”, “dynamic module stub”) so downstream tasks can reference the exact line numbers they must fix.

## 2. Introduce Shared Fake Dependencies
- [ ] 2.1 Scaffold `tests/content_download/fakes/__init__.py` explaining the package goal and how it complements `tests.docparsing.fake_deps`.
- [ ] 2.2 Add `tests/content_download/fakes/README.md` and `MIGRATION_NOTES.md` documenting which external modules are mirrored (starting with `pyalex` and any others discovered during task 1).
- [ ] 2.3 Implement `tests/content_download/fakes/pyalex/__init__.py` exposing `Topics`, `Works`, and a nested `config` module with `mailto`; ensure attributes match the expectations from `tests/content_download/test_networking.py`.
- [ ] 2.4 Create `tests/content_download/stubs.py` (or similar helper) that invokes `tests.docparsing.stubs.dependency_stubs()` and then registers the new content-download-specific fakes.
- [ ] 2.5 Update all content download tests currently fabricating modules via `_stub_module` (notably `test_atomic_writes.py`) to call the new helper instead of constructing `ModuleType` stubs inline.

## 3. Refine Test Utilities And Fixtures
- [ ] 3.1 `test_atomic_writes.py`: adjust `_DummySession.head()` to return `_BaseDummyResponse`, update `_download_with_session()` to return a 4-tuple with the correct types (`WorkArtifact`, `Path`, `Dict[str, Dict[str, Any]]`, `DownloadOutcome`), and pass a real tokenizer instance into `DummyHybridChunker`.
- [ ] 3.2 `test_atomic_writes.py`: ensure chunker/embedding stubs publish `Classification` enums rather than raw strings; add type annotations for manifest logs and helpers.
- [ ] 3.3 `test_networking.py`: consolidate `_make_artifact` definitions into a single helper that accepts keyword overrides and returns a typed `WorkArtifact`; refactor fixtures to import fakes from step 2.
- [ ] 3.4 `test_networking.py`: rewrite `ListLogger`, resolver pipelines, and telemetry helpers so they satisfy both the `AttemptSink` protocol and context-manager requirements (`__enter__` / `__exit__` returning `self` / `None`).
- [ ] 3.5 `test_networking.py`: replace raw string classifications with `Classification` enums and update calls to `_make_artifact` (no unexpected keyword arguments).
- [ ] 3.6 `test_download_strategy_helpers.py`: change `_FakeResponse.iter_content` to declare an iterator return type (`Iterator[bytes]`) and annotate any helper generators accordingly.
- [ ] 3.7 `test_network_unit.py`: update `_session_for_response` so the return annotation reflects the `(Mock, Callable)` tuple actually produced; adjust downstream uses to unpack accordingly.
- [ ] 3.8 `test_runner_download_run.py`: widen `_build_args` to accept `Dict[str, object]` overrides (and annotate `defaults`) so `.update()` conforms to `MutableMapping.update`.
- [ ] 3.9 Ensure all modified tests import the new stubs via `tests.content_download.stubs` and remove now-unused Utilities (e.g. duplicate `_make_artifact`, raw `ModuleType` creation).

## 4. Production Code Tweaks
- [ ] 4.1 Implement `RunTelemetry.__enter__` and `RunTelemetry.__exit__`, delegating to the wrapped sink’s context manager if available, so the class satisfies the `AttemptSink` protocol under mypy.
- [ ] 4.2 Add unit coverage (or extend existing tests) to confirm the new context-manager behaviour works as expected when used with in-memory sinks.

## 5. Dependency Hygiene
- [ ] 5.1 Add `types-requests` (and any additional required stub packages discovered during implementation) to `requirements.in`.
- [ ] 5.2 Re-run the dependency lock process (`pip-compile` or project-specific script) to update `requirements.txt` / other lock files.
- [ ] 5.3 Document the new stub dependency in `openspec/changes/refactor-content-download-mypy/mypy-baseline.md` or README so future contributors understand why it exists.

## 6. Validation
- [ ] 6.1 Re-run `pre-commit run mypy --files tests/content_download/*.py src/DocsToKG/ContentDownload/telemetry.py` and ensure all targeted diagnostics disappear.
- [ ] 6.2 Execute representative pytest selections: `pytest tests/content_download -k atomic_writes`, `pytest tests/content_download -k networking`, and `pytest tests/content_download -k runner_download_run`.
- [ ] 6.3 Confirm `pytest tests/content_download/test_download_strategy_helpers.py` and `pytest tests/content_download/test_network_unit.py` both pass without relying on dynamic stubs.
- [ ] 6.4 Update `tests/content_download/fakes/MIGRATION_NOTES.md` with any additional fake modules or attributes introduced during the change.
