## 1. Baseline the Current Failures
- [x] 1.1 Run `pre-commit run mypy --files tests/content_download/*.py src/DocsToKG/ContentDownload/telemetry.py` and paste the output into `openspec/changes/refactor-content-download-mypy/mypy-baseline.md` for traceability.
- [x] 1.2 Label each diagnostic with a theme (e.g. “bad return type”, “missing stub package”, “dynamic module stub”) so downstream tasks can reference the exact line numbers they must fix.

## 2. Introduce Shared Fake Dependencies
- [x] 2.1 Scaffold `tests/content_download/fakes/__init__.py` explaining the package goal and how it complements `tests.docparsing.fake_deps`.
- [x] 2.2 Add `tests/content_download/fakes/README.md` and `MIGRATION_NOTES.md` documenting which external modules are mirrored (starting with `pyalex` and any others discovered during task 1).
- [x] 2.3 Implement `tests/content_download/fakes/pyalex/__init__.py` exposing `Topics`, `Works`, and a nested `config` module with `mailto`; ensure attributes match the expectations from `tests/content_download/test_networking.py`.
- [x] 2.4 Create `tests/content_download/stubs.py` (or similar helper) that invokes `tests.docparsing.stubs.dependency_stubs()` and then registers the new content-download-specific fakes.
- [x] 2.5 Update all content download tests currently fabricating modules via `_stub_module` (notably `test_atomic_writes.py`) to call the new helper instead of constructing `ModuleType` stubs inline.

## 3. Phase 1 – Refactor `test_atomic_writes.py`
- [x] 3.1 Replace the top-level `_stub_module` invocations with a call to `tests.content_download.stubs.dependency_stubs()`, ensuring the helper is invoked before any imports that rely on fake modules.
- [x] 3.2 Update `_DummySession.head()` to return `_BaseDummyResponse` and adjust `_download_with_session()` so its signature returns a typed 4-tuple (`WorkArtifact`, `Path`, `Dict[str, Dict[str, Any]]`, `DownloadOutcome`).
- [x] 3.3 Ensure helper factories (`DummyHybridChunker`, manifest logger wrappers) accept typed arguments—provide a non-`None` tokenizer instance and annotate manifest log collections as `List[Dict[str, Any]]`.
- [x] 3.4 Replace raw string classifications with `Classification` enums where `DownloadOutcome` instances are built or asserted.
- [x] 3.5 Run `pre-commit run mypy --files tests/content_download/test_atomic_writes.py` and confirm all diagnostics for this module are resolved before proceeding to Phase 2.
- [x] 3.6 Execute `pytest tests/content_download/test_atomic_writes.py` to ensure behavioural parity after refactoring.

## 4. Phase 2 – Refactor `test_networking.py`
- [ ] 4.1 Replace the inline pyalex stubs (`types.ModuleType("pyalex")`, etc.) with imports from `tests.content_download.stubs`, ensuring the helper runs before any `pytest.importorskip("pyalex")` checks.
- [ ] 4.2 Consolidate the three `_make_artifact` definitions into a single helper at the top of the file. The helper should accept keyword overrides (e.g. `pdf_urls`, `title`) and always return a typed `WorkArtifact`.
- [ ] 4.3 Rework logger/telemetry doubles:
  * Introduce a typed `ListLogger`/`AttemptSink` stub that implements `log_attempt`, `log_manifest`, `log_summary`, `close`, `__enter__`, and `__exit__`.
  * Ensure any custom sink used in resolver pipelines (e.g. `_CaptureLogger`) satisfies the same interface.
- [ ] 4.4 Update resolver fixtures (`StubResolver`, fake download callbacks) so `DownloadOutcome` instances use `Classification` enums instead of raw strings, and adjust any assertions accordingly.
- [ ] 4.5 Normalise helper return types:
  * `_session_for_response` should return a tuple with explicit typing (`tuple[Mock, Callable[..., Response]]`) and callers must unpack the tuple.
  * Sequential session helpers should annotate their response iterables and maintain deterministic behaviour for tests.
- [ ] 4.6 Remove redundant state (e.g. duplicated `HAS_PYALEX` logic) once the shared stubs are in place, keeping only a single capability gate for pyalex-dependent tests.
- [ ] 4.7 Run `pre-commit run mypy --files tests/content_download/test_networking.py` and ensure all diagnostics stemming from this module are eliminated.
- [ ] 4.8 Execute focused pytest selections:
  * `pytest tests/content_download/test_networking.py::test_download_candidate_returns_cached`
  * `pytest tests/content_download/test_networking.py::test_retry_budget_honours_max_attempts`
  * `pytest tests/content_download/test_networking.py::test_openalex_attempts_use_session_headers`
  confirming behavioural parity after the refactor.

## 5. Phase 3 – Remaining Suite (Follow-up)
- [ ] 5.1 `test_download_strategy_helpers.py`: change `_FakeResponse.iter_content` to declare an iterator return type (`Iterator[bytes]`) and annotate generator helpers accordingly.
- [ ] 5.2 `test_network_unit.py`: update `_session_for_response` annotations and downstream unpacking to match the tuple actually returned.
- [ ] 5.3 `test_runner_download_run.py`: widen `_build_args` to accept `Dict[str, object]` overrides (and annotate `defaults`) so `.update()` conforms to `MutableMapping.update`.
- [ ] 5.4 Ensure all remaining tests import the shared stubs and remove leftover `ModuleType` construction.

## 5. Production Code Tweaks
- [ ] 5.1 Implement `RunTelemetry.__enter__` and `RunTelemetry.__exit__`, delegating to the wrapped sink’s context manager if available, so the class satisfies the `AttemptSink` protocol under mypy.
- [ ] 5.2 Add unit coverage (or extend existing tests) to confirm the new context-manager behaviour works as expected when used with in-memory sinks.

## 6. Dependency Hygiene
- [ ] 6.1 Add `types-requests` (and any additional required stub packages discovered during implementation) to `requirements.in`.
- [ ] 6.2 Re-run the dependency lock process (`pip-compile` or project-specific script) to update `requirements.txt` / other lock files.
- [ ] 6.3 Document the new stub dependency in `openspec/changes/refactor-content-download-mypy/mypy-baseline.md` or README so future contributors understand why it exists.

## 7. Validation
- [ ] 7.1 Re-run `pre-commit run mypy --files tests/content_download/*.py src/DocsToKG/ContentDownload/telemetry.py` and ensure all targeted diagnostics disappear.
- [ ] 7.2 Execute representative pytest selections: `pytest tests/content_download -k atomic_writes`, `pytest tests/content_download -k networking`, and `pytest tests/content_download -k runner_download_run`.
- [ ] 7.3 Confirm `pytest tests/content_download/test_download_strategy_helpers.py` and `pytest tests/content_download/test_network_unit.py` both pass without relying on dynamic stubs.
- [ ] 7.4 Update `tests/content_download/fakes/MIGRATION_NOTES.md` with any additional fake modules or attributes introduced during the change.
