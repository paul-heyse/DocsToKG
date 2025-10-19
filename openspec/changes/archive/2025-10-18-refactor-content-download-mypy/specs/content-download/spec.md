# Content Download Capability Specification Deltas

## ADDED Requirements
### Requirement: Content Download Tests Must Be Typed
Content download tests MUST rely on shared fake dependency modules and typed helper utilities so the mypy hook runs cleanly without suppressions.

#### Scenario: Atomic Write Tests Pass MyPy
- **GIVEN** `pre-commit run mypy --files tests/content_download/test_atomic_writes.py`
- **WHEN** the command executes
- **THEN** no `return-value`, `call-arg`, or `attr-defined` errors are reported
- **AND** the test imports dependency stubs from shared helper modules rather than constructing `ModuleType` instances inline.

#### Scenario: Networking Tests Use Typed Fixtures
- **GIVEN** `pre-commit run mypy --files tests/content_download/test_networking.py`
- **WHEN** the command executes
- **THEN** artifact factories, telemetry loggers, and optional dependency fakes satisfy MyPy (including `pyalex` symbols)
- **AND** resolver outcomes reference `DocsToKG.ContentDownload.core.Classification` enums instead of raw strings.

#### Scenario: Runner Tests Maintain Typed Config Builders
- **GIVEN** `pre-commit run mypy --files tests/content_download/test_runner_download_run.py`
- **WHEN** the command executes
- **THEN** helpers like `_build_args` and `make_resolved_config` expose concrete `Dict[str, object]` / `ResolvedConfig` types so no `MutableMapping.update` errors arise.

#### Scenario: Telemetry Context Manager Is Concrete
- **GIVEN** `pre-commit run mypy --files src/DocsToKG/ContentDownload/telemetry.py`
- **WHEN** the command executes
- **THEN** `RunTelemetry` implements `__enter__` / `__exit__` and MyPy can instantiate the class without reporting abstract methods.

#### Scenario: Third-Party Stubs Present
- **GIVEN** `pre-commit run mypy --files tests/content_download/test_download_strategy_helpers.py tests/content_download/test_network_unit.py`
- **WHEN** the command executes
- **THEN** no diagnostics appear requesting `types-requests` (or related packages) because the repository declares the necessary stub dependencies.
