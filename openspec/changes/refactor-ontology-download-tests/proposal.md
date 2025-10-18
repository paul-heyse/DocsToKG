# Refactor OntologyDownload Tests for Real Execution Paths

## Why

Existing `tests/ontology_download/**` suites rely heavily on monkeypatching internal modules, session factories, and resolver registries. The patch-based approach makes tests brittle, obscures production behaviour, and can drift from reality as the implementation evolves. Failing to exercise the real networking, resolver selection, storage, and validation code paths means regressions only surface when the downloader is run end-to-end in production.

Option 1—reworking the tests to consume the actual functionality exported from `src/DocsToKG/OntologyDownload`—demands new first-class support within the package so tests can drive full workflows without invasive overrides. We need a plan that eliminates ad-hoc stubs while keeping tests deterministic and hermetic.

## Current State

1. **Tight coupling on globals**: Tests mutate `RESOLVERS`, `STORAGE`, and module-level constants directly. There is no supported API to register temporary resolvers or restore defaults.
2. **Patched network stack**: Download tests replace `requests.Session`, rate limiters, and URL validators. As a result, retry logic, polite headers, and telemetry are never exercised end-to-end.
3. **Environment drift**: Each test configures environment variables via monkeypatch to coerce cache directories, API keys, and storage URLs. The downloader exposes no supported mechanism to create isolated temp environments.
4. **Difficult maintenance**: Because tests diverge from production, refactors (e.g., to resolver plug-ins or storage layout) require extensive test rewrites and provide little confidence in runtime behaviour.

## Proposal

Deliver a supported testing harness and extension points inside `DocsToKG.OntologyDownload` so the suite can drive the real pipeline without patching internals:

1. **Testing harness module**  
   - Add `DocsToKG.OntologyDownload.testing` with a `TestingEnvironment` context manager.  
   - Responsibilities: provision isolated cache/storage/temp directories, spin up a loopback HTTP file server that serves ontology fixtures, seed manifests/caches when needed, and expose helpers to register fixtures by name.  
   - Harness must return ready-to-use `ResolvedConfig`/`DownloadConfiguration` objects, drive downloads through public APIs (`core.fetch_all`, `core.plan_all`), and guarantee teardown resets session pools, token buckets, plugin registries, and environment variables.

2. **First-class resolver and validator registration APIs**  
   - Extend `resolvers.py`/`plugins.py` with `register_resolver`, `unregister_resolver`, and matching validator helpers that manage the shared registry safely.  
   - Provide context-manager helpers (`temporary_resolver`, `temporary_validator`) so tests can register local resolvers backed by loopback HTTP fixtures.  
   - Ensure thread-safety (lock usage), metadata tracking, and error reporting (duplicate registration, missing entries) stay consistent with plugin discovery.

3. **Configurable session and rate-limiter factories**  
   - Introduce optional hooks on `DownloadConfiguration` (or companion factory) allowing callers to supply custom session factories and token-bucket providers.  
   - Plumb these hooks through downloader internals so the harness can inject deterministic networking behaviour without monkeypatching. Defaults must remain unchanged for production users.

4. **Test migration**  
   - Rewrite `tests/ontology_download/**` to exclusively use the new harness, temporary registration helpers, and loopback HTTP fixtures.  
   - Ensure every test exercises real HTTP downloads, checksum computation, manifest parsing, validator execution, and storage writes; add regression coverage verifying production defaults still function.  
   - Remove all monkeypatch usage, dummy session classes, and direct manipulation of private module globals.

5. **Documentation and guardrails**  
   - Add developer documentation describing harness usage, resolver registration APIs, and testing guidelines for downstream integrators.  
   - Implement guardrails (pytest plugin and CI checks) to prevent reintroduction of monkeypatch in ontology download tests.

## Impact

- **Primary capability**: `ontology-download`
- **Affected modules**:  
  - `src/DocsToKG/OntologyDownload/{core,planning,resolvers,plugins,settings,io,validation}` for new hooks and registration APIs.  
  - New module `src/DocsToKG/OntologyDownload/testing.py` (or package) exporting the harness.  
  - `tests/ontology_download/**` rewritten to consume the harness.  
  - Documentation under `docs/` describing test workflow.
- **Risk**: Introducing new extension points touches critical runtime paths. We must ensure defaults remain unchanged and add regression tests for production usage.

## Success Criteria

1. All ontology download tests run without using monkeypatch or direct attribute overrides and still pass.  
2. The testing harness enables deterministic, isolated runs by default (no cross-test leakage).  
3. Production download behaviour remains untouched when the new APIs are unused (verified via regression tests).  
4. Documentation clearly explains how to use the harness and why patching is prohibited going forward.

## Open Questions

1. Do we need additional fixture data (e.g., larger archives, malformed ontologies) hosted within the repo to cover currently stubbed scenarios?  
2. Should the harness expose hooks for long-running integration tests (e.g., spinning up external validators) or remain focused on unit-level coverage?  
3. How do we enforce “no monkeypatch” in CI—linting, runtime assertions, or both?
