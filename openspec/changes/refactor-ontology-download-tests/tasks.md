## 1. Harness Foundation
- [x] 1.1 Create `src/DocsToKG/OntologyDownload/testing/__init__.py` exporting a `TestingEnvironment` context manager.
    - Responsibilities: create per-run cache/storage/tmp dirs, spawn a loopback HTTP server to serve ontology fixtures, track cleanup callbacks.
- [x] 1.2 Implement harness helpers:
    - `register_fixture(name, data_or_path)` to expose ontology test fixtures via HTTP.
    - `build_configs()` returning `(ResolvedConfig, DownloadConfiguration)` pre-wired to harness directories and rate limits.
    - Reset SESSION_POOL, token buckets, plugin registries, and env vars on exit.
- [x] 1.3 Provide pytest fixture (`ontology_download_env`) in `tests/conftest.py` that yields `TestingEnvironment` and ensures teardown.

## 2. Runtime Extension Points
- [x] 2.1 Extend `plugins.py` with public `register_resolver`, `unregister_resolver`, `register_validator`, `unregister_validator` functions.
    - Requirements: use `_PLUGINS_LOCK`, maintain entry metadata, enforce duplicate handling (`overwrite` flag), raise informative errors.
- [x] 2.2 Expose context managers in testing module (`temporary_resolver`, `temporary_validator`) built on the registration APIs.
- [x] 2.3 Update `DownloadConfiguration` (or factory) to accept optional `session_factory` and `bucket_provider` hooks, and plumb those through the downloader code.
- [ ] 2.4 Add unit tests covering new registration APIs (thread-safety, metadata, automatic restoration).

## 3. Harness Utilities & Fixtures
- [x] 3.1 Implement reusable loopback HTTP server utility (e.g., in `testing/http.py`) with logging for assertions.
- [x] 3.2 Ensure ontology fixture assets exist for streams, archives, malformed cases; document how tests register them.
- [x] 3.3 Add helper to seed manifests and cached metadata within the harness (`TestingEnvironment.seed_manifest`).

## 4. Test Suite Migration
- [ ] 4.1 Rewrite `tests/ontology_download/test_download.py` to rely on the harness and temporary resolvers; remove dummy sessions and monkeypatch usage.
- [ ] 4.2 Migrate remaining ontology_download tests (integration, config, validators, CLI, etc.) enumerating each file: adopt harness, temporary resolvers/validators, real HTTP downloads.
- [ ] 4.3 Add regression test ensuring production defaults operate unchanged without harness (baseline fetch using existing API).
- [ ] 4.4 Remove obsolete fixtures/helpers from `tests/ontology_download/conftest.py` (legacy monkeypatch stubs).

## 5. Guardrails & Documentation
- [ ] 5.1 Implement pytest plugin (or fixture) that fails if `pytest.MonkeyPatch` is requested within ontology_download tests.
- [ ] 5.2 Add CI/static check to prevent `monkeypatch` usage reappearing (script or lint rule).
- [ ] 5.3 Document harness usage, resolver registration APIs, and testing guidelines in developer docs (new or existing guide).
