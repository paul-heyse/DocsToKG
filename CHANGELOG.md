# Changelog

All notable changes to DocsToKG are documented in this file.

## [Unreleased]

### Added
- Integration tests for the content download CLI covering CSV logging, staging runs,
  resume manifests with incomplete metadata, threaded execution with domain jitter,
  and head-precheck degradation behaviour.
- Regression safeguards that pin default resolver ordering, manifest entry schemas,
  legacy resume parsing, and legacy resolver configuration inputs.
- Download strategy helpers (`validate_classification`, `handle_resume_logic`,
  `cleanup_sidecar_files`, `build_download_outcome`) with comprehensive unit
  coverage to support artifact-specific logic.
- Regression tests exercising the HTTPX/Hishel transport: cache-hit handling of 304 responses, `Retry-After` backoff accounting, streaming download atomicity, and shared robots cache reuse using `httpx.MockTransport` fixtures.

### Changed
- DocParsing core modules are split into focused packages (`core.discovery`,
  `core.http`, `core.manifest`, `core.planning`, `core.cli_utils`) with a thin
  facade maintaining the stable import surface.
- Structural marker configuration loaders are exported via
  `DocsToKG.DocParsing.config_loaders` / `DocsToKG.DocParsing.config`, including
  stricter error reporting and new unit coverage.
- DocParsing CLIs surface invalid argument combinations via a shared
  `CLIValidationError` hierarchy, producing `[stage] --option: message` failures
  instead of Python tracebacks.
- Embedding runtime now defers imports of optional dependencies (SPLADE and
  vLLM) until required, raising actionable errors when packages are absent and
  supporting import without the heavy extras installed.
- Ontology planner metadata probes now invoke `validate_url_security` before
  issuing HTTP requests, reuse the shared polite networking stack
  (`SessionPool`, token buckets, polite headers, retry/backoff), emit structured
  telemetry for retries/fallbacks, support an opt-out flag
  (`defaults.planner.probing_enabled` / `--no-planner-probes`), and coordinate
  ontology index writes via an ontology-scoped lock with wait-time logging.
- Ontology download networking now streams directly from HTTPX via `io.network.download_stream`, removing the legacy `StreamingDownloader` and `pooch` dependency. Conditional requests honour ETag/Last-Modified validators, cached artefacts short-circuit with `status="cached"`, and tests exercise progress telemetry using `httpx.MockTransport`.
- Content download outcomes now leave `reason` unset for fresh downloads, retain `conditional_not_modified` for genuine 304s, and tag voluntary skips with the dedicated `skip_large_download` code.
- HTTP range resume is hard-disabled; resolver hints prefixed with `resume_` are stripped and telemetry annotates ignored requests via `resume_disabled=true`.
- Manifest warm-up defaults to a lazy `ManifestUrlIndex`, with `--warm-manifest-cache` offered solely for small datasets.
- Raised the documented Python interpreter minimum to 3.13 to align with packaging metadata and deployment tooling.
- Content download networking now consolidates on a shared HTTPX client wrapped in Hishel caching (`DocsToKG.ContentDownload.httpx_transport`), removing the legacy `ThreadLocalSessionFactory`/`create_session` helpers. Transport overrides use `configure_http_client()` and `purge_http_cache()`, Tenacity sleeps are patched via `DocsToKG.ContentDownload.networking.time.sleep`, and all retries operate on `httpx.Response` semantics.
- `_validate_cached_artifact` short-circuits on matching size/mtime and only recomputes digests when `verify_cache_digest=True`, backed by an in-process LRU cache.

### Documentation
- Migration notes outlining the removal of deprecated content download shims,
  new `ApiResolverBase` guidance for resolver authors, staging directory usage,
  and updated CLI flag documentation.
- Updated content download runbooks to describe the HTTPX/Hishel transport, cache directory management (`purge_http_cache()`), telemetry fields emitted by HTTPX event hooks, and the deprecation of `create_session` / `ThreadLocalSessionFactory`.
- CHANGELOG entry detailing the content download robustness refactor for downstream consumers.
- Updated content download architecture, review, and API reference documents to
  describe the resolver modularisation, `DownloadRun` orchestration, and
  strategy pattern.
- Operations runbook and configuration reference now call out planner URL
  validation, polite probe telemetry, ontology index locking, and the
  `defaults.planner.probing_enabled` opt-out flag.
- Migration and observability guides document the new reason code taxonomy, resume deprecation, lazy manifest warm-up behaviour, and updated dashboard expectations (`skip_large_download`, `resume_disabled`).

### Changed
- Refactored `DocsToKG.OntologyDownload` into modular submodules (`config`, `io_safe`,
  `net`, `pipeline`, `storage`, `validation_core`, `plugins`) while keeping the
  public facade stable for downstream consumers.
- Split resolver implementations into `ContentDownload/resolvers/` with
  re-exports in `pipeline.py`, introduced the `DownloadRun` orchestration class,
  and adopted strategy-based download processing for PDF, HTML, and XML
  artifacts. Legacy CLI imports (`_build_download_outcome`, resolver classes)
  remain available for compatibility.

## [0.2.0] - 2025-02-15

### Breaking Changes
- Removed the legacy ``DocsToKG.ContentDownload.resolvers.time`` / ``requests``
  exports and the cached ``_fetch_*`` helper functions; import ``time``,
  ``requests``, and call the shared network helpers directly when upgrading.
- Removed the legacy ``DocsToKG.OntologyDownload.(core|config|download|storage|validators|utils|...)`` module aliases in favour of the public ``DocsToKG.OntologyDownload`` facade and direct ``.ontology_download`` / ``.cli`` imports. Update imports before upgrading.
- Planner metadata enrichment now enforces download host allowlists during planning; configurations that previously logged warnings will now raise ``ConfigError`` when a resolver returns a disallowed host.

### Added
- Regression coverage ensuring validator results are identical between sequential
  and concurrent execution and that streaming normalization flushes chunked
  output for large ontologies.
- Integration test for the ``ontofetch doctor`` command covering filesystem,
  dependency, network, and manifest diagnostics.
- Documentation examples showing how to register custom resolver and validator
  plugins via ``docstokg.ontofetch`` entry points.

### Changed
- Validator configuration docs now highlight the ``max_concurrent_validators``
  limit and the operations runbook covers tuning streaming thresholds and
  validator concurrency for large datasets.
- Migration guide elaborates on removing legacy module aliases and explains how
  to apply ``_migrate_manifest_inplace`` for future schema upgrades.

### Changed
- Validator configuration docs now highlight the ``max_concurrent_validators``
  limit and the operations runbook covers tuning streaming thresholds and
  validator concurrency for large datasets.
- Migration guide elaborates on removing legacy module aliases and explains how
  to apply ``_migrate_manifest_inplace`` for future schema upgrades.
- Zenodo and Figshare resolvers integrated into the modular content download pipeline with defensive error handling and metadata-rich events.
- Unit tests covering shared rate-limit parsing, directory size measurement, datetime normalization helpers, and validator failure handling guard the new single-source utilities.
- Bounded intra-work concurrency controls and HEAD pre-check filtering options documented in the README and resolver guides.
- Migration and developer guides covering resolver import paths, configuration, and extensibility patterns.
- Architecture diagram illustrating the resolver pipeline, conditional request helper, and logging flow.
- Content download CLI now exposes `--concurrent-resolvers`, `--head-precheck/--no-head-precheck`,
  and `--accept` flags for resolver coordination and header customisation.
- Download runs emit JSONL summary records and `.metrics.json` sidecar files capturing aggregate metrics.
- Optional global URL deduplication and domain-level throttling controls, including
  `--global-url-dedup` and `--domain-min-interval` CLI flags for ad-hoc runs.
- HybridSearch module migration guide (`docs/hybrid_search_module_migration.md`) documenting old-to-new import paths.

### Changed
- Centralised HTTP retry helper now logs timeout/connection issues separately and emits warnings when retries are exhausted.
- ``ValidationConfig`` introduces a configurable ``max_concurrent_validators`` field (default ``2``) and the validator runner now executes in a bounded thread pool.
- CLI tooling reuses the shared core helpers for rate-limit parsing, directory sizing, and version timestamp inference to avoid divergent implementations.
- Conditional request helper surfaces detailed error messages when cache metadata is incomplete.
- All resolver providers emit structured error events for HTTP, timeout, connection, and JSON failures.
- Adapter-level retries were removed from content download sessions to eliminate compounded retry behaviour.
- Per-download HEAD requests were retired in favour of streaming hash computation with corruption heuristics.
- JSONL and CSV attempt loggers now guard writes with locks and implement context managers for safe reuse.
- Crossref resolver performs HTTP calls through the shared retry helper while reusing the standard header cache key utility.
- Resolver namespace documents the deprecation timeline for the legacy ``time`` and ``requests`` aliases ahead of their removal.
- Resolver pipeline enforces optional domain rate limits and skips repeat URLs across works when
  global deduplication is enabled.
- HybridSearch modules consolidated: result shaping lives in `ranking`, FAISS similarity and state
  helpers live in `vectorstore`, service orchestration owns pagination/stats, and the CLI is exposed
  solely via `python -m DocsToKG.HybridSearch.validation`.

### Deprecated
- Convenience re-exports of ``time`` and ``requests`` from
  ``DocsToKG.ContentDownload.resolvers`` emit deprecation warnings. Removal is
  scheduled for the 2025.12 minor release (see
  `openspec/changes/enhance-content-download-reliability/notes/deprecation-removal-plan.md`).
- ``DocsToKG.HybridSearch.operations`` and ``DocsToKG.HybridSearch.tools`` now emit
  deprecation warnings and direct users to the consolidated service/vectorstore
  modules and the unified ``DocsToKG.HybridSearch.validation`` CLI entry point.
- ``DocsToKG.HybridSearch.results``, ``.similarity``, ``.retrieval``, and ``.schema``
  exist as shims that re-export the ranking, vectorstore, service, and storage modules
  respectively. All shims will be removed in DocsToKG v0.6.0 after one release cycle.
  See `docs/hybrid_search_module_migration.md` for migration guidance and timelines.

### Fixed
- Tests cover HEAD pre-check redirects, resolver concurrency error isolation, and configuration validation edge cases.
- Planning metadata enrichment no longer passes bare allowlists to ``validate_url_security`` and correctly rejects URLs outside the configured host policy without raising attribute errors.
- Property-based tests ensure conditional request headers and deduplication utilities behave correctly across arbitrary inputs.
