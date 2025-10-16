# Changelog

All notable changes to DocsToKG are documented in this file.

## [Unreleased]

### Breaking Changes
- Removed the legacy ``DocsToKG.OntologyDownload.(core|config|download|storage|validators|utils|...)`` module aliases in favour of the public ``DocsToKG.OntologyDownload`` facade and direct ``.ontology_download`` / ``.cli`` imports. Update imports before upgrading.
- Planner metadata enrichment now enforces download host allowlists during planning; configurations that previously logged warnings will now raise ``ConfigError`` when a resolver returns a disallowed host.

### Added
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

### Deprecated
- Convenience re-exports of ``time`` and ``requests`` from
  ``DocsToKG.ContentDownload.resolvers`` emit deprecation warnings. Removal is
  scheduled for the 2025.12 minor release (see
  `openspec/changes/enhance-content-download-reliability/notes/deprecation-removal-plan.md`).
- ``DocsToKG.HybridSearch.operations`` and ``DocsToKG.HybridSearch.tools`` now emit
  deprecation warnings and direct users to the consolidated service/vectorstore
  modules and the unified ``DocsToKG.HybridSearch.validation`` CLI entry point.

### Fixed
- Tests cover HEAD pre-check redirects, resolver concurrency error isolation, and configuration validation edge cases.
- Planning metadata enrichment no longer passes bare allowlists to ``validate_url_security`` and correctly rejects URLs outside the configured host policy without raising attribute errors.
- Property-based tests ensure conditional request headers and deduplication utilities behave correctly across arbitrary inputs.
## Unreleased

### Added
- Migration guide for the modular resolver architecture, including new
  configuration defaults.
- Developer documentation for adding custom resolver providers and extending the
  registry.
- Property-based tests covering retry backoff, conditional request headers, and
  dedupe behaviour to lift branch coverage.

### Changed
- Resolver namespace now emits deprecation warnings when importing legacy
  ``time`` or ``requests`` aliases.
- ``ResolverPipeline`` captures resolver exceptions consistently across
  sequential and concurrent execution paths.
- README and API docs updated to highlight Zenodo/Figshare resolvers, bounded
  concurrency, and HEAD pre-check filtering.

### Fixed
- HTTP retry helper now validates parameters early and exposes clearer error
  messages for invalid usage.
- Conditional request helper rejects negative content lengths and validates
  response shapes before processing HTTP 304 outcomes.
