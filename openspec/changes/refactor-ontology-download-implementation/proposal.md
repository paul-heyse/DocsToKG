# Refactor Ontology Download Implementation

## Why

The ontology download subsystem within `src/DocsToKG/OntologyDownload` requires comprehensive refactoring to eliminate technical debt, improve reliability, and enhance operational capabilities. Current implementation challenges create friction in production environments and increase maintenance burden. Several critical gaps have been identified through operational experience:

The logging initialization creates import cycles by importing configuration constants from orchestration modules, making unit tests brittle and complicating package initialization. Configuration management carries legacy aliases that create confusion and risk accidental drift. Archive extraction logic is scattered across multiple modules with incomplete format support, creating security and maintainability risks.

Subprocess management for validators directly modifies system paths and imports by file path rather than using standard module execution, breaking packaging and IDE tooling. Optional dependency stubs are implemented as objects rather than proper module types, confusing import machinery and type checkers.

Reliability gaps include sequential resolver planning that is network-bound, lack of download-time fallback when primary resolvers fail after planning succeeds, non-streaming normalization that risks memory exhaustion on large ontologies, and inconsistent retry logic scattered across resolvers and downloaders. Content-type validation is insufficient with scattered MIME alias rules.

Operational capabilities are limited by lack of CLI controls for concurrency and host allowlists, minimal system diagnostics, absence of version management commands, and limited planning introspection. These deficiencies require manual intervention and configuration file changes for routine operational tasks.

## What Changes

### Code Quality and Maintainability

- **Break logging import cycle**: Refactor `logging_config.setup_logging()` to accept directory, level, retention, and size parameters directly without importing from orchestration modules. Have CLI and core pass configuration values explicitly. Read environment variables as fallback defaults only.

- **Deprecate legacy configuration aliases**: Mark `DefaultsConfiguration`, `LoggingConfig`, and `ValidationConfiguration` as deprecated using warnings that fire once on import. Update `__all__` to exclude legacy names. Ensure internal code uses canonical names only.

- **Unify archive extraction**: Create single `extract_archive_safe()` function in `download.py` that dispatches on file suffix to handle ZIP, TAR, TGZ, TXZ formats uniformly. Apply consistent traversal checks and compression bomb detection. Remove duplicate extraction logic from validators.

- **Standardize subprocess worker execution**: Modify `validator_workers.py` to be runnable as module using standard Python module invocation. Update `validators.py` to spawn subprocesses using module execution pattern rather than file path imports. Eliminate all `sys.path` modifications.

- **Convert optional dependency stubs to proper modules**: Wrap stub implementations using `types.ModuleType` before inserting into `sys.modules`. Ensure stubs expose expected attributes and behave like real modules for import machinery and type checkers.

- **Centralize MIME type alias mappings**: Define comprehensive `RDF_MIME_ALIASES` constant in `download.py` covering all acceptable MIME type variations for RDF formats. Use this mapping consistently in downloader media type validation, CLI output summaries, and validator format detection.

- **Extract CLI formatting utilities**: Move table and JSON formatting logic from `cli.py` into reusable functions in `cli_utils.py`. Create `format_plan_rows()`, `format_results_table()` helpers alongside existing `format_validation_summary()`.

### Reliability and Robustness Improvements

- **Implement download-time resolver fallback**: Extend orchestration layer to carry ranked list of resolver candidates. When primary download fails with retryable error, automatically attempt next candidate resolver preserving polite headers and user agent. Record fallback attempts in manifest for audit trail.

- **Add streaming normalization for large ontologies**: Implement memory-bounded normalization path that serializes to N-Triples stream, uses external sort for deterministic ordering, and computes SHA-256 while streaming output. Retain fast in-memory path for small ontologies below configurable threshold.

- **Unify retry mechanisms**: Create single retry helper function accepting callable, retryable error predicate, max attempts, backoff base, and jitter parameters. Replace scattered retry implementations in resolvers and downloader with centralized helper ensuring consistent exponential backoff behavior.

- **Strengthen manifest fingerprint**: Extend fingerprint computation to include schema version, sorted target formats, and normalization mode. Ensure any change to these parameters produces different fingerprint for proper cache invalidation.

- **Parallelize resolver planning**: Modify `plan_all()` to use thread pool for concurrent API calls to OLS, BioPortal, LOV, and other resolver services. Implement per-service token buckets to cap concurrent calls respecting provider limits while accelerating wall-clock time for planning multiple ontologies.

### Operational Capability Enhancements

- **Add CLI concurrency controls**: Introduce `--concurrent-downloads` and `--concurrent-plans` flags to pull and plan commands. Flow values into effective configuration before orchestration begins.

- **Add CLI host allowlist override**: Implement `--allowed-hosts` flag accepting comma-separated list of permitted domains. Merge CLI-specified hosts with configuration allowlist at runtime without requiring file edits.

- **Expand system diagnostics command**: Enhance `doctor` command to verify ROBOT tool presence and version, check available disk space, validate rate limit configuration parsing, and test network egress to one representative host per resolver with proper timeouts. Return structured JSON and human-readable text output.

- **Implement version pruning command**: Create `prune` subcommand accepting `--keep N` and optional `--ids` filters. Sort manifests by version or creation timestamp, delete surplus versions and artifacts, preserve latest symlinks.

- **Add planning introspection commands**: Implement `plan --since DATE` to filter plans based on remote metadata timestamps or HTTP Last-Modified headers. Add `plan diff` capability to compare current plan with previously committed plan file showing changes in URL, version, size, and license.

- **Add manifest schema validation**: Define JSON Schema for manifest structure capturing required fields, value types, and relationships. Generate schema from Pydantic models or hand-write comprehensive schema. Add validation on manifest read to detect corruption or version incompatibilities. Emit schema version in every manifest enabling future schema evolution.

## Impact

### Affected Code

All Python modules under `src/DocsToKG/OntologyDownload/` will be modified:

- **Core orchestration**: `core.py` for fallback logic, parallel planning, fingerprint computation
- **Download execution**: `download.py` for archive extraction, MIME validation, streaming helpers
- **Configuration management**: `config.py` for deprecation warnings, validation
- **CLI interface**: `cli.py` and new `cli_utils.py` for formatting, new commands, flags
- **Logging infrastructure**: `logging_config.py` for pure function refactoring
- **Resolver clients**: `resolvers.py` for unified retry, polite headers
- **Validation pipeline**: `validators.py` for streaming normalization, centralized extraction
- **Worker processes**: `validator_workers.py` for module execution pattern
- **Dependency management**: `optdeps.py` for proper module stubs
- **Storage abstraction**: `storage.py` for version enumeration supporting prune command

### Dependencies

No new mandatory external dependencies required. Changes leverage existing standard library capabilities:

- **Standard library additions**: `types.ModuleType` for proper stub modules, `concurrent.futures.ThreadPoolExecutor` for parallel planning, `tempfile` for streaming normalization intermediates, `subprocess` for external sort and module workers
- **Existing dependencies**: Continue using `requests` for HTTP, `rdflib` for RDF processing, `pydantic` for configuration, `pooch` for downloads
- **Optional dependencies**: Testing enhancements may use `pytest-vcr` or `vcrpy` for contract tests, but these are test-only dependencies not required for runtime

### Breaking Changes

None. All changes maintain strict backward compatibility:

- **Configuration compatibility**: Legacy alias names remain functional but emit deprecation warnings on first use, giving operators clear migration path without immediate breakage
- **Manifest compatibility**: Existing manifest files remain valid, new manifests include additional fields that readers can ignore, no manifest version migration required
- **CLI compatibility**: All existing CLI invocations continue working unchanged, new flags and commands are additive with no changes to existing command behavior
- **API compatibility**: Public API surface unchanged, internal refactoring does not affect downstream consumers of OntologyDownload package

### Testing Requirements

Comprehensive testing strategy spanning unit, integration, and contract test levels:

- **Unit tests**: Streaming normalization determinism across multiple runs and platforms, archive extraction security checks for traversal and compression bombs, retry helper backoff timing and jitter bounds, CLI argument parsing and validation, module stub import machinery, fingerprint computation stability
- **Integration tests**: Local HTTP server simulating network conditions like delays, errors, ETag changes, partial content, concurrent download rate limiting, resolver fallback chain execution with mock failures, CLI commands producing expected JSON and table output, version pruning with mock storage backend
- **Contract tests**: Resolver API interactions using recorded cassettes, verifying URL construction, header inclusion, error handling, timeout behavior, authentication flows for each resolver type
- **Performance tests**: Parallel planning wall-clock time improvement over sequential, streaming normalization memory usage remaining bounded for large ontologies, concurrent download throughput under various rate limits
- **Determinism tests**: Canonical Turtle hash stability across runs, platforms, and configurations, golden fixture validation detecting algorithm changes, cross-platform byte-for-byte output comparison

### Performance Benchmarks

Expected performance improvements for key operations:

- **Parallel planning**: Wall-clock time reduction of sixty to seventy-five percent for batches of ten or more ontologies when resolvers have similar latency, actual improvement depends on network conditions and resolver responsiveness
- **Streaming normalization**: Memory usage bounded to approximately one hundred megabytes regardless of ontology size, compared to memory usage proportional to ontology size in current implementation, enabling processing of multi-gigabyte ontologies that currently fail
- **Resolver fallback**: Download success rate improvement of fifteen to twenty-five percent for ontologies where primary resolver experiences transient failures, reducing operational burden of manual retries
- **Concurrent downloads**: Throughput improvement scaling with configured concurrency up to bandwidth limits, typically two to four times faster for batches with configured concurrency of four to eight workers

### Documentation Updates

Documentation changes required across user guides, API references, and operational runbooks:

- **Configuration schema**: Update schema documentation marking deprecated aliases, document new concurrent_plans and streaming_normalization_threshold_mb parameters, clarify allowed_hosts wildcard syntax, document per-service rate_limits configuration structure
- **CLI reference**: Add help text for new flags including --concurrent-downloads, --concurrent-plans, --allowed-hosts, document new commands including prune and plan diff with examples, update doctor command documentation with new diagnostic checks
- **Operational guides**: Document streaming normalization behavior and when it activates, provide guidance on setting concurrency limits based on environment characteristics, explain resolver fallback mechanism and how to monitor fallback frequency, describe version pruning strategy for storage management
- **Troubleshooting guide**: Enhance doctor command interpretation section with new checks, add common issues section for import cycles and packaging problems now resolved, document how to debug resolver fallback chains using manifest audit trail
- **Migration guide**: Provide instructions for updating code using deprecated configuration aliases, explain manifest fingerprint changes and why old manifests remain valid, document how to regenerate test cassettes when resolver APIs change

### Deployment Considerations

Deployment strategy supporting incremental rollout with minimal risk:

- **No schema migration required**: Changes are internal refactoring with no database or manifest schema changes requiring migration, existing manifests remain valid and can be read by updated code
- **No configuration migration required**: Existing configuration files continue working unchanged with deprecation warnings logged but not blocking execution, operators can migrate to canonical names at their convenience
- **Phased rollout possible**: New CLI flags enable testing with subset of ontologies before updating configuration, operators can validate fallback behavior with --allowed-hosts temporary override before adding to configuration permanently
- **No service disruption**: Changes do not require service restart or downtime, updated package can be deployed to running service with graceful reload picking up new code
- **Rollback capability**: Any deployment can be rolled back to previous version without data migration since no schema changes, only need to revert package version
- **Testing in production-like environment**: Recommend testing with production-like ontology set and network conditions in staging environment, particularly validating streaming normalization on large ontologies, parallel planning concurrency limits, and resolver fallback chain behavior before production deployment
