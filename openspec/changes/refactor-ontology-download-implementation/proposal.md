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

## Impact

- **Affected code**: All Python modules under `src/DocsToKG/OntologyDownload/` including `cli.py`, `core.py`, `config.py`, `download.py`, `logging_config.py`, `resolvers.py`, `validators.py`, `validator_workers.py`, and `optdeps.py`

- **Dependencies**: No new external dependencies required. Changes leverage existing standard library modules including `types`, `subprocess`, `threading`, `tempfile`, and `hashlib`

- **Breaking changes**: None. All changes maintain backward compatibility. Legacy configuration aliases emit deprecation warnings but remain functional. CLI additions are new flags that default to existing behavior when omitted

- **Testing requirements**: New unit tests for streaming normalization determinism, archive extraction safety checks, retry helper behavior, and CLI argument parsing. Integration tests using local HTTP server to exercise concurrency limits, resume behavior, and fallback mechanisms. Contract tests for resolver planning with recorded API responses

- **Documentation updates**: Update configuration schema documentation to reflect deprecated aliases. Add CLI help text for new flags and commands. Document streaming normalization threshold configuration. Update troubleshooting guide with enhanced `doctor` command output interpretation

- **Deployment considerations**: Changes are internal refactoring with no schema or API modifications. Existing manifests remain valid. Existing configuration files continue to work with deprecation warnings. Phased rollout possible by testing with subset of ontologies using CLI overrides before updating configuration files
