## ADDED Requirements
### Requirement: Modular API and CLI separation
The DocsToKG OntologyDownload package SHALL expose its public library surface from `DocsToKG.OntologyDownload.api` while hosting CLI parsing, subcommand handlers, table formatting, and manifest utilities in dedicated modules consumed by the command-line entry point.

#### Scenario: CLI main delegates to dedicated CLI module
- **WHEN** `DocsToKG.OntologyDownload.cli_main` executes
- **THEN** it SHALL obtain its parser and subcommand handlers from `DocsToKG.OntologyDownload.cli`
- **AND** `DocsToKG.OntologyDownload.api` SHALL remain importable without importing CLI-only dependencies (argparse, tabulation helpers, manifest writers).

#### Scenario: Dedicated modules own formatting and manifest helpers
- **WHEN** CLI code renders tables or writes lockfiles
- **THEN** formatting SHALL be performed by `DocsToKG.OntologyDownload.formatters`
- **AND** lockfile/manifest utilities SHALL live in `DocsToKG.OntologyDownload.manifests`
- **AND** those modules SHALL not import `DocsToKG.OntologyDownload.api` to avoid circular dependencies.

### Requirement: Decompose IO concerns by responsibility
The IO subsystem SHALL separate network/session/rate-limit utilities from filesystem/extraction helpers so that each concern is encapsulated in its own module and can be tested independently.

#### Scenario: Streaming downloader composes specialised helpers
- **WHEN** `DocsToKG.OntologyDownload.io.download_stream` executes
- **THEN** it SHALL use `DocsToKG.OntologyDownload.io.network` for session pooling, retries, and streaming downloads
- **AND** rate limiting SHALL be provided by `DocsToKG.OntologyDownload.io.rate_limit`
- **AND** archive extraction and filename handling SHALL be delegated to `DocsToKG.OntologyDownload.io.filesystem`.

#### Scenario: IO helpers remain independently testable
- **WHEN** unit tests import `io.network`, `io.rate_limit`, or `io.filesystem`
- **THEN** each module SHALL expose its public helpers without importing unrelated submodules
- **AND** tests SHALL be able to patch networking or filesystem behaviours in isolation.

### Requirement: Single source for public exports
The DocsToKG OntologyDownload package SHALL generate `PUBLIC_API_MANIFEST`, `__all__`, and related re-exports from a unified manifest definition to avoid manual duplication between modules.

#### Scenario: Adding a new public symbol updates all exports
- **WHEN** a new symbol is registered in the export manifest definition
- **THEN** `DocsToKG.OntologyDownload.exports.PUBLIC_EXPORTS` SHALL define its qualified path and kind
- **AND** both `DocsToKG.OntologyDownload.PUBLIC_API_MANIFEST` and `DocsToKG.OntologyDownload.__all__` SHALL expose it via generated lists
- **AND** `DocsToKG.OntologyDownload.__getattr__` SHALL lazily resolve the symbol without manual updates to either `api.py` or `__init__.py`.

### Requirement: Thread-safe validator plugin loading
The validator plugin loader SHALL coordinate concurrent access with locking, relying on the shared plugin registry to honour reload requests and prevent races.

#### Scenario: Concurrent plugin load remains consistent
- **WHEN** two workers call `DocsToKG.OntologyDownload.validation.load_validator_plugins` concurrently
- **THEN** the loader SHALL serialise access so plugins load exactly once
- **AND** a call with `reload=True` SHALL refresh the registry and make the updated registry visible to subsequent callers
- **AND** the loader SHALL clear any cached state so future calls delegate to `plugins.ensure_plugins_loaded` with the correct `reload` flag.

### Requirement: Cache default configuration resolution
The system SHALL provide a memoised accessor for `ResolvedConfig.from_defaults()` that reuses a cached instance across subsystems until explicitly invalidated.

#### Scenario: Reusing default configuration avoids repeated disk and env reads
- **WHEN** the CLI, planning helpers, and IO helpers request the default resolved configuration multiple times within one process
- **THEN** they SHALL receive the cached instance
- **AND** invoking `DocsToKG.OntologyDownload.settings.invalidate_default_config_cache()` SHALL cause the next access to construct a fresh configuration using `ResolvedConfig.from_defaults()`.

#### Scenario: Environment overrides trigger cache invalidation
- **WHEN** environment overrides applied via CLI options or config reload require new defaults
- **THEN** the code path applying overrides SHALL call the invalidation helper before requesting the cached configuration
- **AND** subsequent calls SHALL observe updated defaults without reusing stale cached values.
