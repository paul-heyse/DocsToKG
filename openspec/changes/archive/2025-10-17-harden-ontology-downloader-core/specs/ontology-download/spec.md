# Ontology Download Specification

## ADDED Requirements

### Requirement: Code Consolidation and Single Source of Truth

The system SHALL consolidate duplicate utility implementations between CLI and core modules into single canonical implementations located in the core module.

#### Scenario: Rate limit parsing consolidation

- **WHEN** the CLI needs to parse a rate limit string (e.g., "4/second")
- **THEN** the CLI SHALL import and use the shared rate limit parsing function from the core module
- **AND** the core module SHALL maintain the single authoritative `_RATE_LIMIT_PATTERN` regex
- **AND** no duplicate rate limit parsing logic SHALL exist in the CLI module

#### Scenario: Directory size measurement consolidation

- **WHEN** the CLI needs to measure directory size for reporting
- **THEN** the CLI SHALL import and use the shared directory size function from the core module
- **AND** no duplicate directory size calculation logic SHALL exist in the CLI module
- **AND** the shared implementation SHALL produce identical results for the same directory

#### Scenario: Datetime parsing consolidation

- **WHEN** the system needs to parse ISO, HTTP header, or version-string timestamps
- **THEN** the system SHALL use consolidated datetime parsing functions from the core module
- **AND** the CLI SHALL not duplicate datetime parsing logic
- **AND** parsing SHALL handle timezone-aware conversions consistently across all use cases

### Requirement: Legacy Import Path Removal

The system SHALL remove all legacy module import aliases to eliminate confusion and enable clear API surface visibility.

#### Scenario: Legacy import alias removal

- **WHEN** the system is initialized
- **THEN** the `_LEGACY_MODULE_MAP` dictionary and its associated installation loop SHALL be removed from `__init__.py`
- **AND** all imports SHALL use the public API from `DocsToKG.OntologyDownload.__all__` or direct module paths (`.ontology_download`, `.cli`)
- **AND** attempting to import from removed legacy paths (e.g., `.core`, `.config`, `.validators`, `.download`, `.storage`) SHALL fail with an `ImportError`

#### Scenario: Migration detection

- **WHEN** a codebase may contain legacy imports
- **THEN** running `rg "from DocsToKG.OntologyDownload\.(core|config|validators|download|storage|optdeps|utils|logging_config|validator_workers|foundation|infrastructure|network|pipeline|settings|validation|cli_utils)" --files-with-matches` SHALL identify all files requiring migration
- **AND** the CHANGELOG SHALL document the breaking change with specific migration instructions

### Requirement: CLI Metadata Enrichment Delegation

The CLI SHALL delegate plan metadata enrichment to the library implementation rather than duplicating HTTP HEAD request logic.

#### Scenario: Plan metadata enrichment by library

- **WHEN** the operator invokes `ontofetch plan` or `ontofetch plan-diff`
- **THEN** the CLI SHALL call `plan_all` from the library
- **AND** the library's `_populate_plan_metadata` function SHALL enrich each `PlannedFetch` with `last_modified`, `etag`, and `content_length` via HTTP HEAD requests
- **AND** the CLI SHALL NOT duplicate this metadata probing logic
- **AND** the CLI SHALL directly serialize the enriched `PlannedFetch.metadata` dictionary

#### Scenario: Removal of duplicate helpers

- **WHEN** the CLI no longer needs separate metadata probing
- **THEN** the `_collect_plan_metadata` and `_extract_response_metadata` helper functions SHALL be removed from the CLI module
- **AND** the CLI SHALL rely solely on library-enriched plans

### Requirement: Storage Backend Delegation for Version Management

The CLI SHALL delegate version management operations to the storage backend rather than duplicating symlink or marker logic.

#### Scenario: Latest version marker update via storage backend

- **WHEN** the `ontofetch prune` command retains the most recent N versions
- **THEN** the CLI SHALL call `STORAGE.set_latest_version(ontology_id, target_path)` to update the "latest" marker
- **AND** the storage backend SHALL handle both symlink creation and fallback text file mechanisms
- **AND** the CLI SHALL NOT implement duplicate symlink management logic
- **AND** the `_update_latest_symlink` function SHALL be removed from the CLI module

### Requirement: Exception Handling Clarity

The system SHALL eliminate duplicate exception handling blocks that obscure control flow and increase maintenance burden.

#### Scenario: Consolidated exception handling in validators

- **WHEN** the `validate_pronto` function encounters an exception during validation
- **THEN** the exception SHALL be caught by a single, well-scoped exception handler
- **AND** the handler SHALL write the error details to the validation JSON file
- **AND** the handler SHALL log a warning with appropriate context
- **AND** the handler SHALL return a `ValidationResult` with `ok=False` and error details
- **AND** there SHALL NOT be multiple consecutive identical `except Exception` blocks

### Requirement: Correct URL Security Validation During Planning

The system SHALL pass the complete HTTP configuration object to URL security validation functions to enable proper host allowlist and IDN safety checks.

#### Scenario: Correct parameter type for URL validation

- **WHEN** the `_populate_plan_metadata` function validates the security of a planned fetch URL
- **THEN** the function SHALL pass `config.defaults.http` (the full `DownloadConfiguration` object) to `validate_url_security`
- **AND** the function SHALL NOT pass only `config.defaults.http.allowed_hosts` (a list)
- **AND** the `validate_url_security` function SHALL successfully access `http_config.normalized_allowed_hosts()` without raising `AttributeError`

#### Scenario: URL security validation with allowlist

- **WHEN** a configuration specifies a non-empty `allowed_hosts` list
- **AND** the planner generates a URL for a host not in the allowlist
- **THEN** `validate_url_security` SHALL raise a `ConfigError` indicating the host is not permitted
- **AND** the planning operation SHALL fail cleanly with an actionable error message

### Requirement: Concurrent Validator Execution with Bounded Parallelism

The system SHALL execute validators concurrently using a thread pool with configurable maximum workers to reduce end-to-end validation latency while preserving safety and result integrity.

#### Scenario: Concurrent validator configuration

- **WHEN** the `ValidationConfig` is initialized
- **THEN** the configuration SHALL include a `max_concurrent_validators` field with a default value of 2
- **AND** the field SHALL accept integer values between 1 and 8 inclusive
- **AND** the field SHALL be user-configurable via the configuration file or environment overrides

#### Scenario: Concurrent validator execution

- **WHEN** the `run_validators` function is called with multiple validation requests
- **THEN** the function SHALL create a `ThreadPoolExecutor` with `max_workers` set to the configured `max_concurrent_validators`
- **AND** the function SHALL submit all validation requests to the executor
- **AND** the function SHALL collect results using `as_completed` to process validators as they finish
- **AND** the function SHALL return a dictionary mapping validator names to `ValidationResult` objects with identical structure to the sequential implementation

#### Scenario: Per-validator artifact preservation

- **WHEN** validators execute concurrently
- **THEN** each validator SHALL write its JSON artifact to a unique file path in the `validation_dir`
- **AND** the logging infrastructure SHALL correctly attribute log messages to the appropriate validator
- **AND** no race conditions SHALL corrupt validator output files or logs

#### Scenario: Exception handling in concurrent validators

- **WHEN** a validator raises an exception during concurrent execution
- **THEN** the exception SHALL be caught within the worker function
- **AND** the validator SHALL record the error in its JSON artifact
- **AND** the validator SHALL return a `ValidationResult` with `ok=False`
- **AND** the exception SHALL NOT prevent other validators from executing

### Requirement: Inter-Process Version Locking

The system SHALL prevent concurrent writes to the same ontology version directory using platform-appropriate file locking mechanisms.

#### Scenario: Version lock acquisition

- **WHEN** a download operation begins for a specific `ontology_id` and `version`
- **THEN** the system SHALL create a lock file at `CACHE_DIR/locks/{safe_id}__{safe_version}.lock`
- **AND** the system SHALL acquire an exclusive file lock using `fcntl.flock` on Unix/Linux or `msvcrt.locking` on Windows
- **AND** the lock acquisition SHALL block until any concurrent process releases the lock

#### Scenario: Lock protection during download

- **WHEN** a process holds a version lock
- **THEN** the process SHALL perform the complete download, extraction, normalization, and validation workflow
- **AND** no other process SHALL be able to write to the same version directory concurrently
- **AND** the lock SHALL remain held until the entire operation completes or fails

#### Scenario: Lock release on completion

- **WHEN** a download operation completes successfully or encounters an error
- **THEN** the system SHALL release the file lock in a `finally` block to guarantee cleanup
- **AND** the lock file SHALL persist on disk for observability but occupy negligible space
- **AND** subsequent processes SHALL be able to acquire the lock immediately

#### Scenario: Automatic lock release on process crash

- **WHEN** a process holding a version lock crashes or is terminated
- **THEN** the operating system SHALL automatically release the file lock
- **AND** waiting processes SHALL unblock and proceed with their downloads
- **AND** no manual intervention SHALL be required to recover from a crashed process

### Requirement: True Streaming Normalization for Large Ontologies

The system SHALL implement genuine streaming normalization that avoids materializing the complete triple set in memory, enabling processing of ontologies exceeding available RAM.

#### Scenario: External sort selection for streaming mode

- **WHEN** the streaming normalization function needs to sort RDF triples
- **THEN** the function SHALL first attempt to use the platform's `sort` command via `subprocess.run` with appropriate timeout
- **AND** if the platform `sort` command is unavailable or fails, the function SHALL fall back to Python merge-sort using `heapq.merge` with memory-bounded chunks
- **AND** the sorted output SHALL be written to a temporary file without loading all triples into Python memory

#### Scenario: Streaming normalization algorithm

- **WHEN** the `normalize_streaming` function is called with a source ontology
- **THEN** the function SHALL serialize the RDFLib Graph to an unsorted N-Triples tempfile
- **AND** the function SHALL sort the N-Triples file using external sort
- **AND** the function SHALL stream-read the sorted triples line by line
- **AND** the function SHALL apply deterministic blank node renumbering using regex substitution with a memoization dictionary
- **AND** the function SHALL write sorted `@prefix` lines first, then a blank line, then canonicalized triples
- **AND** the function SHALL compute SHA-256 incrementally as lines are written
- **AND** the function SHALL never materialize the full triple list in Python memory

#### Scenario: Incremental SHA-256 computation

- **WHEN** the streaming normalization writes canonicalized Turtle output
- **THEN** the function SHALL initialize a `hashlib.sha256()` instance at the start
- **AND** the function SHALL update the hasher with each line as it is written
- **AND** the function SHALL return the final hexadecimal digest without re-reading the output file
- **AND** the SHA-256 SHALL match the result of hashing the complete file content after completion

#### Scenario: Deterministic blank node canonicalization

- **WHEN** the streaming normalization encounters blank nodes in sorted triples
- **THEN** the function SHALL apply the existing `_BNODE_PATTERN` regex to identify blank nodes
- **AND** the function SHALL maintain a memoization dictionary mapping original blank node IDs to canonical IDs (`_:b0`, `_:b1`, etc.)
- **AND** the function SHALL assign canonical IDs deterministically based on first encounter order in the sorted stream
- **AND** multiple runs on the same ontology SHALL produce identical blank node renumbering

### Requirement: Streaming Normalization Threshold Configuration

The system SHALL provide a configurable file size threshold to select between in-memory and streaming normalization modes.

#### Scenario: Threshold configuration

- **WHEN** the `ValidationConfig` is initialized
- **THEN** the configuration SHALL include a `streaming_normalization_threshold_mb` field with a default value of 200 (megabytes)
- **AND** the field SHALL accept integer values of at least 1
- **AND** the field SHALL be user-configurable via the configuration file

#### Scenario: Mode selection based on threshold

- **WHEN** the validator determines which normalization mode to use
- **THEN** the validator SHALL check the ontology file size in megabytes
- **AND** if the file size exceeds `streaming_normalization_threshold_mb`, the validator SHALL invoke streaming normalization
- **AND** if the file size is below the threshold, the validator SHALL invoke in-memory normalization
- **AND** the validator SHALL record the selected mode (e.g., `"streaming"` or `"in-memory"`) in the validation output JSON

#### Scenario: Normalization mode logging

- **WHEN** a validator performs normalization
- **THEN** the validator SHALL log the normalization mode used (streaming or in-memory)
- **AND** the log message SHALL include the file size and the configured threshold
- **AND** the operator SHALL be able to correlate normalization mode with performance characteristics

### Requirement: Resolver Plugin Infrastructure

The system SHALL support registration of custom resolvers via Python package entry points with fail-soft loading to enable ecosystem extensibility without compromising robustness.

#### Scenario: Resolver plugin registration

- **WHEN** a Python package defines an entry point in the `docstokg.ontofetch.resolver` group
- **THEN** the system SHALL discover the entry point during module initialization using `importlib.metadata.entry_points()`
- **AND** the system SHALL attempt to load and instantiate the entry point
- **AND** the system SHALL verify that the loaded object has a `plan` method signature matching `(spec, config, logger) -> FetchPlan`
- **AND** the system SHALL register the resolver in the global `RESOLVERS` dictionary using the plugin's `NAME` attribute or the entry point name as the key

#### Scenario: Successful plugin loading

- **WHEN** a well-formed resolver plugin is registered via entry points
- **AND** the plugin loads without exceptions
- **THEN** the system SHALL add the plugin to the `RESOLVERS` dictionary
- **AND** the system SHALL log an info-level message indicating successful plugin registration
- **AND** the plugin SHALL be available for use in ontology fetch specifications

#### Scenario: Fail-soft plugin loading

- **WHEN** a resolver plugin entry point fails to load due to missing dependencies or errors
- **THEN** the system SHALL catch all exceptions during plugin loading
- **AND** the system SHALL log a warning-level message with the plugin name and exception details
- **AND** the system SHALL continue initialization with built-in resolvers
- **AND** the system SHALL NOT crash or prevent other plugins from loading

### Requirement: Validator Plugin Infrastructure

The system SHALL support registration of custom validators via Python package entry points with fail-soft loading to enable specialized validation without modifying core code.

#### Scenario: Validator plugin registration

- **WHEN** a Python package defines an entry point in the `docstokg.ontofetch.validator` group
- **THEN** the system SHALL discover the entry point during module initialization using `importlib.metadata.entry_points()`
- **AND** the system SHALL attempt to load the entry point as a callable
- **AND** the system SHALL verify that the callable signature matches `(ValidationRequest, logging.Logger) -> ValidationResult`
- **AND** the system SHALL register the validator in the global `VALIDATORS` dictionary using the entry point name as the key

#### Scenario: Successful validator plugin loading

- **WHEN** a well-formed validator plugin is registered via entry points
- **AND** the plugin loads without exceptions
- **THEN** the system SHALL add the plugin to the `VALIDATORS` dictionary
- **AND** the system SHALL log an info-level message indicating successful plugin registration
- **AND** the plugin SHALL be invocable via the `run_validators` function and the `ontofetch validate` CLI command

#### Scenario: Fail-soft validator plugin loading

- **WHEN** a validator plugin entry point fails to load due to missing dependencies or errors
- **THEN** the system SHALL catch all exceptions during plugin loading
- **AND** the system SHALL log a warning-level message with the plugin name and exception details
- **AND** the system SHALL continue initialization with built-in validators
- **AND** the system SHALL NOT crash or prevent other plugins from loading

### Requirement: Manifest Schema Forward Compatibility

The system SHALL support reading manifests created with older schema versions by applying idempotent in-place migrations during the read operation.

#### Scenario: Manifest migration shim initialization

- **WHEN** the system defines a `_migrate_manifest_inplace` function
- **THEN** the function SHALL accept a dictionary payload representing a parsed manifest JSON
- **AND** the function SHALL detect the manifest's `schema_version` field or default to an empty string if missing
- **AND** the function SHALL apply appropriate migrations based on the detected version
- **AND** the function SHALL be idempotent (applying the same migration multiple times SHALL produce the same result)

#### Scenario: Current schema version no-op migration

- **WHEN** the `_migrate_manifest_inplace` function receives a manifest with `schema_version` "1.0" or an empty/missing version
- **THEN** the function SHALL set `schema_version` to "1.0" if it was missing
- **AND** the function SHALL make no other modifications to the payload
- **AND** the function SHALL return without further processing

#### Scenario: Future schema version migration

- **WHEN** the `_migrate_manifest_inplace` function receives a manifest with an older schema version (e.g., hypothetical "0.9")
- **THEN** the function SHALL apply the appropriate migration to upgrade to the current schema
- **AND** the migration SHALL add any missing required fields with default values
- **AND** the migration SHALL remove any deprecated fields
- **AND** the migration SHALL rename fields if necessary
- **AND** the migration SHALL update the `schema_version` field to reflect the new version

#### Scenario: Migration invocation during manifest read

- **WHEN** the `_read_manifest` function parses a manifest JSON file
- **THEN** the function SHALL call `_migrate_manifest_inplace(payload)` on the parsed dictionary
- **AND** the migration SHALL occur before schema validation via `validate_manifest_dict`
- **AND** validation SHALL use the current schema version after migration
- **AND** if migration or validation fails, the function SHALL return `None` or raise an appropriate exception

#### Scenario: Unknown schema version handling

- **WHEN** the `_migrate_manifest_inplace` function encounters a manifest with an unrecognized `schema_version`
- **THEN** the function SHALL log a warning indicating the unknown version
- **AND** the function SHALL attempt to validate the manifest with the current schema
- **AND** the function SHALL NOT crash but MAY fail validation if the schema is incompatible

### Requirement: Comprehensive Security Test Coverage

The system SHALL include unit tests verifying security mechanisms including URL allowlisting, IDN safety, archive safety, and protocol enforcement.

#### Scenario: URL allowlist enforcement test

- **WHEN** a test configures an `allowed_hosts` list containing specific domains
- **AND** the test validates a URL to a host not in the list
- **THEN** the `validate_url_security` function SHALL raise a `ConfigError`
- **AND** the error message SHALL indicate the host is not in the allowlist

#### Scenario: IDN punycode safety test

- **WHEN** a test validates a URL containing homograph or confusable Unicode characters in the domain
- **THEN** the `validate_url_security` function SHALL convert the domain to punycode
- **AND** the function SHALL detect punycode conversion issues if present
- **AND** the function SHALL raise a `ConfigError` for unsafe or ambiguous IDN domains

#### Scenario: Archive traversal protection test

- **WHEN** a test extracts a ZIP or TAR archive containing a member with a path like `../../etc/passwd`
- **THEN** the extraction function SHALL detect the path traversal attempt
- **AND** the function SHALL raise a `ConfigError` rejecting the malicious archive
- **AND** no files SHALL be extracted outside the designated destination directory

#### Scenario: Compression bomb protection test

- **WHEN** a test extracts an archive with an excessive compression ratio (e.g., 1 KB compressed → 10 GB uncompressed)
- **THEN** the extraction function SHALL detect the compression bomb before fully decompressing
- **AND** the function SHALL raise a `ConfigError` indicating a compression bomb threat
- **AND** the extraction SHALL be aborted to prevent resource exhaustion

### Requirement: Comprehensive Concurrency and Locking Test Coverage

The system SHALL include tests verifying concurrent validator execution, inter-process locking, and result integrity under concurrent load.

#### Scenario: Concurrent validator execution test

- **WHEN** a test invokes `run_validators` with multiple validation requests
- **AND** `max_concurrent_validators` is set to a value greater than 1
- **THEN** the test SHALL verify that validators execute concurrently (e.g., via timing or execution order analysis)
- **AND** the test SHALL verify that all validators complete successfully
- **AND** the test SHALL verify that the returned result dictionary contains entries for all requested validators
- **AND** the test SHALL verify that each validator's output JSON file is correctly written

#### Scenario: Inter-process locking test

- **WHEN** a test simulates two concurrent processes attempting to download the same ontology version
- **THEN** one process SHALL acquire the lock immediately
- **AND** the second process SHALL block until the first process releases the lock
- **AND** only one process SHALL write to the version directory at a time
- **AND** no file corruption or race conditions SHALL occur

#### Scenario: Lock file cleanup test

- **WHEN** a test acquires and releases a version lock multiple times
- **THEN** the lock file SHALL be created in `CACHE_DIR/locks/`
- **AND** the lock file SHALL persist after lock release for observability
- **AND** the lock file SHALL occupy minimal disk space (typically empty)
- **AND** subsequent lock acquisitions SHALL reuse the same lock file

### Requirement: Comprehensive Streaming Normalization Test Coverage

The system SHALL include tests verifying streaming normalization behavior, determinism, performance, and correctness for various graph sizes.

#### Scenario: Small graph in-memory normalization test

- **WHEN** a test normalizes a small RDF graph (below the streaming threshold)
- **THEN** the system SHALL use in-memory normalization mode
- **AND** the system SHALL record `normalization_mode = "in-memory"` in validation output
- **AND** the output SHALL be a valid Turtle file with sorted prefixes and triples
- **AND** the SHA-256 hash SHALL be deterministic across multiple runs

#### Scenario: Large graph streaming normalization test

- **WHEN** a test normalizes a large RDF graph (exceeding the streaming threshold)
- **THEN** the system SHALL use streaming normalization mode
- **AND** the system SHALL record `normalization_mode = "streaming"` in validation output
- **AND** the output SHALL be a valid Turtle file with sorted prefixes and triples
- **AND** the SHA-256 hash SHALL be deterministic across multiple runs
- **AND** the SHA-256 SHALL match the hash produced by in-memory normalization for the same input

#### Scenario: Blank node canonicalization consistency test

- **WHEN** a test normalizes an ontology containing blank nodes multiple times
- **THEN** blank nodes SHALL receive identical canonical IDs (e.g., `_:b0`, `_:b1`) in the same order across all runs
- **AND** the normalized Turtle output SHALL be byte-for-byte identical across runs
- **AND** the SHA-256 hash SHALL be identical across runs

#### Scenario: External sort fallback test

- **WHEN** a test runs on a platform where the `sort` command is unavailable or disabled
- **THEN** the streaming normalization SHALL fall back to Python merge-sort using `heapq.merge`
- **AND** the fallback implementation SHALL produce identical output to the platform `sort` implementation
- **AND** the SHA-256 hash SHALL be identical regardless of which sort implementation is used

### Requirement: CLI Integration Test Coverage

The system SHALL include end-to-end tests verifying CLI commands function correctly with proper metadata enrichment, configuration, and output formatting.

#### Scenario: CLI plan command test

- **WHEN** an operator runs `ontofetch plan` with a configuration file
- **THEN** the command SHALL output a plan for each ontology in the configuration
- **AND** each plan SHALL include resolver, URL, media type, service, and version information
- **AND** metadata fields (last_modified, etag, content_length) SHALL be enriched via library-level HTTP HEAD requests
- **AND** the output SHALL be formatted as either ASCII table or JSON based on the `--json` flag

#### Scenario: CLI plan-diff command test

- **WHEN** an operator runs `ontofetch plan-diff --baseline path/to/baseline.json`
- **AND** the baseline file contains a previously saved plan output
- **THEN** the command SHALL generate the current plan state
- **AND** the command SHALL compute additions, removals, and modifications relative to the baseline
- **AND** the output SHALL clearly indicate which ontologies are added, removed, or changed
- **AND** the output SHALL be formatted as either human-readable diff or JSON based on the `--json` flag

#### Scenario: CLI prune command test

- **WHEN** an operator runs `ontofetch prune --keep N`
- **THEN** the command SHALL identify ontologies with more than N versions
- **AND** the command SHALL delete surplus versions, retaining the N most recent
- **AND** the command SHALL call `STORAGE.set_latest_version` to update the "latest" marker
- **AND** the command SHALL report reclaimed disk space
- **AND** the command SHALL support `--dry-run` to preview deletions without executing them

#### Scenario: CLI doctor command test

- **WHEN** an operator runs `ontofetch doctor`
- **THEN** the command SHALL report on directory status (existence, writability)
- **AND** the command SHALL report on disk space (total, free, warning threshold)
- **AND** the command SHALL report on optional dependencies (rdflib, pronto, owlready2, arelle)
- **AND** the command SHALL report on ROBOT tool availability and version
- **AND** the command SHALL report on network connectivity to key services (OLS, BioPortal, Bioregistry)
- **AND** the command SHALL report on manifest schema validity and sample manifest validation
- **AND** the output SHALL be formatted as either human-readable text or JSON based on the `--json` flag

### Requirement: HTTP Client Robustness

The system SHALL maintain existing HTTP client resilience including HEAD 405 fallback, ETag caching, resume capability, size limits, and media type validation.

#### Scenario: HEAD 405 fallback

- **WHEN** the system performs a HEAD request to check resource metadata
- **AND** the server responds with HTTP 405 Method Not Allowed
- **THEN** the system SHALL automatically retry the request using GET with streaming enabled
- **AND** the system SHALL extract metadata from the GET response headers
- **AND** the system SHALL proceed with download if metadata is acceptable

#### Scenario: ETag-based cache hit

- **WHEN** the system downloads an ontology that was previously downloaded with an ETag
- **AND** the system includes the ETag in an `If-None-Match` header in the download request
- **AND** the server responds with HTTP 304 Not Modified
- **THEN** the system SHALL skip downloading the content
- **AND** the system SHALL return download status "cached"
- **AND** the system SHALL preserve the existing local file and manifest

#### Scenario: Resume capability with partial files

- **WHEN** a previous download was interrupted and left a `.part` file
- **AND** the system attempts to resume the download
- **THEN** the system SHALL include a `Range` header with the byte offset of the existing `.part` file
- **AND** if the server supports HTTP 206 Partial Content, the system SHALL append the remaining bytes to the `.part` file
- **AND** the system SHALL verify the complete file hash matches the expected value
- **AND** the system SHALL rename the `.part` file to the final filename upon successful completion

#### Scenario: Download size limit enforcement

- **WHEN** the system receives metadata indicating the content length exceeds `max_download_size_gb`
- **THEN** the system SHALL raise a `ConfigError` before downloading the content
- **AND** the error message SHALL clearly state the size limit and actual content size
- **AND** the system SHALL NOT begin streaming the large file

#### Scenario: Media type validation with RDF tolerance

- **WHEN** the system expects an RDF-based media type (e.g., `application/rdf+xml`)
- **AND** the server returns a related but non-exact media type (e.g., `application/xml`)
- **AND** media type validation is enabled in configuration
- **THEN** the system SHALL tolerate common RDF aliases and variants
- **AND** the system SHALL allow the download to proceed if the media type is an acceptable RDF variant
- **AND** the system SHALL reject the download if the media type is completely unrelated (e.g., `text/html`)
