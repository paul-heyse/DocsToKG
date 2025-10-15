# Implementation Tasks

## 1. Code Quality and Import Structure

### 1.1 Refactor Logging Configuration

- [x] Modify `setup_logging()` function in `logging_config.py` to accept explicit parameters for log directory path, logging level string, retention days integer, and maximum log file size in megabytes
- [x] Remove all imports from orchestration modules within the logging configuration module, specifically eliminating any imports from `core.py` or other downloader components
- [x] Implement environment variable reading as fallback defaults when parameters are not explicitly provided, reading from `ONTOFETCH_LOG_DIR` for directory location
- [x] Update `cli.py` to pass logging configuration values explicitly from parsed configuration or command arguments when initializing logging
- [x] Update `core.py` to pass logging configuration values from resolved configuration when setting up logging for orchestration functions
- [x] Verify that logging module can be imported in isolation without triggering import of other package modules

### 1.2 Deprecate Legacy Configuration Aliases

- [x] Add deprecation warnings module-level for legacy configuration class names including `DefaultsConfiguration`, `LoggingConfig`, and `ValidationConfiguration` in `config.py`
- [x] Implement deprecation warning emission using warnings module that fires exactly once per interpreter session when legacy names are accessed
- [x] Update module `__all__` export list to exclude legacy alias names so they do not appear in public API documentation
- [x] Search all internal package code for uses of legacy names and replace with canonical names including `DefaultsConfig`, `LoggingConfiguration`, and `ValidationConfig`
- [x] Add unit test verifying that importing legacy class name triggers deprecation warning exactly once
- [x] Add unit test verifying that legacy names remain functional despite deprecation to maintain backward compatibility

### 1.3 Centralize Archive Extraction

- [x] Create unified `extract_archive_safe()` function in `download.py` that accepts archive path and destination directory as parameters
- [x] Implement archive format detection logic examining file suffix including consideration of double extensions like `.tar.gz` and `.tar.xz`
- [x] Add dispatch logic routing to appropriate extraction function based on detected format including ZIP for `.zip` and TAR for `.tar`, `.tgz`, `.tar.gz`, `.txz`, `.tar.xz` extensions
- [x] Ensure TAR extraction function includes same security checks as ZIP extraction including path traversal prevention and compression ratio validation
- [x] Implement safe path validation checking for absolute paths, parent directory traversal attempts using `..` segments, and empty path components
- [x] Add compression bomb detection calculating ratio of uncompressed to compressed size and rejecting archives exceeding ten-to-one ratio
- [x] Update all validators currently performing inline ZIP extraction to call centralized `extract_archive_safe()` function
- [x] Remove duplicate extraction implementations from validator modules once centralized function is integrated
- [x] Add error case for unsupported archive formats raising descriptive exception with format name

### 1.4 Standardize Subprocess Worker Execution

- [x] Modify `validator_workers.py` to support execution as module using standard Python module invocation with `python -m` syntax
- [x] Implement proper `if __name__ == "__main__":` guard with command-line argument parsing for worker selection in worker module
- [x] Update `validators.py` subprocess invocation to use module execution pattern building command as list with interpreter path, `-m` flag, full module name, and validator arguments
- [x] Remove all `sys.path` modification code from validator worker module that attempts to add source directories dynamically
- [x] Remove file path imports that locate worker script using `__file__` or relative path resolution
- [x] Verify worker processes inherit correct module search path from parent process without manual path manipulation
- [x] Test worker invocation succeeds in clean virtual environment without development source tree mounted
- [x] Ensure worker can be invoked from any working directory without dependency on relative file paths

### 1.5 Convert Optional Dependency Stubs to Module Types

- [x] Import `types` module at top of `optdeps.py` to access `ModuleType` constructor
- [x] Create helper function accepting module name string and attributes dictionary, returning properly constructed `ModuleType` instance
- [x] Have helper function create new module instance using `ModuleType(name)` constructor
- [x] Have helper function set attributes on module instance by iterating attributes dictionary and using `setattr()`
- [x] Have helper function insert completed module into `sys.modules` dictionary using module name as key
- [x] Update `_PystowFallback` creation to wrap instance in `ModuleType` before storing in module cache
- [x] Update `_StubRDFLib` creation to wrap class in `ModuleType` with `Graph` attribute before caching
- [x] Update `_StubPronto` creation to wrap class in `ModuleType` with `Ontology` attribute before caching
- [x] Update `_StubOwlready2` creation to wrap class in `ModuleType` with `get_ontology` attribute before caching
- [x] Verify that import machinery accepts stub modules without warnings or errors
- [x] Verify that type checkers can resolve imports to stub modules without reporting missing modules

### 1.6 Deduplicate Archive Extraction in Validators

- [x] Search `validators.py` for inline ZIP file opening and extraction code within validator functions
- [x] Identify XBRL validator and any other validators performing direct archive extraction
- [x] Replace inline `zipfile.ZipFile` opening and member extraction with calls to `extract_archive_safe()` from download module
- [x] Ensure extraction destination directories match previous inline extraction behavior for manifest path compatibility
- [x] Remove now-unused imports of `zipfile` module from validators file
- [x] Add tests verifying extracted file paths match expected structure after validator execution
- [x] Verify no duplicate path traversal checks remain in validators after centralization

### 1.7 Centralize MIME Type Alias Mapping

- [x] Define module-level constant `RDF_MIME_ALIASES` in `download.py` as set containing all recognized RDF MIME type strings
- [x] Include primary types `application/rdf+xml`, `text/turtle`, `application/n-triples` in alias set
- [x] Include acceptable variations `application/xml`, `text/xml`, `application/x-turtle`, `text/plain` for RDF formats
- [x] Include additional formats `application/trig`, `application/ld+json` commonly returned by ontology services
- [x] Update `_validate_media_type()` method in `StreamingDownloader` to check actual content type against alias set when expected type is RDF format
- [x] Update CLI output formatting in `cli.py` to use alias set when summarizing downloaded content types
- [x] Update validator format detection logic to reference alias set for identifying parseable RDF formats
- [x] Consider adding secondary mapping for MIME type to format name for consistent labeling across modules

### 1.8 Extract CLI Formatting Utilities

- [x] Create new `cli_utils.py` module in OntologyDownload package directory
- [x] Move `_format_table()` function from `cli.py` to `cli_utils.py` making it public with `format_table` name
- [x] Move `_format_row()` helper function to `cli_utils.py` as private function supporting table formatter
- [x] Keep existing `format_validation_summary()` in `cli_utils.py` since it already exists there
- [x] Create `format_plan_rows()` function accepting list of `PlannedFetch` objects and returning formatted table rows
- [x] Create `format_results_table()` function accepting list of `FetchResult` objects and returning formatted table string
- [x] Update imports in `cli.py` to use formatting functions from `cli_utils` module
- [x] Replace inline table formatting code in `pull` command handler with call to `format_results_table()`
- [x] Replace inline table formatting code in `plan` command handler with call to `format_plan_rows()`
- [x] Add module docstring to `cli_utils.py` describing purpose as CLI output formatting helpers
- [x] Add `__all__` export list to `cli_utils.py` including all public formatting functions

## 2. Reliability and Robustness

### 2.1 Implement Download-Time Resolver Fallback

- [x] Extend `PlannedFetch` dataclass to include `candidates` attribute containing ordered list of alternative resolver fetch plans
- [x] Modify `_resolve_plan_with_fallback()` in `core.py` to populate candidates list with all viable resolver plans during planning phase
- [x] Store resolver name, URL, headers, and media type for each candidate in structured format enabling later retry attempts
- [x] Wrap `download_stream()` call in `fetch_one()` with retry loop that iterates through candidate plans on retryable failures
- [x] Define retryable failures as HTTP 503 service unavailable, HTTP 403 forbidden, network timeouts, and connection errors
- [x] Preserve polite headers and user agent when constructing request for fallback candidate
- [x] Log warning message for each fallback attempt including original resolver failure reason and candidate resolver being attempted
- [x] Record complete fallback chain in manifest including primary attempt and all fallback attempts with their outcomes
- [x] Add `resolver_attempts` field to manifest JSON containing array of dictionaries with resolver name, URL, and result status
- [x] Ensure manifest reflects actual successful resolver used rather than originally planned resolver when fallback occurs
- [x] Test fallback mechanism with mock HTTP server returning 503 for first URL and 200 for second URL
- [x] Verify fallback chain appears correctly in saved manifest after successful fallback

### 2.2 Add Streaming Normalization for Large Ontologies

- [x] Create `normalize_streaming()` function in `validators.py` accepting source file path and optional output file path
- [x] Have function create temporary file using `tempfile.NamedTemporaryFile` for intermediate N-Triples output
- [x] Parse source ontology using rdflib graph and serialize to N-Triples format into temporary file
- [x] Create second temporary file for sorted N-Triples output
- [x] Invoke platform sort command using `subprocess.run()` with input from first temporary and output to second temporary
- [x] Open sorted N-Triples file for reading in binary mode using chunks for memory efficiency
- [x] Initialize SHA-256 hasher using `hashlib.sha256()` for computing canonical hash
- [x] When output path provided, open output file for writing in binary mode alongside hash computation
- [x] Stream through sorted N-Triples reading fixed-size chunks and updating hash with each chunk
- [x] When output path provided, write each chunk to output file in addition to hash computation
- [x] Close all file handles and delete temporary files ensuring cleanup occurs even on exception
- [x] Return computed hexadecimal hash digest as function result
- [x] Modify existing `validate_rdflib()` function to detect large ontologies exceeding configured threshold
- [x] Route large ontologies to streaming normalization path and small ontologies to existing in-memory path
- [x] Add configuration parameter `streaming_normalization_threshold_mb` with default value of two hundred megabytes
- [x] Include fallback to external Python merge sort when platform sort command unavailable for pure-Python execution
- [x] Test determinism by computing hash multiple times from same source and verifying identical results
- [x] Test cross-platform determinism by computing hash on Linux and comparing with hash from same file on macOS

### 2.3 Unify Retry Mechanisms

- [x] Create new `utils.py` module in OntologyDownload package for shared utility functions
- [x] Implement `retry_with_backoff()` function accepting callable, retryable predicate function, maximum attempts integer, backoff base float, and jitter float
- [x] Have retry function iterate from one to maximum attempts executing callable within try block
- [x] Catch all exceptions from callable and check if exception satisfies retryable predicate function
- [x] When exception is not retryable or maximum attempts exhausted, re-raise exception unchanged
- [x] When exception is retryable and attempts remain, calculate sleep duration using exponential backoff formula
- [x] Compute sleep time as backoff base multiplied by two raised to attempt minus one power
- [x] Add random jitter by generating random float between zero and jitter parameter and adding to sleep time
- [x] Sleep for computed duration before next retry attempt
- [x] Return callable result immediately upon successful execution without consuming remaining attempts
- [x] Replace retry logic in resolver `_execute_with_retry()` method with call to unified retry helper
- [x] Replace retry logic in `StreamingDownloader.__call__()` method with call to unified retry helper
- [x] Define retryable predicate for resolver API calls accepting timeout and connection errors but not authentication failures
- [x] Define retryable predicate for download operations accepting connection errors, timeouts, and HTTP 5xx status codes
- [x] Add optional callback parameter to retry function for logging retry attempts with attempt number and error
- [x] Test retry helper with forced exceptions verifying exponential backoff timing and jitter bounds
- [x] Test retry helper with non-retryable exception verifying immediate re-raise without delay

### 2.4 Strengthen Manifest Fingerprint

- [x] Locate fingerprint computation logic in `fetch_one()` function within `core.py`
- [x] Extend `fingerprint_components` list to include `MANIFEST_SCHEMA_VERSION` constant at beginning
- [x] Add sorted target formats by converting `spec.target_formats` to sorted list and joining with comma separator
- [x] Add normalization mode string indicating whether streaming or in-memory normalization was used
- [x] Maintain existing components including ontology ID, resolver name, version, SHA-256 hash, normalized hash, and URL
- [x] Join all components with pipe character as before and compute SHA-256 hash of concatenated string
- [x] Define `MANIFEST_SCHEMA_VERSION` constant as string with current version number at module level
- [x] Emit schema version field in manifest JSON separate from fingerprint to enable version-based parsing
- [x] Test that changing target formats order produces different fingerprint before sorting fix
- [x] Test that changing normalization mode from in-memory to streaming produces different fingerprint
- [x] Test that fingerprint remains stable when components provided in same configuration
- [x] Document fingerprint computation formula in manifest structure documentation

### 2.5 Parallelize Resolver Planning

- [x] Import `ThreadPoolExecutor` and `as_completed` from `concurrent.futures` module in `core.py`
- [x] Modify `plan_all()` function to use thread pool for concurrent execution of planning operations
- [x] Read maximum concurrent plans from configuration with path `defaults.http.concurrent_plans` defaulting to eight workers
- [x] Create thread pool executor with maximum workers set to configured concurrency limit
- [x] Submit `plan_one()` call for each ontology specification as separate future to executor
- [x] Create dictionary mapping future objects to ontology specification for result correlation
- [x] Iterate over completed futures using `as_completed()` to yield results as they finish
- [x] Extract result from each completed future and append to results list
- [x] Handle exceptions from futures by catching and logging without terminating entire batch when continue-on-error enabled
- [x] Close thread pool using context manager to ensure cleanup even on exception
- [x] Maintain per-service token bucket limits within resolver API clients to prevent overwhelming individual services
- [x] Pass service identifier from fetch specification through to resolver so proper token bucket selected
- [x] Configure per-service rate limits with defaults respecting published API rate limits for OLS, BioPortal, LOV
- [x] Test concurrent planning reduces wall-clock time compared to sequential planning for batch of ten ontologies
- [x] Test per-service limits prevent exceeding five concurrent requests to same service even with higher overall concurrency
- [x] Verify ordering of results maintains correspondence with input specification order

## 3. Operational Capabilities

### 3.1 Add CLI Concurrency Controls

- [x] Add `--concurrent-downloads` argument to `pull` command parser accepting positive integer
- [x] Add `--concurrent-plans` argument to `plan` command parser accepting positive integer
- [x] Add `--concurrent-downloads` argument to `plan` command parser for consistency when used with `--dry-run`
- [x] Update argument help text describing flags control maximum simultaneous operations
- [x] Extract concurrent downloads value from parsed arguments in `_handle_pull()` function
- [x] When CLI argument provided, override `config.defaults.http.concurrent_downloads` with argument value before calling orchestration
- [x] Extract concurrent plans value from parsed arguments in `_handle_plan()` function
- [x] When CLI argument provided, create new HTTP configuration section if needed and set concurrent plans limit
- [x] Flow modified configuration to `plan_all()` function ensuring thread pool uses overridden limit
- [x] Validate argument values are positive integers and raise argument error for invalid values
- [x] Add integration test verifying `--concurrent-downloads 3` limits active download threads to three
- [x] Add integration test verifying `--concurrent-plans 5` limits active planning threads to five
- [x] Document flags in CLI help text and user guide with examples of production use cases

### 3.2 Add CLI Host Allowlist Override

- [x] Add `--allowed-hosts` argument to `pull` command parser accepting comma-separated string
- [x] Add `--allowed-hosts` argument to `plan` command parser accepting comma-separated string
- [x] Update argument help text describing flag accepts comma-separated domain list added to allowlist
- [x] Parse comma-separated host string into list by splitting on comma and stripping whitespace from each entry
- [x] Filter empty strings from parsed list after splitting and stripping
- [x] Retrieve existing allowed hosts from configuration or initialize empty list if not configured
- [x] Merge CLI-provided hosts with configuration hosts by converting both to sets and taking union
- [x] Assign merged set back to `config.defaults.http.allowed_hosts` before calling orchestration
- [x] Preserve configuration wildcard prefixes if present in CLI-provided hosts by keeping original string format
- [x] Test merge produces unique list when same host appears in both configuration and CLI argument
- [x] Test wildcard domain in CLI argument works correctly for subdomain matching during download
- [x] Document flag with examples showing temporary allowlist addition for ad-hoc downloads

### 3.3 Expand System Diagnostics Command

- [x] Locate `_doctor_report()` function in `cli.py` and expand checks dictionary
- [x] Add ROBOT tool check using `shutil.which("robot")` to locate robot command in system PATH
- [x] When ROBOT found, execute robot with `--version` flag capturing output to extract version string
- [x] Parse ROBOT version from output using regular expression and include in diagnostics report
- [x] Add disk space check using `shutil.disk_usage()` for ontology directory path
- [x] Calculate free gigabytes by dividing free bytes by one billion and include in report with total space
- [x] Add disk space warning when free space drops below ten gigabytes or ten percent of total whichever is larger
- [x] Add rate limit validation check parsing each configured rate limit string against expected pattern
- [x] Report any invalid rate limit strings in diagnostics with original value and explanation of correct format
- [x] Add network egress check for each resolver service making HEAD request to representative endpoint
- [x] Use short timeout of three seconds for each network check to avoid blocking on unresponsive services
- [x] Record success or failure for each service check including HTTP status code when available
- [x] For OLS check `https://www.ebi.ac.uk/ols4/api/health` endpoint
- [x] For BioPortal check `https://data.bioontology.org` endpoint
- [x] For Bioregistry check `https://bioregistry.io` endpoint
- [x] Report network connectivity status for each service in both JSON and human-readable format
- [x] Update `_print_doctor_report()` to format new checks with clear status indicators and recommendations
- [x] Test doctor command output contains all expected sections with sample configuration
- [x] Test doctor command identifies invalid rate limit pattern and reports it clearly

### 3.4 Implement Version Pruning Command

- [x] Add `prune` subcommand to CLI parser with description about managing ontology version history
- [x] Add `--keep` argument to prune parser accepting positive integer for number of versions to retain
- [x] Add optional `--ids` argument accepting list of ontology identifiers to limit pruning scope
- [x] Add `--dry-run` flag to prune parser for preview mode showing what would be deleted
- [x] Implement `_handle_prune()` function accepting parsed arguments and configuration
- [x] Query storage backend for list of ontology identifiers using `STORAGE.available_ontologies()` if not filtered by IDs
- [x] For each ontology identifier, retrieve available versions using `STORAGE.available_versions()`
- [x] Sort versions by timestamp extracted from version string or manifest creation time
- [x] Identify versions to delete as all except the N most recent where N is keep argument value
- [x] In dry-run mode, print list of versions that would be deleted with file sizes if available
- [x] In normal mode, delete each surplus version by removing version directory and all contained artifacts
- [x] Preserve latest symlink or current version marker ensuring it always points to newest retained version
- [x] Log each deletion including ontology identifier, version string, and freed disk space
- [x] Add safety check refusing to delete when keep value would remove all versions
- [x] Emit summary at end showing total versions deleted and total disk space freed
- [x] Test prune keeps correct number of versions and deletes older ones in correct order
- [x] Test prune dry-run shows expected deletions without actually deleting files
- [x] Test prune with `--ids` argument only affects specified ontologies

### 3.5 Add Planning Introspection Commands

- [x] Add `--since` argument to `plan` command parser accepting date string in YYYY-MM-DD format
- [x] Parse date string into Python datetime object using `datetime.strptime()` with appropriate format string
- [x] Modify planning logic to skip ontologies where last-modified date precedes since date
- [x] Retrieve last-modified information from resolver metadata or HTTP Last-Modified header during planning
- [x] Compare last-modified datetime with since datetime using timezone-aware comparison
- [x] Filter planned fetches removing those with last-modified older than since date before returning results
- [x] Add `plan diff` subcommand to CLI parser for comparing plans
- [x] Implement `_handle_plan_diff()` function loading current plan and previous plan from file
- [x] Define plan file format as JSON containing array of plan objects with URL, version, size, license fields
- [x] Load previous plan file from default location or path specified in `--baseline` argument
- [x] Generate current plan by calling `plan_all()` and converting results to comparable dictionary format
- [x] Compare plans by matching ontology identifiers and detecting changes in URL, version, license, or media type
- [x] Report new ontologies present in current plan but absent from previous plan
- [x] Report removed ontologies present in previous plan but absent from current plan
- [x] Report modified ontologies where any tracked field changed between plans
- [x] Format diff output showing additions with plus prefix, removals with minus prefix, modifications with tilde prefix
- [x] Support JSON output format for diff showing structured changes consumable by automation tools
- [x] Test `--since` filtering excludes ontologies with old timestamps and includes recent ones
- [x] Test plan diff correctly identifies added, removed, and modified ontologies between plans
- [x] Test plan diff JSON output has expected structure for programmatic consumption

## 4. Testing and Validation

### 4.1 Determinism Tests for Canonical Turtle

- [x] Create test fixture directory with complex ontology examples including blank nodes and multiple prefixes
- [x] Generate synthetic ontology with at least one hundred triples using diverse RDF node types
- [x] Include blank nodes in various positions to test sorting stability
- [x] Include multiple namespace prefixes to test prefix handling consistency
- [x] Write test running normalization process five times on same input file
- [x] Compute SHA-256 hash for each normalization output
- [x] Assert all five hashes are identical verifying deterministic output
- [x] Run test on Linux and macOS platforms verifying cross-platform hash consistency
- [x] Store golden hash value for each test fixture in test configuration
- [x] Verify actual hash matches golden value detecting regressions in normalization algorithm
- [x] Add test for streaming normalization producing identical hash as in-memory normalization
- [x] Test edge cases including empty graph, single-triple graph, and graph with only blank nodes

### 4.2 Concurrency Stress Testing via Local HTTP Server

- [x] Create test helper HTTP server using FastAPI or standard library HTTP server
- [x] Implement configurable delay endpoint accepting milliseconds parameter and sleeping before response
- [x] Implement configurable error endpoint accepting status code parameter and returning specified HTTP error
- [x] Implement ETag flip endpoint tracking request count and returning different ETag after threshold
- [x] Implement conditional request handler supporting If-None-Match header returning 304 when ETag matches
- [x] Implement partial content handler supporting Range header returning 206 with correct Content-Range
- [x] Implement Content-Type override endpoint accepting media-type parameter and returning specified type regardless of actual content
- [x] Create integration test spawning local server and configuring downloader to use localhost URLs
- [x] Test scenario where first GET returns 503 and second GET returns 200 verifying retry succeeds
- [x] Test scenario where HEAD returns different Content-Type than GET verifying warning logged
- [x] Test scenario where ETag changes mid-stream verifying download restart behavior
- [x] Test scenario where 304 Not Modified returned verifying cache hit behavior
- [x] Test concurrent downloads from same host verifying token bucket limits requests correctly
- [x] Test concurrent downloads from different hosts verifying no artificial serialization
- [x] Verify JSON output and table output from CLI contain expected fields for each test scenario
- [x] Add timeout protection ensuring test server shuts down cleanly even on test failure

### 4.3 Resolver Contract Tests with Record/Replay

- [x] Choose cassette library for recording HTTP interactions such as pytest-vcr or vcrpy
- [x] Create test fixture directory for storing recorded resolver API responses
- [x] Write contract test for OBO resolver verifying correct URL construction from ontology identifier
- [x] Write contract test for OLS resolver verifying API query includes correct parameters and headers
- [x] Write contract test for BioPortal resolver verifying authorization header included when API key present
- [x] Write contract test for LOV resolver verifying metadata endpoint queried with correct URI parameter
- [x] Write contract test for Ontobee resolver verifying PURL format matches expected pattern
- [x] For each resolver, record minimal API response yielding successful FetchPlan
- [x] Verify plan includes required fields: URL, headers, version, license, media type, service identifier
- [x] Verify polite headers included in request: User-Agent, Accept, X-Request-ID when applicable
- [x] Test failure modes with recorded error responses: missing API key, ontology not found, service unavailable
- [x] Verify resolver raises appropriate exception type with descriptive message for each failure mode
- [x] Test that resolver respects configuration timeouts and rate limits during API interaction
- [x] Add test verifying resolver fallback chain when primary resolver fails
- [x] Verify recorded cassettes scrub sensitive data including API keys and authorization tokens
- [x] Test contract validation runs successfully in CI environment without live network access
