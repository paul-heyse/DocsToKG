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

- [ ] Add prune subcommand to main CLI parser in `_build_parser()` function, creating subparser with description about managing ontology version history and controlling storage consumption
- [ ] Add required `--keep` argument to prune parser accepting positive integer specifying number of most recent versions to retain for each ontology
- [ ] Add optional `--ids` argument accepting list of ontology identifiers to filter which ontologies are pruned, with default behavior of pruning all ontologies if not specified
- [ ] Add `--dry-run` flag to prune parser enabling preview mode that shows what would be deleted without actually removing files, supporting safe exploration of pruning impact
- [ ] Implement `_handle_prune()` function in cli module accepting parsed arguments and configuration, serving as entry point for prune command execution
- [ ] Determine scope of ontologies to prune by checking if ids argument provided, and when provided using only those identifiers, otherwise querying STORAGE backend for complete list of stored ontologies
- [ ] For each ontology in scope, retrieve available versions using STORAGE.available_versions() which returns list of version strings sorted chronologically
- [ ] Sort versions by timestamp when version strings are timestamps, or by manifest creation time when version strings are semantic versions or other non-chronological formats
- [ ] Extract version timestamps from version directory manifest files if needed for sorting by reading manifest.json from each version directory and comparing downloaded_at or version field
- [ ] Identify versions to delete by taking all versions except the N most recent where N is keep argument value, selecting older versions as deletion candidates
- [ ] Validate deletion plan ensures at least one version remains by checking if keep count is less than total versions, refusing to prune when keep count would remove all versions
- [ ] In dry-run mode, iterate over versions to delete computing total size by recursively walking version directory and summing file sizes
- [ ] Print dry-run results showing each version that would be deleted, path to version directory, total size in megabytes or gigabytes, and age since download
- [ ] In normal execution mode, delete each surplus version by removing entire version directory using shutil.rmtree() for recursive deletion
- [ ] Log each deletion action recording ontology identifier, version string, deletion timestamp, freed disk space, and reason for deletion
- [ ] Update latest symlink or current version marker after pruning to ensure it points to newest retained version, recreating symlink if it pointed to deleted version
- [ ] Implement safety check preventing deletion of version marked as current or pinned in configuration, preserving versions that are actively referenced
- [ ] Emit summary at completion showing total number of versions deleted across all ontologies, total disk space freed in gigabytes, and any errors encountered
- [ ] Add test with mock storage containing five versions for ontology, invoking prune with keep=2, and verifying three oldest versions are deleted while two newest remain
- [ ] Add test for dry-run mode verifying no files are actually deleted, only preview output is generated showing what would happen
- [ ] Add test with `--ids` filter verifying only specified ontologies are pruned while others remain untouched, confirming selective pruning works correctly

### 3.5 Add Planning Introspection Commands

- [ ] Add `--since` argument to plan command parser accepting string value in YYYY-MM-DD date format, providing help text explaining flag filters plans to ontologies modified since date
- [ ] Implement date parsing using datetime.strptime() with format string "%Y-%m-%d" to convert string to datetime object, wrapping in try-except to catch ValueError for invalid formats
- [ ] Raise argument parsing error when date format is invalid, providing clear message showing expected format and example valid date
- [ ] Make datetime object timezone-aware by calling replace() with tzinfo parameter set to timezone.utc, ensuring proper comparison with timezone-aware timestamps from APIs
- [ ] Modify planning workflow in `_handle_plan()` function to check if since argument provided, and when provided enabling filtering mode for date-based exclusion
- [ ] Pass since datetime to plan_all() or filter planned fetches after planning completes, choosing approach based on whether filtering should occur before or after resolver API calls
- [ ] Retrieve last-modified information for each ontology during planning by examining resolver metadata returned from API calls, looking for version timestamp, release date, or last updated field
- [ ] Check HTTP Last-Modified header during planning phase by performing HEAD request to download URL and extracting Last-Modified header value when available
- [ ] Parse Last-Modified header using email.utils.parsedate_to_datetime() or similar function to convert HTTP date string to datetime object
- [ ] Compare last-modified datetime with since datetime using timezone-aware comparison, filtering out ontologies where last-modified is older than since date
- [ ] Filter planned fetches by removing entries where last-modified predates since cutoff, resulting in plan list containing only recently modified ontologies
- [ ] Add plan diff subcommand to CLI parser creating new subparser under plan command, providing description about comparing plans to identify changes in resolver metadata
- [ ] Implement `_handle_plan_diff()` function accepting parsed arguments, responsible for loading baseline plan and generating current plan for comparison
- [ ] Define plan file format as JSON containing array of plan objects with fields including ontology_id, resolver, url, version, size if available, license, and media_type
- [ ] Add `--baseline` argument to plan diff parser accepting file path to previous plan file, with default location in repository or cache directory for storing committed plans
- [ ] Load previous plan file using json.load() reading file and parsing as list of dictionaries, validating structure contains expected fields
- [ ] Generate current plan by calling plan_all() with specifications from configuration, converting PlannedFetch results to comparable dictionary format matching baseline structure
- [ ] Compare plans by building maps keyed by ontology identifier, iterating through both maps to identify added entries in current but not baseline, removed entries in baseline but not current, and modified entries present in both
- [ ] For modified entries, compare each tracked field including url, version, license, and media_type, recording which fields changed and their old and new values
- [ ] Format diff output for human consumption using clear indicators showing additions with plus prefix, removals with minus prefix, and modifications with tilde or change arrow
- [ ] Show modification details by listing changed fields with format like "version: 2024-01-01 → 2024-02-01" or "url: <https://old> → <https://new>"
- [ ] Implement JSON output mode for diff when --json flag present, emitting structured change representation with added array, removed array, and modified array containing change details
- [ ] Design JSON structure to support programmatic consumption by tools or scripts, using consistent field names and complete information for each change type
- [ ] Add test providing baseline plan with five ontologies and current plan with six ontologies including one new one, verifying diff correctly identifies the addition
- [ ] Add test with ontology present in baseline but absent from current plan, verifying diff reports removal correctly
- [ ] Add test with ontology where version field changed between baseline and current, verifying diff reports modification with old and new values
- [ ] Test --since filtering by providing date and mock resolver returning mixed last-modified timestamps, verifying only ontologies modified after date are included in plan

## 4. Manifest Schema and Validation

### 4.1 Define and Implement Manifest JSON Schema

- [ ] Create schemas directory within DocsToKG/OntologyDownload package or under repository root for storing JSON Schema definitions
- [ ] Design comprehensive JSON Schema for manifest structure defining required fields including id, resolver, url, filename, version, status, sha256, downloaded_at, and target_formats
- [ ] Define field types in schema specifying id as string, sha256 as string matching hexadecimal pattern with sixty-four characters, downloaded_at as string in ISO-8601 datetime format, target_formats as array of strings
- [ ] Add validation constraints to schema including url must start with https:// using pattern property, status must be enum value from set including "success", "cached", "updated", sha256 must match hex pattern "^[0-9a-f]{64}$"
- [ ] Define optional fields in schema including license allowing null or string, normalized_sha256 allowing null or hex string pattern, etag allowing null or string, last_modified allowing null or string
- [ ] Add schema_version field to schema as required string field, defining current version as "1" to enable future schema evolution tracking
- [ ] Include validation object in schema as required dictionary mapping validator names to result objects, where each result has ok boolean and details object
- [ ] Define artifacts array in schema as array of strings representing file paths, allowing empty array but requiring array type
- [ ] Consider generating JSON Schema from Pydantic Manifest model using model.model_json_schema() method, comparing generated schema against hand-written version for completeness
- [ ] Save completed schema to file named manifest.schema.json in schemas directory, formatting with indentation for human readability
- [ ] Add schema_version constant to core module setting value to "1", exporting constant for use in manifest generation and validation
- [ ] Modify manifest generation in fetch_one function to include schema_version field using constant value, ensuring every new manifest records its schema version
- [ ] Import jsonschema library for validation functionality, adding as dependency if not already present, or implementing basic validation without library if avoiding new dependencies
- [ ] Implement validate_manifest_schema function accepting manifest dictionary and optionally schema path, loading schema from file, calling jsonschema.validate or equivalent to check manifest against schema
- [ ] Handle validation errors by catching ValidationError exceptions, extracting useful error message indicating which field failed validation and why, raising ManifestValidationError with descriptive message
- [ ] Modify _read_manifest function to optionally validate loaded manifest against schema, making validation opt-in through configuration flag or environment variable to avoid breaking existing workflows
- [ ] Add manifest schema validation to doctor command diagnostic checks, loading sample manifest if available and validating against schema to verify schema correctness
- [ ] Create unit test providing valid manifest dictionary and asserting schema validation passes, confirming schema accepts correct manifest structure
- [ ] Create unit test providing manifest missing required field and asserting validation raises appropriate error, testing schema enforcement
- [ ] Create unit test providing manifest with incorrect field type like integer for string field and verifying validation detects type mismatch
- [ ] Consider adding schema versioning support detecting schema_version field in manifest and loading appropriate schema version for validation, enabling forward compatibility when schema evolves

## 5. Testing and Validation

### 5.1 Determinism Tests for Canonical Turtle

- [ ] Create test fixture directory under tests directory specifically for normalization testing, organizing fixtures for different graph characteristics and complexity levels
- [ ] Generate synthetic ontology with at least one hundred triples using rdflib graph construction, adding diverse RDF node types including URIs, blank nodes, and literals
- [ ] Include blank nodes in various triple positions as subjects, objects, and in different patterns to test sorting stability when blank node identifiers are unstable
- [ ] Add multiple namespace prefixes to synthetic ontology including common ontology namespaces like rdf, rdfs, owl, dc, skos ensuring prefix handling is tested
- [ ] Include various literal types in test ontology such as plain literals, language-tagged literals with different language codes, and typed literals with datatype URIs
- [ ] Write test function running normalization process five times on same input file using identical configuration and parameters for each run
- [ ] Compute SHA-256 hash for each normalization output by reading normalized file and calculating hash, storing five hash values for comparison
- [ ] Assert all five hashes are identical by comparing each hash to first hash, verifying deterministic output property holds across multiple runs
- [ ] Design test to run on both Linux and macOS platforms when available in CI environment by using pytest markers or conditional execution based on platform detection
- [ ] Compare hashes across platforms by storing expected hash as fixture data, running test on each platform and comparing actual hash to expected, verifying cross-platform consistency
- [ ] Store golden hash value for each test fixture in test configuration file or as constant in test module, providing reference for regression detection
- [ ] Verify actual hash matches golden value by asserting equality, detecting any regressions in normalization algorithm or rdflib serialization behavior
- [ ] Add test for streaming normalization producing identical hash as in-memory normalization by running both paths on same input and comparing resulting hashes
- [ ] Test edge cases including empty graph with zero triples, single-triple graph with minimal structure, and graph with only blank nodes to stress-test corner cases
- [ ] Add test with graph containing Unicode characters in literals and URIs, verifying proper UTF-8 handling and encoding consistency in normalized output
- [ ] Consider adding test with very large graph approaching memory limits to verify streaming path successfully handles near-limit cases
- [ ] Document expected behavior in test docstrings explaining why determinism is critical for cache validation and manifest fingerprinting

### 5.2 Concurrency Stress Testing via Local HTTP Server

- [ ] Choose HTTP server framework for testing, evaluating options including FastAPI for full-featured server with easy routing, or standard library http.server for lightweight implementation without dependencies
- [ ] Create test helper HTTP server class or module providing configurable endpoints for simulating various network conditions and server behaviors
- [ ] Implement delay endpoint accepting milliseconds parameter in URL path or query string, sleeping for specified duration before returning response to simulate slow servers
- [ ] Implement error endpoint accepting HTTP status code parameter, returning specified status like 503, 500, or 403 to test error handling and retry logic
- [ ] Implement ETag flip endpoint tracking request count using thread-safe counter or global variable, returning different ETag value after request count exceeds threshold
- [ ] Implement conditional request handler examining If-None-Match header in request, comparing against current ETag value, returning 304 Not Modified when ETag matches
- [ ] Implement partial content handler examining Range header, parsing byte range request, returning 206 Partial Content with appropriate Content-Range header and requested byte slice
- [ ] Implement Content-Type override endpoint accepting media-type parameter in URL, returning specified Content-Type header value regardless of actual response body content
- [ ] Design server to support multiple concurrent connections allowing testing of concurrent download scenarios and token bucket rate limiting
- [ ] Create integration test spawning local server in separate thread or subprocess before test execution, binding to localhost on available port
- [ ] Configure downloader in test to use localhost URLs pointing to test server endpoints instead of real resolver URLs
- [ ] Test scenario where first GET returns 503 service unavailable and subsequent GET returns 200 success by configuring server endpoint sequence, verifying retry succeeds
- [ ] Test scenario where HEAD returns different Content-Type than GET by configuring server to return application/xml for HEAD and text/turtle for GET, verifying warning is logged but download proceeds
- [ ] Test scenario where ETag changes mid-stream by having server flip ETag after partial content delivered, verifying downloader detects change and handles appropriately
- [ ] Test scenario where server returns 304 Not Modified for conditional request by providing If-None-Match header matching ETag, verifying cache hit behavior
- [ ] Test concurrent downloads from same host by initiating multiple downloads simultaneously, using server to track active connection count, verifying token bucket limits concurrent requests correctly
- [ ] Test concurrent downloads from different hosts by spawning multiple server instances on different ports or using hostname resolution, verifying no artificial serialization occurs
- [ ] Verify JSON output from CLI contains expected fields by capturing CLI output, parsing JSON, and asserting presence of ontology_id, status, sha256, and other required fields
- [ ] Verify table output from CLI contains expected columns by capturing table output, parsing rows, and checking column headers and data alignment
- [ ] Add timeout protection ensuring test server shuts down cleanly even on test failure by using context manager or cleanup fixtures in test framework
- [ ] Clean up server process or thread after test completion using pytest fixtures with finalizers or unittest tearDown methods, ensuring no orphaned processes

### 5.3 Resolver Contract Tests with Record/Replay

- [ ] Choose cassette library for recording HTTP interactions, comparing options including pytest-vcr for pytest integration, vcrpy for general Python recording, or responses library for mocking without recording
- [ ] Install chosen library as test dependency adding to test requirements file or pyproject.toml test extras
- [ ] Create test fixture directory for storing recorded resolver API responses organizing by resolver type like obo, ols, bioportal with subdirectories for each test case
- [ ] Write contract test for OBO resolver verifying URL construction from ontology identifier by calling resolver.plan() with test spec and examining returned FetchPlan.url
- [ ] Verify OBO resolver URL follows expected pattern like <https://purl.obolibrary.org/obo/{id}.owl> using string matching or regex validation
- [ ] Write contract test for OLS resolver verifying API query structure by recording actual API call during test execution and examining request details in cassette
- [ ] Verify OLS resolver includes correct query parameters in API request such as ontology identifier in path or query string matching OLS API specification
- [ ] Write contract test for BioPortal resolver verifying authorization header present when API key configured by setting API key in test fixture and examining request headers
- [ ] Verify BioPortal authorization header follows expected format like "apikey TOKEN" or bearer token format depending on OntoPortal API specification
- [ ] Write contract test for LOV resolver verifying metadata endpoint query with correct URI parameter by examining API request to LOV service
- [ ] Write contract test for Ontobee resolver verifying PURL format matches expected pattern like <https://purl.obolibrary.org/obo/{prefix}.{format}>
- [ ] For each resolver, record minimal API response yielding successful FetchPlan by running test in record mode capturing actual API HTTP traffic
- [ ] Verify plan includes all required fields by asserting FetchPlan.url is not None and is valid URL, headers dictionary is present, version string is populated when available, license field matches expected SPDX identifier, media_type indicates RDF format, and service identifier is present
- [ ] Verify polite headers included in request by examining recorded request headers in cassette, checking for User-Agent header with library identification, Accept header indicating RDF format preference, X-Request-ID header for request correlation
- [ ] Test failure modes with recorded error responses including missing API key scenario recording 401 or 403 response, ontology not found scenario recording 404 response, service unavailable scenario recording 503 response
- [ ] Verify resolver raises appropriate exception type for each failure mode using pytest.raises() context manager, checking exception message contains useful diagnostic information
- [ ] Test resolver respects configuration timeouts by setting short timeout in config and verifying request fails within expected time window
- [ ] Test resolver respects rate limits during API interaction by tracking token bucket state or measuring time between consecutive requests
- [ ] Add test verifying resolver fallback chain when primary resolver fails by providing spec with multiple resolver candidates, forcing first to fail, verifying second is attempted
- [ ] Verify recorded cassettes scrub sensitive data by examining cassette files and confirming API keys, authorization tokens, and any secrets are redacted or filtered
- [ ] Configure cassette library to filter sensitive headers using before_record hook or filter_headers parameter, removing or masking Authorization, X-API-Key, and similar headers
- [ ] Test contract validation runs successfully in CI environment without live network access by executing tests in replay mode, verifying cassettes provide complete recorded responses
- [ ] Add documentation for regenerating cassettes when resolver APIs change by providing instructions to delete cassette files and run tests in record mode
- [ ] Consider adding cassette expiration checking to detect when recorded responses are stale, prompting regeneration when resolver API versions change
