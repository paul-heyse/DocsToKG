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

- [ ] Create new function `normalize_streaming()` in validators module accepting three parameters: source_path pointing to ontology file, optional output_path for saving normalized Turtle if requested, and optional logger for telemetry
- [ ] Implement temporary file creation for intermediate N-Triples output using tempfile.NamedTemporaryFile with delete=False, storing handle for later cleanup, ensuring file persists beyond context manager for external sort access
- [ ] Parse source ontology using rdflib Graph.parse() reading entire ontology into memory initially since rdflib parsing not streaming-capable, accepting this as unavoidable memory consumption for parsing phase
- [ ] Serialize parsed graph to N-Triples format using rdflib Graph.serialize() specifying format="nt" and destination as first temporary file, producing line-oriented output suitable for external sorting
- [ ] Create second temporary file for sorted N-Triples output using similar tempfile pattern, preparing destination for sort command output
- [ ] Construct platform sort command using subprocess with arguments including input file path, output file path specified with -o flag, ensuring lexical byte-wise sorting for deterministic cross-platform results
- [ ] Execute sort command using subprocess.run() with check=True to raise exception on failure, capturing any error output for logging, waiting for completion before proceeding
- [ ] Implement fallback to pure Python sorting if external sort unavailable or fails by reading N-Triples file into memory line by line, sorting using standard sorted() function with appropriate key function, and writing to output file
- [ ] Open sorted N-Triples file for reading in binary mode using chunk-based reading pattern to avoid loading entire sorted file into memory, maintaining streaming approach throughout
- [ ] Initialize SHA-256 hasher using hashlib.sha256() for computing canonical hash while streaming output, ensuring hash is computed over exactly what will be saved or transmitted
- [ ] When output_path provided, open output file for writing in binary mode preparing to save normalized Turtle, keeping file handle open during streaming phase
- [ ] Implement streaming loop reading fixed-size chunks from sorted N-Triples file, typically one megabyte chunks, updating SHA-256 hasher with each chunk, and when output path provided also writing chunk to output file
- [ ] Ensure streaming loop handles end-of-file correctly by reading until read operation returns empty bytes, properly closing all file handles in finally block to ensure cleanup
- [ ] Close both temporary files and delete them using Path.unlink() or os.unlink(), wrapping deletion in try-except to ignore errors if files already deleted, placing cleanup in finally block for guaranteed execution
- [ ] Return computed hexadecimal hash digest as string from function result, providing canonical hash for manifest recording and cache validation
- [ ] Modify existing `validate_rdflib()` function to detect large ontologies by checking source file size using Path.stat().st_size before beginning validation, comparing against configured threshold
- [ ] Add configuration parameter `streaming_normalization_threshold_mb` to ValidationConfig with default value of two hundred megabytes, allowing operators to tune threshold based on available memory
- [ ] Route large ontologies exceeding threshold to streaming normalization path by calling normalize_streaming() instead of in-memory normalization, while small ontologies continue using fast in-memory path
- [ ] Record which normalization path was used in validator result details or manifest metadata, allowing debugging and verification that correct path was chosen based on file size
- [ ] Test determinism by running normalize_streaming on same input file five times and computing SHA-256 each time, asserting all five hashes are identical to verify sort produces consistent output
- [ ] Test cross-platform determinism by running on both Linux and macOS if available in CI environment, comparing hashes to ensure platform sort implementations produce identical results
- [ ] Create synthetic large ontology for testing by generating graph with configurable number of triples, testing streaming path without requiring multi-gigabyte test fixture download
- [ ] Add configuration for sort collation locale if needed for cross-platform consistency, setting LC_ALL=C environment variable during sort execution to ensure byte-wise comparison independent of locale

### 2.3 Unify Retry Mechanisms

- [ ] Create new module `utils.py` in OntologyDownload package for shared utility functions, structuring as standard Python module with function definitions and docstrings
- [ ] Implement `retry_with_backoff()` function accepting five parameters: callable to execute, predicate function for error classification, max_attempts integer defaulting to five, backoff_base float defaulting to half second, and jitter_max float defaulting to tenth of second
- [ ] Design retry function to iterate from attempt one to max_attempts, executing callable within try block and returning result immediately on success without consuming remaining attempts
- [ ] Implement exception handling that catches all exceptions from callable, checks exception against retryable predicate function, and re-raises immediately if predicate returns false or if maximum attempts exhausted
- [ ] Calculate sleep duration for retryable failures using exponential backoff formula multiplying backoff_base by two raised to attempt minus one power, producing increasing delays between attempts
- [ ] Add random jitter to sleep duration by generating random float between zero and jitter_max using random.random() scaled to range, adding to backoff duration to prevent thundering herd synchronization
- [ ] Sleep for computed duration using time.sleep() before next retry attempt, allowing transient failures to resolve and rate limits to reset
- [ ] Consider adding optional callback parameter accepting attempt number and exception, called before sleeping to enable structured logging of retry attempts without coupling retry logic to logging implementation
- [ ] Design callback invocation to occur after exception caught but before sleeping, passing current attempt number and caught exception, allowing caller to log retry with full context
- [ ] Locate `_execute_with_retry()` method in BaseResolver class within resolvers module, examining current retry implementation to understand retry conditions and backoff calculation
- [ ] Replace resolver retry logic with call to unified retry_with_backoff helper, defining retryable predicate that accepts requests.Timeout exceptions, requests.ConnectionError exceptions, but not requests.HTTPError with 401 or 403 status indicating authentication failure
- [ ] Update resolver retry to pass appropriate max_attempts from configuration, backoff_base from configuration, and jitter based on configuration or default values
- [ ] Locate retry logic in `StreamingDownloader.__call__()` method within download module, identifying current retry conditions and backoff implementation
- [ ] Replace download retry logic with call to unified helper, defining retryable predicate that accepts connection errors, timeouts, SSL errors, and HTTP 5xx status codes but not 4xx client errors besides 429 rate limit
- [ ] Ensure retry helper preserves exception details when re-raising after exhausting attempts, maintaining original traceback for debugging while potentially wrapping in context exception if additional information helpful
- [ ] Add optional timeout parameter to retry helper if needed for operations with strict time constraints, checking total elapsed time and breaking retry loop if timeout exceeded even if attempts remain
- [ ] Create unit test that forces callable to raise retryable exception repeatedly, measuring actual sleep durations between attempts and verifying they follow exponential backoff pattern within jitter bounds
- [ ] Create unit test that provides non-retryable exception and verifies retry helper re-raises immediately without sleeping, confirming predicate classification is respected
- [ ] Add test with callable that succeeds on third attempt, verifying helper returns successful result and logs exactly two retry attempts, testing early success case

### 2.4 Strengthen Manifest Fingerprint

- [ ] Locate fingerprint computation logic in `fetch_one()` function within core module, finding where fingerprint_components list is constructed and SHA-256 hash computed
- [ ] Define module-level constant `MANIFEST_SCHEMA_VERSION` at top of core module, setting initial value to string "1" to indicate first explicit schema version, allowing future schema evolution tracking
- [ ] Extend fingerprint_components list construction by inserting MANIFEST_SCHEMA_VERSION constant as first element, ensuring schema version changes produce different fingerprints
- [ ] Add sorted target formats to fingerprint components by taking spec.target_formats sequence, converting to list, sorting alphabetically using sorted() function, joining with comma delimiter, and appending to components list
- [ ] Add normalization mode indicator to fingerprint components by determining whether streaming or in-memory normalization was used, adding string "streaming" or "inmem" to components list
- [ ] Consider how to detect normalization mode used during current run, potentially adding return value to validation functions indicating path taken, or inferring from file size comparison against threshold
- [ ] Maintain existing fingerprint components including ontology ID, resolver name, version string, original SHA-256 hash, normalized SHA-256 hash if available, and secure URL
- [ ] Verify component ordering places schema version first for easy identification, followed by ontology-specific identifiers, then hash values, maintaining logical grouping
- [ ] Join all components with pipe character delimiter as currently done, ensuring each component is converted to string and no component contains pipe character to avoid ambiguity
- [ ] Compute SHA-256 hash of joined components string after encoding to UTF-8 bytes, producing final fingerprint as hexadecimal digest string
- [ ] Add schema_version field to Manifest dataclass and manifest JSON separately from fingerprint, emitting MANIFEST_SCHEMA_VERSION constant value as top-level manifest field
- [ ] Update _write_manifest() function to include schema_version in manifest JSON output, placing near beginning of JSON structure for visibility
- [ ] Update _read_manifest() function to handle manifests both with and without schema_version field, defaulting to implicit version zero for legacy manifests without field
- [ ] Create test providing same ontology specification with target_formats in different order before sorting, computing fingerprints for both, and asserting they are identical due to sorting normalization
- [ ] Create test computing fingerprint with streaming normalization mode then with in-memory mode, verifying fingerprints differ due to mode component change
- [ ] Create test verifying fingerprint remains stable across multiple computation with same inputs, confirming deterministic behavior
- [ ] Document fingerprint computation formula in code comments and in manifest structure documentation, explaining each component purpose and how changes affect fingerprint

### 2.5 Parallelize Resolver Planning

- [ ] Import ThreadPoolExecutor and as_completed from concurrent.futures module at top of core module, adding to existing imports for concurrent execution support
- [ ] Locate `plan_all()` function in core module, identifying current implementation that likely calls plan_one() sequentially for each specification
- [ ] Add configuration parameter `concurrent_plans` to HTTP configuration section in config module, setting default value to eight workers based on typical number of resolvers and desired parallelism
- [ ] Read maximum concurrent plans from configuration using path defaults.http.concurrent_plans, with fallback to default value if not configured
- [ ] Construct thread pool executor using ThreadPoolExecutor context manager with max_workers parameter set to configured concurrency limit, ensuring pool cleanup on exit
- [ ] Submit plan_one() call for each ontology specification as separate future to executor using submit() method, passing spec, config, correlation_id, and logger parameters
- [ ] Create dictionary mapping future objects to original specifications for result correlation, enabling matching of completed futures back to their source specifications
- [ ] Iterate over completed futures using as_completed() function which yields futures as they finish regardless of submission order, allowing results to stream in as available
- [ ] Extract result from each completed future using future.result() method within try block, appending successful results to accumulator list maintaining order correlation
- [ ] Handle exceptions from futures by catching exception from result() call, logging error with specification details, and deciding whether to continue based on continue_on_error configuration
- [ ] Accumulate results in list preserving original specification order by using index or specification key rather than completing order, ensuring deterministic result ordering
- [ ] Close thread pool using context manager ensuring cleanup occurs even if exception raised during iteration, properly terminating worker threads
- [ ] Design per-service token bucket limits within resolver API clients by modifying BaseResolver to check service identifier from specification and acquiring tokens from appropriate bucket
- [ ] Pass service identifier from fetch specification through to resolver plan method, allowing resolver to identify which service bucket to use for rate limiting
- [ ] Create or retrieve token bucket for service using service name as key, initializing with rate limit from configuration if specified or falling back to general per-host limit
- [ ] Configure default per-service rate limits with values respecting published API limits, setting OLS to five concurrent requests per second, BioPortal to two requests per second, and LOV to three requests per second
- [ ] Ensure token bucket acquisition happens within resolver.plan() method before making API call, consuming tokens to enforce rate limit
- [ ] Create integration test with ten sample ontologies using mock HTTP server tracking concurrent request count per endpoint, verifying maximum concurrency per service not exceeded
- [ ] Create test measuring wall-clock time for sequential planning of ten ontologies versus parallel planning, asserting parallel time is significantly less than sequential
- [ ] Verify result ordering matches input specification order by comparing returned list indices against input list, confirming ordering preservation despite concurrent execution

## 3. Operational Capabilities

### 3.1 Add CLI Concurrency Controls

- [ ] Add `--concurrent-downloads` argument definition to pull command parser in `_build_parser()` function, specifying type as positive integer, providing help text explaining flag controls maximum simultaneous download operations
- [ ] Add `--concurrent-plans` argument definition to plan command parser in similar location, using same pattern as concurrent-downloads with appropriate help text for planning operations
- [ ] Consider adding concurrent-downloads flag to plan command as well if plan supports --dry-run mode that performs downloads, maintaining consistency in flag availability
- [ ] Write help text describing that flags override configuration file values for current invocation only without modifying configuration, emphasizing temporary override nature
- [ ] Add validation to argument parser using type parameter or custom validation function ensuring values are positive integers greater than zero, rejecting zero or negative values with clear error message
- [ ] Locate argument extraction in `_handle_pull()` function where parsed args object is processed, adding code to retrieve concurrent_downloads value using getattr() with None default
- [ ] Implement configuration override by checking if CLI argument provided, and when not None, directly modifying config.defaults.http.concurrent_downloads attribute before calling orchestration functions
- [ ] Verify configuration modification happens before config is passed to fetch_all() or plan_all() functions, ensuring override takes effect for current invocation
- [ ] Repeat similar pattern in `_handle_plan()` function for concurrent_plans argument, extracting value and overriding appropriate configuration attribute
- [ ] Consider whether to create new HTTP configuration object or modify existing one, choosing approach that maintains immutability contracts if configuration objects are meant to be immutable
- [ ] Validate that overridden values are used by thread pool executors by tracing config parameter through call chain to ThreadPoolExecutor construction
- [ ] Add integration test invoking CLI with `--concurrent-downloads 3` flag, using mock HTTP server to count active concurrent connections, verifying limit of three is enforced
- [ ] Add integration test invoking CLI with `--concurrent-plans 5` flag, verifying thread pool has exactly five workers by inspecting executor state or measuring actual concurrency
- [ ] Test that missing flag uses configuration default value, confirming fallback behavior works correctly
- [ ] Document new flags in CLI help output by ensuring argparse help text is clear and complete, and consider adding examples to user guide or documentation

### 3.2 Add CLI Host Allowlist Override

- [ ] Add `--allowed-hosts` argument definition to pull command parser accepting string value containing comma-separated list of hostnames or IP addresses
- [ ] Add identical argument to plan command parser maintaining consistency across commands that perform downloads or check download feasibility
- [ ] Write help text explaining flag adds hosts to allowlist for current invocation, accepts comma-separated list, supports wildcard prefixes like "*.example.org", and does not modify configuration file
- [ ] Locate argument extraction in `_handle_pull()` function retrieving allowed_hosts string value from parsed arguments using getattr with None or empty string default
- [ ] Implement parsing of comma-separated host string by splitting on comma character, stripping leading and trailing whitespace from each segment using strip() method, and filtering out empty strings
- [ ] Retrieve existing allowed hosts from configuration by accessing config.defaults.http.allowed_hosts which may be None or list depending on configuration structure
- [ ] Initialize empty list for merging if configuration value is None, or convert to list if different collection type, preparing for merge operation
- [ ] Merge CLI-provided hosts with configuration hosts by converting both to sets using set() constructor, taking union using set union operator, and converting back to list
- [ ] Preserve wildcard prefixes in CLI-provided hosts by not stripping asterisks or dots, maintaining original string format that allowlist matching logic expects
- [ ] Assign merged list back to config.defaults.http.allowed_hosts before orchestration begins, ensuring download validation uses combined allowlist
- [ ] Consider case-sensitivity when merging, converting to lowercase for comparison if allowlist matching is case-insensitive, maintaining consistency with validation logic
- [ ] Implement test providing `--allowed-hosts example.org,test.com` and configuration allowlist containing other.org, verifying effective allowlist contains all three hosts without duplicates
- [ ] Test wildcard domain in CLI argument by providing `--allowed-hosts *.example.org` and attempting download from subdomain.example.org, verifying download is permitted
- [ ] Test that same host in both CLI and configuration results in single entry in merged list, confirming deduplication works correctly
- [ ] Document flag in help text and user guide with examples showing typical use cases like temporarily allowing new resolver host for testing

### 3.3 Expand System Diagnostics Command

- [ ] Locate `_doctor_report()` function in cli module that currently collects diagnostic information, examining current structure of returned dictionary
- [ ] Add ROBOT tool check by calling shutil.which("robot") to search for robot executable in system PATH, recording whether command is found
- [ ] When ROBOT found, execute robot command with --version flag using subprocess.run() capturing stdout, parsing version string from output using regular expression or string split
- [ ] Extract version number from robot output, handling variations in version format, and include in diagnostics report under robot key with found status and version string
- [ ] When ROBOT not found, include in report with found status false and suggestion to install ROBOT for OBO ontology validation capabilities
- [ ] Add disk space check using shutil.disk_usage() function passing ontology directory path from LOCAL_ONTOLOGY_DIR constant
- [ ] Calculate free gigabytes by dividing free_bytes field by 1024 cubed for binary gigabytes or 1000 cubed for decimal gigabytes, choosing appropriate unit for consistency with disk usage conventions
- [ ] Include total space and free space in report, computing percentage free for easier assessment, reporting in both absolute and percentage terms
- [ ] Add disk space warning when free space drops below ten gigabytes absolute threshold, or below ten percent of total capacity, whichever is larger constraint
- [ ] Add rate limit validation check by iterating over all configured rate limits from configuration including per_host_rate_limit and rate_limits dictionary entries
- [ ] Parse each rate limit string against expected pattern using regex or manual parsing, attempting to extract numeric value and time unit
- [ ] Validate time unit is recognized value like "second", "sec", "s", "minute", "min", "m", "hour", or "h", rejecting unrecognized units
- [ ] Report any invalid rate limit strings in diagnostics with original value and explanation of expected format like "5/second" or "10/minute"
- [ ] Add network egress checks for each resolver service by making HEAD request to representative endpoint with short timeout
- [ ] For OLS check, use endpoint <https://www.ebi.ac.uk/ols4/api/health> which provides service health status, recording response status code and latency
- [ ] For BioPortal check, use endpoint <https://data.bioontology.org/> which is main portal URL, recording whether connection succeeds within timeout
- [ ] For Bioregistry check, use endpoint <https://bioregistry.io/> which is service home page, verifying basic connectivity
- [ ] Use short timeout of three seconds for each network check using timeout parameter to requests.head() call, avoiding long blocks on unresponsive services
- [ ] Record success or failure for each service check including HTTP status code when response received, or error type when request fails
- [ ] Handle exceptions from network checks gracefully by catching requests.RequestException, recording failure with exception message in diagnostics
- [ ] Update `_print_doctor_report()` function to format new diagnostic sections with clear status indicators using checkmarks for success and X marks or warning symbols for failures
- [ ] Format ROBOT section showing either "ROBOT: available (version X.Y.Z)" or "ROBOT: not found (install for OBO validation)"
- [ ] Format disk space section showing "Disk free: X.Y GB (ZZ%)" with warning if below threshold
- [ ] Format rate limit section listing each configured limit with validity status, showing invalid limits with error message
- [ ] Format network connectivity section listing each service with status like "OLS: accessible (123ms)" or "BioPortal: unreachable (timeout)"
- [ ] Add test invoking doctor command with mock configuration containing invalid rate limit pattern, verifying error is reported clearly in output
- [ ] Add test simulating low disk space condition and verifying warning appears in doctor output
- [ ] Consider adding recommendations section suggesting actions based on findings like "Install ROBOT for improved validation" or "Free disk space before downloading large ontologies"

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
