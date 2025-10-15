# Implementation Tasks

## 1. Retry Mechanism Centralization

### 1.1 Simplify HTTP Session Configuration

- [ ] Locate the `_make_session` function in `src/DocsToKG/ContentDownload/download_pyalex_pdfs.py`
- [ ] Remove the instantiation and configuration of the `urllib3.util.retry.Retry` object that currently specifies five total retries with a half-second backoff factor
- [ ] Replace the `HTTPAdapter` initialization to use `max_retries=0` instead of passing the `Retry` instance
- [ ] Preserve the adapter mounting behavior for both HTTP and HTTPS protocols
- [ ] Preserve the header update logic that applies polite crawling headers to the session
- [ ] Update the docstring to clarify that retry behavior is now exclusively delegated to the `http.request_with_retries` helper function
- [ ] Ensure that connection pooling capabilities of the adapter remain active despite zero-retry configuration

### 1.2 Verify Retry Determinism

- [ ] Create a test fixture that instantiates a stub HTTP server capable of returning programmed status code sequences
- [ ] Configure the stub server to return the sequence: 429 (first request), 429 (first retry), 200 (second retry)
- [ ] Execute a download attempt against this stub server using the refactored session
- [ ] Instrument the code path to log timestamps and attempt counts for each outbound HTTP request
- [ ] Assert that the total number of requests equals exactly one initial attempt plus the maximum retry count configured in `http.request_with_retries`
- [ ] Verify that no additional attempts occur beyond what the centralized retry helper specifies
- [ ] Confirm that exponential backoff timing follows the formula defined in `http.request_with_retries` without compounding from adapter-level configuration
- [ ] Document the deterministic retry behavior in test comments to establish regression detection

## 2. Network Efficiency Improvements

### 2.1 Remove Redundant HEAD Requests

- [ ] Navigate to the `download_candidate` function in `download_pyalex_pdfs.py`
- [ ] Identify the code block that constructs `head_headers` by copying the main request headers and removing conditional request headers
- [ ] Locate the try-except block that invokes `request_with_retries` with the HEAD method
- [ ] Remove the entire HEAD request execution block including the response processing that extracts `Content-Type` hints
- [ ] Initialize `content_type_hint` to an empty string directly without attempting to populate it from a preliminary request
- [ ] Preserve the GET request logic that follows, ensuring it continues to receive and process the initial response correctly
- [ ] Verify that the content classification logic relying on the `content_type` variable sourced from the GET response remains functional
- [ ] Confirm that the pipeline-level HEAD precheck mechanism in `ResolverPipeline._head_precheck_url` continues to provide preflight filtering when enabled

### 2.2 Validate Network Call Reduction

- [ ] Configure a test environment with HEAD precheck enabled in the resolver pipeline configuration
- [ ] Instrument network traffic monitoring to count total HTTP requests per candidate URL
- [ ] Execute a download workflow against URLs known to return successful GET responses (status 200)
- [ ] Compare the request count before and after removing the redundant HEAD request
- [ ] Assert that successful downloads now issue approximately one fewer request per candidate URL
- [ ] Document scenarios where the pipeline HEAD precheck may still execute, clarifying that this optimization targets the per-download HEAD request specifically

### 2.3 Refactor Crossref Resolver HTTP Calls

- [ ] Open `src/DocsToKG/ContentDownload/resolvers/providers/crossref.py`
- [ ] Import the centralized `request_with_retries` function from `DocsToKG.ContentDownload.http`
- [ ] Locate the branch in `iter_urls` that checks for `hasattr(session, "get")` to determine the live request path
- [ ] Replace the direct `session.get` invocation with a call to `request_with_retries` passing the session, HTTP method string "GET", the endpoint URL, query parameters, headers dictionary, and timeout value
- [ ] Add the `allow_redirects=True` parameter to match the behavior of the standard session GET method
- [ ] Preserve all existing exception handling blocks for `requests.Timeout`, `requests.ConnectionError`, and `requests.RequestException`
- [ ] Retain the cached request path using `_fetch_crossref_data` for backward compatibility with the cache clearing mechanism in `cache.py`
- [ ] Verify that the resolver continues to emit appropriate error events with metadata when requests fail
- [ ] Update any docstrings that reference the request mechanism to reflect the use of the centralized retry helper

### 2.4 Verify Crossref Retry Behavior

- [ ] Create a mock Crossref API endpoint that can be configured to return specific HTTP status codes
- [ ] Configure the mock to return status 429 (Too Many Requests) to trigger retry logic
- [ ] Execute the Crossref resolver against this mock endpoint while instrumenting retry attempts
- [ ] Assert that retry behavior follows the exponential backoff and maximum retry count defined in `request_with_retries`
- [ ] Verify that `Retry-After` headers, if present in the 429 response, are respected according to the centralized helper's configuration
- [ ] Confirm that successful subsequent requests after retries are processed correctly and yield expected resolver results

## 3. Thread Safety and Resource Management

### 3.1 Add Thread-Safe Logging to JsonlLogger

- [ ] Open `src/DocsToKG/ContentDownload/download_pyalex_pdfs.py` and locate the `JsonlLogger` class
- [ ] Import the `threading` module at the top of the file if not already present
- [ ] Add a `self._lock` instance variable in the `__init__` method, initializing it with `threading.Lock()`
- [ ] Locate the `_write` method that serializes payload dictionaries to JSON and appends them to the log file
- [ ] Wrap the file write and flush operations with a context manager using `with self._lock:`
- [ ] Ensure that the JSON serialization occurs before acquiring the lock to minimize lock hold time
- [ ] Verify that the lock is acquired around both the `write` and `flush` calls to guarantee atomic log record emission
- [ ] Confirm that no other methods in the class perform unsynchronized writes to `self._file`

### 3.2 Add Thread-Safe Logging to CsvAttemptLoggerAdapter

- [ ] Locate the `CsvAttemptLoggerAdapter` class in the same file
- [ ] Add a `self._lock` instance variable in the `__init__` method, initializing it with `threading.Lock()`
- [ ] Locate the `log_attempt` method that writes CSV rows
- [ ] Wrap the `writerow` and `flush` operations with a context manager using `with self._lock:`
- [ ] Ensure that row dictionary construction occurs outside the lock to minimize lock hold time
- [ ] Confirm that the underlying `JsonlLogger` instance passed to this adapter has its own lock, providing independent synchronization for JSONL vs CSV streams
- [ ] Verify that no other methods perform unsynchronized writes to `self._file`

### 3.3 Implement Context Manager Protocol

- [ ] Add an `__enter__` method to the `JsonlLogger` class that returns `self`
- [ ] Add an `__exit__` method that accepts three parameters (exception type, exception value, traceback) and calls `self.close()`
- [ ] Ensure the `__exit__` method returns `None` to allow exceptions to propagate normally
- [ ] Update docstrings to indicate that the class now supports context manager protocol for resource-safe usage
- [ ] Consider updating call sites in `main()` to use the context manager pattern, though backward compatibility requires supporting both patterns

### 3.4 Validate Thread Safety Under Concurrency

- [ ] Create a test fixture that instantiates a `JsonlLogger` targeting a temporary file path
- [ ] Spawn sixteen concurrent threads using `concurrent.futures.ThreadPoolExecutor`
- [ ] Configure each thread to log one thousand unique attempt records with distinct work identifiers
- [ ] Allow all threads to complete their logging operations
- [ ] Close the logger and reopen the output file for reading
- [ ] Parse every line as JSON, asserting that no line contains interleaved or corrupted JSON
- [ ] Count the total number of successfully parsed records and assert it equals sixteen thousand
- [ ] Repeat the same test for `CsvAttemptLoggerAdapter`, parsing the output CSV file
- [ ] Assert that the CSV has the correct number of rows and that no rows have interleaved field values

## 4. Streaming Hash Computation and Byte Counting

### 4.1 Initialize Streaming Hash During Write

- [ ] Locate the section in `download_candidate` where the destination path and part file path are determined based on detected content type
- [ ] Immediately after opening the part file handle in write-binary mode, import the `hashlib` module if not already imported
- [ ] Create a new SHA-256 hasher instance by calling `hashlib.sha256()`
- [ ] Initialize a byte counter variable to zero
- [ ] Identify where the initial `sniff_buffer` contents are written to the file handle
- [ ] Immediately after writing the buffer, update the hasher with the buffer contents by calling the hasher's `update` method
- [ ] Increment the byte counter by the length of the buffer
- [ ] Clear the sniff buffer as currently done to free memory

### 4.2 Update Hash During Streaming Write

- [ ] Locate the section where subsequent chunks are written after transitioning to the WRITING state
- [ ] After writing each chunk to the file handle, update the hasher with the chunk bytes
- [ ] Increment the byte counter by the length of the chunk
- [ ] Ensure that these updates occur for every chunk iteration until the response stream is exhausted
- [ ] Verify that no chunks are skipped in the hash computation

### 4.3 Finalize Hash Without Re-Reading

- [ ] Locate the post-download section where the code currently reopens the part file to compute the SHA-256 digest
- [ ] Remove the block that opens the part file in read-binary mode and iterates over chunks to feed into a new hasher
- [ ] Remove the call to `part_path.stat().st_size` since the byte counter already tracks this value
- [ ] Instead, finalize the hash by calling the hasher's `hexdigest()` method to obtain the hex-encoded digest string
- [ ] Assign the finalized digest to the `sha256` variable
- [ ] Assign the byte counter value to the `content_length` variable
- [ ] Add conditional logic to skip hash finalization if the download is in dry-run mode
- [ ] Proceed to rename the part file to the destination file using `os.replace` as currently done
- [ ] Ensure the cleanup logic for removing orphaned part files remains intact

### 4.4 Validate Hash Accuracy

- [ ] Create a test that downloads a file with known contents and a pre-computed SHA-256 hash
- [ ] Execute the download using the refactored streaming hash computation
- [ ] Compare the computed hash against the reference hash, asserting exact equality
- [ ] Perform a secondary verification by re-reading the downloaded file with an independent SHA-256 implementation
- [ ] Assert that both the streaming computation and the independent verification produce identical hashes
- [ ] Measure wall-clock time for large file downloads (several hundred megabytes) before and after the refactoring
- [ ] Document the observed performance improvement from eliminating the second disk read

## 5. Content Validation and Classification

### 5.1 Implement Size-Based Corruption Detection

- [ ] Navigate to the `_build_download_outcome` function in `download_pyalex_pdfs.py`
- [ ] Locate the section where the classification is "pdf" or "pdf_unknown" and the destination path exists
- [ ] Identify the existing check for the PDF EOF marker using `_has_pdf_eof`
- [ ] After confirming EOF marker presence, add a new check that retrieves the file size using `dest_path.stat().st_size`
- [ ] Define a minimum viable PDF size threshold (one kilobyte) below which files are considered corrupt
- [ ] If the file size is below the threshold, override the classification to "pdf_corrupt"
- [ ] Ensure this check only executes when not in dry-run mode and when a destination path exists
- [ ] Document the threshold value choice in comments, noting that legitimate PDFs are rarely smaller than one kilobyte

### 5.2 Implement HTML Content Detection in PDF Files

- [ ] In the same validation section, add a check that reads the last kilobyte of the file using `dest_path.read_bytes()[-1024:]`
- [ ] Convert the tail bytes to lowercase for case-insensitive matching
- [ ] Search for the byte sequence `b"</html"` within the tail bytes
- [ ] If the HTML closing tag is detected, override the classification to "pdf_corrupt"
- [ ] This check identifies cases where HTML error pages or landing pages were served with PDF content type headers
- [ ] Document that this heuristic catches a common server misconfiguration pattern observed in production crawls

### 5.3 Validate Corruption Detection Accuracy

- [ ] Create a test file containing a minimal valid PDF header ("%PDF-1.4") but only a few hundred bytes total
- [ ] Process this file through the validation logic and assert that it is classified as "pdf_corrupt" due to size threshold
- [ ] Create a test file containing HTML content with proper closing tags but saved with a .pdf extension
- [ ] Process this file and assert that it is classified as "pdf_corrupt" due to HTML detection
- [ ] Create a legitimate multi-kilobyte PDF file with a proper EOF marker
- [ ] Process this file and assert that it passes validation with classification "pdf" or "pdf_unknown" as appropriate
- [ ] Document these test cases to prevent regression in corruption detection logic

### 5.4 Create Filename Extension Inference Helper

- [ ] Create a new helper function named `_infer_suffix` that accepts four parameters: URL string, content type string, response object, and default suffix string
- [ ] Within the function, first attempt to extract a filename from the `Content-Disposition` response header
- [ ] Check for the RFC5987 `filename*=` parameter which may include encoding information
- [ ] Parse the encoded filename by splitting on `''` to separate the encoding declaration from the encoded name
- [ ] URL-decode the filename using `urllib.parse.unquote`
- [ ] If `filename*=` is not present, fall back to parsing the standard `filename=` parameter
- [ ] Strip quotes and whitespace from the extracted filename
- [ ] If the content type header starts with "application/pdf", return ".pdf" immediately as the most reliable signal
- [ ] Otherwise, parse the URL path using `urllib.parse.urlparse` and check if it ends with common extensions
- [ ] Check the extracted filename from `Content-Disposition` for common extensions
- [ ] Return the identified extension, or return the default suffix if no reliable indicator is found
- [ ] Handle exceptions gracefully at each parsing step to prevent crashes from malformed headers

### 5.5 Integrate Filename Inference

- [ ] Locate the section in `download_candidate` where the destination path is determined based on detected content type
- [ ] For HTML content, call the `_infer_suffix` helper with the URL, content type, response object, and ".html" as the default
- [ ] Construct the destination path using the artifact's HTML directory, base stem, and inferred suffix
- [ ] For PDF content, call the helper with ".pdf" as the default
- [ ] Construct the destination path using the artifact's PDF directory, base stem, and inferred suffix
- [ ] Verify that the base stem remains deterministic and unchanged, ensuring that only the extension varies based on server headers
- [ ] Test with mock responses that include various `Content-Disposition` formats including RFC5987 encoded filenames
- [ ] Assert that files are saved with the correct extension regardless of URL patterns

### 5.6 Enhance DOI Normalization

- [ ] Open `src/DocsToKG/ContentDownload/utils.py` and locate the `normalize_doi` function
- [ ] Create a lowercase copy of the input DOI string for case-insensitive prefix matching
- [ ] Define a tuple or list of common DOI prefixes including: "<https://doi.org/>", "<http://doi.org/>", "<https://dx.doi.org/>", "<http://dx.doi.org/>", and "doi:"
- [ ] Iterate through the prefix list and check if the lowercase DOI starts with each prefix
- [ ] When a match is found, slice the original DOI string (preserving case) from the prefix length to the end
- [ ] Break out of the loop after the first match since prefixes are mutually exclusive
- [ ] Strip whitespace from the resulting DOI string
- [ ] Return the normalized DOI or None if the result is empty
- [ ] Update docstring to document all supported prefix formats

### 5.7 Validate DOI Normalization Coverage

- [ ] Create a parameterized test that accepts DOI strings with each of the supported prefixes
- [ ] For each prefix variant, append the same DOI identifier (e.g., "10.1234/example")
- [ ] Pass each variant through the normalization function
- [ ] Assert that all variants normalize to the identical canonical form "10.1234/example"
- [ ] Test mixed-case variants to ensure prefix matching is case-insensitive while preserving DOI case
- [ ] Document the resolver hit-rate improvement expected from this enhancement based on production log analysis

## 6. Configuration and CLI Enhancements

### 6.1 Add Concurrent Resolver CLI Flag

- [ ] Locate the argument parser definition in the `main()` function of `download_pyalex_pdfs.py`
- [ ] Add a new command-line argument `--concurrent-resolvers` that accepts an integer type
- [ ] Set the default value to None to allow detection of whether the user explicitly provided this flag
- [ ] Provide a help string explaining that this controls the maximum number of resolvers executed concurrently per work item
- [ ] Document that the default behavior when unspecified is to use the value from `ResolverConfig` which defaults to 1

### 6.2 Add HEAD Precheck CLI Flags

- [ ] Add a new boolean flag `--head-precheck` with `action="store_true"` and `dest="head_precheck"`
- [ ] Add a corresponding negative flag `--no-head-precheck` with `action="store_false"` and `dest="head_precheck"`
- [ ] Use `parser.set_defaults(head_precheck=True)` to establish that HEAD precheck is enabled by default
- [ ] Provide help strings explaining that HEAD precheck issues lightweight preflight requests to filter obvious HTML responses before attempting full downloads

### 6.3 Add Accept Header CLI Flag

- [ ] Add a new string argument `--accept` that accepts a MIME type string
- [ ] Set the default to None to detect explicit user configuration
- [ ] Provide a help string with an example value such as "application/pdf,text/html;q=0.8,*/*;q=0.5"
- [ ] Document that this header is sent with all resolver HTTP requests to indicate client preferences

### 6.4 Wire CLI Flags to Configuration

- [ ] Locate the `load_resolver_config` function in `download_pyalex_pdfs.py`
- [ ] After processing resolver timeout arguments, check if the parsed arguments contain a `concurrent_resolvers` attribute with a non-None value
- [ ] If present, assign the integer value to `config.max_concurrent_resolvers`
- [ ] Check if the `head_precheck` attribute is present and assign its boolean value to `config.enable_head_precheck`
- [ ] After constructing the polite headers dictionary and before assigning it to config, check for the `accept` argument
- [ ] If the accept argument is present and non-None, insert it into the headers dictionary with key "Accept"
- [ ] Verify that the configuration object correctly reflects all CLI overrides

### 6.5 Validate CLI Argument Parsing

- [ ] Create a test that constructs an argument parser using the same configuration as the main function
- [ ] Parse a command line string including `--concurrent-resolvers 4 --no-head-precheck --accept "application/pdf"`
- [ ] Assert that the parsed arguments namespace contains `concurrent_resolvers=4`, `head_precheck=False`, and `accept="application/pdf"`
- [ ] Pass the parsed arguments through the configuration loading function
- [ ] Assert that the resulting `ResolverConfig` object has `max_concurrent_resolvers=4`, `enable_head_precheck=False`, and `polite_headers` containing the Accept header
- [ ] Test the round-trip behavior to ensure flags are correctly propagated to runtime behavior

## 7. Machine-Readable Run Summaries

### 7.1 Emit Summary to Manifest Stream

- [ ] Locate the end of the `main()` function where metrics are collected and the logger is closed
- [ ] After calling `metrics.summary()` to retrieve the metrics dictionary, wrap the subsequent code in a try-except block
- [ ] Within the try block, call `attempt_logger.log_summary(summary)` to write the summary as a JSONL record to the manifest stream
- [ ] Verify that this method exists on both `JsonlLogger` and `CsvAttemptLoggerAdapter` classes (it currently exists on both)
- [ ] In the except block, catch any exception and log a warning message indicating that summary emission failed
- [ ] Use the logger's warning method with `exc_info=True` to include stack trace details in the warning
- [ ] Ensure the logger is closed in a finally block to guarantee resource cleanup regardless of summary emission success

### 7.2 Export Sidecar Metrics JSON

- [ ] After closing the attempt logger, add a new try-except block for metrics export
- [ ] Determine the output path by using the manifest path if it exists in the local scope, otherwise constructing it from `args.out / "manifest.jsonl"`
- [ ] Derive the metrics JSON path by replacing the `.jsonl` suffix with `.metrics.json`
- [ ] Construct a metrics document dictionary containing keys: "processed", "saved", "html_only", "skipped", and "summary"
- [ ] Populate these keys with the respective counter values accumulated during the run and the metrics summary dictionary
- [ ] Serialize the metrics document to JSON with indentation for human readability and sorted keys for deterministic output
- [ ] Write the JSON string to the metrics path using the path's `write_text` method
- [ ] In the except block, catch any exception and log a warning indicating metrics JSON export failure
- [ ] Include exception details in the warning for debugging purposes

### 7.3 Validate Metrics Export

- [ ] Create a test that executes a minimal download job with a small number of works
- [ ] Configure the job to write manifest output to a temporary directory
- [ ] After job completion, verify that a `.metrics.json` file exists alongside the manifest JSONL file
- [ ] Parse the JSON file and assert that it contains the expected keys
- [ ] Compare the counter values in the JSON against values printed to the console output
- [ ] Assert that the resolver summary section contains per-resolver attempt counts and success rates
- [ ] Document the schema of the metrics JSON file for consumers building dashboards or monitoring systems

## 8. Code Organization and Decoupling

### 8.1 Extract Shared Headers Utility

- [ ] Create a new file `src/DocsToKG/ContentDownload/resolvers/providers/headers.py`
- [ ] Import the `Dict` and `Tuple` types from the `typing` module
- [ ] Define a function `headers_cache_key` that accepts a headers dictionary parameter
- [ ] Within the function, iterate through the dictionary items and create tuples of lowercase keys paired with original-case values
- [ ] Sort these tuples to ensure deterministic ordering regardless of dictionary iteration order
- [ ] Return a tuple of the sorted key-value tuples suitable for use as an LRU cache key
- [ ] Add docstring explaining that this utility creates hashable cache keys from HTTP header dictionaries
- [ ] Document that key lowercasing ensures case-insensitive header matching while preserving value case

### 8.2 Update Unpaywall Resolver

- [ ] Open `src/DocsToKG/ContentDownload/resolvers/providers/unpaywall.py`
- [ ] Locate the `_headers_cache_key` function definition
- [ ] Replace the function implementation with an import statement that imports the function from the new `headers` module
- [ ] Alternatively, keep a module-level alias that delegates to the shared implementation for backward compatibility
- [ ] Update the `_fetch_unpaywall_data` function to use the imported utility
- [ ] Verify that the function's public export in `__all__` remains if other modules depend on it

### 8.3 Update Crossref Resolver Imports

- [ ] Open `src/DocsToKG/ContentDownload/resolvers/providers/crossref.py`
- [ ] Locate the import statement that reads `from .unpaywall import _headers_cache_key`
- [ ] Replace this import with `from .headers import headers_cache_key as _headers_cache_key`
- [ ] This eliminates the hidden dependency on Unpaywall resolver internals
- [ ] Verify that the `_fetch_crossref_data` function continues to use the cache key utility correctly
- [ ] Test that cache clearing operations invoked via `cache.py` continue to function

### 8.4 Update Cache Clearing Utilities

- [ ] Open `src/DocsToKG/ContentDownload/resolvers/cache.py`
- [ ] Verify that imports reference the correct resolver provider modules
- [ ] Confirm that `_fetch_crossref_data.cache_clear()` and similar calls continue to work after the refactoring
- [ ] If any imports need adjustment, update them to reference the current location of cached functions
- [ ] Document that the cache clearing interface remains stable for backward compatibility

### 8.5 Validate Utility Decoupling

- [ ] Create a test that imports the `headers_cache_key` function from the new `headers` module
- [ ] Pass a dictionary with mixed-case keys such as `{"User-Agent": "test", "accept": "text/html"}`
- [ ] Assert that the returned tuple contains lowercase keys but preserves the original value casing
- [ ] Pass the same dictionary multiple times and assert that the cache key is deterministic
- [ ] Import the function from the Unpaywall resolver and verify it produces identical output
- [ ] Import the function from the Crossref resolver's internal usage and verify consistency
- [ ] Document that this decoupling allows future resolver implementations to use the utility without circular dependencies

## 9. Legacy Code Deprecation

### 9.1 Document Time and Requests Export Deprecation

- [ ] Open `src/DocsToKG/ContentDownload/resolvers/__init__.py`
- [ ] Locate the `_LEGACY_EXPORTS` dictionary that maps names to the `time` and `requests` modules
- [ ] Verify that the `__getattr__` function currently emits `DeprecationWarning` when these modules are accessed
- [ ] Add a comment block above the `_LEGACY_EXPORTS` definition documenting the deprecation timeline
- [ ] Specify that these exports will be removed in the next minor version release
- [ ] Update the `CHANGELOG.md` file to announce the deprecation of these convenience re-exports
- [ ] Recommend that callers import `time` and `requests` directly from the standard library and PyPI package respectively

### 9.2 Plan Removal of Legacy Exports

- [ ] Create a task or issue in the project tracking system noting that the next minor version bump should remove these exports
- [ ] Document that removal involves: deleting `time` and `requests` from the `__all__` list, removing their entries from `_LEGACY_EXPORTS`, and simplifying the `__getattr__` function
- [ ] Note that the grace period allows downstream consumers to adapt their imports without immediate breakage
- [ ] Recommend searching the codebase for any internal imports that rely on these re-exports and updating them proactively

## 10. Testing and Validation

### 10.1 Concurrency Stress Testing

- [ ] Create a test module for multi-threaded download scenarios
- [ ] Implement a test that spawns multiple worker threads each processing a batch of work items
- [ ] Configure the test to use a shared logger instance with thread-safe locking
- [ ] Verify that all logged records are complete and well-formed after concurrent execution
- [ ] Measure the total execution time and verify that concurrency provides expected parallelism benefits
- [ ] Document any race conditions discovered during testing and their resolutions

### 10.2 Network Behavior Verification

- [ ] Create mock HTTP servers that can be programmed to return specific response sequences
- [ ] Test retry behavior under various failure conditions: timeouts, connection errors, rate limiting
- [ ] Verify that the retry count and timing follow the centralized retry helper configuration
- [ ] Test HEAD precheck behavior with servers that return different content types
- [ ] Validate that redundant network calls have been eliminated where specified
- [ ] Document network behavior expectations for operational runbooks

### 10.3 Performance Benchmarking

- [ ] Establish baseline performance metrics by running the current codebase against a representative dataset
- [ ] Measure: average download time per PDF, total disk I/O operations, network bandwidth utilization, CPU time for hash computation
- [ ] Execute the same workload after implementing the streaming hash computation optimization
- [ ] Compare the measurements and calculate percentage improvements
- [ ] Document performance gains in the pull request description and project documentation
- [ ] Identify any scenarios where performance regressed and investigate root causes

### 10.4 Integration Testing

- [ ] Execute full end-to-end download workflows using the refactored codebase
- [ ] Test with various CLI flag combinations to ensure configuration wiring works correctly
- [ ] Verify that manifest JSONL and metrics JSON files are correctly generated
- [ ] Validate that all resolvers continue to function and produce expected results
- [ ] Test resume functionality to ensure previous manifests are correctly loaded and processed
- [ ] Confirm that dry-run mode continues to provide accurate coverage estimates without writing files
- [ ] Document any integration issues discovered and their resolutions

## 11. Optional Enhancements

### 11.1 Global URL Deduplication (Optional)

- [ ] In `src/DocsToKG/ContentDownload/resolvers/pipeline.py`, add instance variables to the `ResolverPipeline` class
- [ ] Add `self._global_seen_urls` as an empty set to track URLs across all works in the pipeline's lifetime
- [ ] Add `self._global_lock` as a `threading.Lock()` to synchronize access to the global set
- [ ] In the `_process_result` method, before adding the URL to the per-work seen set, acquire the global lock
- [ ] Check if the URL already exists in the global seen set
- [ ] If present, log an attempt record with status "skipped" and reason "duplicate-url-global"
- [ ] Record a skip event in the metrics with the same reason
- [ ] Return None to skip downloading this URL
- [ ] If not present, add the URL to the global set before releasing the lock
- [ ] Document that this feature is opt-in and should only be enabled for broad crawls where URL sharing across works is common
- [ ] Add a configuration flag to enable/disable this feature, defaulting to disabled

### 11.2 Domain-Level Rate Limiting (Optional)

- [ ] Add a new configuration field `domain_min_interval_s` as a dictionary mapping domain names to minimum interval floats
- [ ] In the `ResolverPipeline` class, add `self._last_host_hit` as a `defaultdict(lambda: 0.0)` tracking last request time per host
- [ ] Add `self._host_lock` as a `threading.Lock()` for synchronizing domain-level rate limiting
- [ ] Create a helper method `_respect_domain_limit` that accepts a URL string
- [ ] Parse the URL to extract the network location (hostname) using `urllib.parse.urlsplit`
- [ ] Convert the hostname to lowercase for case-insensitive matching
- [ ] Look up the minimum interval for this domain in the configuration
- [ ] If no interval is configured, return immediately without sleeping
- [ ] Acquire the host lock and calculate the time since the last request to this domain
- [ ] If insufficient time has elapsed, sleep for the remaining duration
- [ ] Update the last request timestamp for this domain before releasing the lock
- [ ] Call this helper in `_process_result` just before invoking the download function
- [ ] Document that this provides per-domain rate limiting independent of per-resolver limits

### 11.3 Validate Optional Features

- [ ] For global URL deduplication, create a test with two work items that reference the same PDF URL
- [ ] Verify that only the first work downloads the PDF and the second logs a "duplicate-url-global" skip event
- [ ] For domain-level rate limiting, configure a minimum interval of 0.5 seconds for a test domain
- [ ] Execute multiple requests to that domain and measure the inter-request timing
- [ ] Assert that each request to the domain is separated by at least 0.5 seconds
- [ ] Document the use cases where these optional features provide value
- [ ] Note that these features should remain disabled by default to preserve backward compatibility

## 12. Documentation Updates

### 12.1 Update Module Docstrings

- [ ] Review each modified Python file and update the module-level docstring to reflect changes
- [ ] For `download_pyalex_pdfs.py`, note the removal of double-retry behavior and addition of streaming hash computation
- [ ] For `utils.py`, document the expanded DOI normalization coverage
- [ ] For `pipeline.py`, note the optional global deduplication and domain rate limiting features if implemented
- [ ] For the new `headers.py` module, provide a complete docstring explaining its purpose and usage

### 12.2 Update Function Docstrings

- [ ] For `_make_session`, clarify that retry logic is delegated to the centralized helper
- [ ] For `download_candidate`, document the removal of redundant HEAD request and streaming hash computation
- [ ] For `normalize_doi`, list all supported prefix formats in the docstring
- [ ] For new helper functions like `_infer_suffix`, provide comprehensive docstrings with parameter descriptions and return value semantics

### 12.3 Update CHANGELOG

- [ ] Add entries for each major change under an "Unreleased" or version-specific section
- [ ] Group changes by category: Performance, Reliability, Configuration, Deprecations
- [ ] Note breaking changes prominently (none expected for this change set)
- [ ] Document new CLI flags and their default behavior
- [ ] Mention the deprecation of `time` and `requests` re-exports with removal timeline

### 12.4 Update User-Facing Documentation

- [ ] If a user guide or README exists for the Content Download component, update it to reflect new CLI options
- [ ] Provide examples of using `--concurrent-resolvers`, `--head-precheck`, and `--accept` flags
- [ ] Document the metrics JSON sidecar file format for users building monitoring dashboards
- [ ] Explain the performance benefits of streaming hash computation for users processing large files
- [ ] Note the improved reliability from centralized retry logic for operators managing production crawls

## 13. Deployment Preparation

### 13.1 Code Review Checklist

- [ ] Verify that all thread safety mechanisms use appropriate locking primitives
- [ ] Confirm that no new race conditions have been introduced
- [ ] Check that all file handles are properly closed in error conditions
- [ ] Review exception handling to ensure failures are logged with sufficient context
- [ ] Validate that backward compatibility is maintained for existing manifest files and configuration formats

### 13.2 Pre-Deployment Testing

- [ ] Execute the full test suite including new tests for all implemented changes
- [ ] Run integration tests against staging infrastructure with production-like data volumes
- [ ] Perform load testing to validate thread safety under high concurrency
- [ ] Verify that metrics export and logging produce expected output formats
- [ ] Test failure scenarios including network outages, disk full conditions, and permission errors

### 13.3 Deployment Plan

- [ ] Document that changes can be deployed incrementally as each PR is merged
- [ ] Note that no configuration migration is required since new features are opt-in
- [ ] Recommend clearing resolver caches after deploying Crossref refactoring to ensure fresh behavior
- [ ] Suggest monitoring error rates and performance metrics closely for the first 24 hours after deployment
- [ ] Prepare rollback plan in case unexpected issues arise in production

### 13.4 Post-Deployment Monitoring

- [ ] Set up dashboards to track key metrics: successful download rate, retry counts, error classifications, performance percentiles
- [ ] Monitor thread pool utilization if concurrent resolvers are enabled
- [ ] Watch for any increase in log file corruption reports after thread safety changes
- [ ] Validate that metrics JSON files are being generated correctly and consumed by monitoring systems
- [ ] Collect feedback from operators on the impact of CLI enhancements and configuration options
