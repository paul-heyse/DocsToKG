# Implementation Tasks

## 1. High-Impact Correctness Fixes

### 1.1 Consolidate CLI Context Management

- [x] 1.1.1 Remove first `with JsonlLogger(manifest_path) as base_logger:` block and incomplete `_session_factory()` docstring
- [x] 1.1.2 Ensure single `contextlib.ExitStack()` manages all context resources (JSONL logger, CSV adapter if enabled)
- [x] 1.1.3 Move `load_previous_manifest()` and `clear_resolver_caches()` calls inside unified context
- [x] 1.1.4 Define `_session_factory()` exactly once with single return statement
- [x] 1.1.5 Remove all references to abandoned first context block
- [x] 1.1.6 Verify CSV adapter enters context via `stack.enter_context()` when `--log-format csv` specified

### 1.2 Fix CSV Resource Leak

- [x] 1.2.1 Modify `CsvAttemptLoggerAdapter.close()` to close CSV file handle under lock
- [x] 1.2.2 Ensure close() checks `self._file.closed` before closing to enable idempotency
- [x] 1.2.3 Update `__exit__()` to safely handle already-closed file
- [x] 1.2.4 Add unit test verifying file descriptor closure after explicit `close()` call
- [x] 1.2.5 Add integration test confirming clean process exit on all platforms including Windows

### 1.3 Remove Duplicate Session Factory

- [x] 1.3.1 Delete unreachable second `return _make_session(config.polite_headers)` statement
- [x] 1.3.2 Ensure `_session_factory()` returns result of `_make_session(config.polite_headers)` exactly once
- [x] 1.3.3 Verify function contains complete docstring describing session configuration

### 1.4 Unify HEAD Precheck Implementation

- [x] 1.4.1 Create `head_precheck(session, url, timeout)` function in `network.py`
- [x] 1.4.2 Implement HEAD request with `request_with_retries` using `max_retries=1` and `min(timeout, 5.0)`
- [x] 1.4.3 Handle exceptions by returning True (conservative pass-through)
- [x] 1.4.4 Check status codes in {200, 302, 304} set for success
- [x] 1.4.5 Reject responses with Content-Type containing "text/html" or Content-Length of "0"
- [x] 1.4.6 Implement optional HEAD-to-GET degradation for 405/501 status codes
- [x] 1.4.7 Replace `_head_precheck_candidate()` in CLI with call to unified function
- [x] 1.4.8 Replace `ResolverPipeline._head_precheck_url()` with call to unified function
- [x] 1.4.9 Preserve per-resolver opt-out via `ResolverConfig.resolver_head_precheck` consultation
- [x] 1.4.10 Add unit tests for 200/pdf, 200/html, 405, 501, timeout, and connection error cases
- [x] 1.4.11 Verify dry-run behavior remains deterministic (no streaming downloads)

## 2. Legacy Code Removal

### 2.1 Remove __getattr__ Compatibility Shims

- [x] 2.1.1 Delete `__getattr__` function exporting `time` and `requests` from `resolvers.py`
- [x] 2.1.2 Remove `time` and `requests` entries from `__all__` export list
- [x] 2.1.3 Search codebase for any remaining `from DocsToKG.ContentDownload.resolvers import time` patterns
- [x] 2.1.4 Replace located imports with direct `import time` or `import requests as _requests`
- [x] 2.1.5 Add CHANGELOG entry documenting breaking change with migration instructions

### 2.2 Delete request_with_retries Proxy

- [x] 2.2.1 Remove `request_with_retries()` proxy function definition from `resolvers.py`
- [x] 2.2.2 Add direct import `from DocsToKG.ContentDownload.network import request_with_retries` at module top
- [x] 2.2.3 Search all resolver implementations for calls to local `request_with_retries`
- [x] 2.2.4 Verify all calls now resolve to network module import
- [x] 2.2.5 Update test mocks/patches to target `network.request_with_retries` instead of `resolvers.request_with_retries`

### 2.3 Eliminate Session-Less Resolver Branches

- [x] 2.3.1 Remove `hasattr(session, "get")` conditional branches from `CrossrefResolver`
- [x] 2.3.2 Remove `hasattr(session, "get")` conditional branches from `UnpaywallResolver`
- [x] 2.3.3 Remove `hasattr(session, "get")` conditional branches from `SemanticScholarResolver`
- [x] 2.3.4 Delete `_fetch_crossref_data()` LRU cache function and `@lru_cache` decorator
- [x] 2.3.5 Delete `_fetch_unpaywall_data()` LRU cache function
- [x] 2.3.6 Delete `_fetch_semantic_scholar_data()` LRU cache function
- [x] 2.3.7 Replace deleted function calls with direct `request_with_retries(session, ...)` invocations
- [x] 2.3.8 Remove `headers_cache_key()` utility function if no longer referenced
- [x] 2.3.9 Verify all resolvers now require and use `session` parameter

### 2.4 Consolidate Resolver Toggle Defaults

- [x] 2.4.1 Identify single authoritative location for toggle defaults (recommend `_DEFAULT_RESOLVER_TOGGLES` in resolvers module)
- [x] 2.4.2 Remove duplicate default computation from `apply_config_overrides()` in CLI module
- [x] 2.4.3 Remove duplicate default computation from `load_resolver_config()` in CLI module
- [x] 2.4.4 Import authoritative defaults into CLI module and reference directly
- [x] 2.4.5 Verify toggle behavior remains identical for enabled/disabled resolvers
- [x] 2.4.6 Add unit test confirming single source of truth for defaults

## 3. HTTP Behavior Standardization

### 3.1 Replace Direct session.get() Calls

- [x] 3.1.1 Replace `session.get()` in `UnpaywallResolver.iter_urls()` with `request_with_retries(session, "GET", ...)`
- [x] 3.1.2 Replace any remaining `session.post()` calls with `request_with_retries(session, "POST", ...)`
- [x] 3.1.3 Search codebase for pattern `session\.(get|post|head|put|delete)\(` and verify all use retry helper
- [x] 3.1.4 Ensure all replacements preserve `params`, `json`, `headers`, `timeout` keyword arguments

### 3.2 Enforce config.get_timeout() Usage

- [x] 3.2.1 Audit all `request_with_retries` calls in resolver implementations for timeout parameter
- [x] 3.2.2 Replace hardcoded `config.timeout` references with `config.get_timeout(self.name)`
- [x] 3.2.3 Replace `timeout=30.0` literals with `timeout=config.get_timeout(self.name)`
- [x] 3.2.4 Verify each resolver uses its own name for timeout lookups
- [x] 3.2.5 Add unit test confirming per-resolver timeout overrides function correctly

### 3.3 Standardize Polite Headers Application

- [x] 3.3.1 Audit resolver HTTP calls for headers parameter
- [x] 3.3.2 Ensure base case passes `headers=config.polite_headers` to request_with_retries
- [x] 3.3.3 For resolvers needing custom headers, use `headers = dict(config.polite_headers); headers.update(...)` pattern
- [x] 3.3.4 Verify API key headers (CORE, Semantic Scholar, DOAJ) properly extend polite headers rather than replace
- [x] 3.3.5 Confirm User-Agent and mailto headers present in all outbound requests during integration tests

### 3.4 Implement HEAD-to-GET Degradation

- [x] 3.4.1 In unified `head_precheck()` function, detect 405 and 501 status codes
- [x] 3.4.2 On detection, issue short GET request with `stream=True` and `timeout=min(timeout, 5.0)`
- [x] 3.4.3 Read only first chunk from response iterator without downloading full body
- [x] 3.4.4 Inspect Content-Type header from GET response
- [x] 3.4.5 Return False if Content-Type contains "html", True otherwise
- [x] 3.4.6 Ensure GET degradation wrapped in try-except returning True on any exception
- [x] 3.4.7 Add unit test simulating 405 response with PDF content-type
- [x] 3.4.8 Add unit test simulating 405 response with HTML content-type

## 4. Code Reduction Through Abstraction

### 4.1 Introduce ApiResolverBase Class

- [x] 4.1.1 Define `ApiResolverBase` class inheriting from `RegisteredResolver` in resolvers module
- [x] 4.1.2 Implement `_request_json(session, method, url, *, config, timeout=None, params=None, json=None, headers=None)` method
- [x] 4.1.3 Wrap `request_with_retries` call in try-except for `requests.Timeout` returning error event
- [x] 4.1.4 Handle `requests.ConnectionError` returning event with reason "connection-error"
- [x] 4.1.5 Handle `requests.RequestException` returning event with reason "request-error"
- [x] 4.1.6 Check response status code equals 200, otherwise return error event with http_status
- [x] 4.1.7 Parse response JSON with try-except for `ValueError`, returning json-error event on failure
- [x] 4.1.8 Return tuple of (parsed_data, None) on success or (None, error_event) on any failure
- [x] 4.1.9 Include content preview in json-error metadata (first 200 characters of response.text)

### 4.2 Refactor Resolvers to Use ApiResolverBase

- [x] 4.2.1 Change `DoajResolver` to inherit from `ApiResolverBase` instead of `RegisteredResolver`
- [x] 4.2.2 Replace DOAJ resolver's manual HTTP and error handling with `_request_json()` call
- [x] 4.2.3 Simplify `DoajResolver.iter_urls()` to focus on data extraction from parsed JSON
- [x] 4.2.4 Refactor `ZenodoResolver` to use `ApiResolverBase._request_json()`
- [x] 4.2.5 Refactor `EuropePmcResolver` to use `ApiResolverBase._request_json()`
- [x] 4.2.6 Refactor `HalResolver` to use `ApiResolverBase._request_json()`
- [x] 4.2.7 Refactor `OsfResolver` to use `ApiResolverBase._request_json()`
- [x] 4.2.8 Verify each refactored resolver maintains identical URL extraction logic
- [x] 4.2.9 Measure line count reduction across refactored resolvers (target: 100-200 lines)

### 4.3 Extract HTML Scraping Helpers

- [x] 4.3.1 Create `find_pdf_via_meta(soup, base_url)` function extracting citation_pdf_url meta tag
- [x] 4.3.2 Create `find_pdf_via_link(soup, base_url)` function finding alternate link with PDF type
- [x] 4.3.3 Create `find_pdf_via_anchor(soup, base_url)` function finding anchors with .pdf hrefs or "pdf" text
- [x] 4.3.4 Ensure each helper returns absolute URL using existing `_absolute_url()` utility or None
- [x] 4.3.5 Refactor `LandingPageResolver.iter_urls()` to call helpers in sequence
- [x] 4.3.6 Yield `ResolverResult` with metadata indicating which pattern matched
- [x] 4.3.7 Add unit tests for each helper with sample HTML fragments

### 4.4 Refactor Logging to AttemptSink Protocol

- [x] 4.4.1 Define `AttemptSink` Protocol with `log_attempt()`, `log_manifest()`, `log_summary()`, `close()` methods
- [x] 4.4.2 Rename `JsonlLogger` to `JsonlSink` and implement `AttemptSink` protocol
- [x] 4.4.3 Create `CsvSink` class implementing `AttemptSink` protocol (no longer wrapping JSONL)
- [x] 4.4.4 Implement `MultiSink` class accepting list of sinks and forwarding all method calls
- [x] 4.4.5 Update CLI `main()` to build sink list based on `--log-format` argument
- [x] 4.4.6 Use `MultiSink([JsonlSink(...), CsvSink(...)])` when CSV format requested
- [x] 4.4.7 Remove `CsvAttemptLoggerAdapter` class after confirming no external references
- [x] 4.4.8 Update resolver pipeline to accept any `AttemptSink`-compatible logger
- [x] 4.4.9 Verify all sink methods called correctly during dry-run and live execution

## 5. Robustness Enhancements

### 5.1 Add Conditional Request Pre-Validation

- [x] 5.1.1 In `ConditionalRequestHelper.build_headers()`, check if etag or last_modified present
- [x] 5.1.2 When conditional headers present, validate sha256, content_length, and path are all non-None
- [x] 5.1.3 If validation fails, log warning "resume-metadata-incomplete: falling back to full fetch"
- [x] 5.1.4 Return empty headers dict to force 200 response instead of 304
- [x] 5.1.5 Add unit test with partial metadata triggering fallback
- [x] 5.1.6 Add integration test confirming graceful handling of corrupted resume manifests

### 5.2 Implement Jittered Domain Throttling

- [x] 5.2.1 Locate `ResolverPipeline._respect_domain_limit()` method
- [x] 5.2.2 Add `+ random.random() * 0.05` to computed sleep duration
- [x] 5.2.3 Ensure jitter applied after lock-protected last hit time update
- [x] 5.2.4 Verify jitter prevents synchronized wakeup across concurrent resolver threads
- [x] 5.2.5 Add integration test measuring sleep time distribution with multiple workers

### 5.3 Harden PDF Detection Heuristics

- [x] 5.3.1 In `classify_payload()`, check if content_type equals "application/octet-stream"
- [x] 5.3.2 For octet-stream, return None to force continued sniffing regardless of URL extension
- [x] 5.3.3 Move URL extension check (.pdf) after all other heuristics as final fallback
- [x] 5.3.4 Add unit test with octet-stream and .pdf URL confirming sniff-based detection
- [x] 5.3.5 Add unit test with octet-stream and %PDF header confirming correct classification

### 5.4 Enhance Corruption Detection

- [x] 5.4.1 In `_build_download_outcome()`, verify `head_precheck_passed` flag propagated correctly
- [x] 5.4.2 Apply tiny PDF (<1 KiB) rejection only when `head_precheck_passed` is False
- [x] 5.4.3 Ensure HEAD-validated tiny PDFs are not rejected as corrupt
- [x] 5.4.4 Maintain existing tail buffer HTML detection logic unchanged
- [x] 5.4.5 Preserve %%EOF validation for all PDF classifications
- [x] 5.4.6 Add unit test with 800-byte HEAD-validated PDF confirming acceptance
- [x] 5.4.7 Add unit test with 800-byte non-validated PDF confirming rejection

## 6. Observability Improvements

### 6.1 Add Staging Directory Mode

- [x] 6.1.1 Add `--staging` boolean CLI flag with help text explaining timestamped run directories
- [x] 6.1.2 When enabled, compute run directory as `args.out / datetime.now(UTC).strftime("%Y%m%d_%H%M")`
- [x] 6.1.3 Set `pdf_dir = run_dir / "PDF"` and `html_dir = run_dir / "HTML"`
- [x] 6.1.4 Set `manifest_path = run_dir / "manifest.jsonl"`
- [x] 6.1.5 Ensure `ensure_dir()` creates run directory structure before downloads begin
- [x] 6.1.6 Preserve existing non-staging behavior as default
- [x] 6.1.7 Add integration test verifying isolated staging directory creation

### 6.2 Generate Manifest Index

- [x] 6.2.1 Define index format as `{"work_id": {"pdf_path": str, "sha256": str, "classification": str}}`
- [x] 6.2.2 Accumulate index entries as manifest entries are written
- [x] 6.2.3 On successful PDF downloads, store path and sha256 under work_id key
- [x] 6.2.4 On completion, write `manifest_path.with_suffix(".index.json")` with sorted JSON
- [x] 6.2.5 Handle exceptions during index write gracefully with logged warning
- [x] 6.2.6 Document index as derived artifact for fast resumption queries
- [x] 6.2.7 Add unit test parsing sample manifest and generating correct index

### 6.3 Produce Last-Attempt CSV

- [x] 6.3.1 When `--log-format csv` specified, also initialize `manifest.last.csv` writer
- [x] 6.3.2 Maintain dict mapping work_id to latest ManifestEntry as processing proceeds
- [x] 6.3.3 Define columns: work_id, title, publication_year, resolver, url, classification, path, sha256, content_length, etag, last_modified
- [x] 6.3.4 On completion, write accumulated entries to last-attempt CSV
- [x] 6.3.5 Ensure CSV sorted by work_id for reviewability
- [x] 6.3.6 Add integration test confirming one row per work in output

### 6.4 Improve CLI Help Organization

- [x] 6.4.1 Create argparse group "Resolver Settings" using `parser.add_argument_group()`
- [x] 6.4.2 Move `--resolver-config`, `--resolver-order`, `--unpaywall-email`, `--core-api-key`, `--semantic-scholar-api-key`, `--doaj-api-key` into group
- [x] 6.4.3 Move `--disable-resolver`, `--enable-resolver`, `--max-resolver-attempts`, `--resolver-timeout`, `--concurrent-resolvers` into group
- [x] 6.4.4 Move `--head-precheck`, `--no-head-precheck`, `--domain-min-interval` into group
- [x] 6.4.5 Remove `--log-path` argument or mark as deprecated hidden argument
- [x] 6.4.6 Update `--manifest` help text to mention it replaces deprecated `--log-path`
- [x] 6.4.7 Verify `--help` output shows organized sections

## 7. Testing and Validation

### 7.1 Unit Tests

- [x] 7.1.1 Add test for consolidated context management lifecycle
- [x] 7.1.2 Add test for CSV close() idempotency
- [x] 7.1.3 Add tests for unified head_precheck with various status codes
- [x] 7.1.4 Add test for ApiResolverBase error handling paths
- [x] 7.1.5 Add tests for HTML scraping helper functions
- [x] 7.1.6 Add test for conditional request pre-validation
- [x] 7.1.7 Add test for PDF classification with octet-stream
- [x] 7.1.8 Add test for corruption detection with head_precheck_passed flag

### 7.2 Integration Tests

- [x] 7.2.1 Run dry-run with 3-5 works ensuring single manifest written
- [ ] 7.2.2 Run with --log-format csv confirming both JSONL and CSV produced
- [ ] 7.2.3 Run with --staging confirming timestamped directory created
- [ ] 7.2.4 Run with --resume-from using manifest with partial metadata
- [ ] 7.2.5 Run with --workers 3 confirming domain throttling with jitter
- [ ] 7.2.6 Run with --head-precheck against HEAD-hostile URL
- [ ] 7.2.7 Verify all resolvers produce consistent attempt records

### 7.3 Regression Testing

- [ ] 7.3.1 Compare dry-run coverage metrics before and after refactor
- [ ] 7.3.2 Verify identical resolver ordering and selection logic
- [ ] 7.3.3 Confirm manifest JSONL format compatibility with pre-refactor versions
- [ ] 7.3.4 Validate resume behavior with old manifest files
- [ ] 7.3.5 Test backward compatibility of resolver configuration files

## 8. Documentation and Migration

### 8.1 Update Documentation

- [ ] 8.1.1 Document breaking changes in CHANGELOG.md
- [ ] 8.1.2 Add migration guide for import changes
- [ ] 8.1.3 Update CLI help text for new flags
- [ ] 8.1.4 Document ApiResolverBase for custom resolver authors
- [ ] 8.1.5 Describe staging mode usage patterns

### 8.2 Code Cleanup

- [ ] 8.2.1 Remove commented-out legacy code blocks
- [ ] 8.2.2 Update module docstrings reflecting new architecture
- [ ] 8.2.3 Ensure consistent formatting across modified files
- [ ] 8.2.4 Run linters and fix reported issues
- [ ] 8.2.5 Verify no unused imports remain after refactor
