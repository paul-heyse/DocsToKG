# Content Download Capability Specification

## ADDED Requirements

### Requirement: Unified CLI Context Management

The CLI orchestrator SHALL manage all context resources through a single `contextlib.ExitStack` instance ensuring deterministic lifecycle management for logging, file handles, and resolver state.

#### Scenario: Single context initialization

- **WHEN** the CLI main() function begins execution
- **THEN** exactly one ExitStack context manager is created
- **AND** the JSONL logger enters the context via stack.enter_context()
- **AND** any CSV sink (if enabled) enters the same context via stack.enter_context()
- **AND** all context resources are guaranteed closed on exit regardless of exceptions

#### Scenario: Session factory definition

- **WHEN** the CLI requires HTTP session creation
- **THEN** exactly one _session_factory() function is defined
- **AND** the function contains a single return statement invoking _make_session(config.polite_headers)
- **AND** the function includes complete docstring describing session configuration

#### Scenario: No duplicate context blocks

- **WHEN** scanning the main() function implementation
- **THEN** no abandoned or duplicate context manager blocks exist
- **AND** no unreachable return statements are present
- **AND** all resource initialization occurs within the unified ExitStack

### Requirement: Idempotent Resource Cleanup

All context managers and closeable resources SHALL implement idempotent close() methods that safely handle multiple invocations without errors or resource leaks.

#### Scenario: CSV file handle closure

- **WHEN** CsvSink.close() is invoked
- **THEN** the CSV file handle is checked for closed status under lock
- **AND** the CSV file is closed if not already closed
- **AND** subsequent close() calls do not raise exceptions

#### Scenario: Multiple close invocations

- **WHEN** an AttemptSink implementation close() method is called multiple times
- **THEN** the first call releases all resources
- **AND** subsequent calls complete successfully without side effects
- **AND** no file descriptor leaks occur on any platform including Windows

### Requirement: Unified HEAD Preflight Validation

The network layer SHALL provide a single `head_precheck()` function implementing HEAD request validation with optional GET degradation for HEAD-hostile servers.

#### Scenario: Successful HEAD validation

- **WHEN** head_precheck() issues HEAD request to URL
- **AND** response status code is 200, 302, or 304
- **AND** Content-Type header does not contain "text/html"
- **AND** Content-Length header is not "0"
- **THEN** the function returns True indicating likely PDF

#### Scenario: HTML content rejection

- **WHEN** head_precheck() receives HEAD response with Content-Type "text/html"
- **THEN** the function returns False rejecting the URL

#### Scenario: HEAD method not allowed degradation

- **WHEN** head_precheck() receives 405 or 501 status code
- **THEN** the function issues short GET request with stream=True
- **AND** reads only first chunk without downloading full body
- **AND** inspects Content-Type header from GET response
- **AND** returns False if Content-Type contains "html"
- **AND** returns True for non-HTML content types

#### Scenario: Exception handling

- **WHEN** head_precheck() encounters any exception during request
- **THEN** the function returns True for conservative pass-through
- **AND** no exceptions propagate to caller

#### Scenario: Timeout budget

- **WHEN** head_precheck() issues HEAD or GET request
- **THEN** timeout is limited to minimum of provided timeout and 5.0 seconds
- **AND** requests use max_retries=1 for quick failure

### Requirement: Network Layer Import Unification

All resolver implementations SHALL import HTTP retry functionality directly from the network module eliminating proxy functions and compatibility shims.

#### Scenario: Direct network imports

- **WHEN** resolver code requires HTTP request capabilities
- **THEN** resolvers import request_with_retries from DocsToKG.ContentDownload.network
- **AND** no proxy function exists in resolvers module
- **AND** no **getattr** shim re-exports network functionality

#### Scenario: Legacy export removal

- **WHEN** scanning resolvers module exports
- **THEN** **getattr** function does not exist
- **AND** **all** list does not include 'time' or 'requests'
- **AND** no deprecation warnings are emitted for network imports

### Requirement: Standardized Resolver HTTP Behavior

All resolver implementations SHALL use the unified request_with_retries() helper with consistent timeout and header application ensuring uniform backoff and rate limit handling.

#### Scenario: Unified retry invocation

- **WHEN** resolver issues HTTP request
- **THEN** the resolver calls request_with_retries() from network module
- **AND** no direct session.get() or session.post() calls exist
- **AND** all requests benefit from exponential backoff and Retry-After header respect

#### Scenario: Per-resolver timeout application

- **WHEN** resolver constructs HTTP request
- **THEN** timeout parameter uses config.get_timeout(self.name)
- **AND** resolver-specific timeout overrides are honored
- **AND** default timeout fallback is applied when no override exists

#### Scenario: Polite header propagation

- **WHEN** resolver issues HTTP request without custom headers
- **THEN** headers parameter includes config.polite_headers
- **AND** User-Agent and mailto headers are present
- **AND** requests identify as DocsToKG with contact information

#### Scenario: Custom header extension

- **WHEN** resolver requires API key or custom headers
- **THEN** resolver creates dict copy of config.polite_headers
- **AND** resolver updates copy with custom headers
- **AND** polite headers are extended not replaced

### Requirement: Session-Less Branch Elimination

Resolver implementations SHALL require and use the provided requests.Session instance eliminating fallback code paths that bypass unified retry and header logic.

#### Scenario: Session parameter requirement

- **WHEN** resolver iter_urls() method is invoked
- **THEN** session parameter is used for all HTTP requests
- **AND** no hasattr(session, "get") conditional branches exist
- **AND** no direct requests.get() calls bypass session

#### Scenario: LRU cache removal

- **WHEN** scanning resolver implementations
- **THEN** no @lru_cache decorated _fetch_* functions exist
- **AND** no global HTTP calls cache responses
- **AND** all caching occurs through explicit conditional request headers

### Requirement: API Resolver Base Class

The resolver infrastructure SHALL provide ApiResolverBase class offering _request_json() helper that standardizes JSON API interaction patterns including error handling and event generation.

#### Scenario: JSON request success

- **WHEN** ApiResolverBase._request_json() is called with valid endpoint
- **AND** response status code is 200
- **AND** response body contains valid JSON
- **THEN** method returns tuple (parsed_data, None)

#### Scenario: HTTP error handling

- **WHEN** _request_json() receives non-200 status code
- **THEN** method returns tuple (None, error_event)
- **AND** error_event contains event="error" and event_reason="http-error"
- **AND** error_event metadata includes http_status

#### Scenario: Connection error handling

- **WHEN** _request_json() encounters requests.ConnectionError
- **THEN** method returns tuple (None, error_event)
- **AND** error_event contains event_reason="connection-error"
- **AND** error detail is preserved in metadata

#### Scenario: Timeout handling

- **WHEN** _request_json() encounters requests.Timeout
- **THEN** method returns tuple (None, error_event)
- **AND** error_event contains event_reason="timeout"
- **AND** timeout value is preserved in metadata

#### Scenario: JSON parse error handling

- **WHEN** _request_json() receives 200 response with invalid JSON
- **THEN** method returns tuple (None, error_event)
- **AND** error_event contains event_reason="json-error"
- **AND** metadata includes first 200 characters of response text as content_preview

### Requirement: HTML Scraping Helper Extraction

The resolver infrastructure SHALL provide reusable helper functions for extracting PDF URLs from landing page HTML using common academic publisher patterns.

#### Scenario: Citation meta tag extraction

- **WHEN** find_pdf_via_meta() receives HTML containing citation_pdf_url meta tag
- **THEN** function extracts content attribute value
- **AND** returns absolute URL resolved against base URL
- **AND** returns None if meta tag absent or content empty

#### Scenario: Alternate link extraction

- **WHEN** find_pdf_via_link() receives HTML with link rel="alternate" type="application/pdf"
- **THEN** function extracts href attribute value
- **AND** returns absolute URL resolved against base URL
- **AND** returns None if no matching link found

#### Scenario: Anchor heuristic extraction

- **WHEN** find_pdf_via_anchor() receives HTML with anchor containing .pdf href or "pdf" text
- **THEN** function identifies first matching anchor element
- **AND** returns absolute URL resolved against base URL
- **AND** returns None if no pdf-related anchors found

### Requirement: Protocol-Based Logging Architecture

The CLI SHALL compose logging outputs using AttemptSink protocol enabling symmetric multi-sink composition without adapter wrapping patterns.

#### Scenario: JSONL sink implementation

- **WHEN** JsonlSink is instantiated with file path
- **THEN** sink implements log_attempt() writing attempt records
- **AND** sink implements log_manifest() writing manifest entries
- **AND** sink implements log_summary() writing summary records
- **AND** sink implements close() closing file handle

#### Scenario: CSV sink implementation

- **WHEN** CsvSink is instantiated with file path
- **THEN** sink writes CSV header row if file does not exist
- **AND** sink implements log_attempt() appending CSV rows
- **AND** sink serializes metadata as JSON string in CSV cell
- **AND** sink owns file handle directly without wrapping another sink

#### Scenario: MultiSink composition

- **WHEN** MultiSink is instantiated with list of sinks
- **THEN** calling log_attempt() forwards to all sinks in order
- **AND** calling log_manifest() forwards to all sinks
- **AND** calling log_summary() forwards to all sinks
- **AND** calling close() closes all sinks

#### Scenario: CLI sink selection

- **WHEN** CLI runs with default log format
- **THEN** only JsonlSink is instantiated
- **WHEN** --log-format csv is specified
- **THEN** MultiSink([JsonlSink, CsvSink]) is created
- **AND** both JSONL and CSV outputs are produced

### Requirement: Conditional Request Pre-Validation

The conditional request helper SHALL validate metadata completeness before building headers preventing late failures when resume manifests contain partial information.

#### Scenario: Complete metadata validation

- **WHEN** ConditionalRequestHelper.build_headers() is called
- **AND** prior_etag or prior_last_modified is present
- **AND** prior_sha256, prior_content_length, and prior_path are all present
- **THEN** conditional headers are generated normally

#### Scenario: Incomplete metadata detection

- **WHEN** build_headers() detects conditional header candidates
- **AND** any of prior_sha256, prior_content_length, or prior_path is missing
- **THEN** warning is logged with reason "resume-metadata-incomplete"
- **AND** empty headers dict is returned forcing fresh 200 fetch
- **AND** no 304 response is possible preventing ValueError

#### Scenario: No conditional headers

- **WHEN** neither prior_etag nor prior_last_modified is present
- **THEN** empty headers dict is returned
- **AND** no validation is performed

### Requirement: Jittered Domain Throttling

The resolver pipeline SHALL add random jitter to domain-level throttling delays preventing synchronized thread wakeup across concurrent resolver workers.

#### Scenario: Domain throttle jitter application

- **WHEN** _respect_domain_limit() computes required sleep duration
- **THEN** jitter of random.random() * 0.05 seconds is added
- **AND** sleep occurs with jittered duration
- **AND** last hit time is recorded under lock after sleep
- **AND** concurrent threads wake at staggered times

#### Scenario: Jitter distribution

- **WHEN** multiple resolver threads throttle same domain
- **THEN** sleep durations vary across 50ms range
- **AND** thundering herd effect is mitigated

### Requirement: Hardened PDF Detection

The payload classification logic SHALL treat ambiguous content types as unknown requiring content sniffing regardless of URL extensions preventing misclassification of octet-stream responses.

#### Scenario: Octet-stream suspicion

- **WHEN** classify_payload() receives Content-Type "application/octet-stream"
- **THEN** function returns None forcing continued sniffing
- **AND** URL extension .pdf does not override octet-stream treatment

#### Scenario: Sniff-based classification

- **WHEN** content_type is ambiguous or octet-stream
- **AND** head_bytes contain %PDF signature
- **THEN** classification returns "pdf" based on content not content-type

#### Scenario: URL extension as fallback

- **WHEN** all other heuristics fail to classify payload
- **AND** URL ends with .pdf
- **THEN** "pdf" classification is returned as final fallback

### Requirement: Enhanced Corruption Detection

The download outcome builder SHALL apply size-based corruption detection only to non-validated downloads preserving HEAD-validated tiny PDFs that legitimately exist below 1 KiB threshold.

#### Scenario: Tiny PDF with HEAD validation

- **WHEN** _build_download_outcome() processes PDF classification
- **AND** content_length is less than 1024 bytes
- **AND** head_precheck_passed flag is True
- **THEN** PDF is accepted as valid
- **AND** no size-based rejection occurs

#### Scenario: Tiny PDF without validation

- **WHEN** PDF classification has content_length less than 1024
- **AND** head_precheck_passed is False
- **THEN** file is deleted
- **AND** classification changes to "pdf_corrupt"
- **AND** outcome path is set to None

#### Scenario: EOF marker validation

- **WHEN** PDF classification outcome is built
- **AND** dry_run is False
- **THEN** _has_pdf_eof() is invoked checking for %%EOF marker
- **AND** missing EOF causes rejection with pdf_corrupt classification

#### Scenario: HTML tail detection

- **WHEN** tail_bytes contain "</html" case-insensitive
- **THEN** file is recognized as HTML stub not PDF
- **AND** classification changes to "pdf_corrupt"
- **AND** file is deleted

### Requirement: Staging Directory Isolation

The CLI SHALL support optional --staging mode creating timestamped run directories enabling side-by-side comparison and trivial rollback of large-scale crawls.

#### Scenario: Staging mode activation

- **WHEN** CLI runs with --staging flag
- **THEN** run directory is created with format YYYYMMDD_HHMM using UTC time
- **AND** PDF subdirectory is created as run_dir/PDF
- **AND** HTML subdirectory is created as run_dir/HTML
- **AND** manifest path is set to run_dir/manifest.jsonl

#### Scenario: Default non-staging behavior

- **WHEN** --staging flag is not provided
- **THEN** pdf_dir equals args.out directly
- **AND** html_dir equals args.html_out or args.out.parent/HTML
- **AND** manifest_path equals args.out/manifest.jsonl
- **AND** behavior is unchanged from pre-refactor versions

#### Scenario: Staging directory structure

- **WHEN** staging run completes
- **THEN** complete directory structure is self-contained
- **AND** directory can be renamed or moved as atomic unit
- **AND** multiple staging runs coexist in parent directory

### Requirement: Manifest Index Generation

The CLI SHALL generate derived manifest.index.json file mapping work_id to best PDF path and SHA256 enabling fast resumption queries without scanning full JSONL manifest.

#### Scenario: Index accumulation

- **WHEN** manifest entry is written with PDF classification
- **THEN** index entry is accumulated mapping work_id to pdf_path and sha256
- **AND** subsequent entries for same work_id update existing index entry
- **AND** only successful PDF downloads appear in index

#### Scenario: Index persistence

- **WHEN** download run completes successfully
- **THEN** manifest.index.json is written alongside manifest.jsonl
- **AND** JSON is formatted with sorted keys for readability
- **AND** write failures are logged as warnings without failing run

#### Scenario: Index structure

- **WHEN** index file is parsed
- **THEN** top-level object maps work_id strings to entry objects
- **AND** each entry contains pdf_path, sha256, and classification fields
- **AND** entries are sorted by work_id for deterministic output

### Requirement: Last-Attempt CSV Generation

When CSV logging is enabled, the CLI SHALL produce manifest.last.csv containing exactly one row per work_id summarizing final outcome for human review workflows.

#### Scenario: Last-attempt accumulation

- **WHEN** manifest entries are written during execution
- **AND** --log-format csv is specified
- **THEN** dict mapping work_id to latest ManifestEntry is maintained
- **AND** subsequent entries for same work_id replace previous entries

#### Scenario: Last-attempt CSV structure

- **WHEN** run completes and last-attempt CSV is written
- **THEN** CSV contains columns: work_id, title, publication_year, resolver, url, classification, path, sha256, content_length, etag, last_modified
- **AND** exactly one row exists per processed work_id
- **AND** rows are sorted by work_id

#### Scenario: Last-attempt independence

- **WHEN** last-attempt CSV is produced
- **THEN** full attempts CSV contains all attempts as before
- **AND** last-attempt CSV provides summary view
- **AND** both CSVs can be analyzed independently

### Requirement: Organized CLI Help

The CLI argument parser SHALL group resolver-related flags under dedicated section improving discoverability and reducing help text cognitive overhead.

#### Scenario: Resolver settings group

- **WHEN** CLI --help is invoked
- **THEN** "Resolver Settings" argument group is displayed
- **AND** group contains --resolver-config, --resolver-order flags
- **AND** group contains API key flags for Unpaywall, CORE, Semantic Scholar, DOAJ
- **AND** group contains --disable-resolver, --enable-resolver flags
- **AND** group contains --max-resolver-attempts, --resolver-timeout, --concurrent-resolvers flags
- **AND** group contains --head-precheck, --no-head-precheck, --domain-min-interval flags

#### Scenario: Legacy flag deprecation

- **WHEN** scanning argument definitions
- **THEN** --log-path flag does not exist or is marked hidden
- **AND** --manifest flag help text mentions it replaces --log-path
- **AND** users are directed to use --manifest for clarity

### Requirement: Resolver Toggle Default Centralization

The resolver configuration SHALL define toggle defaults (enabled/disabled by default) in single authoritative location preventing drift between configuration loaders and runtime defaults.

#### Scenario: Single source of truth

- **WHEN** resolver toggle defaults are needed
- **THEN** _DEFAULT_RESOLVER_TOGGLES in resolvers module is consulted
- **AND** no duplicate default definitions exist in CLI configuration code
- **AND** apply_config_overrides() references authoritative defaults

#### Scenario: Default propagation

- **WHEN** load_resolver_config() builds ResolverConfig
- **THEN** resolver toggles are populated from authoritative defaults
- **AND** configuration file overrides replace defaults
- **AND** CLI --enable-resolver and --disable-resolver flags override both

#### Scenario: Backward compatibility

- **WHEN** existing resolver configuration files are loaded
- **THEN** toggle behavior remains identical to pre-refactor versions
- **AND** openaire, hal, osf remain disabled by default
- **AND** all other resolvers remain enabled by default

### Requirement: Module Independence Preservation

The ContentDownload module SHALL remain independent from DocParsing module with no imports or dependencies on parsing logic ensuring clean separation of concerns.

#### Scenario: No parsing imports

- **WHEN** scanning ContentDownload module imports
- **THEN** no imports from DocsToKG.DocParsing exist
- **AND** no imports from DocsToKG.EmbeddingPipeline exist
- **AND** no dependencies on parsing outputs in resolver logic

#### Scenario: HTML text extraction isolation

- **WHEN** --extract-html-text flag is used
- **THEN** trafilatura import occurs within try-except
- **AND** extraction happens only after download completes
- **AND** text file is written as sibling artifact not embedded in manifest

#### Scenario: Manifest content limits

- **WHEN** manifest entries are created
- **THEN** entries contain only download metadata (paths, sizes, hashes, URLs, timestamps)
- **AND** no parsing results are embedded in manifest records
- **AND** downstream parsing can reconstruct corpus from manifest paths alone

### Requirement: Backward Manifest Compatibility

The manifest JSONL format and resume functionality SHALL remain fully compatible with pre-refactor versions ensuring existing workflows continue operating without modification.

#### Scenario: Manifest format preservation

- **WHEN** manifest entries are written
- **THEN** record_type, timestamp, work_id, resolver, url, path, classification fields are identical
- **AND** sha256, content_length, etag, last_modified fields remain in same format
- **AND** dry_run boolean field has same semantics

#### Scenario: Resume from old manifests

- **WHEN** --resume-from references pre-refactor manifest
- **THEN** load_previous_manifest() parses entries successfully
- **AND** conditional request metadata is extracted correctly
- **AND** completed work_id set is built using same classification rules

#### Scenario: Resume with partial metadata

- **WHEN** old manifest contains entries with missing etag or sha256
- **THEN** pre-validation detects incomplete metadata
- **AND** fresh fetch is forced without 304 attempt
- **AND** no ValueError occurs during resume

### Requirement: Configuration File Compatibility

Resolver configuration files (JSON/YAML) SHALL remain fully compatible with pre-refactor versions preserving all existing configuration options and override semantics.

#### Scenario: Configuration field preservation

- **WHEN** resolver configuration file is loaded
- **THEN** all pre-refactor fields are recognized: resolver_order, resolver_toggles, max_attempts_per_work, timeout, polite_headers, unpaywall_email, core_api_key, semantic_scholar_api_key, doaj_api_key, resolver_timeouts, resolver_min_interval_s, domain_min_interval_s, enable_head_precheck, resolver_head_precheck, mailto, max_concurrent_resolvers, enable_global_url_dedup
- **AND** deprecated resolver_rate_limits field is still supported with automatic migration

#### Scenario: Override application order

- **WHEN** configuration is loaded with CLI overrides
- **THEN** file configuration is applied first
- **AND** environment variables override file configuration
- **AND** CLI flags override both file and environment
- **AND** override precedence is unchanged from pre-refactor behavior

#### Scenario: Default value fallbacks

- **WHEN** configuration option is absent from all sources
- **THEN** default value matches pre-refactor default
- **AND** ResolverConfig.**post_init**() applies same defaults
