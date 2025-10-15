# Content Download Specification

## ADDED Requirements

### Requirement: Centralized HTTP Retry Logic

The system SHALL implement a single unified retry mechanism for all HTTP requests through the `http.request_with_retries` helper function that provides deterministic exponential backoff behavior.

The HTTP session layer SHALL be configured with zero adapter-level retries to prevent compounding of retry attempts with application-level retry logic.

#### Scenario: Failed request triggers deterministic retry count

- **WHEN** an HTTP request fails with status code 429 (Too Many Requests)
- **AND** the centralized retry helper is configured with maximum three retries
- **THEN** the system SHALL execute exactly one initial request plus three retry attempts for a total of four requests
- **AND** no additional retry attempts SHALL occur beyond those specified in the centralized configuration

#### Scenario: Retry timing follows exponential backoff with jitter

- **WHEN** an HTTP request fails and triggers retry logic
- **THEN** the delay before the first retry SHALL be base_delay × 2^0 plus random jitter between 0 and 0.1 seconds
- **AND** the delay before the second retry SHALL be base_delay × 2^1 plus jitter
- **AND** subsequent retries SHALL continue the exponential backoff pattern
- **AND** if the server provides a Retry-After header, that value SHALL override the calculated delay when it exceeds the calculated value

#### Scenario: Status code filtering determines retryability

- **WHEN** an HTTP request returns status code 200 (OK)
- **THEN** the response SHALL be returned immediately without retry attempts
- **WHEN** an HTTP request returns status code 429, 500, 502, 503, or 504
- **THEN** the retry logic SHALL be triggered according to the configured retry policy
- **WHEN** an HTTP request returns status code 404 (Not Found)
- **THEN** the response SHALL be returned immediately without retry attempts

### Requirement: Elimination of Redundant HEAD Requests

The content download function SHALL NOT issue preliminary HEAD requests before GET requests since content classification occurs through streaming byte analysis and the pipeline provides configurable HEAD preflight filtering.

#### Scenario: Successful download issues single GET request

- **WHEN** a candidate URL is downloaded
- **AND** the pipeline HEAD precheck has been executed (if enabled)
- **THEN** the download function SHALL issue exactly one GET request to retrieve the content
- **AND** no separate HEAD request SHALL be issued by the download function
- **AND** content type determination SHALL rely on response headers from the GET request and byte stream analysis

#### Scenario: Content classification proceeds without HEAD response

- **WHEN** the download function begins retrieving a candidate URL
- **THEN** content type classification SHALL be performed using the Content-Type header from the GET response combined with analysis of the initial bytes received from the response stream
- **AND** the classification process SHALL NOT depend on any preliminary HEAD request data from the download function

### Requirement: Thread-Safe Logging Infrastructure

The JSONL and CSV logging components SHALL implement thread-safe write operations using threading locks to prevent record interleaving when multiple workers log concurrently.

#### Scenario: Concurrent log writes produce well-formed records

- **WHEN** sixteen threads each write one thousand log records concurrently to a shared logger instance
- **THEN** the resulting JSONL file SHALL contain exactly sixteen thousand lines
- **AND** every line SHALL be valid JSON that can be parsed without errors
- **AND** no line SHALL contain interleaved content from multiple threads

#### Scenario: CSV output remains structured under concurrent access

- **WHEN** multiple threads write CSV attempt records concurrently to a shared CSV logger adapter
- **THEN** every row in the resulting CSV file SHALL contain the correct number of fields
- **AND** no row SHALL contain field values interleaved from different threads
- **AND** the CSV SHALL be parseable by standard CSV libraries without errors

#### Scenario: Logger supports context manager protocol

- **WHEN** a JSONL logger is used within a with-statement block
- **THEN** the logger SHALL return itself from the **enter** method
- **AND** the logger SHALL close its file handle when the with-statement block exits
- **AND** exceptions occurring within the block SHALL propagate normally after resource cleanup

### Requirement: Streaming Hash Computation

The system SHALL compute SHA-256 checksums and byte counts incrementally during the initial file write operation to eliminate redundant disk reads.

#### Scenario: Hash computed during streaming download

- **WHEN** content bytes are retrieved from the HTTP response stream
- **AND** bytes are written to the destination file
- **THEN** the same bytes SHALL be simultaneously fed to a SHA-256 hasher instance
- **AND** a running byte counter SHALL be incremented by the length of each written chunk
- **AND** no separate file read operation SHALL occur to compute the hash

#### Scenario: Finalized hash matches reference computation

- **WHEN** the download completes and the hash is finalized
- **THEN** the hex digest from the streaming hasher SHALL match the SHA-256 hash that would be computed by re-reading the entire file
- **AND** the byte counter value SHALL equal the file size reported by the filesystem

#### Scenario: Hash computation bypassed in dry-run mode

- **WHEN** the system operates in dry-run mode
- **THEN** no files SHALL be written to disk
- **AND** no hash computation SHALL occur
- **AND** hash and content_length fields in the outcome SHALL be set to None

### Requirement: Early Content Corruption Detection

The system SHALL detect corrupted or mislabeled PDF files through size threshold checks and HTML content inspection to prevent inclusion of invalid files in the corpus.

#### Scenario: Tiny files flagged as corrupt

- **WHEN** a file is classified as PDF during download
- **AND** the file size is less than 1024 bytes
- **THEN** the classification SHALL be overridden to "pdf_corrupt"
- **AND** the file MAY be deleted to avoid polluting the download directory

#### Scenario: HTML content detected in PDF files

- **WHEN** a file is classified as PDF during download
- **AND** the final 1024 bytes of the file contain the byte sequence "</html" (case-insensitive)
- **THEN** the classification SHALL be overridden to "pdf_corrupt"
- **AND** this indicates a server error where HTML was served with incorrect Content-Type headers

#### Scenario: Valid PDFs pass corruption detection

- **WHEN** a PDF file is downloaded successfully
- **AND** the file size exceeds 1024 bytes
- **AND** the file terminates with the "%%EOF" marker
- **AND** the file does not contain HTML closing tags
- **THEN** the classification SHALL remain "pdf" or "pdf_unknown" as appropriate
- **AND** the file SHALL be retained in the download directory

### Requirement: Robust Filename Extension Inference

The system SHALL determine appropriate file extensions from HTTP response headers including Content-Disposition, Content-Type, and URL patterns while preserving deterministic base filename stems.

#### Scenario: RFC5987 encoded filename extracted from headers

- **WHEN** a Content-Disposition header contains "filename*=utf-8''example%20file.pdf"
- **THEN** the filename SHALL be URL-decoded to "example file.pdf"
- **AND** the extension ".pdf" SHALL be extracted for determining the file suffix
- **AND** the base filename stem SHALL remain unchanged and deterministic based on work metadata

#### Scenario: Content-Type takes precedence for PDF identification

- **WHEN** the Content-Type header is "application/pdf"
- **THEN** the file extension SHALL be set to ".pdf" regardless of URL or filename patterns
- **AND** this ensures correct extension even when URLs do not indicate PDF format

#### Scenario: Default extension used when headers are ambiguous

- **WHEN** Content-Disposition is absent or unparseable
- **AND** Content-Type is non-specific or missing
- **AND** the URL does not end with a recognized extension
- **THEN** the default extension appropriate for the detected content type SHALL be used
- **AND** for detected PDF content the default SHALL be ".pdf"
- **AND** for detected HTML content the default SHALL be ".html"

### Requirement: Enhanced DOI Normalization

The system SHALL normalize DOI identifiers from multiple common prefix formats to produce canonical identifiers for resolver queries.

#### Scenario: HTTPS DOI.org prefix normalized

- **WHEN** a DOI is provided as "<https://doi.org/10.1234/example>"
- **THEN** the normalized form SHALL be "10.1234/example"

#### Scenario: HTTP DOI.org prefix normalized

- **WHEN** a DOI is provided as "<http://doi.org/10.1234/example>"
- **THEN** the normalized form SHALL be "10.1234/example"

#### Scenario: DX.DOI.org prefix normalized

- **WHEN** a DOI is provided as "<https://dx.doi.org/10.1234/example>" or "<http://dx.doi.org/10.1234/example>"
- **THEN** the normalized form SHALL be "10.1234/example"

#### Scenario: DOI colon prefix normalized

- **WHEN** a DOI is provided as "doi:10.1234/example"
- **THEN** the normalized form SHALL be "10.1234/example"

#### Scenario: Prefix matching is case-insensitive

- **WHEN** a DOI is provided with mixed-case prefix like "<HTTPS://DOI.ORG/10.1234/example>"
- **THEN** the prefix SHALL be recognized and removed
- **AND** the resulting DOI portion SHALL preserve its original case

### Requirement: Crossref Resolver HTTP Integration

The Crossref resolver SHALL use the centralized retry helper function for all HTTP requests to ensure consistent retry behavior across resolver implementations.

#### Scenario: Crossref queries use centralized retry logic

- **WHEN** the Crossref resolver issues a request to the Crossref API
- **THEN** the request SHALL be made through the `request_with_retries` function
- **AND** retry behavior SHALL follow the centralized exponential backoff configuration
- **AND** no direct session.get() calls SHALL bypass the retry helper

#### Scenario: Crossref retry behavior matches other resolvers

- **WHEN** the Crossref API returns status code 429
- **THEN** the retry behavior SHALL be identical to that of other resolvers using the same retry helper
- **AND** timing and attempt count SHALL be deterministic based on centralized configuration

### Requirement: Resolver Utility Decoupling

Shared utility functions used by multiple resolvers SHALL be extracted to dedicated modules to eliminate hidden inter-resolver dependencies.

#### Scenario: Headers cache key utility independently importable

- **WHEN** a resolver requires the headers cache key utility
- **THEN** the utility SHALL be importable from a shared headers utility module
- **AND** resolvers SHALL NOT import utilities from other resolver implementation modules
- **AND** this eliminates coupling between Crossref and Unpaywall resolver implementations

#### Scenario: Cache key function produces deterministic output

- **WHEN** a headers dictionary is passed to the cache key utility
- **AND** the dictionary contains mixed-case keys
- **THEN** the resulting cache key tuple SHALL normalize keys to lowercase
- **AND** the tuple SHALL be sorted for deterministic ordering
- **AND** header values SHALL preserve their original case

### Requirement: CLI Configuration Options

The command-line interface SHALL expose configuration options for concurrent resolver execution, HEAD preflight requests, and custom Accept headers.

#### Scenario: Concurrent resolver limit configurable via CLI

- **WHEN** the user specifies "--concurrent-resolvers 4" on the command line
- **THEN** the resolver configuration SHALL set max_concurrent_resolvers to 4
- **AND** up to four resolvers SHALL execute concurrently per work item
- **WHEN** the flag is not specified
- **THEN** the configuration SHALL use the default value of 1 (sequential execution)

#### Scenario: HEAD precheck toggleable via CLI flags

- **WHEN** the user specifies "--head-precheck" on the command line
- **THEN** the resolver configuration SHALL enable HEAD preflight filtering
- **WHEN** the user specifies "--no-head-precheck"
- **THEN** the resolver configuration SHALL disable HEAD preflight filtering
- **WHEN** neither flag is specified
- **THEN** HEAD precheck SHALL be enabled by default

#### Scenario: Custom Accept header injected via CLI

- **WHEN** the user specifies '--accept "application/pdf,text/html;q=0.8,*/*;q=0.5"'
- **THEN** the polite headers dictionary SHALL include an Accept header with that value
- **AND** the Accept header SHALL be sent with all HTTP requests made by resolvers

### Requirement: Machine-Readable Run Summaries

The system SHALL export structured run metrics to both the JSONL manifest stream and a sidecar JSON file to enable automated monitoring and operational dashboards.

#### Scenario: Summary record appended to JSONL manifest

- **WHEN** a download run completes successfully
- **THEN** a record with type "summary" SHALL be appended to the manifest JSONL file
- **AND** the summary record SHALL contain aggregated counts: processed, saved, html_only, skipped
- **AND** the summary record SHALL contain per-resolver metrics: attempts, successes, html results, skip reasons, failures

#### Scenario: Sidecar JSON file created for monitoring

- **WHEN** a download run completes successfully
- **THEN** a JSON file with suffix ".metrics.json" SHALL be created alongside the manifest JSONL file
- **AND** the JSON file SHALL contain the same aggregated metrics as the JSONL summary record
- **AND** the JSON SHALL be formatted with indentation for human readability
- **AND** keys SHALL be sorted alphabetically for deterministic output

#### Scenario: Metrics export failures do not block completion

- **WHEN** metrics export encounters an error such as permission denied or disk full
- **THEN** a warning SHALL be logged indicating the failure
- **AND** the error SHALL include stack trace details for troubleshooting
- **AND** the run SHALL continue to completion without crashing
- **AND** the attempt logger SHALL be closed properly to ensure manifest records are not lost

### Requirement: Legacy Export Deprecation

The resolver package facade SHALL emit deprecation warnings when the time and requests module re-exports are accessed and SHALL document the removal timeline.

#### Scenario: Time module re-export triggers deprecation warning

- **WHEN** code imports time from DocsToKG.ContentDownload.resolvers
- **THEN** a DeprecationWarning SHALL be emitted
- **AND** the warning message SHALL state that the re-export will be removed in the next minor version
- **AND** the warning SHALL recommend importing time directly from the standard library

#### Scenario: Requests module re-export triggers deprecation warning

- **WHEN** code imports requests from DocsToKG.ContentDownload.resolvers
- **THEN** a DeprecationWarning SHALL be emitted
- **AND** the warning message SHALL state that the re-export will be removed in the next minor version
- **AND** the warning SHALL recommend importing requests directly from the PyPI package

#### Scenario: Other resolver exports remain functional

- **WHEN** code imports resolver classes, types, or legitimate utilities from the resolver package
- **THEN** no deprecation warnings SHALL be emitted
- **AND** all documented public APIs SHALL continue to function correctly

### Requirement: Optional Global URL Deduplication

The system SHALL provide an optional capability to track URLs globally across all work items processed by a pipeline instance to prevent redundant downloads of shared assets.

This capability SHALL be disabled by default and enabled only through explicit configuration to preserve backward compatibility.

#### Scenario: Shared URL downloaded only once when enabled

- **WHEN** global URL deduplication is enabled in the pipeline configuration
- **AND** two work items both resolve to the same PDF URL
- **THEN** the first work item SHALL download the PDF successfully
- **AND** the second work item SHALL skip the download with reason "duplicate-url-global"
- **AND** an attempt record SHALL be logged for the second work with status "skipped"

#### Scenario: Per-work deduplication continues to function

- **WHEN** global URL deduplication is disabled (default configuration)
- **AND** two work items both resolve to the same PDF URL
- **THEN** each work item SHALL attempt to download the PDF independently
- **AND** per-work URL deduplication SHALL prevent multiple attempts within the same work

### Requirement: Optional Domain-Level Rate Limiting

The system SHALL provide an optional capability to enforce minimum intervals between requests to specific domain names independently of per-resolver rate limiting.

This capability SHALL be disabled by default with no domains configured for domain-level limiting.

#### Scenario: Domain interval enforced for configured domains

- **WHEN** domain-level rate limiting is configured with "example.org" requiring 0.5 second intervals
- **AND** two resolvers produce URLs both pointing to example.org
- **THEN** the second request SHALL be delayed until at least 0.5 seconds have elapsed since the first request
- **AND** this delay is independent of per-resolver rate limiting

#### Scenario: Unconfigured domains not affected

- **WHEN** domain-level rate limiting is enabled but a domain is not in the configuration
- **THEN** requests to that domain SHALL proceed without domain-level delays
- **AND** only per-resolver rate limits SHALL apply

#### Scenario: Domain matching is case-insensitive

- **WHEN** domain rate limiting is configured for "Example.Org"
- **AND** a URL targets "example.org" or "EXAMPLE.ORG"
- **THEN** the rate limiting SHALL apply regardless of hostname case
