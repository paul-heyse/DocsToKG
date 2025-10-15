## ADDED Requirements

### Requirement: Automatic Retry on Transient HTTP Errors

The system SHALL automatically retry HTTP requests that fail due to transient errors (429, 502, 503, 504) using exponential backoff with Retry-After header support, up to a configured maximum of 5 attempts, to eliminate misses from temporary network issues.

#### Scenario: 503 Service Unavailable retried successfully

- **WHEN** a resolver API returns HTTP 503 on first attempt
- **THEN** the system waits backoff_factor * 2^attempt seconds, retries, and succeeds on second attempt without logging permanent failure

#### Scenario: 429 Rate Limit with Retry-After honored

- **WHEN** a resolver returns 429 with Retry-After: 5 header
- **THEN** the system waits at least 5 seconds before retry, respects the server's guidance

#### Scenario: 404 Not Found fails immediately without retry

- **WHEN** a download URL returns 404
- **THEN** the system logs permanent failure immediately, does not retry, moves to next candidate

#### Scenario: Maximum retry budget exhausted

- **WHEN** a URL returns 503 on all 5 retry attempts
- **THEN** the system logs final failure with reason "max-retries-exhausted" and moves to next resolver

### Requirement: Atomic File Writes with Integrity Verification

The system SHALL write downloaded files to temporary .part files, compute SHA-256 digests and content length, then atomically rename to final paths to prevent partial file corruption and enable deduplication.

#### Scenario: Successful download with atomic rename

- **WHEN** a PDF downloads completely
- **THEN** the system writes to *.part, computes SHA-256, performs os.replace() to final path, logs digest and size

#### Scenario: Download interrupted mid-stream

- **WHEN** download is killed via SIGKILL while writing
- **THEN** only *.part file exists, final path does not exist, no corrupted PDF in storage

#### Scenario: Digest enables deduplication

- **WHEN** two works have same PDF at different URLs
- **THEN** manifest records show identical SHA-256, allowing downstream dedup by digest

### Requirement: Unambiguous Rate Limit Configuration

The system SHALL use explicit `resolver_min_interval_s` configuration field to specify minimum seconds between calls per resolver, replacing ambiguous `resolver_rate_limits`, with deprecation warning and auto-migration for old configs.

#### Scenario: Config specifies min_interval_s

- **WHEN** config contains `resolver_min_interval_s: {"unpaywall": 1.0}`
- **THEN** system enforces minimum 1.0 seconds between Unpaywall calls (max ~1 QPS)

#### Scenario: Legacy resolver_rate_limits config auto-migrates

- **WHEN** config contains deprecated `resolver_rate_limits: {"unpaywall": 1.0}`
- **THEN** system logs deprecation warning, auto-migrates value to min_interval_s, continues normally

#### Scenario: Misconfiguration with both fields present

- **WHEN** config contains both resolver_rate_limits and resolver_min_interval_s
- **THEN** system logs error "Conflicting rate limit fields" and fails gracefully with clear remediation message

### Requirement: LRU Cache for Resolver API Responses

The system SHALL cache resolver API responses (Unpaywall, Crossref, Semantic Scholar) in-memory using LRU cache (maxsize=1000) keyed by (resolver_name, identifier) to eliminate redundant API calls within batch runs.

#### Scenario: Same DOI queried twice uses cache

- **WHEN** two works share the same DOI and Unpaywall resolver is invoked for both
- **THEN** only one HTTP request is made to Unpaywall API, second lookup served from cache

#### Scenario: Cache cleared on resume

- **WHEN** --resume-from flag is used to restart a batch
- **THEN** resolver caches are cleared before processing to force fresh API lookups

#### Scenario: Cache hit logged in metrics

- **WHEN** a resolver API response is served from cache
- **THEN** metrics show cache_hit=true in resolver metadata, elapsed_ms near zero

### Requirement: Conditional Request Support for Idempotent Re-runs

The system SHALL support HTTP conditional requests using If-None-Match (ETag) and If-Modified-Since (Last-Modified) headers to enable fast 304 responses on unchanged resources when re-running batches.

#### Scenario: ETag match returns 304 cached

- **WHEN** manifest records previous ETag for a URL and re-run fetches same URL
- **THEN** system sends If-None-Match with ETag, receives 304, returns DownloadOutcome(classification='cached', path=existing_path) without re-downloading

#### Scenario: Last-Modified unchanged returns 304

- **WHEN** manifest records Last-Modified and resource unchanged
- **THEN** system sends If-Modified-Since, receives 304, skips download

#### Scenario: Resource updated returns 200 with new content

- **WHEN** ETag sent but resource changed since last fetch
- **THEN** system receives 200, downloads new version, updates manifest with new ETag and SHA-256

### Requirement: Unified JSONL Logging

The system SHALL log all resolver attempts and work summaries to a single JSONL file with structured records (timestamp, record_type, work_id, resolver_name, status, http_status, sha256, content_length, elapsed_ms) replacing separate CSV and manifest files, with CSV export script provided for backward compatibility.

#### Scenario: Attempt logged as JSONL record

- **WHEN** a resolver attempts a URL
- **THEN** logger writes JSON line: `{"timestamp": "ISO8601", "record_type": "attempt", "work_id": "W123", "resolver_name": "unpaywall", "url": "...", "status": "pdf", "http_status": 200, "sha256": "abc...", "content_length": 123456, "elapsed_ms": 1234.5}`

#### Scenario: Work summary logged as JSONL record

- **WHEN** a work completes processing (success or failure)
- **THEN** logger writes JSON line with record_type="summary" containing total_attempts, resolvers_used, final_status, html_paths

#### Scenario: CSV export from JSONL

- **WHEN** operator runs `scripts/export_attempts_csv.py attempts.jsonl`
- **THEN** script outputs CSV with columns matching legacy CsvAttemptLogger format for backward compatibility

#### Scenario: Machine-readable queries on JSONL

- **WHEN** operator runs `jq '.resolver_name == "unpaywall" and .status == "pdf"' attempts.jsonl`
- **THEN** structured JSONL enables efficient filtering without parsing CSV

### Requirement: Bounded Parallel Execution

The system SHALL support bounded parallelism via --workers N flag using ThreadPoolExecutor to process multiple works concurrently (2-5x throughput) while maintaining per-work sequential pipeline and thread-safe per-resolver rate limiting.

#### Scenario: Parallel processing with --workers=3

- **WHEN** batch run invoked with --workers=3
- **THEN** up to 3 works are processed concurrently via ThreadPoolExecutor, each with own session

#### Scenario: Rate limiting enforced across parallel workers

- **WHEN** multiple workers call same resolver (e.g., Unpaywall) concurrently
- **THEN** shared `_last_invocation` dict with threading.Lock enforces global min_interval_s, preventing rate limit violations

#### Scenario: Sequential mode backward compatible

- **WHEN** --workers=1 (default) used
- **THEN** system processes works sequentially as before, no threading overhead

#### Scenario: Worker failure isolated

- **WHEN** one worker encounters exception during work processing
- **THEN** other workers continue normally, failed work logged with traceback, batch completes

### Requirement: Dry Run Mode for Coverage Measurement

The system SHALL support --dry-run flag that measures resolver coverage and logs attempts without writing files to enable quick testing and coverage analysis.

#### Scenario: Dry run logs attempts without files

- **WHEN** batch run with --dry-run flag
- **THEN** all resolver attempts logged with classifications (pdf/html/miss), no files written to disk, summary shows coverage percentages

#### Scenario: Dry run metadata includes flag

- **WHEN** dry run completes
- **THEN** all JSONL records include `dry_run: true` field, final summary includes "DRY RUN: no files written"

### Requirement: Resume from Manifest

The system SHALL support --resume-from manifest.jsonl flag to skip works already successfully processed, enabling efficient idempotent re-runs after interruptions.

#### Scenario: Completed works skipped on resume

- **WHEN** --resume-from manifest.jsonl used and manifest contains work W123 with status 'success'
- **THEN** system skips W123 during iteration, logs "Skipping W123 (already completed)", processes only missed works

#### Scenario: Failed works retried on resume

- **WHEN** manifest contains work W456 with status 'miss'
- **THEN** system re-attempts W456 with full resolver pipeline

### Requirement: HTML Text Extraction

The system SHALL optionally extract plaintext from HTML fallbacks using trafilatura when --extract-html-text flag is set, saving *.html.txt files alongside raw HTML for downstream parsers.

#### Scenario: HTML text extracted with trafilatura

- **WHEN** resolver returns HTML and --extract-html-text flag set
- **THEN** system saves raw HTML to *.html, extracts plaintext with trafilatura, saves to*.html.txt, logs extracted_text_path in manifest

#### Scenario: Trafilatura missing logs warning

- **WHEN** --extract-html-text set but trafilatura not installed
- **THEN** system logs warning "trafilatura not available, skipping text extraction", continues with raw HTML only

### Requirement: Extended Resolver Coverage

The system SHALL support optional OpenAIRE, HAL, and OSF Preprints resolvers (disabled by default) to expand EU OA and preprint coverage beyond existing 10 resolvers.

#### Scenario: OpenAIRE resolver enabled

- **WHEN** config sets `resolver_toggles: {"openaire": true}` and work has DOI
- **THEN** OpenAIRE resolver queries <https://api.openaire.eu/search/publications?doi=>... and yields PDF links from instances

#### Scenario: HAL resolver for French OA

- **WHEN** HAL resolver enabled and work has DOI
- **THEN** resolver queries <https://api.archives-ouvertes.fr/search/?q=doiId_s>:... and yields file_s PDF URLs

#### Scenario: OSF Preprints resolver

- **WHEN** OSF resolver enabled and work has DOI
- **THEN** resolver queries <https://api.osf.io/v2/preprints/?filter[doi]=>... and yields links.download URLs

#### Scenario: New resolvers disabled by default

- **WHEN** config does not explicitly enable openaire/hal/osf
- **THEN** these resolvers are skipped, no API calls made, no impact on default runs

## MODIFIED Requirements

### Requirement: Enhanced User-Agent for Crossref Compliance

The system SHALL include mailto address directly in User-Agent string for Crossref API requests (in addition to separate header) following Crossref best practices: `DocsToKGDownloader/1.0 (+mailto:user@example.com)`.

#### Scenario: User-Agent includes mailto

- **WHEN** Crossref resolver makes API request with config.mailto set
- **THEN** User-Agent header is `DocsToKGDownloader/1.0 (+mailto:user@example.com; mailto:user@example.com)`

#### Scenario: Crossref polite pool access

- **WHEN** properly formatted User-Agent with mailto sent to Crossref API
- **THEN** requests are processed via Crossref polite pool with higher rate limits and priority

### Requirement: Refactored Download State Machine

The system SHALL use explicit DownloadState enum (PENDING/WRITING) in `download_candidate()` with single outcome-building function to improve readability and testability while preserving all existing validation (EOF check, sniff logic).

#### Scenario: State transitions from PENDING to WRITING

- **WHEN** streaming download and payload classification completes (detected = 'pdf')
- **THEN** state transitions from DownloadState.PENDING to DownloadState.WRITING, file handle opened

#### Scenario: Single outcome builder invoked

- **WHEN** download completes or errors
- **THEN** `_build_download_outcome()` function constructs DownloadOutcome with consistent fields (classification, path, http_status, sha256, content_length, elapsed_ms)

#### Scenario: Existing validations preserved

- **WHEN** PDF downloaded
- **THEN** EOF check (`%%EOF` in tail bytes) still executed, corrupt PDFs detected and rejected as before

### Requirement: Shared Utility Functions

The system SHALL extract duplicated normalization utilities (`normalize_doi`, `normalize_pmcid`, `strip_prefix`, `dedupe`) from `download_pyalex_pdfs.py` and `resolvers/__init__.py` into `ContentDownload/utils.py` to reduce code duplication by ~15% and improve maintainability.

#### Scenario: DOI normalization reused

- **WHEN** both main module and resolvers need to normalize DOI
- **THEN** both import `from ContentDownload.utils import normalize_doi` and use same implementation

#### Scenario: Dedupe preserves order

- **WHEN** `utils.dedupe(['b', 'a', 'b', 'c'])` called
- **THEN** returns `['b', 'a', 'c']` preserving first occurrence order

## REMOVED Requirements

None. All existing requirements remain, only enhanced with additional capabilities.
