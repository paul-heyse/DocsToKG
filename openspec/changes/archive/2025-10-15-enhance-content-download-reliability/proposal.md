# Enhance Content Download Reliability and Performance

## Why

The Content Download component at `src/DocsToKG/ContentDownload` currently exhibits several reliability and performance issues that impact large-scale PDF acquisition workflows. These issues manifest as compounded retry behavior causing unpredictable latency, redundant network calls increasing bandwidth costs, race conditions in multi-threaded logging producing corrupted output files, and inefficient disk I/O patterns that degrade performance on large downloads.

Field observations from production crawls reveal that the double-retry mechanism (HTTP adapter retries compounded with application-level retries) creates non-deterministic backoff behavior that makes capacity planning difficult and occasionally violates rate limit agreements with external APIs. The redundant HEAD request pattern generates unnecessary preflight traffic that doubles network calls for successful downloads. Thread safety issues in logging infrastructure have resulted in interleaved JSONL records and corrupted CSV files when operating with multiple workers. The post-download SHA-256 computation requires reading large files twice from disk, adding measurable overhead for multi-gigabyte PDFs.

These architectural deficiencies limit the system's ability to scale reliably and efficiently handle production workloads while maintaining good standing with external content providers.

## What Changes

This proposal introduces thirteen coordinated improvements to the Content Download pipeline, organized into functional categories:

### Retry and Session Management

- **Centralize retry mechanism**: Remove `urllib3.Retry` configuration from HTTP adapter in favor of exclusive use of the centralized `http.request_with_retries` function, eliminating compounded backoff and ensuring deterministic retry counts
- **Simplify session construction**: Configure HTTP adapters with zero retries (`max_retries=0`) to delegate all retry logic to the application layer

### Network Efficiency

- **Eliminate redundant HEAD requests**: Remove the preliminary HEAD request in `download_candidate` function since the pipeline already provides HEAD precheck capability and content classification occurs through streaming sniff buffer analysis
- **Refactor Crossref resolver HTTP calls**: Migrate Crossref resolver to use centralized `request_with_retries` helper instead of direct session methods, ensuring consistent retry behavior across all resolvers

### Thread Safety and Data Integrity

- **Implement thread-safe logging**: Add `threading.Lock` synchronization to `JsonlLogger` and `CsvAttemptLoggerAdapter` classes to prevent concurrent write interleaving
- **Add context manager protocol**: Enhance `JsonlLogger` with `__enter__` and `__exit__` methods to support resource-safe usage patterns
- **Stream-based hash computation**: Calculate SHA-256 digests and byte counts incrementally during initial file write operations, eliminating the need to reopen and re-read downloaded files

### Content Validation and Classification

- **Early corruption detection**: Implement size threshold checks and HTML content detection in PDF validation logic to identify corrupted or mislabeled files before they pollute the corpus
- **Robust filename extraction**: Add helper function to infer correct file extensions from `Content-Disposition` headers, `Content-Type` MIME types, and URL patterns while preserving deterministic base filename stems
- **Enhanced DOI normalization**: Extend DOI parsing to handle additional common prefixes (`http://doi.org/`, `dx.doi.org`, `doi:`) beyond the current `https://doi.org/` pattern

### Configuration and Observability

- **Expand CLI options**: Add command-line flags for concurrent resolver configuration (`--concurrent-resolvers`), HEAD precheck toggling (`--head-precheck`/`--no-head-precheck`), and custom Accept headers (`--accept`)
- **Machine-readable run summaries**: Export structured metrics to sidecar JSON files alongside manifest JSONL logs, enabling automated dashboard generation and operational monitoring
- **Optional global URL deduplication**: Provide opt-in cross-work URL tracking to prevent redundant downloads of assets shared across multiple scholarly works

### Code Organization

- **Decouple resolver utilities**: Extract shared `headers_cache_key` utility from Unpaywall resolver into standalone module to eliminate hidden cross-resolver dependencies
- **Phase out legacy exports**: Document deprecation timeline for convenience re-exports of standard library modules (`time`, `requests`) from resolver package namespace

## Impact

**Affected specs:**

- `content-download` (new spec being introduced by this change)

**Affected code:**

- `src/DocsToKG/ContentDownload/download_pyalex_pdfs.py` — Core download orchestration and CLI
- `src/DocsToKG/ContentDownload/utils.py` — Identifier normalization utilities
- `src/DocsToKG/ContentDownload/resolvers/pipeline.py` — Resolver orchestration engine
- `src/DocsToKG/ContentDownload/resolvers/types.py` — Configuration dataclasses
- `src/DocsToKG/ContentDownload/resolvers/providers/crossref.py` — Crossref API integration
- `src/DocsToKG/ContentDownload/resolvers/providers/headers.py` — New shared utility module
- `src/DocsToKG/ContentDownload/resolvers/__init__.py` — Package facade
- `src/DocsToKG/ContentDownload/resolvers/cache.py` — Cache invalidation utilities

**Breaking changes:**
None. All changes maintain backward compatibility through optional flags, internal refactoring, or deprecation warnings with grace periods.

**Testing requirements:**

- Concurrency stress tests for logging infrastructure (16 threads × 1000 records)
- Retry determinism verification with stub servers returning controlled status sequences
- SHA-256 computation accuracy validation against reference implementations
- File corruption detection accuracy tests with pathological payloads
- CLI argument parsing round-trip verification
- Performance benchmarking for large file downloads (before/after comparisons)

**Deployment considerations:**

- Changes can be deployed incrementally as each sub-component becomes available
- No configuration migration required; new features are opt-in
- Existing manifest JSONL files remain compatible
- Resolver cache clearing may be advisable after Crossref refactoring to ensure fresh behavior
