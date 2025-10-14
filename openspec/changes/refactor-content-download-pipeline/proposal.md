## Why

The ContentDownload module (`src/DocsToKG/ContentDownload`) successfully downloads PDFs from OpenAlex with resolver fallbacks, but lacks operational polish that causes reliability issues in production: transient HTTP failures abort downloads, partial files corrupt storage on kill/timeout, rate-limit semantics are ambiguous leading to misconfiguration, and duplicate code/logging adds maintenance burden. Additionally, single-threaded work processing leaves throughput on the table despite per-resolver rate limits being respected.

## What Changes

### Quick Wins (Operational Polish)

- Mount HTTPAdapter with urllib3 Retry on shared session to handle transient 429/502/503/504 with exponential backoff and Retry-After header support, eliminating one-off failures
- Implement atomic file writes using *.part + os.replace() pattern, and compute SHA-256 digests + content-length for manifest integrity verification and deduplication
- Rename `resolver_rate_limits` config field to `min_interval_s` for unambiguous units (seconds between calls, not QPS)
- Enhance Crossref User-Agent to include mailto: directly in UA string per Crossref best practices (in addition to separate header)
- Add in-memory LRU cache for resolver API responses (Unpaywall/Crossref/S2) keyed by (resolver, DOI) to eliminate duplicate lookups within batch runs
- Support conditional requests (If-None-Match/If-Modified-Since) for idempotent re-runs using ETag/Last-Modified from manifest

### Streamlining (Reduce Code, Same Behavior)

- Extract duplicated normalization utilities (`_normalize_doi`, `_normalize_pmcid`, `_strip_prefix`) from both `download_pyalex_pdfs.py` and `resolvers/__init__.py` into `ContentDownload/utils.py`
- Create reusable `dedupe(seq)` utility for "preserve order while deduplicating" pattern used across location URL collection and resolver link lists
- Standardize OpenAlex candidate handling to always return structured result `{outcome, url, html_paths[]}` instead of mixed tuple/(outcome, url) returns
- Unify CsvAttemptLogger + ManifestLogger into single JSONL logger with per-attempt records and per-work summaries (CSV export remains available)
- Refactor `download_candidate` to use state enum (PENDING/WRITING) and single outcome-building function, reducing branching while preserving all validation

### Functionality Upgrades

- Add `--workers N` flag for bounded ThreadPoolExecutor parallelism across works while maintaining sequential per-work pipeline and per-resolver rate limiting (2-5x throughput gain)
- Implement `--dry-run` mode to measure resolver coverage without writing files, and `--resume-from manifest.jsonl` to only process previously-missed works
- Enhance HTML fallback with readability/trafilatura extraction to save plaintext (`*.html.txt`) alongside raw HTML for downstream parsers
- Document topic resolution default behavior (text → Topic ID resolution for improved recall)
- Add optional OpenAIRE, HAL, and OSF Preprints resolvers for expanded EU OA and preprint coverage

### Testing & Observability

- Add tests for 429 handling + Retry-After, corrupt HTML with wrong Content-Type, Wayback "not available" path, manifest/attempts alignment
- Ensure polite headers propagate to pre-pipeline OpenAlex attempts for consistency
- Update documentation with rate limit unit clarifications and new CLI flags

## Impact

- Affected specs: existing `content-download` capability (already deployed, now enhanced)
- Affected code: `src/DocsToKG/ContentDownload/download_pyalex_pdfs.py`, `src/DocsToKG/ContentDownload/resolvers/__init__.py`, new `src/DocsToKG/ContentDownload/utils.py`
- **BREAKING**: Rename `resolver_rate_limits` → `min_interval_s` in YAML configs; provide migration guide
- **BREAKING**: JSONL logging replaces separate CSV/manifest files; CSV export script provided for backward compatibility
- Improved reliability: automatic retry on transient failures, no partial file corruption
- Reduced maintenance: ~15% code reduction via shared utilities and unified logging
- Increased throughput: 2-5x with `--workers` flag (opt-in)
- Enhanced observability: SHA-256 digests in manifests, conditional request support for idempotency
- **Implementation Detail**: 120 tasks include complete code blocks, function signatures, exact line numbers, and test templates for unambiguous AI agent execution
