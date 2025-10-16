# Refactor ContentDownload Robustness and Remove Legacy Code

## Why

The ContentDownload module (`src/DocsToKG/ContentDownload/`) contains critical infrastructure for acquiring open-access PDFs through resolver pipelines, but currently suffers from duplicate code paths, resource management issues, and legacy compatibility shims that increase cognitive overhead and maintenance burden. The module exhibits several high-impact correctness issues including duplicate context management blocks, unclosed file handles, and inconsistent HTTP retry behavior across resolvers. Additionally, approximately 30% of the resolver codebase consists of dead or near-duplicate code paths (session-less branches, duplicate HEAD precheck implementations, legacy export shims) that obscure the actual business logic and create maintenance hazards.

This refactoring consolidates duplicate implementations, removes legacy compatibility layers, standardizes HTTP behavior across all resolvers, and improves resource management—all while preserving the module's independence from downstream parsing stages. The changes eliminate merge artifacts, reduce code volume by an estimated 400-600 lines, and establish uniform patterns for resolver implementations that will simplify future additions.

## What Changes

### High-Impact Correctness Fixes (Priority 1)

- Consolidate duplicate CLI context management blocks in `main()` function that create conflicting logger lifecycles
- Fix CSV file handle resource leak in `CsvAttemptLoggerAdapter.close()` method
- Remove duplicate `_session_factory()` definitions and unreachable return statements
- Unify HEAD preflight check implementations into single `network.head_precheck()` function

### Legacy Code Removal (Priority 1)

- Remove `__getattr__` compatibility shims exporting `time` and `requests` modules with deprecation warnings
- Delete `resolvers.request_with_retries()` proxy function that unnecessarily wraps network helper
- Eliminate session-less resolver branches using `hasattr(session, "get")` checks and associated `_fetch_*` LRU cache functions
- Consolidate resolver toggle default definitions from multiple locations into single authoritative source

### HTTP Behavior Standardization (Priority 2)

- Replace all direct `session.get()` calls with unified `request_with_retries()` for consistent backoff and `Retry-After` handling
- Enforce use of `config.get_timeout(resolver_name)` for all resolver HTTP requests instead of mixed timeout strategies
- Ensure all resolver requests include `config.polite_headers` unless resolver-specific headers are required
- Implement optional HEAD-to-GET degradation for servers returning 405/501 status codes

### Code Reduction Through Abstraction (Priority 2)

- Introduce `ApiResolverBase` class providing `_request_json()` helper with standardized error handling
- Extract HTML scraping helpers (`find_pdf_via_meta`, `find_pdf_via_link`, `find_pdf_via_anchor`) from `LandingPageResolver`
- Refactor manifest/attempt logging from adapter pattern to `AttemptSink` protocol with `MultiSink` composition

### Robustness Enhancements (Priority 3)

- Add conditional request metadata validation before issuing requests to prevent late 304 failures
- Implement jittered domain throttling to prevent thundering herd across concurrent resolver threads
- Harden PDF detection to treat `application/octet-stream` as suspicious and require content sniffing
- Enhance corruption detection to flag PDFs missing HEAD precheck validation

### Observability Improvements (Priority 3)

- Add optional `--staging` mode creating timestamped run directories with isolated artifact collections
- Generate `manifest.index.json` derived index mapping work_id to best PDF path and SHA for fast resumption
- Produce `manifest.last.csv` containing final outcome per work for human review
- Group CLI resolver flags under dedicated argparse group for improved help text organization

## Impact

### Affected Specifications

- `content-download` (new capability specification)

### Affected Code

- `src/DocsToKG/ContentDownload/download_pyalex_pdfs.py` - CLI orchestration, main() consolidation, manifest logging
- `src/DocsToKG/ContentDownload/network.py` - unified HEAD precheck, session factory
- `src/DocsToKG/ContentDownload/resolvers.py` - removal of legacy exports, HTTP standardization, ApiResolverBase, all resolver implementations
- `src/DocsToKG/ContentDownload/utils.py` - no changes expected (normalization helpers remain independent)

### Breaking Changes

- **BREAKING**: External code importing `time` or `requests` from `DocsToKG.ContentDownload.resolvers` will fail
- **BREAKING**: External code importing `request_with_retries` from `resolvers` must change to import from `network`
- **BREAKING**: Custom resolver implementations using session-less patterns must provide `requests.Session` instances
- **BREAKING**: Removal of `--log-path` CLI flag (use `--manifest` instead)

### Migration Path

- For broken imports: Update to `from DocsToKG.ContentDownload.network import request_with_retries`
- For custom resolvers: Ensure `iter_urls()` implementations use provided session parameter
- For CLI users: Replace `--log-path PATH` with `--manifest PATH`

### Compatibility Preservation

- All resolver toggles and configuration options remain backward compatible
- Manifest JSONL format unchanged (new optional indexes are additive)
- Dry-run behavior semantics preserved
- Resume functionality continues working with existing manifest files
- PDF/HTML artifact output paths and naming unchanged

### Risk Assessment

- **Low risk**: Context management fixes, CSV close(), duplicate removal (fixes bugs, no behavioral change)
- **Medium risk**: HTTP standardization (behavior becomes uniform but may surface latent issues in specific resolvers)
- **Medium risk**: Legacy removal (breaking for external consumers but internal usage is unaffected)
- **Low risk**: Observability additions (opt-in features, no impact when unused)
