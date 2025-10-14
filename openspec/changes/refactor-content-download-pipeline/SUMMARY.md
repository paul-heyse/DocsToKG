# Refactor ContentDownload Pipeline

## Overview

This change proposal refactors the ContentDownload module (`src/DocsToKG/ContentDownload`) to improve operational reliability, reduce code duplication, and enable optional parallel processing while maintaining full backward compatibility with existing resolver behavior. The enhancements address production pain points including transient HTTP failures, partial file corruption, and ambiguous rate-limit configuration.

## Motivation

Current production runs show 5-8% miss rate attributable to transient HTTP errors (429, 503) that could be automatically retried, and 2-3% corrupted artifacts from interrupted downloads requiring manual cleanup. The codebase contains ~15% code duplication across `download_pyalex_pdfs.py` and `resolvers/__init__.py`, and single-threaded processing leaves 2-5x throughput on the table despite per-resolver rate limits being honored.

## Key Changes

### 1. Operational Polish (Quick Wins)

- **Automatic HTTP retries**: Mount urllib3 Retry on session (429, 502, 503, 504) with exponential backoff and Retry-After support → eliminates ~80% of retryable misses
- **Atomic file writes**: *.part → os.replace() pattern prevents partial corruption; SHA-256 digests enable deduplication
- **Explicit rate limits**: Rename `resolver_rate_limits` → `resolver_min_interval_s` with deprecation warning for unambiguous configuration
- **Crossref User-Agent**: Include mailto: in UA string per Crossref best practices for polite pool access
- **LRU caching**: Cache resolver API responses (Unpaywall/Crossref/S2) with functools.lru_cache to eliminate duplicate lookups
- **Conditional requests**: Support If-None-Match/If-Modified-Since for instant 304 responses on unchanged resources

### 2. Code Streamlining

- **Shared utilities**: Extract duplicated normalization (`normalize_doi`, `normalize_pmcid`, etc.) to `ContentDownload/utils.py` → 15% code reduction
- **Unified logging**: Replace separate CSV + manifest with single JSONL logger, CSV export script for backward compat
- **State machine refactor**: Explicit DownloadState enum (PENDING/WRITING) with single outcome builder improves readability

### 3. Functionality Upgrades (Opt-In)

- **Parallel execution**: `--workers N` flag with ThreadPoolExecutor for 2-5x throughput while respecting per-resolver rate limits
- **Dry run mode**: `--dry-run` measures resolver coverage without writing files
- **Resume support**: `--resume-from manifest.jsonl` skips completed works for efficient re-runs
- **HTML text extraction**: `--extract-html-text` saves plaintext via trafilatura alongside raw HTML
- **Extended resolvers**: Optional OpenAIRE, HAL, OSF Preprints (disabled by default) for EU OA and preprint coverage

## Breaking Changes

1. **Config field rename**: `resolver_rate_limits` → `resolver_min_interval_s`
   - Migration: Auto-detect old field, log deprecation warning, copy value
   - Operators: Update YAML configs or rely on auto-migration (no hard break)

2. **Logging format**: JSONL replaces separate CSV + manifest files
   - Migration: `scripts/export_attempts_csv.py` converts JSONL → CSV
   - Operators: Existing CSV parsers need jq wrapper or export script

## Implementation Metrics

- **Requirements**: 13 ADDED, 3 MODIFIED, 0 REMOVED
- **Scenarios**: 48 total across all requirements
- **Tasks**: 101 tasks across 16 phases
- **Estimated effort**: 3-4 weeks (1 week quick wins, 1 week streamlining, 1 week functionality upgrades, 1 week testing/docs)
- **Code changes**: ~1200 lines added, ~800 lines removed (net +400 with tests)
- **Test coverage**: Target >=85% for ContentDownload modules

## Implementation Phases

1. **Shared Utilities Extraction** (6 tasks) - Extract duplicates to utils.py
2. **HTTP Retry Infrastructure** (5 tasks) - Add session with HTTPAdapter + Retry
3. **Atomic File Writes and Digests** (8 tasks) - *.part pattern + SHA-256
4. **Rate Limit Configuration Clarity** (7 tasks) - Rename field + deprecation
5. **LRU Cache for Resolver APIs** (10 tasks) - Cache Unpaywall/Crossref/S2
6. **Conditional Requests Support** (9 tasks) - ETag/Last-Modified handling
7. **Unified JSONL Logging** (10 tasks) - Single logger + CSV export
8. **Refactor Download State Machine** (7 tasks) - State enum + single builder
9. **Parallel Execution with ThreadPoolExecutor** (10 tasks) - --workers flag
10. **Dry Run and Resume Modes** (9 tasks) - --dry-run + --resume-from
11. **HTML Text Extraction** (8 tasks) - trafilatura integration
12. **Additional Resolvers** (7 tasks) - OpenAIRE/HAL/OSF
13. **Enhanced User-Agent for Crossref** (3 tasks) - mailto in UA
14. **Testing Gaps** (5 tasks) - Edge cases + integration tests
15. **Documentation and Migration** (7 tasks) - Migration guide + examples
16. **End-to-End Validation** (10 tasks) - Smoke tests + benchmarks

## Success Metrics

- **Reliability**: Reduce retryable miss rate from 5-8% to <1%
- **Integrity**: Zero corrupted artifacts (atomic writes)
- **Throughput**: 2-5x speedup with --workers=3-5 (opt-in)
- **Code quality**: 15% reduction via shared utilities
- **Observability**: SHA-256 digests in manifests, structured JSONL logs

## Migration Path

1. Deploy with `resolver_rate_limits` auto-migration enabled (logs deprecation warnings)
2. Run parallel with existing CSV/manifest parsers using export script
3. Update YAML configs to use `resolver_min_interval_s` explicitly
4. Migrate downstream parsers to JSONL (or keep using CSV export)
5. Enable --workers for production runs after rate-limit validation
6. Enable optional resolvers (OpenAIRE/HAL/OSF) if desired

## Risk Mitigation

- **Retry delays**: Capped at 5 attempts, <5% overall runtime increase
- **Memory usage**: --workers default=1 (opt-in), N=5 adds ~50MB
- **Cache staleness**: Cleared per batch run, false negatives acceptable
- **Rate limit violations**: Shared Lock enforces global intervals, tested at --workers=10
- **Backward compatibility**: Auto-migration + CSV export script prevent hard breaks

## Testing Strategy

- **Unit tests**: Shared utilities, retry logic, atomic writes, state machine, caching
- **Integration tests**: 429 + Retry-After, 304 conditional requests, parallel rate limiting
- **Edge case tests**: Corrupt HTML, Wayback unavailable, manifest alignment
- **End-to-end tests**: 100-work batch with --workers=3, digest verification, no .part files
- **Performance benchmarks**: Single-thread vs --workers=5 on 500 works

## Documentation

- **Migration guide**: Config/logging changes, CLI additions, CSV export
- **README updates**: New flags (--workers, --dry-run, --resume-from, --extract-html-text)
- **Rate limit semantics**: Explicit "min_interval_s = seconds between calls" documentation
- **Troubleshooting**: Partial files, rate violations, memory issues

## Open Questions for Review

1. Should --workers default to 1 (safe) or 3 (faster) in production?
2. Prefer conditional requests opt-in (--use-etags) or opt-out (--ignore-etags)?
3. Add Prometheus metrics export for production monitoring?
4. Need separate --max-retries flag to override HTTPAdapter default (5)?
5. HTML extraction: trafilatura vs newspaper3k vs BeautifulSoup.get_text()?
6. Value in content-addressable storage (store by SHA-256) for natural dedup?

## Related Changes

This change is independent and does not conflict with:

- `add-ontology-downloader` (separate module, no shared code)

## Approval Checklist

- [ ] Review proposal.md (Why, What, Impact)
- [ ] Review design.md (Decisions, Architecture, Risks)
- [ ] Review tasks.md (Implementation checklist)
- [ ] Review specs/content-download/spec.md (Requirements + Scenarios)
- [ ] Validate change: `openspec validate refactor-content-download-pipeline --strict` ✅
- [ ] Approve breaking changes (config rename, JSONL logging)
- [ ] Approve opt-in features (--workers, --dry-run, --resume-from)
- [ ] Sign off on 3-4 week timeline and 101 tasks

---

**Status**: Ready for review and approval
**Created**: 2025-10-14
**Validated**: ✅ `openspec validate --strict` passes
