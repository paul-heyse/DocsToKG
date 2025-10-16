# Design Document: ContentDownload Robustness Refactoring

## Context

The ContentDownload module implements a multi-resolver pipeline for acquiring open-access PDFs from academic works indexed by OpenAlex. The current implementation evolved through incremental additions resulting in duplicate code paths, inconsistent HTTP behavior across resolvers, and legacy compatibility layers that obscure the core business logic. This refactoring addresses technical debt while preserving the module's architectural independence from downstream parsing stages.

### Stakeholders

- **AI Coding Agents**: Primary users of DocsToKG for research data acquisition
- **Researchers**: Consumers of downloaded PDF corpora
- **Maintainers**: Developers extending resolver implementations or modifying pipeline behavior

### Constraints

- Must preserve independence from DocParsing module (no coupling to parsing logic)
- Cannot break manifest JSONL format (resume functionality depends on it)
- Must maintain backward compatibility for resolver configuration files
- HTTP behavior changes must not violate rate limits or politeness requirements
- All refactoring must be testable without external API calls in unit tests

### Background

The module consists of four key components:

1. **CLI Orchestrator** (`download_pyalex_pdfs.py`): Argument parsing, manifest logging, work iteration
2. **Network Layer** (`network.py`): Session factory, HTTP retries with backoff, conditional requests
3. **Resolver Infrastructure** (`resolvers.py`): Pipeline orchestration, 15+ resolver implementations, rate limiting
4. **Utilities** (`utils.py`): Normalization helpers for DOI, PMCID, arXiv identifiers

The refactoring focuses on consolidating duplicate implementations, removing dead code paths, and establishing uniform patterns while preserving all existing capabilities.

## Goals

### Primary Goals

1. **Eliminate duplicate implementations**: Single HEAD precheck, single context manager lifecycle, single resolver HTTP pattern
2. **Remove legacy compatibility layers**: Delete `__getattr__` shims, proxy functions, and session-less fallbacks
3. **Standardize HTTP behavior**: All resolvers use unified retry logic, timeouts, and headers
4. **Improve resource management**: Deterministic file handle cleanup, idempotent close operations
5. **Reduce code volume**: Target 400-600 line reduction through abstraction and removal

### Secondary Goals

1. **Enhanced observability**: Staging directories, derived indexes, last-attempt summaries
2. **Improved robustness**: Jittered throttling, pre-validated conditional requests, hardened detection
3. **Better developer experience**: Clearer CLI help, protocol-based composition, documented patterns

## Non-Goals

- Introducing new resolver implementations (preserves existing 15 resolvers)
- Changing manifest or attempt record schemas (backward compatibility required)
- Performance optimization beyond throttling improvements (functionality focus)
- Integration with parsing stage (maintains independence)
- Distributed or cloud-native execution (local execution model unchanged)

## Decisions

### Decision 1: Unified HEAD Precheck in Network Layer

**Choice**: Move HEAD preflight logic from both CLI and pipeline into single `network.head_precheck()` function

**Rationale**:

- Network layer already centralizes retry and session logic
- Both existing implementations use identical timeout budgets and status code checks
- Single implementation enables consistent testing and behavior
- Natural layering: network helpers serve both CLI and pipeline

**Alternatives Considered**:

- Keep separate implementations: Rejected due to maintenance overhead and drift risk
- Move to resolvers module: Rejected because HEAD checks are HTTP concern, not resolver concern
- Make it a method on ResolverPipeline: Rejected because CLI also needs it independently

**Trade-offs**:

- ✓ Single source of truth for HEAD behavior
- ✓ Easier to add 405/501 degradation logic once
- ✗ Small increase in network module surface area (acceptable, still cohesive)

**Implementation Notes**:

- Function signature: `head_precheck(session: requests.Session, url: str, timeout: float) -> bool`
- Returns True on exceptions (conservative pass-through)
- Optional degradation to short GET with `stream=True` for HEAD-hostile servers
- Pipeline consults `ResolverConfig.resolver_head_precheck` before calling

### Decision 2: ApiResolverBase for JSON API Pattern

**Choice**: Introduce base class with `_request_json()` helper for resolvers querying JSON APIs

**Rationale**:

- Seven resolvers (DOAJ, Zenodo, EuropePMC, HAL, OSF, OpenAIRE, Core) share identical pattern:
  1. Build headers from config
  2. GET/POST to JSON endpoint
  3. Parse response with error handling
  4. Extract URLs from structured data
- Current implementations duplicate error handling logic (timeouts, connection errors, JSON parse errors)
- Abstraction reduces code volume and makes resolver additions simpler

**Alternatives Considered**:

- Keep each resolver fully independent: Rejected due to code duplication and maintenance burden
- Create standalone helper function: Rejected because method better encapsulates resolver context
- Use mixin instead of base class: Considered equivalent; chose base class for clarity
- Build full resolver framework: Rejected as over-engineering; minimal base class sufficient

**Trade-offs**:

- ✓ 100-200 line reduction across refactored resolvers
- ✓ Uniform error event generation and logging
- ✓ Easier to add new API-based resolvers
- ✗ Additional class in hierarchy (acceptable complexity)
- ✗ Scrapers (landing page, Wayback) don't benefit (acceptable, different pattern)

**Implementation Notes**:

- `_request_json()` returns `(data, None)` on success or `(None, error_event)` on failure
- Error events include `event_reason` distinguishing timeout/connection/http/json errors
- Preserves existing timeout and header customization capabilities
- Refactor proceeds incrementally (DOAJ, Zenodo, EuropePMC first, then others)

### Decision 3: AttemptSink Protocol with MultiSink Composition

**Choice**: Replace CSV adapter wrapping JSONL logger with protocol-based sink composition

**Rationale**:

- Current adapter asymmetry: JSONL is primary, CSV wraps it, leading to special-case methods
- Protocol-based design enables clean composition: `MultiSink([JsonlSink(...), CsvSink(...)])`
- Future extensibility: Can add database sink, metrics sink, etc. without adapter chains
- Symmetric design: All sinks equal, composed uniformly

**Alternatives Considered**:

- Keep adapter pattern: Rejected due to asymmetry and single-responsibility violation
- Use observer pattern: Considered equivalent; chose explicit sink list for clarity
- Single monolithic logger: Rejected; protocol enables independent testing of each sink

**Trade-offs**:

- ✓ Clean separation of concerns (JSONL vs CSV logic independent)
- ✓ Easier testing (mock sink protocol, not wrapped adapters)
- ✓ Future extensibility (add sinks without modifying existing code)
- ✗ Small protocol definition overhead (minimal, worthwhile)
- ✗ Requires refactoring main() setup (one-time cost)

**Implementation Notes**:

- `AttemptSink` protocol requires: `log_attempt()`, `log_manifest()`, `log_summary()`, `close()`
- `MultiSink` iterates sink list for each method call
- `JsonlSink` is renamed `JsonlLogger` (minimal change)
- `CsvSink` owns its file handle directly (no logger wrapping)

### Decision 4: Remove Legacy Exports and Session-Less Branches

**Choice**: Delete `__getattr__` shims, `request_with_retries` proxy, and `hasattr(session, "get")` fallbacks

**Rationale**:

- `__getattr__` exports (`time`, `requests`) were temporary compatibility layer; codebase matured
- `request_with_retries` proxy serves no purpose (no import cycle exists)
- Session-less branches are dead code (tests always provide session, `hasattr` always True)
- Removal reduces cognitive overhead and maintenance surface

**Alternatives Considered**:

- Deprecate gradually: Rejected; already deprecated, time to remove
- Keep for external users: Rejected; external usage should be minimal, breaking change acceptable
- Convert to runtime errors: Rejected; clean break clearer than runtime surprises

**Trade-offs**:

- ✓ Simpler module structure
- ✓ No more duplicate retry logic
- ✓ Clearer dependency graph
- ✗ Breaking change for external importers (documented in CHANGELOG)
- ✗ External code must update imports (low impact, easily fixed)

**Migration Path**:

```python
# Old (breaks)
from DocsToKG.ContentDownload.resolvers import request_with_retries, time

# New
from DocsToKG.ContentDownload.network import request_with_retries
import time
```

### Decision 5: Conditional Request Pre-Validation

**Choice**: Check metadata completeness in `build_headers()` and emit empty dict if incomplete

**Rationale**:

- Current behavior: incomplete metadata causes 304 → ValueError at response interpretation
- Pre-validation provides clearer error message and forces fresh 200 fetch earlier
- Avoids network round-trip with malformed conditional headers

**Alternatives Considered**:

- Keep validation in `interpret_response()`: Rejected; fails after network call
- Raise exception in `build_headers()`: Rejected; prefer graceful degradation
- Warn but send conditional headers anyway: Rejected; 304 would still fail

**Trade-offs**:

- ✓ Clearer error messages ("resume-metadata-incomplete" upfront)
- ✓ Avoids network call when request would fail
- ✓ Graceful fallback to full fetch
- ✗ Slightly more complex header builder (acceptable)

### Decision 6: Staging Directory Mode (Opt-In)

**Choice**: Add `--staging` flag creating timestamped run directories with isolated artifacts

**Rationale**:

- Large-scale crawls benefit from run isolation (compare coverage, rollback easily)
- Current flat directory structure complicates multi-run analysis
- Staging mode optional: default behavior unchanged

**Alternatives Considered**:

- Always use staging: Rejected; breaking change, users may depend on flat structure
- Use timestamps in filenames: Rejected; directory structure cleaner for side-by-side reviews
- Database-backed staging: Rejected; filesystem sufficient, maintains simplicity

**Trade-offs**:

- ✓ Clean run isolation
- ✓ Side-by-side comparison of resolver coverage across runs
- ✓ Trivial rollback (delete directory)
- ✗ Slightly more complex path logic (acceptable)
- ✗ Opt-in means users must discover feature (document in help text)

**Implementation Notes**:

- Directory format: `{out}/YYYYMMDD_HHMM/{PDF,HTML,manifest.jsonl}`
- UTC timestamps avoid timezone confusion
- Non-staging mode unchanged (default `out/` directory)

## Risks and Trade-offs

### Risk: Breaking Changes for External Consumers

**Likelihood**: Medium
**Impact**: Low-Medium
**Mitigation**:

- Document all breaking changes in CHANGELOG with migration examples
- Provide deprecation period for `--log-path` (warn but keep working)
- Clear error messages if external code imports legacy symbols

### Risk: HTTP Behavior Changes Surface Resolver Bugs

**Likelihood**: Low-Medium
**Impact**: Medium
**Mitigation**:

- Comprehensive integration testing with dry-run coverage checks
- Phased rollout: correctness fixes first, HTTP standardization second, abstractions third
- Preserve per-resolver timeout and head-precheck overrides for safety valves

### Risk: ApiResolverBase Doesn't Fit All API Resolvers

**Likelihood**: Low
**Impact**: Low
**Mitigation**:

- Refactor 2-3 resolvers first, evaluate fit before continuing
- Base class remains optional; resolvers can skip inheritance if needed
- `_request_json()` helper easily overridden for special cases

### Risk: Staging Mode Path Logic Bugs

**Likelihood**: Low
**Impact**: Medium (data loss if paths wrong)
**Mitigation**:

- Integration tests explicitly verify staging directory structure
- Idempotent `ensure_dir()` prevents failures if directories exist
- Keep non-staging path logic unchanged (regression risk isolated)

## Migration Plan

### Phase 1: Correctness Fixes (Low Risk)

1. Consolidate CLI context management
2. Fix CSV close() resource leak
3. Remove duplicate session factory returns
4. Test: dry-run with CSV logging on Windows/Linux/macOS

### Phase 2: Legacy Removal (Breaking but Clean)

1. Delete `__getattr__` shims and update internal imports
2. Remove `request_with_retries` proxy
3. Update external usage documentation
4. Test: ensure no internal broken imports

### Phase 3: HTTP Standardization (Medium Risk)

1. Unify HEAD precheck in network layer
2. Remove session-less resolver branches
3. Standardize timeout and header usage
4. Test: dry-run coverage comparison before/after

### Phase 4: Code Reduction (Medium Risk)

1. Introduce ApiResolverBase
2. Refactor DOAJ, Zenodo, EuropePMC resolvers
3. Extract HTML scraping helpers
4. Test: resolver-specific unit tests

### Phase 5: Observability (Low Risk, Opt-In)

1. Add staging directory mode
2. Generate manifest indexes
3. Produce last-attempt CSV
4. Test: staging mode with small dataset

### Phase 6: Robustness (Low Risk)

1. Add conditional request pre-validation
2. Implement jittered domain throttling
3. Harden PDF detection heuristics
4. Test: edge cases (partial metadata, tiny PDFs, octet-stream)

### Rollback Strategy

- Each phase independently committable
- Phases 1-3 improve correctness without changing capabilities (safe to keep)
- Phases 4-6 opt-in or backward-compatible (can be reverted individually if needed)
- Manifest format unchanged (old manifests always work)

## Open Questions

1. **Should we add explicit caching layer above `request_with_retries` for API resolvers?**
   - Current decision: No explicit cache; removed LRU caches for transparency
   - Alternative: Small per-run LRU cache keyed by (url, params)
   - Resolution: Start without cache; add if profiling shows redundant API calls

2. **Should HEAD-to-GET degradation be opt-in or automatic?**
   - Current decision: Automatic for 405/501 status codes
   - Alternative: Require explicit flag `--head-degrade-on-405`
   - Resolution: Automatic (conservative, improves coverage); per-resolver disable remains available

3. **Should we provide migration script for external import updates?**
   - Current decision: No script, manual migration with clear docs
   - Alternative: `import_updater.py` scanning codebase for legacy imports
   - Resolution: Manual sufficient; breaking change limited to importers

4. **Should ApiResolverBase support paginated API responses?**
   - Current decision: No pagination support; resolvers handle manually if needed
   - Alternative: `_paginated_request_json()` helper yielding pages
   - Resolution: Wait for concrete use case; current resolvers don't need it

## Success Metrics

### Code Quality

- **Target**: 400-600 line reduction across ContentDownload module
- **Measure**: Line count before/after via `cloc` or similar

### Correctness

- **Target**: Zero new bugs introduced (dry-run coverage metrics unchanged)
- **Measure**: Compare resolver success rates before/after on reference dataset

### Maintainability

- **Target**: 30% reduction in duplicate code patterns
- **Measure**: Manual audit of HTTP call patterns, error handling blocks

### Testing

- **Target**: 90%+ test coverage for new/modified code
- **Measure**: pytest coverage report for ContentDownload module

### Documentation

- **Target**: All breaking changes documented with migration examples
- **Measure**: CHANGELOG completeness audit

## Timeline Estimate

- Phase 1 (Correctness): 1-2 days
- Phase 2 (Legacy Removal): 1 day
- Phase 3 (HTTP Standardization): 2-3 days
- Phase 4 (Code Reduction): 3-4 days
- Phase 5 (Observability): 2-3 days
- Phase 6 (Robustness): 1-2 days
- Testing & Documentation: 2-3 days

**Total**: 12-20 days for complete implementation
