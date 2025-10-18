# Design: ContentDownload Refactoring

## Context

The `ContentDownload` module is responsible for downloading academic content (PDFs, HTML, XML) from multiple sources using a resolver-based architecture. The system integrates with OpenAlex for work discovery and supports 17+ content providers (Unpaywall, CrossRef, arXiv, PubMed Central, etc.).

Current implementation has grown organically to 11,875 lines across 13 modules, with significant architectural debt:

1. **Monolithic resolver registry**: `pipeline.py` contains 4,710 lines mixing registry logic with resolver implementations
2. **Impure configuration**: `resolve_config` performs I/O side effects during configuration resolution
3. **Coupled orchestration**: The runner loop combines operational concerns with business logic
4. **Complex download processing**: Large helper functions combine multiple concerns without clear boundaries

The refactoring addresses these issues while maintaining backward compatibility with existing CLI usage, configuration files, and telemetry formats.

### Stakeholders

- **Contributors**: Need clear boundaries to add new resolvers or artifact types
- **Operators**: Need configurable concurrency, resilient throttling, and telemetry
- **Researchers**: Need reliable content acquisition with resume capabilities
- **Maintainers**: Need testable, modular code with low cognitive load

### Constraints

- **No breaking changes** to public CLI interface
- **Backward compatible** with existing configuration files
- **Performance neutral** (within 5% of baseline)
- **Test coverage** maintained at >= 85%
- **Python 3.9+** compatibility

## Goals / Non-Goals

### Goals

1. **Modularize resolvers**: Split `pipeline.py` into focused resolver modules
2. **Pure configuration**: Make `resolve_config` side-effect-free and testable
3. **Composable orchestration**: Decompose runner loop into testable stages
4. **Extensible download processing**: Use strategy pattern for artifact types

### Non-Goals

1. **Resolver algorithm changes**: No changes to resolver logic or priority
2. **Telemetry format changes**: Maintain existing JSONL/CSV/SQLite formats
3. **New features**: No new artifact types or extraction flows (deferred)
4. **API redesign**: Public CLI interface remains unchanged

## Decisions

### Decision 1: Resolver Module Structure

**Choice**: Use flat module structure in `resolvers/` directory

```
ContentDownload/resolvers/
├── __init__.py          # Registry + exports
├── base.py              # Base classes
├── arxiv.py             # Concrete resolvers (one per file)
├── core.py
├── ...
```

**Alternatives considered**:

1. **Hierarchical grouping** (`resolvers/api/`, `resolvers/heuristic/`)
   - *Rejected*: Adds cognitive overhead without clear value; resolvers don't form natural hierarchies

2. **Single `resolvers.py` module with classes**
   - *Rejected*: Only reduces line count without addressing modularity concerns

3. **Plugin directory with dynamic discovery**
   - *Rejected*: Increases complexity for marginal benefit; static registry is sufficient

**Rationale**: Flat structure is simple, scales linearly with resolver count, and aligns with Python conventions. Each resolver is self-contained and can be tested independently.

### Decision 2: Configuration Immutability

**Choice**: Make `ResolvedConfig` frozen dataclass with separate `bootstrap_run_environment` function

```python
@dataclass(frozen=True)
class ResolvedConfig:
    ...

def resolve_config(args, parser) -> ResolvedConfig:
    # Pure computation, no side effects

def bootstrap_run_environment(resolved: ResolvedConfig) -> None:
    # Directory creation, telemetry setup
```

**Alternatives considered**:

1. **Mutable dataclass with lazy initialization**
   - *Rejected*: Doesn't solve testability; side effects still occur during config resolution

2. **Builder pattern with explicit `build()` method**
   - *Rejected*: Over-engineered for this use case; dataclass is sufficient

3. **Separate `Config` (immutable) and `RuntimeState` (mutable) classes**
   - *Rejected*: Increases complexity without clear benefit; single frozen dataclass is clearer

**Rationale**: Frozen dataclass enforces immutability at the type level, enables serialization/deserialization, and makes side effects explicit in the caller.

### Decision 3: Runner Orchestration Pattern

**Choice**: `DownloadRun` class with composable stage methods

```python
class DownloadRun:
    def __init__(self, resolved: ResolvedConfig): ...
    def setup_sinks(self) -> MultiSink: ...
    def setup_resolver_pipeline(self) -> ResolverPipeline: ...
    def run(self) -> RunResult: ...
```

**Alternatives considered**:

1. **Functional pipeline with generator composition**
   - *Rejected*: Python generators don't compose well with exception handling and resource management

2. **Builder pattern with chained method calls**
   - *Rejected*: Adds boilerplate without improving testability

3. **Separate orchestrator functions** (`setup_sinks()`, `setup_resolvers()`, `run_pipeline()`)
   - *Rejected*: Lacks cohesion; state must be threaded through multiple functions

**Rationale**: Class-based approach provides natural state management, clear stage boundaries, and straightforward testing via method mocking.

### Decision 4: Download Strategy Pattern

**Choice**: Protocol-based strategy with concrete implementations per artifact type

```python
class DownloadStrategy(Protocol):
    def should_download(...) -> bool: ...
    def process_response(...) -> Classification: ...
    def finalize_artifact(...) -> DownloadOutcome: ...

class PdfDownloadStrategy(DownloadStrategy): ...
class HtmlDownloadStrategy(DownloadStrategy): ...
```

**Alternatives considered**:

1. **Chain of Responsibility pattern**
   - *Rejected*: Overkill for classification-based dispatch; strategy is more direct

2. **Polymorphic artifact classes** (`PdfArtifact`, `HtmlArtifact`)
   - *Rejected*: Requires changing core `WorkArtifact` dataclass; too invasive

3. **Conditional branching with helper functions**
   - *Rejected*: Already exists; doesn't improve testability or extensibility

**Rationale**: Strategy pattern provides clean extension points for new artifact types, isolates classification-specific logic, and enables independent testing of each concern.

## Architecture Diagrams

### Current Architecture (Simplified)

```
┌─────────────┐
│   CLI       │
└──────┬──────┘
       │
       v
┌──────────────────┐
│ resolve_config   │ ← Mutates args, writes dirs, pulls env
└──────┬───────────┘
       │
       v
┌──────────────────┐
│ run()            │ ← Monolithic orchestration
│  - Setup sinks   │
│  - Setup resolvers│
│  - Worker loop   │
│  - Budget check  │
└──────┬───────────┘
       │
       v
┌──────────────────┐
│ pipeline.py      │ ← 4,710 lines, 17 resolvers + registry
└──────┬───────────┘
       │
       v
┌──────────────────┐
│ process_one_work │ ← Download + classification + logging
└──────────────────┘
```

### Target Architecture (Refactored)

```
┌─────────────┐
│   CLI       │
└──────┬──────┘
       │
       ├─────> resolve_config() → ResolvedConfig (immutable)
       │
       ├─────> bootstrap_run_environment() → Creates dirs
       │
       v
┌──────────────────┐
│ DownloadRun      │
│  .setup_sinks()  │ ← Composable stages
│  .setup_resolvers│
│  .run()          │
└──────┬───────────┘
       │
       v
┌──────────────────────────────────┐
│ resolvers/                       │
│  ├── base.py                     │ ← Base classes + helpers
│  ├── arxiv.py                    │ ← Focused modules (17 total)
│  ├── core.py                     │
│  └── ...                         │
└──────┬──────────────────────────┘
       │
       v
┌──────────────────────────────────┐
│ DownloadStrategy                 │
│  ├── PdfDownloadStrategy         │ ← Strategy per artifact type
│  ├── HtmlDownloadStrategy        │
│  └── XmlDownloadStrategy         │
└──────────────────────────────────┘
```

## Implementation Strategy

### Phase 1: Resolver Modularization (Week 1, Days 1-3)

**Objective**: Split `pipeline.py` into focused resolver modules

**Steps**:

1. Create `resolvers/` directory structure
2. Move `RegisteredResolver`, `ApiResolverBase` to `resolvers/base.py`
3. Extract each `*Resolver` class into dedicated module
4. Update imports in `pipeline.py`, `runner.py`, `args.py`, tests
5. Verify resolver auto-discovery works
6. Remove resolver implementations from `pipeline.py`

**Success criteria**:

- All tests pass
- `pipeline.py` reduced to ~800 lines
- Each resolver in separate module under `resolvers/`

**Rollback plan**: Revert commits, restore original `pipeline.py`

### Phase 2: Configuration Purity (Week 1, Days 4-5)

**Objective**: Make `resolve_config` side-effect-free

**Steps**:

1. Add `frozen=True` to `ResolvedConfig`
2. Create `bootstrap_run_environment` function
3. Move directory creation to `bootstrap_run_environment`
4. Remove `args` mutations from `resolve_config`
5. Update `cli.py` to call both functions
6. Update tests to remove filesystem mocks

**Success criteria**:

- `resolve_config` has no side effects (testable assertion)
- All configuration tests pass without filesystem mocks
- Integration tests pass with `bootstrap_run_environment`

**Rollback plan**: Revert commits, restore mutable configuration

### Phase 3: Runner Decomposition (Week 2, Days 1-3)

**Objective**: Decompose `run()` into `DownloadRun` class with composable stages

**Steps**:

1. Create `DownloadRun` class skeleton
2. Implement stage methods (`setup_sinks`, `setup_resolver_pipeline`, etc.)
3. Migrate logic from `run()` to stage methods
4. Implement orchestration in `DownloadRun.run()`
5. Update `cli.py` to use `DownloadRun`
6. Add unit tests for each stage

**Success criteria**:

- Each stage method testable in isolation
- Integration tests pass with `DownloadRun`
- Old `run()` function removed

**Rollback plan**: Revert commits, restore original `run()` function

### Phase 4: Download Strategy Pattern (Week 2, Days 4-5)

**Objective**: Use strategy pattern for artifact-specific download processing

**Steps**:

1. Define `DownloadStrategy` protocol
2. Implement concrete strategies (`PdfDownloadStrategy`, etc.)
3. Extract focused helper functions (`validate_classification`, etc.)
4. Refactor `process_one_work` to use strategy
5. Add unit tests for strategies and helpers

**Success criteria**:

- Each strategy independently testable
- New artifact types can be added without modifying existing code
- Integration tests pass with strategy pattern

**Rollback plan**: Revert commits, restore original download processing

### Phase 5: Validation and Cleanup (Week 3)

**Objective**: Validate refactoring and prepare for merge

**Steps**:

1. Run full test suite
2. Benchmark performance (baseline vs. refactored)
3. Update documentation
4. Add migration guide
5. Request code review
6. Address feedback
7. Merge to main

**Success criteria**:

- All tests pass
- Performance within 5% of baseline
- Documentation updated
- Code review approved

**Rollback plan**: Maintain feature branch until approval

## Risks / Trade-offs

### Risk: Import breakage for downstream consumers

**Likelihood**: Low (internal refactoring only)

**Impact**: High if external code imports internal modules

**Mitigation**:

- Maintain backward-compatible imports in `pipeline.py` via re-exports
- Document internal API instability in top-level `__init__.py`
- Use deprecation warnings for 1 release cycle before removing

### Risk: Performance regression from indirection

**Likelihood**: Low (modern Python handles abstraction well)

**Impact**: Medium (5-10% slowdown unacceptable)

**Mitigation**:

- Benchmark critical paths before and after
- Profile hot paths with `cProfile`
- Optimize identified bottlenecks (e.g., inline hot functions)
- Accept trade-off if regression < 5%

### Risk: Test maintenance burden

**Likelihood**: Medium (109 tasks include extensive testing)

**Impact**: Low (tests are investment in maintainability)

**Mitigation**:

- Use fixtures to reduce boilerplate
- Share test utilities across resolver modules
- Document testing patterns in contributor guide

### Risk: Incomplete resolver migration

**Likelihood**: Low (automated discovery catches missing resolvers)

**Impact**: High (missing resolver breaks functionality)

**Mitigation**:

- Add test to verify all expected resolvers are registered
- Use resolver registry auto-discovery with fallback to explicit list
- Manual validation during code review

### Trade-off: Line count increase

**Context**: Refactoring adds ~500 lines (module boundaries, docstrings, tests)

**Rationale**: Cognitive load reduction justifies slight increase

**Acceptance**: Maintainability > raw line count

## Migration Plan

### For Contributors

**Adding a new resolver**:

Before:

```python
# Edit pipeline.py (4,710 lines)
class NewResolver(RegisteredResolver):
    ...
```

After:

```python
# Create resolvers/new_resolver.py (~50 lines)
from DocsToKG.ContentDownload.resolvers.base import RegisteredResolver

class NewResolver(RegisteredResolver):
    ...
```

**Adding a new artifact type**:

Before:

```python
# Edit download.py, add conditionals
if classification == Classification.PDF:
    ...
elif classification == Classification.HTML:
    ...
elif classification == Classification.NEW_TYPE:  # ← Add everywhere
    ...
```

After:

```python
# Create NewTypeDownloadStrategy
class NewTypeDownloadStrategy(DownloadStrategy):
    def should_download(self, ...): ...
    def process_response(self, ...): ...
    def finalize_artifact(self, ...): ...
```

### For Operators

**No changes required** to existing:

- CLI invocations
- Configuration files
- Environment variables
- Telemetry parsing scripts

**Optional improvements**:

- Configuration serialization for distributed execution
- Per-stage telemetry hooks (future enhancement)

### Rollback Procedure

If critical issues discovered post-merge:

1. Revert merge commit
2. Deploy previous release tag
3. File regression issue with reproduction steps
4. Address in feature branch before re-merge

## Open Questions

### Q1: Should we introduce a resolver plugin directory for user-defined resolvers?

**Status**: Deferred

**Rationale**: No user requests for custom resolvers; premature optimization

**Revisit**: If 2+ users request custom resolver support

### Q2: Should configuration be serializable to YAML/JSON for distributed execution?

**Status**: Future enhancement

**Rationale**: Frozen dataclass enables this, but no immediate use case

**Revisit**: If users request batch job support or Kubernetes deployment

### Q3: Should we add telemetry hooks for per-stage metrics?

**Status**: Future enhancement

**Rationale**: `DownloadRun` stages enable this, but requires telemetry format changes

**Revisit**: If operators request finer-grained metrics

## Success Metrics

**Quantitative**:

- `pipeline.py`: 4,710 lines → ~800 lines (83% reduction)
- Resolver modules: 1 file → 19 files (base + 17 resolvers + init)
- Test coverage: Maintain >= 85%
- Performance: Within 5% of baseline
- Build time: < 10 minutes

**Qualitative**:

- New contributor onboarding time reduced
- Code review feedback focuses on logic, not navigation
- Bugs localized to specific modules, not monolith
- Feature requests easier to implement

## References

- Original code review: `docs/ContentDownloadReview.md`
- Resolver configuration: `docs/resolver-configuration.md`
- Module organization: `docs/MODULE_ORGANIZATION_GUIDE.md`
- Architecture overview: `docs/architecture.md`
