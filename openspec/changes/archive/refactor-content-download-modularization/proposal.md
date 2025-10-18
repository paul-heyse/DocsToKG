# Refactor ContentDownload for Modularity and Testability

## Why

The `ContentDownload` module currently suffers from significant architectural debt that impairs maintainability, testability, and extensibility. A comprehensive code review has identified four critical design flaws:

1. **Resolver orchestration sprawl**: The `pipeline.py` module has grown to 4,710 lines, intermingling resolver registry logic with a dozen concrete resolver implementations. This monolithic structure makes it difficult to navigate the codebase, reason about resolver interactions, understand failure modes, and add new content sources.

2. **Configuration side-effects**: The `resolve_config` function in `args.py` mutates the parsed argument namespace, writes directories to disk, and pulls credentials from the environment—all within a single pass. This tight coupling prevents lightweight unit testing and makes configuration reuse difficult for alternative entry points (notebooks, programmatic APIs, batch jobs).

3. **Runner loop coupling**: The `run` function in `runner.py` coordinates sink setup, manifest rotation, OpenAlex pagination, resolver metrics, and worker pools within a single 359-line function. Operational concerns are tightly coupled with business logic, making it difficult to adjust concurrency policies, swap telemetry backends, or test individual stages independently.

4. **Download processing complexity**: The `download.py` module contains very large helper functions (1,770 lines total) that combine classification validation, resume handling, sidecar cleanup, and manifest logging. This increases the cognitive load for contributors and raises the risk of regressions when introducing new artifact types (XML, DOCX) or extraction flows (OCR, semantic parsing).

These issues compound as the system scales: adding a new resolver requires editing a 4,700-line file; testing configuration logic requires mocking filesystem operations; debugging worker concurrency requires instrumenting a monolithic runner; and extending download classification requires modifying deeply nested conditionals.

The proposed refactoring addresses these systemic issues by decomposing the ContentDownload pipeline into focused, composable modules that support independent testing, flexible configuration, and incremental enhancement.

## What Changes

### 1. Modularize Resolver Implementations

**Current state**: All resolver logic lives in `pipeline.py` (4,710 lines).

**Target state**: Split into focused modules:

```
ContentDownload/
├── resolvers/
│   ├── __init__.py           # Public exports, ResolverRegistry
│   ├── base.py               # RegisteredResolver, ApiResolverBase
│   ├── arxiv.py              # ArxivResolver
│   ├── core.py               # CoreResolver
│   ├── crossref.py           # CrossrefResolver
│   ├── doaj.py               # DoajResolver
│   ├── europe_pmc.py         # EuropePmcResolver
│   ├── figshare.py           # FigshareResolver
│   ├── hal.py                # HalResolver
│   ├── landing_page.py       # LandingPageResolver
│   ├── openaire.py           # OpenAireResolver
│   ├── openalex.py           # OpenAlexResolver
│   ├── osf.py                # OsfResolver
│   ├── pmc.py                # PmcResolver
│   ├── semantic_scholar.py   # SemanticScholarResolver
│   ├── unpaywall.py          # UnpaywallResolver
│   ├── wayback.py            # WaybackResolver
│   └── zenodo.py             # ZenodoResolver
├── pipeline.py               # ResolverPipeline orchestration, config loading
└── ...
```

**Changes**:

- Extract each `*Resolver` class into a dedicated module under `resolvers/`
- Keep shared base classes (`RegisteredResolver`, `ApiResolverBase`) in `resolvers/base.py`
- Move `ResolverRegistry` to `resolvers/__init__.py` with public exports
- Consolidate `ResolverResult`, `ResolverEvent`, `ResolverEventReason`, `ResolverMetrics`, `ResolverConfig`, `ResolverPipeline` in `pipeline.py`
- Extract common request/JSON parsing helpers into `resolvers/base.py` for shared use
- Retain resolver configuration loading (`load_resolver_config`, `read_resolver_config`) in `pipeline.py`

**Benefits**:

- New resolvers can be added without touching existing implementations
- Each resolver can be tested in isolation
- Easier to share common patterns (API polling, rate limiting, retry logic)
- Reduced cognitive load when working on a specific content source

### 2. Separate Configuration Resolution from I/O Side Effects

**Current state**: `resolve_config` in `args.py` mutates `args`, writes directories, pulls credentials, and bootstraps telemetry state.

**Target state**: Pure configuration resolution with explicit side-effect handling:

```python
@dataclass(frozen=True)
class ResolvedConfig:
    """Immutable configuration derived from CLI arguments."""
    args: argparse.Namespace  # Not mutated
    run_id: str
    query: Works
    pdf_dir: Path
    html_dir: Path
    xml_dir: Path
    manifest_path: Path
    csv_path: Optional[Path]
    sqlite_path: Path
    resolver_instances: List[Any]
    resolver_config: Any
    previous_url_index: Dict[str, Dict[str, Any]]
    persistent_seen_urls: Set[str]
    robots_checker: Optional[RobotsCache]
    concurrency_product: int


def resolve_config(args: argparse.Namespace, parser: argparse.ArgumentParser) -> ResolvedConfig:
    """Convert CLI arguments into immutable configuration."""
    # Pure computation: no directory creation, no mutation
    # Returns frozen dataclass
    ...


def bootstrap_run_environment(resolved: ResolvedConfig) -> None:
    """Handle filesystem initialization and telemetry setup."""
    ensure_dir(resolved.pdf_dir)
    ensure_dir(resolved.html_dir)
    ensure_dir(resolved.xml_dir)
    # Telemetry bootstrapping moved here
    ...
```

**Changes**:

- Make `ResolvedConfig` immutable (`@dataclass(frozen=True)`)
- Remove directory creation from `resolve_config`
- Remove `args` mutation (e.g., `args.extract_html_text = ...`)
- Move `ensure_dir` calls to a new `bootstrap_run_environment` function
- Move telemetry initialization to `bootstrap_run_environment`
- Update `cli.py` to call `bootstrap_run_environment` before `run(resolved)`

**Benefits**:

- Configuration can be tested without filesystem mocks
- Configuration can be reused across multiple runs
- Configuration can be serialized/deserialized for distributed execution
- Clear separation of concerns: pure logic vs. side effects

### 3. Refactor Runner Loop into Composable Stages

**Current state**: The `run` function in `runner.py` coordinates sink setup, manifest rotation, OpenAlex pagination, resolver metrics, and worker pools in a single 359-line function.

**Target state**: Decompose into dedicated helpers and a `DownloadRun` class:

```python
class DownloadRun:
    """Orchestrates the content download pipeline with composable stages."""

    def __init__(self, resolved: ResolvedConfig):
        self.resolved = resolved
        self.metrics = RunMetrics()
        self.telemetry = None
        self.resolver_pipeline = None
        self.work_provider = None
        self.download_state = None

    def setup_sinks(self) -> MultiSink:
        """Initialize telemetry sinks based on configuration."""
        ...

    def setup_resolver_pipeline(self) -> ResolverPipeline:
        """Create resolver pipeline with metrics tracking."""
        ...

    def setup_work_provider(self) -> WorkProvider:
        """Create OpenAlex work provider with pagination."""
        ...

    def setup_download_state(self, session_factory, robots_cache) -> DownloadState:
        """Initialize download state for artifact tracking."""
        ...

    def setup_worker_pool(self) -> ThreadPoolExecutor:
        """Create worker pool based on concurrency settings."""
        ...

    def process_work_item(self, work, options) -> ProcessResult:
        """Process a single work item through the pipeline."""
        ...

    def run(self) -> RunResult:
        """Execute the download pipeline."""
        self.telemetry = self.setup_sinks()
        self.resolver_pipeline = self.setup_resolver_pipeline()
        self.work_provider = self.setup_work_provider()
        # ... orchestrate stages ...
```

**Changes**:

- Create `DownloadRun` class in `runner.py`
- Extract sink wiring into `setup_sinks`
- Extract resolver pipeline creation into `setup_resolver_pipeline`
- Extract OpenAlex provider creation into `setup_work_provider`
- Extract worker pool management into `setup_worker_pool`
- Keep main orchestration in `DownloadRun.run()`
- Update `cli.py` to instantiate `DownloadRun` and call `.run()`

**Benefits**:

- Each stage can be tested independently
- Concurrency policies can be adjusted without touching business logic
- Telemetry backends can be swapped easily

### 4. Decompose Download Processing into Smaller, Testable Units

**Current state**: `download.py` contains very large helper functions (1,770 lines) combining classification validation, resume handling, sidecar cleanup, and manifest logging.

**Target state**: Focused functions and strategy objects:

```python
# download.py
def validate_classification(classification, artifact, options) -> ValidationResult:
    """Validate that the downloaded payload matches expected classification."""
    ...

def handle_resume_logic(artifact, previous_index, options) -> ResumeDecision:
    """Determine if download should be skipped based on previous attempts."""
    ...

def cleanup_sidecar_files(artifact, classification, options) -> None:
    """Remove auxiliary files based on classification outcome."""
    ...

def build_download_outcome(artifact, classification, attempts) -> DownloadOutcome:
    """Construct outcome record for telemetry logging."""
    ...

class DownloadStrategy:
    """Strategy pattern for different artifact types."""

    def should_download(self, artifact, context) -> bool:
        ...

    def process_response(self, response, artifact, context) -> Classification:
        ...

    def finalize_artifact(self, artifact, classification, context) -> DownloadOutcome:
        ...


class PdfDownloadStrategy(DownloadStrategy):
    """Handle PDF-specific download and validation logic."""
    ...


class HtmlDownloadStrategy(DownloadStrategy):
    """Handle HTML-specific download and extraction logic."""
    ...


class XmlDownloadStrategy(DownloadStrategy):
    """Handle XML-specific download and parsing logic."""
    ...
```

**Changes**:

- Extract `_build_download_outcome` logic into `build_download_outcome` function
- Extract resume logic into `handle_resume_logic` function
- Extract sidecar cleanup into `cleanup_sidecar_files` function
- Extract classification validation into `validate_classification` function
- Introduce `DownloadStrategy` protocol/interface
- Implement `PdfDownloadStrategy`, `HtmlDownloadStrategy`, `XmlDownloadStrategy`
- Refactor `process_one_work` to use strategy pattern
- Reduce `download_candidate` complexity by delegating to strategies

**Benefits**:

- Each concern can be tested in isolation
- New artifact types can be added without modifying existing code
- Extraction flows can be customized per classification
- Reduced cognitive load when debugging specific download scenarios

## Impact

### Affected Specs

This change introduces a new capability:

- **content-download**: Specification for the content download pipeline, including resolver orchestration, configuration management, runner orchestration, and download processing

### Affected Code

**Modules requiring significant refactoring**:

- `src/DocsToKG/ContentDownload/pipeline.py` (4,710 lines → ~800 lines + 17 resolver modules)
- `src/DocsToKG/ContentDownload/args.py` (689 lines → ~600 lines, `resolve_config` becomes pure)
- `src/DocsToKG/ContentDownload/runner.py` (359 lines → ~400 lines with `DownloadRun` class)
- `src/DocsToKG/ContentDownload/download.py` (1,770 lines → ~1,200 lines + strategy classes)

**New modules**:

- `src/DocsToKG/ContentDownload/resolvers/__init__.py`
- `src/DocsToKG/ContentDownload/resolvers/base.py`
- `src/DocsToKG/ContentDownload/resolvers/*.py` (17 resolver modules)

**Breaking changes**:

- None (internal refactoring only, public CLI interface unchanged)

**Compatibility**:

- Existing CLI usage remains unchanged
- Existing configuration files remain compatible
- Existing telemetry formats unchanged
- Existing tests may require updates to import paths

### Migration Path

1. **Phase 1: Resolver modularization** (Week 1)
   - Extract resolver classes into `resolvers/` modules
   - Update imports in `pipeline.py`, `runner.py`, tests
   - Validate resolver discovery and registration

2. **Phase 2: Configuration purity** (Week 1)
   - Make `ResolvedConfig` immutable
   - Extract `bootstrap_run_environment` function
   - Update `cli.py` to call both functions
   - Update configuration tests to remove filesystem mocks

3. **Phase 3: Runner decomposition** (Week 2)
   - Implement `DownloadRun` class with stage methods
   - Migrate existing `run` logic into stages
   - Update `cli.py` to use `DownloadRun`
   - Add integration tests for each stage

4. **Phase 4: Download strategy pattern** (Week 2)
   - Define `DownloadStrategy` protocol
   - Implement `PdfDownloadStrategy`, `HtmlDownloadStrategy`, `XmlDownloadStrategy`
   - Refactor `process_one_work` to use strategies
   - Add unit tests for each strategy

5. **Phase 5: Validation and cleanup** (Week 3)
   - Run full integration test suite
   - Validate performance benchmarks
   - Update documentation
   - Clean up deprecated code paths

### Risks and Mitigations

**Risk: Breaking existing integrations**

- *Mitigation*: Maintain backward-compatible public API (CLI, entry points)
- *Mitigation*: Comprehensive integration test suite before merge

**Risk: Performance regression**

- *Mitigation*: Benchmark critical paths before and after refactoring
- *Mitigation*: Optimize hot paths identified during profiling

**Risk: Incomplete resolver migration**

- *Mitigation*: Use resolver registry auto-discovery to catch missing resolvers
- *Mitigation*: Add tests to verify all resolvers are registered

**Risk: Configuration deserialization failures**

- *Mitigation*: Add validation tests for configuration serialization/deserialization
- *Mitigation*: Document immutable field constraints

### Success Criteria

1. All resolver classes moved to `resolvers/` modules
2. `pipeline.py` reduced from 4,710 lines to ~800 lines
3. `resolve_config` becomes pure (no filesystem writes, no mutations)
4. `DownloadRun` class encapsulates runner orchestration
5. Download processing uses strategy pattern
6. All existing tests pass
7. Integration test coverage >= 85%
8. Performance within 5% of baseline
9. Documentation updated to reflect new architecture
