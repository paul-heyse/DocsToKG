# Implementation Tasks

## 1. Resolver Modularization

- [ ] 1.1 Create `src/DocsToKG/ContentDownload/resolvers/` directory
- [ ] 1.2 Create `resolvers/__init__.py` with `ResolverRegistry` and public exports
- [ ] 1.3 Create `resolvers/base.py` with `RegisteredResolver`, `ApiResolverBase`, shared helpers
- [ ] 1.4 Extract `ArxivResolver` → `resolvers/arxiv.py`
- [ ] 1.5 Extract `CoreResolver` → `resolvers/core.py`
- [ ] 1.6 Extract `CrossrefResolver` → `resolvers/crossref.py`
- [ ] 1.7 Extract `DoajResolver` → `resolvers/doaj.py`
- [ ] 1.8 Extract `EuropePmcResolver` → `resolvers/europe_pmc.py`
- [ ] 1.9 Extract `FigshareResolver` → `resolvers/figshare.py`
- [ ] 1.10 Extract `HalResolver` → `resolvers/hal.py`
- [ ] 1.11 Extract `LandingPageResolver` → `resolvers/landing_page.py`
- [ ] 1.12 Extract `OpenAireResolver` → `resolvers/openaire.py`
- [ ] 1.13 Extract `OpenAlexResolver` → `resolvers/openalex.py`
- [ ] 1.14 Extract `OsfResolver` → `resolvers/osf.py`
- [ ] 1.15 Extract `PmcResolver` → `resolvers/pmc.py`
- [ ] 1.16 Extract `SemanticScholarResolver` → `resolvers/semantic_scholar.py`
- [ ] 1.17 Extract `UnpaywallResolver` → `resolvers/unpaywall.py`
- [ ] 1.18 Extract `WaybackResolver` → `resolvers/wayback.py`
- [ ] 1.19 Extract `ZenodoResolver` → `resolvers/zenodo.py`
- [ ] 1.20 Update `pipeline.py` to import from `resolvers/`
- [ ] 1.21 Update `runner.py` imports to use `resolvers/`
- [ ] 1.22 Update `args.py` imports to use `resolvers/`
- [ ] 1.23 Update tests to import from `resolvers/`
- [ ] 1.24 Verify resolver auto-discovery works correctly
- [ ] 1.25 Add test to verify all expected resolvers are registered
- [ ] 1.26 Update NAVMAP annotations in new modules
- [ ] 1.27 Remove resolver implementations from `pipeline.py`

## 2. Configuration Purity

- [x] 2.1 Add `frozen=True` to `ResolvedConfig` dataclass
- [x] 2.2 Create `bootstrap_run_environment(resolved: ResolvedConfig) -> None` function
- [x] 2.3 Move `ensure_dir(pdf_dir)` from `resolve_config` to `bootstrap_run_environment`
- [x] 2.4 Move `ensure_dir(html_dir)` from `resolve_config` to `bootstrap_run_environment`
- [x] 2.5 Move `ensure_dir(xml_dir)` from `resolve_config` to `bootstrap_run_environment`
- [x] 2.6 Remove `args.extract_html_text = ...` mutation from `resolve_config`
- [x] 2.7 Compute `extract_html_text` derived field in `ResolvedConfig` dataclass
- [x] 2.8 Update `cli.py` to call `bootstrap_run_environment(resolved)` before `run(resolved)`
- [x] 2.9 Update tests to remove filesystem mocks from `resolve_config` tests
- [x] 2.10 Add test for `resolve_config` purity (no side effects)
- [x] 2.11 Add test for `bootstrap_run_environment` directory creation
- [x] 2.12 Document immutability contract in `ResolvedConfig` docstring

## 3. Runner Decomposition

- [ ] 3.1 Create `DownloadRun` class in `runner.py`
- [ ] 3.2 Add `__init__(self, resolved: ResolvedConfig)` to initialize state
- [ ] 3.3 Implement `setup_sinks(self) -> MultiSink` method
- [ ] 3.4 Implement `setup_resolver_pipeline(self) -> ResolverPipeline` method
- [ ] 3.5 Implement `setup_work_provider(self) -> WorkProvider` method
- [ ] 3.6 Implement `setup_download_state(self, session_factory, robots_cache) -> DownloadState` method
- [ ] 3.7 Implement `setup_worker_pool(self) -> ThreadPoolExecutor` method
- [ ] 3.8 Implement `process_work_item(self, work, options) -> ProcessResult` method
- [ ] 3.9 Implement `check_budget_limits(self) -> bool` method
- [ ] 3.10 Implement `DownloadRun.run(self) -> RunResult` orchestration method
- [ ] 3.11 Migrate sink wiring logic from `run()` to `setup_sinks()`
- [ ] 3.12 Migrate resolver pipeline creation from `run()` to `setup_resolver_pipeline()`
- [ ] 3.13 Migrate OpenAlex provider creation from `run()` to `setup_work_provider()`
- [ ] 3.14 Migrate worker pool management from `run()` to `setup_worker_pool()`
- [ ] 3.15 Migrate budget enforcement from `run()` to `check_budget_limits()`
- [ ] 3.16 Update `cli.py` to instantiate `DownloadRun` and call `.run()`
- [ ] 3.17 Add unit tests for `setup_sinks()`
- [ ] 3.18 Add unit tests for `setup_resolver_pipeline()`
- [ ] 3.19 Add unit tests for `setup_work_provider()`
- [ ] 3.20 Add unit tests for `setup_download_state()`
- [ ] 3.21 Add unit tests for `setup_worker_pool()`
- [ ] 3.22 Add unit tests for `check_budget_limits()`
- [ ] 3.23 Add integration test for `DownloadRun.run()`
- [ ] 3.24 Remove old `run()` function
- [ ] 3.25 Update documentation to reference `DownloadRun` class

## 4. Download Strategy Pattern

- [ ] 4.1 Define `DownloadStrategy` protocol in `download.py`
- [ ] 4.2 Add `should_download(artifact, context) -> bool` method to protocol
- [ ] 4.3 Add `process_response(response, artifact, context) -> Classification` method to protocol
- [ ] 4.4 Add `finalize_artifact(artifact, classification, context) -> DownloadOutcome` method to protocol
- [ ] 4.5 Implement `PdfDownloadStrategy` class
- [ ] 4.6 Implement `HtmlDownloadStrategy` class
- [ ] 4.7 Implement `XmlDownloadStrategy` class
- [ ] 4.8 Extract `validate_classification(classification, artifact, options) -> ValidationResult` function
- [ ] 4.9 Extract `handle_resume_logic(artifact, previous_index, options) -> ResumeDecision` function
- [ ] 4.10 Extract `cleanup_sidecar_files(artifact, classification, options) -> None` function
- [ ] 4.11 Extract `build_download_outcome(artifact, classification, attempts) -> DownloadOutcome` function
- [ ] 4.12 Refactor `_build_download_outcome` to use new `build_download_outcome` function
- [ ] 4.13 Refactor `process_one_work` to accept strategy parameter
- [ ] 4.14 Add strategy selection logic based on classification
- [ ] 4.15 Delegate to strategy in `download_candidate`
- [ ] 4.16 Add unit tests for `validate_classification`
- [ ] 4.17 Add unit tests for `handle_resume_logic`
- [ ] 4.18 Add unit tests for `cleanup_sidecar_files`
- [ ] 4.19 Add unit tests for `build_download_outcome`
- [ ] 4.20 Add unit tests for `PdfDownloadStrategy`
- [ ] 4.21 Add unit tests for `HtmlDownloadStrategy`
- [ ] 4.22 Add unit tests for `XmlDownloadStrategy`
- [ ] 4.23 Add integration test for strategy selection
- [ ] 4.24 Remove inlined classification/resume/cleanup logic
- [ ] 4.25 Update documentation for strategy pattern usage

## 5. Validation and Cleanup

- [ ] 5.1 Run full test suite and fix failing tests
- [ ] 5.2 Run integration tests against sample OpenAlex data
- [ ] 5.3 Benchmark resolver pipeline performance (baseline vs. refactored)
- [ ] 5.4 Benchmark download processing performance (baseline vs. refactored)
- [ ] 5.5 Profile hot paths and optimize if regressions detected
- [ ] 5.6 Update `docs/ContentDownloadReview.md` with new architecture
- [ ] 5.7 Update `docs/architecture.md` with resolver modularization
- [ ] 5.8 Add docstrings to all new public functions and classes
- [ ] 5.9 Add type hints to all new functions and methods
- [ ] 5.10 Run `ruff` linter and fix violations
- [ ] 5.11 Run `black` formatter
- [ ] 5.12 Update `CHANGELOG.md` with refactoring summary
- [ ] 5.13 Create migration guide for custom resolver implementations
- [ ] 5.14 Remove deprecated code paths marked for deletion
- [ ] 5.15 Verify import paths in all modules
- [ ] 5.16 Verify resolver discovery in production-like environment
- [ ] 5.17 Add end-to-end test with real OpenAlex query (if safe)
- [ ] 5.18 Request code review from maintainers
- [ ] 5.19 Address review feedback
- [ ] 5.20 Merge to main branch

## Notes

- **Resolver modularization** (27 tasks) should be completed first to unlock parallel work on other tracks
- **Configuration purity** (12 tasks) can proceed in parallel with resolver work
- **Runner decomposition** (25 tasks) depends on configuration purity
- **Download strategy** (25 tasks) can proceed in parallel with runner work
- **Validation** (20 tasks) must wait for all tracks to complete

**Total tasks**: 109

**Estimated effort**: 3 weeks (15 working days) with 1-2 full-time contributors
