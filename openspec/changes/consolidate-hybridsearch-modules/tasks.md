# Implementation Tasks

## 1. Preparation and Analysis

- [ ] 1.1 Document current module dependency graph by analyzing all import statements in `src/DocsToKG/HybridSearch/*.py`
- [ ] 1.2 Catalog all public exports from `__init__.py` to establish backward compatibility requirements
- [ ] 1.3 Identify all test files that import from modules targeted for consolidation
- [ ] 1.4 Review CI configurations and scripts for references to `Tools/` directory executables
- [ ] 1.5 Create tracking issue for deprecation timeline and user migration communications

## 2. Move ResultShaper to ranking.py

- [ ] 2.1 Copy `ResultShaper` class from `results.py` to `ranking.py`, preserving all method signatures and docstrings
- [ ] 2.2 Verify `ResultShaper` constructor signature matches existing `device` and `resources` parameter threading pattern used by `apply_mmr_diversification`
- [ ] 2.3 Update imports in `service.py` from `from .results import ResultShaper` to `from .ranking import ResultShaper`
- [ ] 2.4 Update `__init__.py` to export `ResultShaper` from `.ranking` module path
- [ ] 2.5 Convert `results.py` to deprecation shim: retain file with re-export statement `from .ranking import ResultShaper  # noqa: F401` and emit `DeprecationWarning` with migration guidance
- [ ] 2.6 Run test suite targeting `tests/hybrid_search/test_ranking*.py` to verify no behavioral regressions
- [ ] 2.7 Update any test imports from `.results` to `.ranking` module path

## 3. Integrate similarity.py into vectorstore.py

- [ ] 3.1 Move `normalize_rows` function from `similarity.py` to `vectorstore.py`, placing it near FAISS utility functions
- [ ] 3.2 Move `cosine_against_corpus_gpu` function ensuring it retains `device` and `resources` parameters
- [ ] 3.3 Move `pairwise_inner_products` function maintaining compatibility with existing callers in `ranking.py` and `validation.py`
- [ ] 3.4 Move `max_inner_product` function preserving signature for FAISS-based similarity queries
- [ ] 3.5 Update imports in `ranking.py` from `from .similarity import cosine_batch, pairwise_inner_products` to `from .vectorstore import cosine_batch, pairwise_inner_products`
- [ ] 3.6 Update imports in `validation.py` for all similarity functions to reference `vectorstore` module
- [ ] 3.7 Update `__init__.py` to avoid exporting similarity functions directly (they remain internal to vectorstore unless explicitly needed)
- [ ] 3.8 Convert `similarity.py` to deprecation shim with re-exports and `DeprecationWarning` emission
- [ ] 3.9 Run GPU-enabled tests (`tests/hybrid_search/test_vectorstore*.py`) to confirm similarity computations remain bit-exact
- [ ] 3.10 Verify no performance degradation in benchmarks that exercise similarity operations

## 4. Distribute operations.py functions

- [ ] 4.1 Move `verify_pagination` function to `service.py` under a "Service-level operations" section
- [ ] 4.2 Move `build_stats_snapshot` function to `service.py` ensuring it maintains access to FAISS, OpenSearch, and registry parameters
- [ ] 4.3 Move `should_rebuild_index` function to `service.py` preserving its heuristic logic for delete thresholds
- [ ] 4.4 Move `serialize_state` function to `vectorstore.py` as a public module-level function accepting `FaissIndexManager` and `ChunkRegistry`
- [ ] 4.5 Move `restore_state` function to `vectorstore.py` ensuring exception handling for missing or corrupt payloads
- [ ] 4.6 Update imports in `validation.py` from `from .operations import serialize_state, restore_state` to `from .vectorstore import serialize_state, restore_state`
- [ ] 4.7 Update imports in tests and service code for pagination/stats operations to reference `service` module
- [ ] 4.8 Update `__init__.py` to export service-level ops from `.service` and state ops from `.vectorstore`
- [ ] 4.9 Convert `operations.py` to deprecation shim with conditional re-exports routing to appropriate modules
- [ ] 4.10 Run operational tests (`tests/hybrid_search/test_operations*.py` or equivalent validation harness) to confirm no regressions

## 5. Merge schema.py with storage.py

- [ ] 5.1 Verify `schema.py` exists as separate file; if not, skip this section as schema may already be integrated
- [ ] 5.2 Move `OpenSearchIndexTemplate` dataclass to `storage.py` under "Index template management" section
- [ ] 5.3 Move `OpenSearchSchemaManager` class to `storage.py` placing it near `OpenSearchSimulator` for lexical storage cohesion
- [ ] 5.4 Update any imports in `service.py`, `ingest.py`, or tests from `from .schema import ...` to `from .storage import ...`
- [ ] 5.5 Update `__init__.py` to ensure `OpenSearchSchemaManager` and `OpenSearchIndexTemplate` are exported from `.storage`
- [ ] 5.6 Convert `schema.py` to deprecation shim if it existed as a separate file
- [ ] 5.7 Run storage-related tests to verify template management and namespace isolation remain functional

## 6. Eliminate retrieval.py by consolidating into service.py

- [ ] 6.1 Review `retrieval.py` to identify any functions or classes not already present in `service.py`
- [ ] 6.2 If `retrieval.py` contains unique orchestration logic, move it into `service.py` under appropriate method or helper function
- [ ] 6.3 Update all imports in tests and validation code from `from .retrieval import ...` to `from .service import ...`
- [ ] 6.4 Update `__init__.py` to ensure all previously exported symbols from `retrieval` are now sourced from `service`
- [ ] 6.5 Convert `retrieval.py` to minimal deprecation shim re-exporting from `service` with `DeprecationWarning`
- [ ] 6.6 Run full integration test suite to verify end-to-end search request flow remains unchanged

## 7. Retire Tools directory and consolidate CLI entry points

- [ ] 7.1 Review functionality in `HybridSearch/tools/run_hybrid_tests.py` and identify which validation scenarios it covers
- [ ] 7.2 Ensure `validation.py:main()` CLI accepts equivalent arguments to provide same functionality as `run_hybrid_tests.py`
- [ ] 7.3 Review `HybridSearch/tools/run_real_vector_ci.py` and migrate its CI-specific logic into `validation.py` as optional modes or separate subcommands
- [ ] 7.4 Update `pyproject.toml` or setup configuration to expose console script entry point for validation module if not already present
- [ ] 7.5 Update CI workflow files (`.github/workflows/*.yml` or equivalent) to invoke `python -m DocsToKG.HybridSearch.validation` with appropriate arguments
- [ ] 7.6 Delete `HybridSearch/tools/__init__.py` after verifying no external references exist
- [ ] 7.7 Delete `HybridSearch/tools/run_hybrid_tests.py` after confirming equivalent functionality exists in `validation.py`
- [ ] 7.8 Delete `HybridSearch/tools/run_real_vector_ci.py` after migration completion
- [ ] 7.9 Remove `tools/` directory entirely once all files are deleted
- [ ] 7.10 Run CI pipeline locally or in test environment to confirm validation entry points function correctly

## 8. Update public interface and maintain backward compatibility

- [ ] 8.1 Audit `__init__.py` to ensure all previously exported symbols remain available at their original import paths
- [ ] 8.2 Add explicit re-exports in `__init__.py` for deprecated modules using `sys.modules` aliasing pattern: `sys.modules[__name__ + ".similarity"] = sys.modules[__name__ + ".vectorstore"]`
- [ ] 8.3 Verify deprecation warnings are emitted when users import from deprecated module paths
- [ ] 8.4 Document the deprecation timeline in `CHANGELOG.md` with specific guidance on migration paths
- [ ] 8.5 Update module-level docstrings in consolidated modules to reflect their expanded responsibilities
- [ ] 8.6 Ensure `__all__` exports in each module accurately reflect public API surface

## 9. Update test suite for new module structure

- [ ] 9.1 Create mapping of all test files in `tests/hybrid_search/` to modules they test
- [ ] 9.2 Update import statements in test files to use new module paths (preferring new paths over deprecated shims)
- [ ] 9.3 Verify no tests rely on internal implementation details exposed only through now-consolidated modules
- [ ] 9.4 Add tests specifically validating deprecation warnings are emitted when using old import paths
- [ ] 9.5 Run full test suite with coverage analysis to ensure no coverage gaps introduced by consolidation
- [ ] 9.6 Update test utilities or fixtures that may have hardcoded module paths

## 10. Update documentation and examples

- [ ] 10.1 Search documentation files (`docs/hybrid_search*.md`) for import examples using deprecated module paths
- [ ] 10.2 Update all code examples to demonstrate imports from consolidated modules
- [ ] 10.3 Add migration guide section to relevant documentation explaining the consolidation and providing import path translations
- [ ] 10.4 Update architecture diagrams or module dependency graphs to reflect nine-module structure
- [ ] 10.5 Ensure API reference documentation (if auto-generated) correctly resolves to new module locations
- [ ] 10.6 Update docstrings that reference other modules by path to use correct post-consolidation paths

## 11. Validation and quality assurance

- [ ] 11.1 Run complete test suite on CPU-only environment to verify non-GPU functionality unaffected
- [ ] 11.2 Run GPU-enabled test suite to confirm device/resource threading through ranking and vectorstore unchanged
- [ ] 11.3 Execute validation harness (`python -m DocsToKG.HybridSearch.validation --mode scale`) against realistic dataset
- [ ] 11.4 Perform static type checking with mypy or equivalent to catch any import-related type inference issues
- [ ] 11.5 Run linting tools to ensure code style consistency maintained across consolidated modules
- [ ] 11.6 Execute performance benchmarks comparing pre-consolidation and post-consolidation timing for search operations
- [ ] 11.7 Verify no memory leaks or GPU resource leaks introduced by module reorganization using memory profiling tools
- [ ] 11.8 Conduct code review focusing on dependency arrows to ensure unidirectional flow preserved

## 12. Deployment and communication

- [ ] 12.1 Prepare release notes detailing the consolidation, deprecation warnings, and migration paths
- [ ] 12.2 Update CHANGELOG.md with comprehensive description of module structure changes
- [ ] 12.3 Create migration guide document for external users detailing old-to-new import mappings
- [ ] 12.4 Schedule deprecation removal for subsequent release (e.g., if current is v0.5.0, plan removal for v0.6.0)
- [ ] 12.5 Announce changes in project communication channels with links to migration documentation
- [ ] 12.6 Tag release with appropriate version number following semantic versioning conventions
