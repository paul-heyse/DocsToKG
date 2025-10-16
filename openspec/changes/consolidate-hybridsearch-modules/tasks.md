# Implementation Tasks

## 1. Preparation and Analysis

- [x] 1.1 Document current module dependency graph by analyzing all import statements in `src/DocsToKG/HybridSearch/*.py`
- [x] 1.2 Catalog all public exports from `__init__.py` to establish backward compatibility requirements
- [x] 1.3 Identify all test files that import from modules targeted for consolidation
- [x] 1.4 Review CI configurations and scripts for references to `Tools/` directory executables
- [x] 1.5 Create tracking issue for deprecation timeline and user migration communications

## 2. Move ResultShaper to ranking.py

- [x] 2.1 Copy `ResultShaper` class from `results.py` to `ranking.py`, preserving all method signatures and docstrings
- [x] 2.2 Verify `ResultShaper` constructor signature matches existing `device` and `resources` parameter threading pattern used by `apply_mmr_diversification`
- [x] 2.3 Update imports in `service.py` from `from .results import ResultShaper` to `from .ranking import ResultShaper`
- [x] 2.4 Update `__init__.py` to export `ResultShaper` from `.ranking` module path
- [x] 2.5 Convert `results.py` to deprecation shim: retain file with re-export statement `from .ranking import ResultShaper  # noqa: F401` and emit `DeprecationWarning` with migration guidance
- [ ] 2.6 Run test suite targeting `tests/hybrid_search/test_ranking*.py` to verify no behavioral regressions
      Attempted `.venv/bin/python -m pytest tests/hybrid_search/test_suite.py -k ranking` but collection fails because `numpy`
      is unavailable in the execution environment after partial bootstrap. 【a83029†L1-L19】
- [x] 2.7 Update any test imports from `.results` to `.ranking` module path

## 3. Integrate similarity.py into vectorstore.py

- [x] 3.1 Move `normalize_rows` function from `similarity.py` to `vectorstore.py`, placing it near FAISS utility functions
- [x] 3.2 Move `cosine_against_corpus_gpu` function ensuring it retains `device` and `resources` parameters
- [x] 3.3 Move `pairwise_inner_products` function maintaining compatibility with existing callers in `ranking.py` and `validation.py`
- [x] 3.4 Move `max_inner_product` function preserving signature for FAISS-based similarity queries
- [x] 3.5 Update imports in `ranking.py` from `from .similarity import cosine_batch, pairwise_inner_products` to `from .vectorstore import cosine_batch, pairwise_inner_products`
- [x] 3.6 Update imports in `validation.py` for all similarity functions to reference `vectorstore` module
- [x] 3.7 Update `__init__.py` to avoid exporting similarity functions directly (they remain internal to vectorstore unless explicitly needed)
- [x] 3.8 Convert `similarity.py` to deprecation shim with re-exports and `DeprecationWarning` emission
- [ ] 3.9 Run GPU-enabled tests (`tests/hybrid_search/test_vectorstore*.py`) to confirm similarity computations remain bit-exact
      Attempted `.venv/bin/python -m pytest tests/hybrid_search/test_suite.py -k vectorstore`; collection halts due to missing
      `numpy` dependency. 【4e2db8†L1-L19】
- [ ] 3.10 Verify no performance degradation in benchmarks that exercise similarity operations
      Unable to execute benchmarks in this container—GPU dependencies could not be installed (see bootstrap failure) and
      real-vector datasets are unavailable.

## 4. Distribute operations.py functions

- [x] 4.1 Move `verify_pagination` function to `service.py` under a "Service-level operations" section
- [x] 4.2 Move `build_stats_snapshot` function to `service.py` ensuring it maintains access to FAISS, OpenSearch, and registry parameters
- [x] 4.3 Move `should_rebuild_index` function to `service.py` preserving its heuristic logic for delete thresholds
- [x] 4.4 Move `serialize_state` function to `vectorstore.py` as a public module-level function accepting `FaissIndexManager` and `ChunkRegistry`
- [x] 4.5 Move `restore_state` function to `vectorstore.py` ensuring exception handling for missing or corrupt payloads
- [x] 4.6 Update imports in `validation.py` from `from .operations import serialize_state, restore_state` to `from .vectorstore import serialize_state, restore_state`
- [x] 4.7 Update imports in tests and service code for pagination/stats operations to reference `service` module
- [x] 4.8 Update `__init__.py` to export service-level ops from `.service` and state ops from `.vectorstore`
- [x] 4.9 Convert `operations.py` to deprecation shim with conditional re-exports routing to appropriate modules
- [ ] 4.10 Run operational tests (`tests/hybrid_search/test_operations*.py` or equivalent validation harness) to confirm no regressions
      Attempted `.venv/bin/python -m pytest tests/hybrid_search/test_suite.py -k operations`; collection aborts because `numpy`
      is not installed. 【a821d3†L1-L19】

## 5. Merge schema.py with storage.py

- [x] 5.1 Verify `schema.py` exists as separate file; if not, skip this section as schema may already be integrated
- [x] 5.2 Move `OpenSearchIndexTemplate` dataclass to `storage.py` under "Index template management" section
- [x] 5.3 Move `OpenSearchSchemaManager` class to `storage.py` placing it near `OpenSearchSimulator` for lexical storage cohesion
- [x] 5.4 Update any imports in `service.py`, `ingest.py`, or tests from `from .schema import ...` to `from .storage import ...`
- [x] 5.5 Update `__init__.py` to ensure `OpenSearchSchemaManager` and `OpenSearchIndexTemplate` are exported from `.storage`
- [x] 5.6 Convert `schema.py` to deprecation shim if it existed as a separate file
- [ ] 5.7 Run storage-related tests to verify template management and namespace isolation remain functional
      Attempted `.venv/bin/python -m pytest tests/hybrid_search/test_suite.py -k storage`, which fails during import with
      `ModuleNotFoundError: numpy`. 【ab7172†L1-L19】

## 6. Eliminate retrieval.py by consolidating into service.py

- [x] 6.1 Review `retrieval.py` to identify any functions or classes not already present in `service.py`
- [x] 6.2 If `retrieval.py` contains unique orchestration logic, move it into `service.py` under appropriate method or helper function
- [x] 6.3 Update all imports in tests and validation code from `from .retrieval import ...` to `from .service import ...`
- [x] 6.4 Update `__init__.py` to ensure all previously exported symbols from `retrieval` are now sourced from `service`
- [x] 6.5 Convert `retrieval.py` to minimal deprecation shim re-exporting from `service` with `DeprecationWarning`
- [ ] 6.6 Run full integration test suite to verify end-to-end search request flow remains unchanged
      Attempted `.venv/bin/python -m pytest tests/hybrid_search/test_suite.py` but the suite cannot start without `numpy` in the
      environment. 【223d11†L1-L19】

## 7. Retire Tools directory and consolidate CLI entry points

- [x] 7.1 Review functionality in `HybridSearch/tools/run_hybrid_tests.py` and identify which validation scenarios it covers
- [x] 7.2 Ensure `validation.py:main()` CLI accepts equivalent arguments to provide same functionality as `run_hybrid_tests.py`
- [x] 7.3 Review `HybridSearch/tools/run_real_vector_ci.py` and migrate its CI-specific logic into `validation.py` as optional modes or separate subcommands
- [x] 7.4 Update `pyproject.toml` or setup configuration to expose console script entry point for validation module if not already present
- [x] 7.5 Update CI workflow files (`.github/workflows/*.yml` or equivalent) to invoke `python -m DocsToKG.HybridSearch.validation` with appropriate arguments
- [x] 7.6 Delete `HybridSearch/tools/__init__.py` after verifying no external references exist
- [x] 7.7 Delete `HybridSearch/tools/run_hybrid_tests.py` after confirming equivalent functionality exists in `validation.py`
- [x] 7.8 Delete `HybridSearch/tools/run_real_vector_ci.py` after migration completion
- [x] 7.9 Remove `tools/` directory entirely once all files are deleted
- [ ] 7.10 Run CI pipeline locally or in test environment to confirm validation entry points function correctly
      CLI smoke test via `.venv/bin/python -m DocsToKG.HybridSearch.validation --help` fails immediately because `numpy` is
      unavailable; CI emulation is blocked until dependencies can be installed. 【6d8f80†L1-L9】

## 8. Update public interface and maintain backward compatibility

- [x] 8.1 Audit `__init__.py` to ensure all previously exported symbols remain available at their original import paths  
      Verified exports for service pagination helpers, vectorstore state ops, and ranking shaper, ensuring shims surface the legacy names without altering warning behaviour.
- [x] 8.2 Add explicit re-exports in `__init__.py` for deprecated modules using `sys.modules` aliasing pattern: `sys.modules[__name__ + ".similarity"] = sys.modules[__name__ + ".vectorstore"]`  
      Confirmed compatibility via dedicated shim modules so importing legacy paths still raises deprecation warnings only when explicitly used (avoids eager warnings during package import).
- [x] 8.3 Verify deprecation warnings are emitted when users import from deprecated module paths
- [x] 8.4 Document the deprecation timeline in `CHANGELOG.md` with specific guidance on migration paths
- [x] 8.5 Update module-level docstrings in consolidated modules to reflect their expanded responsibilities
- [x] 8.6 Ensure `__all__` exports in each module accurately reflect public API surface

## 9. Update test suite for new module structure

- [x] 9.1 Create mapping of all test files in `tests/hybrid_search/` to modules they test  
      Mapping recorded: `test_suite.py` exercises ingest/service/vectorstore/storage/ranking/config/features/validation plus shim warning coverage.
- [x] 9.2 Update import statements in test files to use new module paths (preferring new paths over deprecated shims)
- [x] 9.3 Verify no tests rely on internal implementation details exposed only through now-consolidated modules  
      Adjusted hybrid search stack fixture to use the public `FaissIndexManager.dim` property instead of touching the private `_dim` attribute.
- [x] 9.4 Add tests specifically validating deprecation warnings are emitted when using old import paths
- [ ] 9.5 Run full test suite with coverage analysis to ensure no coverage gaps introduced by consolidation
      Coverage execution not attempted: even focused pytest runs fail due to missing `numpy`, so coverage tooling cannot run.
- [x] 9.6 Update test utilities or fixtures that may have hardcoded module paths  
      Normalised all fixture imports to consolidated modules and added explicit shim warning assertions to guard against regressions.

## 10. Update documentation and examples

- [x] 10.1 Search documentation files (`docs/hybrid_search*.md`) for import examples using deprecated module paths
- [x] 10.2 Update all code examples to demonstrate imports from consolidated modules
- [x] 10.3 Add migration guide section to relevant documentation explaining the consolidation and providing import path translations
- [x] 10.4 Update architecture diagrams or module dependency graphs to reflect nine-module structure  
      Added a mermaid dependency graph to the migration guide documenting the consolidated module DAG.
- [x] 10.5 Ensure API reference documentation (if auto-generated) correctly resolves to new module locations
- [x] 10.6 Update docstrings that reference other modules by path to use correct post-consolidation paths

## 11. Validation and quality assurance

- [ ] 11.1 Run complete test suite on CPU-only environment to verify non-GPU functionality unaffected
      Blocked: pytest collection fails on import because the bootstrap step cannot install `numpy` (see above command logs).
- [ ] 11.2 Run GPU-enabled test suite to confirm device/resource threading through ranking and vectorstore unchanged
      Blocked: environment lacks CUDA drivers and the bootstrap script cannot fetch NVIDIA-hosted wheels due to SSL issues. 【a26ec9†L1-L109】
- [ ] 11.3 Execute validation harness (`python -m DocsToKG.HybridSearch.validation --mode scale`) against realistic dataset
      Attempted `.venv/bin/python -m DocsToKG.HybridSearch.validation --mode scale`; import fails with `ModuleNotFoundError:
      numpy`. 【e8b215†L1-L9】
- [ ] 11.4 Perform static type checking with mypy or equivalent to catch any import-related type inference issues
      `mypy src/DocsToKG/HybridSearch` reports numerous missing dependency stubs and pre-existing typing issues across ingest and
      validation modules; remediation deferred pending restored environment. 【8f464a†L1-L11】【db2312†L1-L46】
- [x] 11.5 Run linting tools to ensure code style consistency maintained across consolidated modules
      Reordered validation imports to satisfy ruff import-sorting rules; `ruff check` now passes for the touched modules. 【4f3ad8†L1-L2】
- [ ] 11.6 Execute performance benchmarks comparing pre-consolidation and post-consolidation timing for search operations
      Deferred: performance fixtures require GPU-enabled datasets that cannot be provisioned in this sandbox.
- [ ] 11.7 Verify no memory leaks or GPU resource leaks introduced by module reorganization using memory profiling tools
      Deferred pending access to GPU profiling infrastructure (CUDA stack installation is blocked). 【a26ec9†L1-L109】
- [x] 11.8 Conduct code review focusing on dependency arrows to ensure unidirectional flow preserved
      Manually audited `HybridSearch` module imports to confirm the consolidated structure retains the service → vectorstore →
      storage layering with no new cycles introduced.

## 12. Deployment and communication

- [x] 12.1 Prepare release notes detailing the consolidation, deprecation warnings, and migration paths
- [x] 12.2 Update CHANGELOG.md with comprehensive description of module structure changes
- [x] 12.3 Create migration guide document for external users detailing old-to-new import mappings
- [x] 12.4 Schedule deprecation removal for subsequent release (e.g., if current is v0.5.0, plan removal for v0.6.0)
- [ ] 12.5 Announce changes in project communication channels with links to migration documentation
      Pending coordination with project maintainers; no automated channel available from this environment.
- [ ] 12.6 Tag release with appropriate version number following semantic versioning conventions
      Pending repository maintainer action once validation tasks complete.
