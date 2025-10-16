# HybridSearch Module Structure Specification

## ADDED Requirements

### Requirement: Consolidated Module Organization

The HybridSearch subsystem SHALL organize functionality into exactly nine core modules, each with a single, well-defined responsibility that minimizes cross-module coupling and eliminates adapter modules that exist solely for indirection.

#### Scenario: Developer locates ranking functionality

- **WHEN** a developer needs to understand or modify result ranking behavior
- **THEN** all ranking-related code (Reciprocal Rank Fusion, MMR diversification, and result shaping including deduplication and highlighting) SHALL be located in the `ranking` module
- **AND** the developer SHALL NOT need to navigate to separate `results` or `similarity` modules to trace the complete ranking pipeline

#### Scenario: Developer locates FAISS operations

- **WHEN** a developer needs to understand or modify vector store operations
- **THEN** all FAISS-related functionality (index management, GPU resource creation, similarity computations, and state serialization) SHALL be located in the `vectorstore` module
- **AND** no FAISS-specific logic SHALL exist in separate utility modules

#### Scenario: Developer locates service orchestration

- **WHEN** a developer needs to understand the hybrid search request flow
- **THEN** all orchestration logic (channel execution, fusion, pagination, and stats) SHALL be located in the `service` module
- **AND** no thin adapter modules SHALL exist between the service entry point and its implementation

### Requirement: GPU Resource Lifecycle Management

The HybridSearch subsystem SHALL manage GPU resources through a single authoritative source with explicit parameter threading, ensuring GPU device handles and StandardGpuResources instances have unambiguous ownership and lifecycle boundaries.

#### Scenario: FAISS index manager creates GPU resources

- **WHEN** a `FaissIndexManager` instance is constructed with a GPU-enabled `DenseIndexConfig`
- **THEN** the manager SHALL create exactly one `faiss.StandardGpuResources` instance
- **AND** the manager SHALL expose this resource via a public `gpu_resources` property
- **AND** the manager SHALL expose the configured GPU device ordinal via a public `device` property
- **AND** no other module SHALL create or cache GPU resources independently

#### Scenario: Ranking functions receive GPU resources explicitly

- **WHEN** the service orchestrator invokes `apply_mmr_diversification` for result diversification
- **THEN** the function SHALL receive `device` and `resources` as explicit keyword-only parameters sourced from the `FaissIndexManager` instance
- **AND** the function SHALL NOT attempt to create or discover GPU resources through global state or module-level caching

#### Scenario: Similarity operations execute on specified GPU

- **WHEN** ranking or validation code invokes GPU similarity functions (e.g., `pairwise_inner_products`)
- **THEN** each function SHALL accept `device` and `resources` as explicit parameters
- **AND** all GPU tensor operations SHALL execute on the specified device without CPU fallback
- **AND** the function SHALL raise an exception if GPU resources are unavailable rather than silently falling back to CPU

### Requirement: Backward Compatibility Through Deprecation Shims

The HybridSearch subsystem SHALL maintain backward compatibility for deprecated import paths through module shims that emit DeprecationWarning messages, allowing external code one full release cycle to migrate before removal.

#### Scenario: User imports from deprecated retrieval module

- **WHEN** a user imports `HybridSearchService` from `DocsToKG.HybridSearch.retrieval`
- **THEN** the import SHALL succeed without errors
- **AND** a `DeprecationWarning` SHALL be emitted to stderr indicating the module is deprecated
- **AND** the warning message SHALL specify the new import path (`DocsToKG.HybridSearch.service`)
- **AND** the warning message SHALL indicate the planned removal version

#### Scenario: User imports from deprecated results module

- **WHEN** a user imports `ResultShaper` from `DocsToKG.HybridSearch.results`
- **THEN** the import SHALL succeed and provide the correct `ResultShaper` class from the `ranking` module
- **AND** a `DeprecationWarning` SHALL be emitted guiding migration to `DocsToKG.HybridSearch.ranking`

#### Scenario: User imports from deprecated similarity module

- **WHEN** a user imports GPU similarity functions from `DocsToKG.HybridSearch.similarity`
- **THEN** the imports SHALL succeed and resolve to implementations in `DocsToKG.HybridSearch.vectorstore`
- **AND** a `DeprecationWarning` SHALL be emitted for each import statement
- **AND** all function signatures SHALL remain identical to pre-consolidation versions

#### Scenario: User imports from deprecated operations module

- **WHEN** a user imports `serialize_state` from `DocsToKG.HybridSearch.operations`
- **THEN** the import SHALL succeed and resolve to the implementation in `DocsToKG.HybridSearch.vectorstore`
- **AND** when a user imports `verify_pagination` from `DocsToKG.HybridSearch.operations`
- **THEN** the import SHALL succeed and resolve to the implementation in `DocsToKG.HybridSearch.service`

### Requirement: Functional Cohesion Within Modules

The HybridSearch subsystem SHALL group functionally related operations within individual modules such that modifications to a feature area require changes to at most one module, avoiding fragmentation of related logic across multiple files.

#### Scenario: Modifying ranking pipeline behavior

- **WHEN** a developer needs to change result ranking logic (e.g., adjusting MMR lambda parameter handling or modifying deduplication threshold)
- **THEN** all required changes SHALL be localized to the `ranking` module
- **AND** no changes SHALL be required to separate `results` or `fusion` modules

#### Scenario: Modifying FAISS index configuration

- **WHEN** a developer needs to change vector store behavior (e.g., enabling IVFPQ precomputation or adjusting nprobe)
- **THEN** all required changes SHALL be localized to the `vectorstore` module
- **AND** GPU similarity operations SHALL be co-located in the same module for joint optimization

#### Scenario: Modifying OpenSearch lexical storage

- **WHEN** a developer needs to change lexical storage behavior (e.g., updating BM25 scoring or adding new index template fields)
- **THEN** all required changes SHALL be localized to the `storage` module
- **AND** schema management and storage simulation SHALL be unified within this single module

### Requirement: Unidirectional Module Dependencies

The HybridSearch subsystem SHALL maintain a directed acyclic graph (DAG) of module dependencies where foundational modules (types, config) are imported by higher-level modules, and no circular imports exist between modules at the same architectural level.

#### Scenario: Types module has no imports from HybridSearch

- **WHEN** the `types` module is loaded
- **THEN** it SHALL NOT import from any other HybridSearch module
- **AND** it SHALL define only data structures and exception types
- **AND** all other HybridSearch modules MAY import from `types` without creating cycles

#### Scenario: Service module is a root node

- **WHEN** the `service` module is loaded
- **THEN** it MAY import from `config`, `features`, `vectorstore`, `storage`, `ranking`, `types`, and `observability`
- **AND** no other HybridSearch module SHALL import from `service` (except `validation` for testing)
- **AND** the service module SHALL act as the orchestration root without being depended upon by domain modules

#### Scenario: Vectorstore module has limited upward dependencies

- **WHEN** the `vectorstore` module is loaded
- **THEN** it MAY import only from `config` and `types`
- **AND** it SHALL NOT import from `service`, `ranking`, `storage`, or `ingest`
- **AND** all GPU-related functionality SHALL be self-contained within this module

### Requirement: CLI Entry Point Consolidation

The HybridSearch subsystem SHALL provide validation and testing capabilities through a single CLI entry point in the `validation` module, eliminating redundant scripts in the Tools directory that duplicate argument parsing and execution logic.

#### Scenario: Running basic validation suite

- **WHEN** a developer invokes `python -m DocsToKG.HybridSearch.validation --mode basic --dataset fixtures/test.jsonl --config fixtures/config.json`
- **THEN** the validation harness SHALL execute standard checks (ingest integrity, dense self-hit, sparse relevance, namespace filters, pagination, highlights, backup/restore)
- **AND** the tool SHALL produce a validation summary in JSON format
- **AND** no separate `HybridSearch/tools/run_hybrid_tests.py` script SHALL be required

#### Scenario: Running scale validation suite

- **WHEN** a developer invokes `python -m DocsToKG.HybridSearch.validation --mode scale --dataset fixtures/scale.jsonl --config fixtures/config.json --query-sample-size 120`
- **THEN** the validation harness SHALL execute comprehensive scale checks (data sanity, CRUD/namespace, dense metrics, channel relevance, fusion/MMR, pagination, shaping, backup/restore, ACL, performance, stability, calibration)
- **AND** the tool SHALL output detailed metrics to a timestamped report directory
- **AND** no separate CI-specific script SHALL be required

#### Scenario: CI pipeline invokes validation module

- **WHEN** a CI workflow executes validation as part of the build pipeline
- **THEN** the workflow SHALL invoke `python -m DocsToKG.HybridSearch.validation` with appropriate arguments
- **AND** the workflow SHALL NOT reference any file in a `HybridSearch/tools/` directory
- **AND** the validation module SHALL provide consistent argument parsing and output formatting

### Requirement: Observability Integration

The HybridSearch subsystem SHALL integrate observability capabilities (metrics collection, tracing spans, structured logging) through a unified `Observability` facade that remains available as a separate module for injection into service and ingestion pipelines.

#### Scenario: Service orchestrator records search metrics

- **WHEN** the `HybridSearchService` executes a search request
- **THEN** it SHALL use the injected `Observability` instance to record timing metrics for each channel (BM25, SPLADE, dense)
- **AND** it SHALL emit structured log events containing query, namespace, result count, and per-channel timings
- **AND** the observability module SHALL remain separate from `service.py` to allow independent testing and configuration

#### Scenario: Ingestion pipeline tracks upsert metrics

- **WHEN** the `ChunkIngestionPipeline` processes document upserts
- **THEN** it SHALL use the injected `Observability` instance to increment counters for chunks upserted and deleted
- **AND** it SHALL create tracing spans for ingest_document and ingest_dual_write operations
- **AND** the observability abstraction SHALL allow swapping implementations without modifying ingest or service code

### Requirement: Configuration Isolation

The HybridSearch subsystem SHALL isolate all configuration management (dataclasses for Chunking, Dense, Fusion, Retrieval settings and the thread-safe HybridSearchConfigManager) in a dedicated `config` module that serves as a pure data layer without operational logic.

#### Scenario: Service loads configuration at runtime

- **WHEN** the `HybridSearchService` is constructed with a `HybridSearchConfigManager` instance
- **THEN** the service SHALL call `config_manager.get()` to retrieve the current configuration
- **AND** the configuration MAY be reloaded via `config_manager.reload()` without restarting the service
- **AND** the config module SHALL NOT import from `service`, `vectorstore`, or any operational modules

#### Scenario: Dense index configuration drives FAISS setup

- **WHEN** a `FaissIndexManager` is constructed with a `DenseIndexConfig` instance
- **THEN** the manager SHALL apply all config parameters (index_type, nlist, nprobe, pq_m, device, etc.) during index creation
- **AND** configuration changes SHALL propagate through the manager without requiring code changes in `vectorstore.py`

### Requirement: Test Suite Import Path Updates

The HybridSearch test suite SHALL use consolidated module import paths preferentially over deprecated shim paths, ensuring tests validate the intended module structure and serve as migration examples for external users.

#### Scenario: Ranking tests import from ranking module

- **WHEN** test files in `tests/hybrid_search/` validate ranking functionality
- **THEN** they SHALL import `ReciprocalRankFusion`, `apply_mmr_diversification`, and `ResultShaper` from `DocsToKG.HybridSearch.ranking`
- **AND** they SHALL NOT import from deprecated `DocsToKG.HybridSearch.results` paths

#### Scenario: Vectorstore tests import similarity functions correctly

- **WHEN** test files validate GPU similarity operations
- **THEN** they SHALL import `pairwise_inner_products`, `cosine_against_corpus_gpu`, etc. from `DocsToKG.HybridSearch.vectorstore`
- **AND** they SHALL NOT use deprecated `DocsToKG.HybridSearch.similarity` imports

#### Scenario: Service tests import service operations directly

- **WHEN** test files validate pagination or stats functionality
- **THEN** they SHALL import `verify_pagination`, `build_stats_snapshot` from `DocsToKG.HybridSearch.service`
- **AND** state management functions SHALL be imported from `DocsToKG.HybridSearch.vectorstore`

### Requirement: Documentation Consistency

The HybridSearch documentation SHALL reflect the consolidated nine-module structure with accurate import examples, updated architecture diagrams, and explicit migration guidance for users transitioning from deprecated paths.

#### Scenario: API documentation shows correct import paths

- **WHEN** a user reads API documentation for `ResultShaper`
- **THEN** the documentation SHALL show the import statement `from DocsToKG.HybridSearch.ranking import ResultShaper`
- **AND** no examples SHALL reference `DocsToKG.HybridSearch.results`

#### Scenario: Architecture diagram reflects module structure

- **WHEN** a developer views the HybridSearch architecture diagram
- **THEN** the diagram SHALL depict exactly nine core modules (config, features, ingest, observability, ranking, service, storage, types, validation, vectorstore)
- **AND** dependency arrows SHALL be unidirectional from higher-level modules to foundational modules
- **AND** no deprecated modules SHALL appear in the diagram

#### Scenario: Migration guide provides import mappings

- **WHEN** a user consults the migration guide for the consolidation change
- **THEN** the guide SHALL provide a complete table mapping old import paths to new paths
- **AND** the guide SHALL include code examples demonstrating before/after import statements
- **AND** the guide SHALL specify the deprecation removal timeline

### Requirement: Behavioral Equivalence Post-Consolidation

The HybridSearch subsystem SHALL produce identical search results, scores, and GPU execution patterns after consolidation compared to pre-consolidation behavior, ensuring the refactoring introduces no semantic changes.

#### Scenario: Search results remain identical

- **WHEN** the same `HybridSearchRequest` is executed pre-consolidation and post-consolidation with identical configuration
- **THEN** the `HybridSearchResponse.results` SHALL contain the same chunks in the same order
- **AND** each result's `score` and `fused_rank` SHALL be numerically identical (within floating-point epsilon)
- **AND** highlights, provenance offsets, and diagnostics SHALL match exactly

#### Scenario: GPU device placement unchanged

- **WHEN** MMR diversification is invoked with a specific GPU device parameter
- **THEN** all tensor operations SHALL execute on the same GPU device as pre-consolidation
- **AND** FAISS index searches SHALL use the same GPU resources and device ordinal
- **AND** no CPU fallback codepaths SHALL be introduced

#### Scenario: FAISS state serialization remains compatible

- **WHEN** a FAISS index is serialized using `serialize_state` post-consolidation
- **THEN** the serialized payload SHALL be compatible with `restore_state` implementations pre-consolidation
- **AND** the restored index SHALL produce identical search results to the original index
- **AND** the state format SHALL remain a base64-encoded FAISS index byte string with registered vector IDs
