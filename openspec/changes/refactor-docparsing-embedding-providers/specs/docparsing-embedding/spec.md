# DocParsing Embedding Specification Deltas

## ADDED Requirements

### Requirement: Embedding Providers Encapsulate Backend Dependencies
`DocsToKG.DocParsing.embedding.backends` SHALL define narrow base interfaces (`DenseEmbeddingBackend`, `SparseEmbeddingBackend`, `LexicalEmbeddingBackend`) that expose `name`, `open(cfg)`, `embed|encode|vector(...)`, and `close()` methods while raising a structured `ProviderError` for failures. Backend-specific imports, caches, batching, and concurrency management SHALL live inside provider implementations.

#### Scenario: Runtime Consumes Providers Through Interfaces
- **WHEN** `DocsToKG.DocParsing.embedding.runtime.process_chunk_file_vectors` executes with dense or sparse embeddings enabled
- **THEN** it SHALL obtain provider instances from `ProviderFactory`
- **AND** it SHALL interact with those instances only via the interface methods (`open`, `embed/encode/vector`, `close`)
- **AND** `embedding.runtime` SHALL NOT import or reference vLLM, sentence-transformers, SPLADE, or BM25 helper classes directly.

#### Scenario: Providers Own Concurrency and Caching
- **WHEN** a backend requires batching, queueing, or model caching (e.g., Qwen vLLM, TEI HTTP backends)
- **THEN** the provider SHALL implement the necessary queue/cache logic internally using hints supplied in `EmbedCfg`
- **AND** the runtime SHALL not define or manage backend-specific queues such as `QwenEmbeddingQueue` or `_QWEN_LLM_CACHE`.

### Requirement: Concrete Providers Cover Dense, Sparse, and Lexical Backends
DocsToKG SHALL ship provider modules for dense (`dense.qwen_vllm`, `dense.sentence_transformers`, `dense.tei`), sparse (`sparse.splade_st`), and lexical (`lexical.local_bm25`) backends. Each provider SHALL implement the appropriate interface, honour cross-cutting configuration, and publish a stable `provider_name` used in telemetry and manifests.

#### Scenario: Default Configuration Mirrors Existing Behaviour
- **WHEN** users run the embed pipeline without overriding provider keys
- **THEN** the factory SHALL select `dense.qwen_vllm`, `sparse.splade_st`, and `lexical.local_bm25`
- **AND** those providers SHALL produce vectors identical (within existing tolerance) to today’s implementation, including dimension, normalization, and ordering.

#### Scenario: TEI and Optional Backends Validate Required Keys
- **WHEN** `dense.backend=tei`
- **THEN** the TEI provider SHALL require a non-empty `dense.tei.url` (or `DOCSTOKG_TEI_URL`) and raise `ProviderError(category="init", retryable=False)` if missing or invalid
- **AND** it SHALL enforce inflight concurrency caps, TLS defaults, and timeout configuration without altering runtime code.

#### Scenario: Backends Can Be Disabled Explicitly
- **WHEN** `dense.backend=none`, `sparse.backend=none`, or `lexical.backend=none`
- **THEN** the factory SHALL return a no-op provider shim
- **AND** the runtime SHALL skip the corresponding work while preserving manifests, telemetry, and writer behaviour for the remaining providers.

### Requirement: Provider Factory Governs Lifecycle and Hints
`ProviderFactory` SHALL accept a fully merged `EmbedCfg`, instantiate at most one provider per backend, call `open()` before use, and guarantee `close()` executes even on failure. It SHALL propagate shared hints (`embedding.batch_size`, `embedding.max_concurrency`, `embedding.normalize_l2`, `embedding.device`, `embedding.dtype`) to providers at construction time.

#### Scenario: Providers Open Once Per Run and Always Close
- **WHEN** the embed pipeline starts Pass B (vector writing)
- **THEN** the factory SHALL create provider instances exactly once, call `open()` before processing any chunks, and ensure `close()` executes in a `finally` block even if batch processing raises errors.

#### Scenario: Runtime Supplies Batch and Concurrency Hints
- **WHEN** `EmbedCfg` specifies `embedding.batch_size` or `embedding.max_concurrency`
- **THEN** the factory SHALL pass those values (and any per-backend overrides) to the provider constructor or `open()` method
- **AND** providers SHALL honour or adapt the hints without the runtime manually splitting queues.

### Requirement: Embedding Configuration Keys Are Normalized
`EmbedCfg` SHALL expose provider-centric keys with documented defaults and precedence: cross-cutting keys under `embedding.*`, dense keys under `dense.*`, sparse keys under `sparse.*`, and lexical keys under `lexical.*`. Allowed keys include (non-exhaustive): `embedding.device` (default `auto`), `embedding.dtype` (`auto`), `embedding.batch_size` (int ≥1, default provider-defined), `embedding.max_concurrency` (defaults to `files_parallel`), `embedding.normalize_l2` (default `true`), `embedding.offline` (default `false`), `embedding.cache_dir`, `embedding.telemetry_tags`, `dense.backend` (default `qwen_vllm`), `dense.qwen_vllm.model_id`, `dense.qwen_vllm.download_dir`, `dense.qwen_vllm.batch_size`, `dense.tei.url`, `sparse.backend` (default `splade_st`), `sparse.splade_st.model_id`, `lexical.backend` (default `local_bm25`), `lexical.local_bm25.k1`, `lexical.local_bm25.b`.

#### Scenario: Configuration Precedence Is Deterministic
- **WHEN** conflicting values exist for the same key across config file, environment variable (`DOCSTOKG_*`), and CLI flag
- **THEN** the resolved `EmbedCfg` SHALL choose CLI values over environment values, environment values over config file values, and config values over defaults
- **AND** the final config SHALL be the source of truth passed into `ProviderFactory`.

#### Scenario: Legacy Flags and Env Vars Map to New Keys
- **WHEN** users provide legacy options (e.g., `--bm25-k1`, `--qwen-batch`, `DOCSTOKG_QWEN_DIR`)
- **THEN** the CLI/config loader SHALL translate them into the corresponding `lexical.local_bm25.k1`, `dense.qwen_vllm.batch_size`, `dense.qwen_vllm.download_dir` keys
- **AND** it SHALL emit a single-line deprecation warning while allowing the new keys to win if both are supplied.

### Requirement: Telemetry and Errors Are Standardized
Providers SHALL emit structured telemetry per batch (or per file) covering `provider_name`, `provider_version`, `device`, `dtype`, `batch_size_effective`, `max_inflight_requests`, `time_open_ms`, `time_embed_ms`, `time_close_ms`, `normalize_l2`, and `fallback_used`. All provider failures SHALL surface as `ProviderError` instances with populated `provider`, `category`, `detail`, and `retryable` fields that the runtime logs and serializes into manifests.

#### Scenario: Telemetry Augments Vector Manifests
- **WHEN** embedding completes for a chunk file
- **THEN** the manifest/telemetry event SHALL include the provider fields listed above for each enabled backend
- **AND** existing schema keys SHALL remain untouched so downstream consumers continue to parse manifests.

#### Scenario: Provider Errors Propagate with Taxonomy
- **WHEN** a provider raises `ProviderError` during `open`, `embed/encode/vector`, or `close`
- **THEN** the runtime SHALL capture the error, annotate telemetry/manifests with `provider`, `category`, and `retryable`, present a human-oriented message (using `detail`), and decide on retries based on `retryable` without converting the exception to an unstructured string.

### Requirement: Behaviour Parity and Validation Safeguards
The refactor SHALL preserve observable behaviour for default configurations and introduce regression tests that exercise provider selection, configuration precedence, and dependency handling.

#### Scenario: Default Providers Produce Parity Outputs
- **WHEN** the embed pipeline runs with the default configuration on the existing test corpus
- **THEN** JSONL and Parquet outputs SHALL match pre-refactor baselines within floating point tolerances
- **AND** normalization, dimension, and ordering SHALL remain unchanged.

#### Scenario: Config and Dependency Validation Is Tested
- **WHEN** automated tests run
- **THEN** they SHALL cover provider selection across all supported backends (including `none`), legacy alias mapping, TEI URL validation, SPLADE/BM25 parameter validation, and optional dependency skip behaviour (marking tests as skipped when libraries are unavailable).
