# DocParsing Embedding Specification Deltas

## ADDED Requirements

### Requirement: Embedding Providers Encapsulate Backend Dependencies
`DocsToKG.DocParsing.embedding.backends` SHALL define base interfaces (`DenseEmbeddingBackend`, `SparseEmbeddingBackend`, `LexicalEmbeddingBackend`) that expose:
- `name: str` (stable identifier, e.g., `dense.qwen_vllm`)
- `open(cfg: ProviderConfig) -> None`
- `close() -> None`
- `embed(texts: Sequence[str], *, batch_hint: int | None = None) -> list[list[float]]` for dense providers
- `encode(texts: Sequence[str]) -> list[list[tuple[str, float]]]` for sparse providers
- `accumulate_stats(chunks: Iterable[Chunk]) -> BM25Stats` and `vector(text: str, stats: BM25Stats) -> list[tuple[str, float]]` for lexical providers

Providers SHALL raise `ProviderError(provider: str, category: Literal["init", "validation", "network", "runtime"], detail: str, retryable: bool, wrapped: BaseException | None)` when initialization, request handling, or cleanup fails. Backend-specific imports, caches, batching, and concurrency management SHALL live inside provider implementations, not in the runtime.

#### Scenario: Runtime Consumes Providers Through Interfaces
- **WHEN** `DocsToKG.DocParsing.embedding.runtime.process_chunk_file_vectors` executes with dense or sparse embeddings enabled
- **THEN** it SHALL obtain provider instances from `ProviderFactory`
- **AND** it SHALL interact with those instances only via the interface methods described above
- **AND** `embedding.runtime` SHALL NOT import or reference vLLM, sentence-transformers, SPLADE, or BM25 helper classes directly.

#### Scenario: Providers Own Concurrency, Caching, and Retry Logic
- **WHEN** a backend requires batching, queueing, caching, or retry control (e.g., Qwen vLLM, TEI HTTP backends)
- **THEN** the provider SHALL implement the required logic internally using hints supplied in `EmbedCfg`
- **AND** the runtime SHALL NOT define backend-specific queues such as `QwenEmbeddingQueue`, caches like `_QWEN_LLM_CACHE`, or ad-hoc retry loops for those providers.

#### Scenario: Provider Errors Expose Structured Taxonomy
- **WHEN** a provider fails during `open`, `embed/encode/vector`, or `close`
- **THEN** it SHALL raise a `ProviderError` with the correct `category` (`init`, `validation`, `network`, or `runtime`) and `retryable` flag
- **AND** the runtime SHALL pass the `detail` message through to CLI/manifests and include the `category` and `retryable` values in telemetry.

### Requirement: Concrete Providers Cover Dense, Sparse, and Lexical Backends
DocsToKG SHALL ship provider modules for dense (`dense.qwen_vllm`, `dense.sentence_transformers`, `dense.tei`), sparse (`sparse.splade_st`), and lexical (`lexical.local_bm25`) backends. Each provider SHALL implement the appropriate interface, honour cross-cutting configuration, and publish a stable `provider_name` used in telemetry and manifests.

#### Scenario: Default Configuration Mirrors Existing Behaviour
- **WHEN** users run the embed pipeline without overriding provider keys
- **THEN** the factory SHALL select `dense.qwen_vllm`, `sparse.splade_st`, and `lexical.local_bm25`
- **AND** those providers SHALL produce vectors identical (within existing tolerance) to today’s implementation, including dimension, normalization, and ordering.

#### Scenario: Provider Lifecycle Handles Optional Dependencies
- **WHEN** an optional library required by a provider (e.g., vLLM, sentence-transformers) is missing
- **THEN** the provider SHALL raise `ProviderError(category="init", retryable=False, detail=<missing dependency message>)`
- **AND** the runtime SHALL surface the error while leaving other providers unaffected.

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
`EmbedCfg` SHALL expose provider-centric keys with documented defaults and precedence. The configuration surface MUST include at least the keys listed below; implementers MAY add future keys provided they follow the namespace pattern:

| Key | Type / Allowed Values | Default | Notes |
| --- | --- | --- | --- |
| `embedding.device` | `auto` \\| `cpu` \\| `cuda` \\| `cuda:<index>` | `auto` | Providers resolve to actual device and report fallback decisions. |
| `embedding.dtype` | `auto` \\| `float32` \\| `float16` \\| `bfloat16` | `auto` | Dense providers coerce unsupported dtypes back to `float32` with warning. |
| `embedding.batch_size` | `int >= 1` | Provider default | Global batch hint; overridden by `dense.*.batch_size` or `sparse.*.batch_size` when set. |
| `embedding.max_concurrency` | `int >= 1` | `files_parallel` | Caps provider-level concurrency/queue depth. |
| `embedding.normalize_l2` | `bool` | `true` | Dense providers normalize outputs unless disabled. |
| `embedding.offline` | `bool` | `false` | Providers fail fast if downloads required while offline. |
| `embedding.cache_dir` | `Path | None` | Tool default | Shared cache root for model artifacts. |
| `embedding.telemetry_tags` | `dict[str, str]` | `{}` | Applied to every provider telemetry event. |
| `dense.backend` | `qwen_vllm` \\| `sentence_transformers` \\| `tei` \\| `none` | `qwen_vllm` | Determines dense provider. |
| `dense.qwen_vllm.model_id` | `str` | Existing default | Must match current Qwen model. |
| `dense.qwen_vllm.download_dir` | `Path | None` | `None` | Overrides weight cache root. |
| `dense.qwen_vllm.batch_size` | `int >= 1` | Existing default | Provider-specific batch override. |
| `dense.qwen_vllm.queue_depth` | `int >= 1` | Derived from `embedding.max_concurrency` | Controls request queue length. |
| `dense.tei.url` | `str` (validated URL) | _required when backend=tei_ | Must be HTTPS; provider enforces TLS. |
| `dense.tei.timeout_seconds` | `float > 0` | 30 | Per-request timeout. |
| `dense.tei.max_inflight` | `int >= 1` | `embedding.max_concurrency` | Caps HTTP inflight requests. |
| `dense.sentence_transformers.model_id` | `str` | Current default | Local SBERT/SPLADE-compatible identifier. |
| `sparse.backend` | `splade_st` \\| `none` | `splade_st` | Determines sparse provider. |
| `sparse.splade_st.model_id` | `str` | Present default | Matches current SPLADE checkpoint. |
| `sparse.splade_st.batch_size` | `int >= 1` | Existing default | Without override, uses `embedding.batch_size`. |
| `sparse.splade_st.attn_backend` | `auto` \\| `flash` \\| `sdpa` \\| `eager` | `auto` | Provider handles fallback ordering. |
| `lexical.backend` | `local_bm25` \\| `none` | `local_bm25` | Determines lexical provider. |
| `lexical.local_bm25.k1` | `float > 0` | 1.5 | Preserves legacy default. |
| `lexical.local_bm25.b` | `float` where `0 <= b <= 1` | 0.75 | Preserves legacy default. |

Implementers MUST document new keys in provider docs and update validation logic accordingly.

#### Scenario: Configuration Precedence Is Deterministic
- **WHEN** conflicting values exist for the same key across config file, environment variable (`DOCSTOKG_*`), and CLI flag
- **THEN** the resolved `EmbedCfg` SHALL choose CLI values over environment values, environment values over config file values, and config values over defaults
- **AND** the final config SHALL be the source of truth passed into `ProviderFactory`.

#### Scenario: Legacy Flags and Env Vars Map to New Keys
- **WHEN** users provide legacy options (e.g., `--bm25-k1`, `--qwen-batch`, `--sparse-model`, `DOCSTOKG_QWEN_DIR`, `DOCSTOKG_TEI_URL`)
- **THEN** the CLI/config loader SHALL translate them into the corresponding `lexical.local_bm25.k1`, `dense.qwen_vllm.batch_size`, `sparse.splade_st.model_id`, `dense.qwen_vllm.download_dir`, `dense.tei.url` keys
- **AND** it SHALL emit a single-line deprecation warning while allowing new-key values to override legacy values when both are present.

#### Scenario: Configuration Validation Provides Remediation Guidance
- **WHEN** validation fails (e.g., `dense.tei.url` missing, `lexical.local_bm25.k1` ≤ 0, unsupported dtype)
- **THEN** the raised `EmbeddingCLIValidationError` SHALL identify the offending key and include a remediation message (e.g., “Provide --dense-tei-url or set dense.backend=sentence_transformers.”)

### Requirement: Telemetry and Errors Are Standardized
Providers SHALL emit structured telemetry per batch (or per file) covering `provider_name`, `provider_version`, `device`, `dtype`, `batch_size_effective`, `max_inflight_requests`, `time_open_ms`, `time_embed_ms`, `time_close_ms`, `normalize_l2`, and `fallback_used`. All provider failures SHALL surface as `ProviderError` instances with populated `provider`, `category`, `detail`, and `retryable` fields that the runtime logs and serializes into manifests.

#### Scenario: Telemetry Augments Vector Manifests
- **WHEN** embedding completes for a chunk file
- **THEN** the manifest/telemetry event SHALL include the provider fields listed above for each enabled backend
- **AND** existing schema keys SHALL remain untouched so downstream consumers continue to parse manifests.

#### Scenario: Provider Errors Propagate with Taxonomy
- **WHEN** a provider raises `ProviderError` during `open`, `embed/encode/vector`, or `close`
- **THEN** the runtime SHALL capture the error, annotate telemetry/manifests with `provider`, `category`, and `retryable`, present a human-oriented message (using `detail`), and decide on retries based on `retryable` without converting the exception to an unstructured string.

#### Scenario: Telemetry Captures Fallback Behaviour
- **WHEN** a provider triggers an internal fallback (e.g., TEI unavailable so Sentence-Transformers is used)
- **THEN** telemetry SHALL set `fallback_used=true` and include a `fallback_target` tag describing the substitute backend.

### Requirement: Behaviour Parity and Validation Safeguards
The refactor SHALL preserve observable behaviour for default configurations and introduce regression tests that exercise provider selection, configuration precedence, and dependency handling.

#### Scenario: Default Providers Produce Parity Outputs
- **WHEN** the embed pipeline runs with the default configuration on the existing test corpus
- **THEN** JSONL and Parquet outputs SHALL match pre-refactor baselines within floating point tolerances
- **AND** normalization, dimension, and ordering SHALL remain unchanged.

#### Scenario: Config and Dependency Validation Is Tested
- **WHEN** automated tests run
- **THEN** they SHALL cover provider selection across all supported backends (including `none`), legacy alias mapping, TEI URL validation, SPLADE/BM25 parameter validation, and optional dependency skip behaviour (marking tests as skipped when libraries are unavailable).

#### Scenario: Parity Tests Cover Plan-Only and Tracemalloc Modes
- **WHEN** the regression suite executes plan-only mode or tracemalloc-enabled runs
- **THEN** results SHALL match current behaviour (no premature provider initialization) and the suite SHALL assert that providers are not opened when no embedding work is performed.
