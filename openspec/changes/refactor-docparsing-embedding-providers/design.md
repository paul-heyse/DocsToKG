## Context
`DocsToKG.DocParsing.embedding.runtime` currently owns every detail of dense, sparse, and lexical vectorisation: it imports vLLM, sentence-transformers, and SPLADE helpers; manages bespoke queues/caches; and stitches outputs straight into vector writers. Configuration is spread across `EmbedCfg`, CLI flags, and environment variables with inconsistent naming. As new backends (TEI, PySerini, alternative BM25 providers) come online, the runtime has become a tangle of optional imports, branching logic, and duplicated batching/concurrency code. Downstream teams want to experiment with different providers without modifying the runtime core, while ops wants stronger provenance (which backend produced which vectors) and predictable failure semantics.

## Goals / Non-Goals
- **Goals**
  - Decouple backend-specific code from `embedding.runtime` via narrow provider interfaces and a factory that honours configuration precedence.
  - Support dense providers (Qwen vLLM, Sentence-Transformers, TEI HTTP), SPLADE sparse encoding, and BM25 lexical weighting as swappable modules with consistent telemetry and error handling.
  - Normalize configuration keys (`embedding.*`, `dense.*`, `sparse.*`, `lexical.*`) and map legacy flags/env vars without breaking existing pipelines.
  - Guarantee parity: default configuration must emit the same vectors (within floating point tolerance) and manifests as today, including JSONL and Parquet writers.
  - Deliver observability hooks so manifests/logs capture provider name, model version, device/dtype, batching, and timing metadata.
- **Non-Goals**
  - Refactor retrieval-time provider usage (HybridSearch) or change vector file formats/schema.
  - Introduce new providers beyond the scoped set (e.g., PySerini BM25, hybrid fallbacks) although stubs may be prepared.
  - Modify writer abstractions (`VectorWriter`, `create_vector_writer`) or chunk discovery/manifests.

## Decisions
- **Provider interfaces**: Define `DenseEmbeddingBackend`, `SparseEmbeddingBackend`, and `LexicalEmbeddingBackend` as small Protocols (or ABCs) with `open(cfg: ProviderConfig) -> None`, `embed|encode|vector(batch, *, batch_hint=None) -> Sequence[...]`, and `close() -> None` plus a `name` property. Providers raise `ProviderError(provider, category, detail, retryable, wrapped)` so callers can distinguish init vs runtime vs network issues and surface remediation hints.
- **Module layout**: New package `DocsToKG.DocParsing.embedding.backends` hosts `base.py`, `factory.py`, shared `utils.py`, and subpackages `dense/`, `sparse/`, `lexical/`. Concrete modules include `dense/sentence_transformers.py`, `dense/tei.py`, `dense/qwen_vllm.py`, `sparse/splade_st.py`, and `lexical/local_bm25.py`. The package exposes a `ProviderBundle` typing alias summarising the three providers returned to the runtime.
- **Factory behaviour**: `ProviderFactory` accepts a fully merged `EmbedCfg`, resolves backend names (`dense.backend`, etc.), lazy-imports the requested module, instantiates provider objects with shared hints (device, dtype, batch size, concurrency, cache directory, normalization flags), and coordinates lifecycle (ensuring `open`/`close` occurs exactly once). Passing `backend=none` returns a `NullProvider` shim that the runtime treats as disabled. Factory exposes a context manager API so runtime integrations require minimal boilerplate.
- **Configuration schema**: Rework `EmbedCfg` to expose cross-cutting keys under `embedding.*` (device, dtype, batch size, max_concurrency, normalize_l2, offline, cache_dir, telemetry tags) and nested provider keys (`dense.backend`, `dense.qwen_vllm.model_id`, `dense.tei.url`, `sparse.backend`, `sparse.splade_st.model_id`, `lexical.backend`, `lexical.local_bm25.k1`, `lexical.local_bm25.b`, etc.). CLI > ENV > config > defaults precedence is enforced centrally, and every legacy flag/env populates the new keys with single-line deprecation warnings.
- **Runtime integration**: `process_pass_a` and `process_chunk_file_vectors` request providers from the factory, use interface methods only, and rely on provider-managed queues/caches. Backend-specific helpers (`_get_vllm_components`, `_QWEN_LLM_CACHE`, `_get_sparse_encoder_cls`, `_qwen_embed_direct`, `QwenEmbeddingQueue`, `_ensure_*_dependencies`) are removed from the runtime. Concurrency hints like `embedding.batch_size` and `embedding.max_concurrency` are passed through the factory instead of being hard-coded in the runtime.
- **Telemetry**: Providers emit structured telemetry via callbacks supplied by the factory, producing per-batch tags: `provider_name`, `provider_version`, `device`, `dtype`, `batch_size_effective`, `max_inflight_requests`, `time_open_ms`, `time_embed_ms`, `time_close_ms`, `normalize_l2`, and `fallback_used`. Errors attach `ProviderError` fields and propagate to manifests and CLI messaging. Telemetry inherits `embedding.telemetry_tags` so downstream analytics can correlate runs.
- **Developer ergonomics**: A provider onboarding checklist, configuration mapping table, and doc updates accompany the change so junior contributors can add providers by following explicit steps (config key allocation, telemetry wiring, tests, docs).

## Data Flow Overview
1. **Configuration ingestion**
   - Typer CLI, environment variables, and config files populate `EmbedCfg`. A compatibility layer rewrites legacy inputs (e.g., `--bm25-k1`) into nested keys and logs deprecation warnings.
   - `EmbedCfg.finalize()` resolves defaults, applies precedence, expands relative paths, and derives provider hints such as `batch_hint`, `queue_depth`, and `cache_dir`.
2. **Provider factory**
   - `ProviderFactory.build(cfg, telemetry_sink)` lazily imports provider modules, instantiates each backend (`dense`, `sparse`, `lexical`), and injects shared hints (device, dtype, batch size, max concurrency, normalize_l2, cache directory, telemetry tags).
   - The factory returns a context-managed bundle (`with ProviderFactory.bundle(cfg) as providers:`) that calls `open()` for each provider on entry and guarantees `close()` on exit even when processing errors occur.
3. **Runtime execution**
   - `process_pass_a` uses the lexical provider to compute BM25 stats rather than directly invoking `BM25StatsAccumulator`.
   - `process_chunk_file_vectors` iterates chunk batches, delegates dense/sparse vector generation to providers (`provider.embed(...)`, `provider.encode(...)`, `provider.vector(...)`), validates outputs, merges telemetry, and writes vectors using existing writers.
   - Disabled providers (`backend=none`) return neutral shims; runtime skips vector generation for those categories without altering manifest semantics.
4. **Telemetry & manifests**
   - Providers emit telemetry events through supplied callbacks; runtime augments manifest entries with provider metadata (name, version, device/dtype, normalization flag, fallback state).
   - Failures raise `ProviderError`, which runtime logs, records in manifests, and converts into CLI error messages while preserving existing retry behaviour.
5. **Shutdown**
   - Runtime `finally` blocks (or context manager exit) call `provider.close()` to free GPU memory, HTTP sessions, queues, and other resources deterministically.

## Risks / Trade-offs
- **Parity drift**: Moving logic risks subtle behavioural changes (e.g., different normalization defaults). Mitigation: copy existing implementations into providers, add regression tests comparing outputs, and run focused integration tests on default configs.
- **Dependency availability**: Providers import heavy libs lazily; if imports fail, factory must produce clear `ProviderError` messages without breaking other providers. Document fallback expectations and ensure runtime skips disabled providers cleanly.
- **Complexity**: Introducing multiple modules increases surface area. Keep interfaces minimal, share utilities judiciously, and document provider contract so future contributors stay aligned.
- **Telemetry overhead**: Capturing additional metadata may incur slight runtime cost. Instrumentation should reuse existing telemetry paths and avoid expensive computations (e.g., GPU metrics) unless available.

## Migration Plan
1. Land provider scaffolding and dense providers (Qwen, Sentence-Transformers, TEI), guarded behind factory selection switches.
2. Port SPLADE and BM25 logic into providers while keeping runtime adapters that call into the factory.
3. Replace runtime references to old helpers, ensuring compatibility shims exist until tests confirm parity.
4. Enable new configuration keys and alias legacy flags/env vars; update CLI option parsing and docs simultaneously.
5. Run integration suite (JSONL/Parquet) and compare against baselines; adjust tolerances if needed.
6. Remove obsolete runtime helpers once parity and documentation updates are confirmed.
7. Publish updated telemetry/manifest field documentation and the provider onboarding checklist; communicate configuration changes to downstream consumers.

## Open Questions
- Should we add an explicit provider fallback chain (e.g., try TEI, then Sentence-Transformers) in this refactor or leave it for a follow-up once telemetry proves useful?
- Do we need to version provider configurations (e.g., hash of model + tokenizer) in manifests now, or is emitting raw strings sufficient for current compliance needs?
- How aggressively should we enforce device availability (fail vs. warn when `cuda:N` missing)? Current plan is to downgrade with warnings; stakeholders might want a strict mode flag.
