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
- **Provider interfaces**: Define `DenseEmbeddingBackend`, `SparseEmbeddingBackend`, and `LexicalEmbeddingBackend` as small Protocols (or ABCs) with `open(cfg)`, `embed|encode|vector(...)`, and `close()` plus a `name` property. Providers raise `ProviderError(provider, category, detail, retryable)` so callers can distinguish init vs runtime vs network issues.
- **Module layout**: New package `DocsToKG.DocParsing.embedding.backends` hosts `base.py`, `factory.py`, and submodules: `dense/sentence_transformers.py`, `dense/tei.py`, `dense/qwen_vllm.py`, `sparse/splade_st.py`, `lexical/local_bm25.py`. Optional helpers live in `utils.py` to avoid runtime dependency leakage.
- **Factory behaviour**: `ProviderFactory` accepts fully merged `EmbedCfg`, resolves backend names (`dense.backend` etc.), lazy-imports the requested module, instantiates provider objects, and coordinates lifecycle (ensuring `open`/`close` occurs exactly once). Passing `backend=none` returns a `NullProvider` shim that the runtime understands as disabled.
- **Configuration schema**: Rework `EmbedCfg` to expose cross-cutting keys under `embedding.*` (device, dtype, batch size, concurrency, normalize_l2, offline, cache_dir, telemetry tags) and nested provider keys (`dense.backend`, `dense.qwen_vllm.model_id`, `sparse.splade_st.model_id`, `lexical.local_bm25.k1`, etc.). CLI > ENV > config > defaults precedence is enforced centrally, and legacy flags/envs populate the new keys with deprecation warnings.
- **Runtime integration**: `process_pass_a` and `process_chunk_file_vectors` request providers from the factory, then only use interface methods. Backend-specific helpers (`_get_vllm_components`, `_QWEN_LLM_CACHE`, `_get_sparse_encoder_cls`, `QwenEmbeddingQueue`) are deleted; concurrency/queue handling moves into providers, each respecting hints like `embedding.batch_size` and `embedding.max_concurrency`.
- **Telemetry**: Providers emit structured telemetry via call-backs or context managers supplied by the runtime/factory, producing per-batch tags: `provider_name`, `provider_version`, `device`, `dtype`, `batch_size_effective`, `max_inflight_requests`, `time_open_ms`, `time_embed_ms`, `time_close_ms`, `normalize_l2`, and `fallback_used`. Errors attach `ProviderError` fields and propagate to manifests.

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

## Open Questions
- Should we add an explicit provider fallback chain (e.g., try TEI, then Sentence-Transformers) in this refactor or leave it for a follow-up once telemetry proves useful?
- Do we need to version provider configurations (e.g., hash of model + tokenizer) in manifests now, or is emitting raw strings sufficient for current compliance needs?
- How aggressively should we enforce device availability (fail vs. warn when `cuda:N` missing)? Current plan is to downgrade with warnings; stakeholders might want a strict mode flag.
