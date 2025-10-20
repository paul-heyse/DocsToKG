## Why
Embedding currently binds backend-specific logic directly into `DocsToKG.DocParsing.embedding.runtime`. The module imports vLLM and sentence-transformers, owns bespoke queues/caches (for example `_QWEN_LLM_CACHE` and `QwenEmbeddingQueue`), and decides how to batch or retry each backend. Configuration knobs are fragmented across `EmbedCfg`, CLI flags, and ad-hoc environment variables, so adding a new backend requires touching runtime plumbing, config loaders, telemetry, and docs in tandem. That coupling makes provider swaps (e.g., TEI vs. local Qwen) risky for junior contributors, obscures provenance in telemetry, and complicates offline runs where optional dependencies are missing.

## What Changes
- Build a dedicated `DocsToKG.DocParsing.embedding.backends` package containing:
  - Base protocols (`DenseEmbeddingBackend`, `SparseEmbeddingBackend`, `LexicalEmbeddingBackend`) that define exact method signatures (`open(cfg: ProviderConfig) -> None`, `embed(texts: Sequence[str], batch_hint: int | None) -> list[list[float]]`, etc.) and the `ProviderError` taxonomy (`provider`, `category`, `detail`, `retryable`).
  - A `ProviderFactory` responsible for merging configuration, lazy-importing providers, managing lifecycle (`open`/`close`), injecting telemetry hooks, and returning null-provider shims when a backend is disabled.
- Ship concrete providers with full lifecycle ownership:
  - `dense.qwen_vllm` absorbs vLLM imports, caching, queue depth logic derived from `files_parallel`, warm-up, and normalization defaults.
  - `dense.sentence_transformers` encapsulates SBERT/SPLADE local models, cache directory usage, offline validation, and dtype/device handling.
  - `dense.tei` wraps the HTTP client (TLS, timeout, concurrency hints) and error handling for TEI deployments.
  - `sparse.splade_st` owns SPLADE dependency checks, attention backend fallbacks, and sparsity threshold reporting.
  - `lexical.local_bm25` manages stats accumulation, per-token weights, and deterministic ordering.
- Reshape configuration so provider selection and tuning live under nested keys with documented defaults:
  - `embedding.*` cross-cutting keys (`device`, `dtype`, `batch_size`, `max_concurrency`, `normalize_l2`, `offline`, `cache_dir`, `telemetry_tags`).
  - `dense.*`, `sparse.*`, `lexical.*` namespaces containing backend identifiers, per-backend overrides, and validation rules.
  - CLI/env/config precedence rules (CLI > ENV > config > defaults) plus a compatibility layer that maps every legacy flag/env (e.g., `--bm25-k1`, `DOCSTOKG_QWEN_DIR`) into the new structure while emitting single-line deprecation notices.
- Refactor `embedding.runtime` so it obtains providers from the factory, calls interface methods only, and drops direct imports of vLLM or sentence-transformers, as well as helpers like `_get_vllm_components`, `_QWEN_LLM_CACHE`, `QwenEmbeddingQueue`, `_get_sparse_encoder_cls`, `_ensure_*_dependencies`, and related globals.
- Expand telemetry coverage: every provider call reports `provider_name`, model revision, device/dtype, effective batch size, inflight concurrency, timing metrics (open/embed/close), normalization flag, fallback usage, and structured errors that propagate to manifests.
- Strengthen validation and testing to make the refactor safe for junior contributors: unit tests for provider selection/config precedence/legacy aliasing/error categories, provider-level smoke tests (shape, normalization, network retry semantics), integration parity tests for JSONL and Parquet outputs (including plan-only/tracemalloc flows), and documentation walkthroughs showcasing new backend selection.
- Update user and maintainer docs (DocParsing README/API reference, ProviderOverview, operations guides) with configuration examples, offline expectations, telemetry fields, and instructions for adding new providers or fallbacks.

## Impact
- Affected specs: docparsing-embedding
- Affected code: `src/DocsToKG/DocParsing/embedding/runtime.py`, `src/DocsToKG/DocParsing/embedding/config.py`, new `src/DocsToKG/DocParsing/embedding/backends/*`, CLI config loaders, telemetry emitters, manifest builders, unit/integration tests, and DocParsing documentation/handbooks.
