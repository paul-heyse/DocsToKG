## Why
Embedding currently binds backend-specific logic directly into `DocsToKG.DocParsing.embedding.runtime`: the module imports vLLM and sentence-transformers, manages bespoke queues/caches, and wires dense, sparse, and lexical behaviour through ad-hoc helpers. Configuration is likewise fragmented across `EmbedCfg`, CLI flags, and environment variables. This tight coupling makes it painful to swap providers (e.g., TEI vs. local Qwen), tempts regressions when optional dependencies are missing, and forces every enhancement to touch runtime plumbing, tests, and docs in tandem.

## What Changes
- Introduce `DocsToKG.DocParsing.embedding.backends` with base protocols (`DenseEmbeddingBackend`, `SparseEmbeddingBackend`, `LexicalEmbeddingBackend`) and a structured `ProviderError`, plus a `ProviderFactory` that builds providers from `EmbedCfg` inputs.
- Ship concrete providers: `dense.sentence_transformers`, `dense.tei`, `dense.qwen_vllm`, `sparse.splade_st`, and `lexical.local_bm25`, each responsible for their own imports, batching, caching, and lifecycle.
- Reshape `EmbedCfg` so provider selection lives under `embedding.*`, `dense.*`, `sparse.*`, and `lexical.*` keys with explicit defaults, CLI/env/config precedence, and legacy flag/env aliases that emit deprecation notices.
- Refactor `embedding.runtime` (and helpers) to request providers from the factory, call `open/embed|encode/vector/close`, and drop direct use of `_get_vllm_components`, `_QWEN_LLM_CACHE`, `QwenEmbeddingQueue`, `_get_sparse_encoder_cls`, and related globals.
- Expand telemetry: every provider emits `provider_name`, model/device/dtype, effective batch/concurrency, timings, normalization flag, and structured error signals so manifests and logs capture backend provenance.
- Strengthen validation and testing: unit cases for provider selection, config precedence, legacy alias mapping, telemetry emission, and parity runs that confirm default dense/sparse/lexical outputs match today’s JSONL/Parquet artefacts.
- Update docs (DocParsing README/API, provider best practices) and developer tooling to describe backend choices, required keys (e.g., `dense.tei.url`), offline behaviour, and migration guidance.

## Impact
- Affected specs: docparsing-embedding
- Affected code: `src/DocsToKG/DocParsing/embedding/runtime.py`, `src/DocsToKG/DocParsing/embedding/config.py`, new `src/DocsToKG/DocParsing/embedding/backends/*`, CLI config loaders, telemetry emitters, unit/integration tests, and DocParsing documentation
