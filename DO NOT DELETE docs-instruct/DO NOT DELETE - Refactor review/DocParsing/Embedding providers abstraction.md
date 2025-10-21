Absolutely—here’s a **formal change proposal (RFC)** for the next opportunity on the roadmap: **moving dense / sparse / lexical embedding logic behind “providers” with a tiny, stable interface**. This is the design we’ll implement as **PR-4: Providers Abstraction**.

---

# RFC: Embedding Providers Abstraction (PR-4)

## 0) Summary

We will carve dense, sparse (SPLADE), and lexical (BM25) embedding logic out of `embedding/runtime.py` into small, swappable **provider** modules. The runtime (and, after PR-5, the unified **Runner**) will no longer know about vLLM, Sentence-Transformers, TEI, SPLADE, or BM25 details. Providers own imports, model lifecycles, queues, batching, device selection, HTTP sessions, and error taxonomy; the runtime calls a **3–5 method** interface and writes outputs.

**Outcomes**

* Smaller runtime; fewer conditionals.
* Backends are pluggable (e.g., flip Qwen vLLM → TEI via config).
* Deterministic semantics (normalization, tokenizer revisions, provenance).
* Easier performance tuning and failure handling localized to the backend.
* Clean seam for future backends (e.g., OpenAI E5, GTE-Xa, new SPLADE checkpoints).

---

## 1) Motivation

* The current runtime mixes orchestration with backend details (imports, caches, batching, queues, error handling). This increases complexity, makes changes risky, and spreads similar logic across code paths.
* We want **library-first** reuse and **neat surfaces** (your strategic goal), while supporting multiple execution modes (local GPU, remote HTTP, CPU fallback).

---

## 2) Goals / Non-Goals

**Goals**

1. Introduce **tiny provider interfaces**: `DenseEmbeddingBackend`, `SparseEmbeddingBackend`, `LexicalBackend`.
2. Move **all** backend-specific logic (imports, sessions, queues, batching, device, dtype, caching, retries) into provider modules.
3. Expose **one factory** to construct providers from config.
4. Keep **runtime semantics unchanged** (same outputs, manifests, vector schemas).
5. Add **uniform error taxonomy** (`ProviderError` with `category`, `retryable`) and **telemetry** tags (provider name, model id/rev, device, dtype, batch size).
6. Support **Parquet footers** & manifest fields for provenance.

**Non-Goals (this PR)**

* No changes to vector file formats (already defined in PR-8).
* No new retrieval features or indexers.
* No orchestrator (Prefect) work (comes later).
* No quantization/IVF/PQ (future).

---

## 3) Current State (pain points)

* `embedding/runtime.py` initializes vLLM, queues requests, manages batch sizes, and handles ST/TEI differences inline.
* SPLADE model selection and post-processing are embedded in the runtime.
* BM25 term weighting and corpus stats are interleaved with file-parallel logic.
* Error handling is ad-hoc (e.g., GPU OOM vs HTTP 429 treated differently per code path).
* Configuration is parsed in multiple places; provenance is partly in manifests and partly implicit.

---

## 4) High-Level Design

### 4.1 Provider Interfaces

**DenseEmbeddingBackend**

* `name -> str` (e.g., `"dense.qwen_vllm"`)
* `open(cfg) -> None` (initialize weights/server/queue)
* `embed(texts: list[str], *, batch_size: int|None = None) -> list[list[float]]`
* `close() -> None`

**SparseEmbeddingBackend** (SPLADE)

* `name -> str` (e.g., `"sparse.splade_st"`)
* `open(cfg) -> None`
* `encode(texts: list[str]) -> list[tuple[list[int], list[float]]]`
  (parallel `indices`, `weights`; `nnz = len(indices) = len(weights)`)
* `close() -> None`

**LexicalBackend** (BM25)

* `name -> str` (e.g., `"lexical.local_bm25"`)
* `open(cfg) -> None`
* `accumulate_stats(chunks_iter) -> Stats` (avgdl, N, df map; policy: per-file or corpus)
* `vector(text: str, stats: Stats) -> tuple[list[int|str], list[float]]`
* `close() -> None`

**Uniform error type**: `ProviderError(provider, category, retryable, detail)`

### 4.2 Providers we will ship

* `dense_qwen_vllm` (local GPU via vLLM)
* `dense_tei` (HTTP Text-Embeddings-Inference)
* `dense_sentence_transformers` (local CPU/GPU)
* `sparse_splade_st` (SPLADE via HF / torch)
* `lexical_local_bm25` (in-process BM25)

### 4.3 Provider Factory

`ProviderFactory.from_cfg(embed_cfg) -> (dense: DenseEmbeddingBackend|None, sparse: SparseEmbeddingBackend|None, lexical: LexicalBackend|None)`

* Validates required fields (e.g., TEI `url`, ST `model_id`).
* Maps legacy flags/env → new settings (PR-7 compatibility).
* No heavy imports at module import time; providers perform **lazy imports** inside `open()`.

### 4.4 Lifecycle with Runner (PR-5)

* `before_stage` hook: `dense.open(cfg)`, `sparse.open(cfg)`, `lexical.open(cfg)`
* `worker(item)`: read chunk file; call providers; write vectors; return counts.
* `after_stage`: `provider.close()`, write corpus summary.

### 4.5 Configuration & Precedence

* Settings live under:

  * `embed.dense.backend` (`qwen_vllm|tei|sentence_transformers|none`)
  * `embed.dense.qwen_vllm.*` (model_id, device, dtype, TP, batch_size, mem util, etc.)
  * `embed.dense.tei.*` (url, api_key, timeout, max_inflight, batch_size, compression)
  * `embed.dense.sentence_transformers.*` (model_id, device, dtype, batch_size, max_seq_len, trust_remote_code, threads)
  * `embed.sparse.splade_st.*` (model_id, device, dtype, batch_size, postproc/pruning, tokenizer, attn backend)
  * `embed.lexical.local_bm25.*` (k1, b, stopwords, tokenizer, min_df, max_df_ratio)
* **Precedence** (PR-7): **CLI > ENV > profile file > defaults**
* Legacy flags/env (e.g., `--bm25-k1`, `DOCSTOKG_QWEN_DIR`) map to new keys with **one-line deprecations** (new wins).

### 4.6 Provenance (Parquet footers + manifests)

* Parquet footers already defined (PR-8); providers must set:
  `docparse.family`, `docparse.provider`, `docparse.model_id`, `docparse.dim` (dense), `docparse.dtype`, `docparse.device` (if relevant), `docparse.cfg_hash`, `docparse.created_at`, etc.
* Per-file manifest extras include: `provider_name`, `model_id@rev`, `dim` (dense), `avg_nnz` (sparse), `vector_format`, timing.

### 4.7 Concurrency, Batching, Backpressure

* Runner handles **file-parallelism** only.
* Providers own **micro-batching** and inflight request limits (e.g., TEI `max_inflight_requests`, vLLM queue depth).
* The Runner’s `workers` must not exceed provider’s safe inflight; provider can **block/backoff** if inflight is maxed.

### 4.8 Caching (optional, safe by default)

* Provider-local hot cache (bounded LRU `{text_hash → vector}`) to save repeated boilerplate.
* Respect `embedding.offline=true` (no downloads).

### 4.9 Error Taxonomy Mapping

| Provider situation              | `category` | `retryable`                                          |
| ------------------------------- | ---------- | ---------------------------------------------------- |
| HTTP 429/5xx (TEI)              | `network`  | true                                                 |
| Connect timeout (TEI)           | `network`  | true                                                 |
| GPU OOM (vLLM)                  | `runtime`  | false *(unless provider automatically down-batches)* |
| Model missing with offline=true | `init`     | false                                                |
| Import failure / invalid dtype  | `init`     | false                                                |
| Tokenizer/model mismatch        | `config`   | false                                                |

---

## 5) Module Layout (new files)

```
src/DocsToKG/DocParsing/embedding/
  backends/
    base.py                        # interfaces + ProviderError
    factory.py                     # ProviderFactory
    dense_qwen_vllm.py             # vLLM-backed provider
    dense_tei.py                   # TEI HTTP provider
    dense_sentence_transformers.py # ST local provider
    sparse_splade_st.py            # SPLADE provider
    lexical_local_bm25.py          # BM25 provider
```

> `embedding/runtime.py` will import only `ProviderFactory` and never import torch/vLLM/transformers directly.

---

## 6) Public API changes

* New import surface: `from DocsToKG.DocParsing.embedding.backends import ProviderFactory`
* No change to CLI surface in this PR except **backend selector** flags (already specified in earlier work):
  `--dense-backend`, `--tei-...`, `--qwen-...`, `--st-...`, `--splade-...`, `--bm25-...`

Legacy flags: still accepted → mapped → one-line deprecation message.

---

## 7) Migration & Compatibility

* Step 1 (internal): providers created but runtime still calls old functions (behind a feature flag) → validate parity.
* Step 2: flip runtime to call providers; keep old code under `compat/` module for one minor.
* Step 3: remove legacy code paths.

**No changes** to file formats, vector schemas, or manifest keys beyond adding provider provenance.

---

## 8) Rollout (safe, incremental)

1. **Commit A — Scaffolding**: `base.py`, `factory.py`, empty provider stubs; unit tests for factory selection.
2. **Commit B — Dense providers**: implement Qwen vLLM / TEI / ST; feature flag `DOCSTOKG_EMBED_PROVIDERS=on` to switch runtime use.
3. **Commit C — Sparse provider (SPLADE)**: move SPLADE logic; add post-proc pruning.
4. **Commit D — Lexical provider (BM25)**: move stats / vectorization; maintain same defaults.
5. **Commit E — Runtime switch**: replace inline calls with providers; remove heavy imports; keep a `compat` fallback for one minor.
6. **Commit F — Provenance & telemetry**: footer metadata + manifest provider fields; counters.
7. **Commit G — Docs & deprecations**: provider authoring guide + mapping table; deprecate legacy flags in help.
8. **Commit H — Tests & parity**: run A/B corpus, compare vector counts and fp tolerance; enable by default; remove `compat` after deprecation.

---

## 9) Testing Strategy

**Unit (fast)**

* Factory selection by backend names; missing required fields yield clear `ValidationError`.
* Dense (mocked): correct vector length, L2 norm when enabled, dtype transforms.
* TEI: retry on 429; timeout surfaces as `ProviderError(network, retryable=True)`.
* SPLADE: non-negative weights; `nnz == len(indices) == len(weights)`.
* BM25: parameter ranges; deterministic vectors for fixed text.

**Integration (mini corpus)**

* Per-family parity: vectors count & shapes match legacy; cosine deltas within tolerance for dense; nnz distributions match for SPLADE.
* End-to-end runtime with providers: JSONL/Parquet outputs identical to baseline (schema & counts).

**Performance sanity**

* Measure p50/p95 for per-file embedding under each backend; keep a baseline chart.

**Observability**

* Metrics exposed: `provider_batch_ms`, `provider_inflight`, `provider_errors_total{category}`.
* Manifests/footers contain provider metadata.

---

## 10) Security & Privacy

* Default `trust_remote_code=false`.
* `offline=true` respected: if missing weights, providers **fail early** with remediation.
* TEI `api_key` never logged or written to manifests; redact in config snapshots.
* Device enumeration is informational (not sensitive).

---

## 11) Risks & Mitigations

* **Behavior drift** (vectors differ):
  *Mitigation*: strict normalization rules (L2 on by default), tokenizer revisions pinned, parity tests on a mini corpus before enabling by default.
* **Import bloat / slow CLI**:
  *Mitigation*: lazy imports in `open()` only; the CLI stays snappy.
* **Backpressure mismatch** (Runner vs provider):
  *Mitigation*: default provider inflight ≤ `workers`; block rather than queue unbounded; document the relationship in help.
* **GPU OOM**:
  *Mitigation*: default conservative batch sizes; support down-batching (future), or surface clear error with remediation (“reduce batch-size”).
* **HTTP flakiness** (TEI):
  *Mitigation*: bounded retry + backoff + jitter; surface `retryable=True` to Runner.

---

## 12) Acceptance Criteria (“done”)

* `embedding/runtime.py` **does not import** vLLM/torch/transformers/requests; it depends solely on `ProviderFactory`.
* All three families (dense/sparse/lexical) run through providers with **identical outputs** (schema, dims/nnz) as legacy code (within fp tolerance for dense).
* Manifests and Parquet footers include provider provenance (name, model id, dim/dtype, device, cfg_hash, created_at).
* CLI exposes backend selectors; legacy flags map to new keys with deprecation lines.
* Unit + integration + parity tests green; performance similar or better (due to localized batching).

---

## 13) Work Breakdown (developer-friendly checklist)

**A. Scaffolding**

* [ ] Create `embedding/backends/base.py`: interfaces + `ProviderError`
* [ ] Create `embedding/backends/factory.py`: config→provider mapping, lazy import checks

**B. Dense**

* [ ] Implement `dense_qwen_vllm`: move import cache, model open, queue/batching; expose `batch_size`, `tensor_parallelism`, `gpu_memory_utilization`, `device`, `dtype`, `warmup`.
* [ ] Implement `dense_tei`: HTTP pool, retries, inflight cap, compression; honor `timeout_s`, `verify_tls`.
* [ ] Implement `dense_sentence_transformers`: model id/rev, device/dtype, batch size, max_seq_length, threads; memory-map weights if enabled.

**C. Sparse**

* [ ] Implement `sparse_splade_st`: model id/rev, device/dtype, batch size; post-proc pruning, topk, doclen norm.

**D. Lexical**

* [ ] Implement `lexical_local_bm25`: tokenizer options, k1/b, stopwords, df thresholds; stats accumulate policy.

**E. Runtime switch**

* [ ] Replace inline calls with providers; integrate with Runner stage hooks.
* [ ] Delete heavy imports from runtime; guard with feature flag during bake.

**F. Provenance & telemetry**

* [ ] Ensure footer metadata set; manifest extras populated; counters emitted.

**G. Docs & deprecations**

* [ ] Provider guide; config key tables; legacy mapping; deprecation banners in help.

**H. Tests**

* [ ] Unit for each provider + factory.
* [ ] Parity suite vs legacy outputs; flip default after green.

---

## 14) Example sequences (for reviewers)

**Dense (TEI)**

```
Runner.before_stage → factory → dense_tei.open()
for each chunk file:
  worker → read chunk texts
  dense_tei.embed(texts)  # HTTP POST, retries, inflight cap
  writer → vectors parquet/jsonl
Runner.after_stage → dense_tei.close() → write __corpus__ summary
```

**Dense (vLLM)**

```
Runner.before_stage → dense_qwen_vllm.open()  # lazy import, load model, warmup
worker → micro-batch texts → dense_qwen_vllm.embed()
writer → parquet/jsonl
Runner.after_stage → close()
```

**Sparse (SPLADE)**

```
open() → load model/tokenizer
encode(texts) → (indices, weights), nnz per doc
writer → parquet/jsonl
```

**Lexical (BM25)**

```
open() → prepare tokenizer/stopwords
accumulate_stats(chunks) → avgdl, df map (policy)
vector(text, stats) → (indices|terms, weights)
writer → parquet/jsonl
```

---

If you’d like, I can immediately follow this with a **commit-level diff plan** (file paths + function names to author) and a **parity test worksheet** (exact assertions, cosine thresholds, nnz checks) so your agents can implement PR-4 with minimal ambiguity.
