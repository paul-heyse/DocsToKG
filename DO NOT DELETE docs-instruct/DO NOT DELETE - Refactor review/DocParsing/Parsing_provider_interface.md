Here’s a **surgical, narrative-only implementation plan** for **PR-4: Push “providers” behind small interfaces (swap heavy code for libraries)**. It’s written so an AI programming agent can execute it step-by-step—no guessing, no Python code.

---

# Scope & intent

**Goals**

1. Isolate SPLADE, BM25, and dense-embedding logic behind **small provider interfaces** so the runtime no longer hard-codes library details (vLLM, sentence-transformers, local loops) or queue/caching mechanics.
2. Enable **drop-in backends**:

   * Dense: `sentence-transformers` (local), **TEI** (HTTP service), **Qwen via vLLM** (local GPU).
   * Sparse: SPLADE via `sentence-transformers` (keep), with room for `pyserini` SPLADE later.
   * Lexical: BM25 current local implementation (default) with an optional `pyserini` provider.
3. Keep **runtime orchestration** focused on file planning, telemetry, and manifest I/O; delete per-backend queues/caches from the runtime, moving them into providers. Today the runtime embeds Qwen queueing/caching and vLLM imports inline.
4. Preserve external behavior (CLI flags, outputs, manifests), and keep existing `VectorWriter` abstractions and `create_vector_writer` unchanged. The runtime already advertises these writer seams.

**Non-goals (deferred)**

* No changes to vector schema, filenames, or directory layout (PR-2/PR-3 already covered I/O and CLI consolidation).
* No retrieval-time provider refactors (HybridSearch) in this PR.

---

# Why this is needed (inventory of current tight coupling)

* **Dense/Qwen is entangled with runtime control logic**: the runtime defines `_qwen_embed_direct`, a process-wide LLM cache, and **`QwenEmbeddingQueue`** worker thread to serialize requests, plus direct vLLM imports via a local import-cache. All of this sits inside `embedding/runtime.py`, not behind an abstraction.
* **Backend selection is implicit**: runtime helpers `_get_sparse_encoder_cls()` and `_get_vllm_components()` import/cross-check optional deps; they should live with their provider.
* **Runtime owns queues/caches** (`_QWEN_LLM_CACHE`, `QwenEmbeddingQueue`) and wires them into file-level parallelism. This should be delegated so the runtime only requests “embed these texts”; concurrency is a backend detail.

---

# Target design (the “after” picture)

1. **Thin provider interfaces** under `src/DocsToKG/DocParsing/embedding/backends/`:

   * **DenseEmbeddingBackend**: `name`, `open(cfg)`, `embed(texts, *, batch_size=None) -> list[list[float]]`, `close()`.
   * **SparseEmbeddingBackend** (SPLADE): `open(cfg)`, `encode(texts) -> list[(tokens, weights)]`.
   * **LexicalBackend** (BM25): `open(cfg)`, `accumulate_stats(chunks) -> stats`, `vector(text, stats) -> (terms, weights)`.
     Each provider owns its import checks, device selection, caching, and any internal concurrency or queues. The runtime **never** imports vLLM/sentence-transformers directly after this PR. (Today it does.)

2. **Concrete dense backends**:

   * `dense_sentence_transformers`: local SBERT family.
   * `dense_tei`: HTTP client for Hugging Face **TEI** (or equivalent) with `TEI_URL` env support.
   * `dense_qwen_vllm`: wraps vLLM and **absorbs** `_QWEN_LLM_CACHE` and **`QwenEmbeddingQueue`** semantics internally. (Remove both from the runtime.)

3. **Concrete sparse backend**:

   * `sparse_splade_st`: today’s `SparseEncoder` path moved from runtime helper into the provider; it owns `ensure_splade_dependencies()` and environment probing.

4. **Concrete lexical backend**:

   * `lexical_local_bm25`: wraps the current in-process stats/weighting (your two-pass architecture).
   * Optional future: `lexical_pyserini_bm25` (index once, query chunks to get stable BM25 term weights), staged for a later PR.

5. **Factory and selection**:

   * `ProviderFactory` (tiny module): chooses provider by `EmbedCfg` and CLI (`--dense-backend`, `--sparse-backend`, `--lexical-backend`).
   * Backends can be **absent** (e.g., `--dense-backend none`) without changing runtime logic.

6. **Runtime simplification**:

   * `process_chunk_file_vectors` pulls providers from the factory and **only** calls `provider.embed/encode/vector`.
   * Remove: `_get_vllm_components`, `_get_sparse_encoder_cls`, `_QWEN_LLM_CACHE`, and the top-level `QwenEmbeddingQueue` definition from runtime. (Providers own those.)

7. **Writers stay as-is**: continue using your `VectorWriter`/`JsonlVectorWriter`/`create_vector_writer`; no change to Parquet/JSONL emission added in recent work.

---

# Step-by-step implementation plan

## 0) Pre-flight

* Branch: `codex/pr4-embedding-providers`.
* Inventory callsites to detach: `_qwen_embed_direct`, `qwen_embed`, `QwenEmbeddingQueue`, `_get_vllm_components`, `_get_sparse_encoder_cls`, `_QWEN_LLM_CACHE`, and where they are used in **file-parallel** Pass B.

## 1) Create the provider folders & base interfaces

* New package: `DocParsing/embedding/backends/` with `__init__.py` and `base.py`.
* Define the three Protocol-like surfaces described above and a tiny `ProviderError` type.
* Add a `ProviderFactory` module that accepts `EmbedCfg` and returns `(dense, sparse, lexical)` triples; no heavy logic here (one page of code once implemented).

**Acceptance**: New modules import cleanly; no consumers yet.

## 2) Implement **dense backends**

**a) `dense_sentence_transformers`**

* Resolve model id/path from `EmbedCfg` (or environment); instantiate `SentenceTransformer` or `TextEmbedding`-style helper; control device.
* `embed` handles batching, dtype normalization (to float), and returns `List[List[float]]`. Runtime **does not** see the model object.

**b) `dense_tei`**

* Read `TEI_URL` (and optional token) from env or config.
* `open()` stores session; `embed()` POSTs `{"inputs": texts}` and returns vectors.
* Handle server-side errors with `ProviderError`.

**c) `dense_qwen_vllm`**

* Move in the current vLLM lifecycle: import caching (`LLM`, `PoolingParams` via internal helper), **local LLM cache**, and **background queue** if needed for multi-file concurrency—**inside the provider**. Delete `_QWEN_LLM_CACHE` and `QwenEmbeddingQueue` from the runtime.
* Respect existing config knobs (dtype, TP, mem util, quantization, batch size) to preserve behavior of `_qwen_embed_direct`.
* `close()` shuts down the LLM if the cache policy requires.

**Acceptance**: Each backend passes a minimal, provider-local smoke test; no integration with runtime yet.

## 3) Implement **sparse SPLADE backend**

* New `sparse_splade_st` wraps your `_get_sparse_encoder_cls` logic (moved here) and owns dependency checks. Replace the runtime’s helper with a provider call.
* `encode(texts)` returns `(tokens, weights)` pairs per input; apply any model-specific normalization you already expect.

**Acceptance**: Backend returns a plausible activation list on tiny inputs.

## 4) Implement **lexical BM25 backend(s)**

* `lexical_local_bm25` wraps the **existing** two-pass stats and vector computation used by `process_pass_a/process_chunk_file_vectors`. No behavior change—just relocate library-coupled math into a provider so the runtime calls `accumulate_stats()` once, then `vector()` per chunk. (Your runtime already distinguishes Pass A vs Pass B.)
* (Optional stub) `lexical_pyserini_bm25` provider with a TODO marker; not wired by default.

**Acceptance**: Parity on `avgdl/N/k1/b` usage when building `BM25Vector` objects (same numbers as today). Your vector rows already use these fields.

## 5) Refactor the **runtime** to use providers

* In `embedding/runtime.py`:

  * **Construct providers** once per run via `ProviderFactory` at the start of Pass B.
  * Replace direct calls to `splade_encode`, `_qwen_embed_direct/qwen_embed`, and in-module BM25 helpers with calls to the respective providers.
  * Delete or **stop exporting**: `_get_vllm_components`, `_get_sparse_encoder_cls`, `QwenEmbeddingQueue`, `_QWEN_LLM_CACHE` from runtime. Keep `__all__` clean. (The module currently re-exports many symbols—trim provider-specific ones.)
  * Keep everything else: plan-only mode, manifest logging, file-parallel ThreadPool, quarantine behavior, and vector writing remain unchanged. You already log per-file success/failure and durations.

**Acceptance**: The runtime’s Pass B loop no longer references vLLM or sentence-transformers directly; it only knows “dense/sparse/lexical providers”.

## 6) CLI & config alignment (non-breaking)

* Add three **selector options** to the embed CLI: `--dense-backend`, `--sparse-backend`, `--lexical-backend` with defaults matching current behavior (`qwen_vllm`, `splade_st`, `local_bm25`).
* Map existing flags into provider configs without renaming user-visible options (e.g., keep `--batch-size-qwen`, but also accept `--dense-batch-size` as an alias).
* The package already centralizes CLI exports in `embedding/__init__.py`; ensure these re-exports still work after removing provider-specific symbols from runtime.

**Acceptance**: `--help` shows backend selectors; old flags still work; defaults preserve current behavior.

## 7) Telemetry & error mapping

* Wrap provider calls with a tiny timing decorator in the runtime to record durations, batches, and error counts in your existing **manifest and telemetry** events (you already report success/failure per file with durations).
* Map provider exceptions to the existing `EmbeddingProcessingError` so the runtime’s error paths and quarantine logic stay unchanged. (Runtime currently quarantines on `ValueError` and logs failure manifests.)

**Acceptance**: Telemetry remains identical (same fields and stages).

## 8) Tests

* **Provider unit tests** (fast):

  * Dense-ST: returns correct length vectors, no device exceptions without GPU.
  * Dense-TEI: mock HTTP; error → `ProviderError`.
  * Dense-Qwen-vLLM: mark as GPU/optional; smoke via a tiny batch or skip if deps missing; test cache hit path.
  * SPLADE-ST: returns non-negative weights; list lengths match.
  * Local BM25: parity with current numbers on a tiny corpus (avgdl, N) and stable term ordering.

* **Runtime integration tests**:

  * Embed end-to-end with default backends; outputs unchanged (JSONL/Parquet). Your repository recently added **Parquet vector** paths—ensure both formats still work.
  * Plan-only mode unchanged (you recently ensured tracemalloc stops properly—keep that behavior).

**Acceptance**: All tests green with providers; optional tests skip gracefully when vLLM/TEI are unavailable.

## 9) Documentation updates

* Update `AGENTS.md` and the DocParsing README to mention backend selection and environment variables (`TEI_URL`, `DOCSTOKG_QWEN_DIR`, etc.). Your docs already describe SPLADE/Qwen env setup; extend those sections with backend names and defaults.
* Update the API page for `embedding` to remove provider-internal symbols from the public surface (e.g., `QwenEmbeddingQueue`), leaving only runtime/CLI entry points and models. The module currently lists many exports in `__all__`.

---

# Deletions / moves

* **Remove from runtime** (moved into providers): `_get_vllm_components`, `_get_sparse_encoder_cls`, `_QWEN_LLM_CACHE`, `QwenEmbeddingQueue`, `_qwen_embed_direct`.
* **Keep in runtime**: all file planning, manifest logging, quarantine handling, progress bars, and vector writing; these are already clean seams.

---

# Risks & mitigations

* **Behavior drift** (batch size, normalization): mirror the current `_qwen_embed_direct` defaults inside the `dense_qwen_vllm` provider; write a parity test.
* **Hidden import costs**: providers must lazy-import heavy libs on `open()` to avoid impacting unrelated backends; this mirrors your current cache/import-helper pattern but isolates it.
* **Concurrency edge cases**: if the runtime’s file-parallel pool outpaces a single vLLM instance, the provider’s internal queue will bound requests. Keep the queue depth derived from `files_parallel` (provider can accept a hint). The runtime currently passes a queue object; after the refactor, **delete that coupling** and pass a numeric hint only.

---

# Work breakdown (small, reviewable commits)

1. **Commit A — Scaffolding**

   * Create `embedding/backends/{base.py,factory.py}` and add empty provider stubs. No runtime changes.

2. **Commit B — Dense providers**

   * Implement `dense_sentence_transformers`, `dense_tei`, `dense_qwen_vllm`; move vLLM cache/queue in here. Delete `_QWEN_LLM_CACHE` and `QwenEmbeddingQueue` from runtime.

3. **Commit C — Sparse provider**

   * Implement `sparse_splade_st`; move import checks from runtime.

4. **Commit D — Lexical provider**

   * Implement `lexical_local_bm25` wrapping current stats/weight computation.

5. **Commit E — Runtime refactor**

   * Replace direct calls with providers; remove provider-specific exports from `__all__`. Ensure Pass A/B shape and manifests unchanged.

6. **Commit F — CLI+config & docs**

   * Add backend selectors; update docs/AGENTS.md/README; keep old flags.

7. **Commit G — Tests**

   * Provider unit tests + runtime parity tests (JSONL/Parquet, plan-only tracemalloc behavior).

---

# Acceptance criteria (“done”)

* `embedding/runtime.py` **no longer imports** vLLM or sentence-transformers directly and **does not define** `QwenEmbeddingQueue` nor `_QWEN_LLM_CACHE`.
* The embed stage runs end-to-end with default backends and produces **identical vector rows** as before (both JSONL and Parquet). Recent Parquet support remains functional.
* CLI exposes `--dense-backend/--sparse-backend/--lexical-backend`, with legacy flags still supported.
* Unit/integration tests cover at least one backend per category; optional tests skip cleanly if deps absent.
* Docs reflect backend selection and environment variables.

---

# Rollback plan

* If a provider breaks behavior, revert only its module and the factory wiring (isolated commit), keeping the runtime intact.
* If vLLM performance regresses under provider queueing, temporarily re-enable the previous `_qwen_embed_direct` path behind a “compat” flag in the provider while you tune queue depth.

---

# Why this shrinks code & risk immediately

* Centralizes **all** third-party model calls, caching, and queueing in a predictable place, deleting bespoke runtime mechanics (queue/cache/import paths) that currently live alongside orchestration.
* Leaves your **good seams** (planning, manifests, writers, telemetry) untouched, so blast radius is small. The runtime already has robust per-file success/failure logging and vector serialization.

If you want next, I can enumerate the **provider config keys** (dense/sparse/lexical) and map each existing CLI flag/env var to the new provider fields so the agent can wire the factory with zero ambiguity.
