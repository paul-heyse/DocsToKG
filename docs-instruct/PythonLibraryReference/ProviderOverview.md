Here’s an AI-agent-ready **technical guide to “providers”**—what they are, why they exist, how to design and operate them, plus best practices and advanced techniques tailored to your DocParsing stack (dense, sparse/SPLADE, lexical/BM25).

---

# What a “provider” is (and isn’t)

**Definition:** A *provider* is a small, swappable component that implements a **narrow contract** (dense, sparse, or lexical vectorization) behind a stable interface. Providers encapsulate **all** third-party details (libraries, models, processes, devices, HTTP services, queues) so the *runtime* only asks for “vectors for these texts” and writes results.

**Not a provider:** Anything concerned with file discovery, manifests, JSONL/Parquet writing, progress bars, plan-only analysis, or CLI orchestration. Those stay in the runtime and `core/*`.

**Why:**

* Reduce bespoke glue and drift.
* Make backends swappable (e.g., switch from local vLLM to TEI HTTP) without touching orchestration.
* Localize performance tricks, caching, and failure handling.
* Enable clean A/B experiments and fallbacks.

---

# Design philosophy

1. **One job, tiny surface.**
   Each provider has a **3–5 method** interface: `open(cfg)`, `embed|encode|vector(...)`, `close()`. No incidental complexity leaks out.

2. **Inversion of dependencies.**
   The runtime depends on interfaces; **providers depend on libraries** (vLLM, Sentence-Transformers, PySerini, TEI client, etc.). Imports for heavy libs live *only* in providers.

3. **Explicit lifecycle.**
   Providers own initialization (weights download, HTTP sessions, GPU warmup) and cleanup (free VRAM, close sockets).

4. **Pure(ish) core with impure edges.**
   Provider methods are referentially transparent *for a fixed config* (same inputs → same outputs), but may use caches/queues internally for performance—never changing the result.

5. **Determinism over cleverness.**
   Prefer deterministic preprocessing, fixed revisions, pinned tokenizers, and L2-normalized outputs (unless explicitly disabled).

6. **Fail fast, degrade gracefully.**
   Fail early on missing required keys (e.g., `TEI_URL`). Offer optional **fallbacks** (e.g., ST if vLLM fails), but never emit partial or shape-mismatched vectors.

---

# Core interfaces (conceptual)

## DenseEmbeddingBackend

* **open(cfg)** → prepares model/session/queue
* **embed(texts, *, batch_size=None)** → `List[List[float]]`
  Invariants:

  * Output length is constant per model (embedding dim).
  * Empty/whitespace inputs return the **model’s defined neutral** (often zeros) and are **still normalized** if normalization is enabled.
  * Text order preserved.
* **close()** → release resources (GPU, threads, sockets)
* **name** (property) → stable id like `dense.qwen_vllm`, `dense.tei`, `dense.sentence_transformers`

## SparseEmbeddingBackend (SPLADE)

* **open(cfg)**
* **encode(texts)** → `List[List[(token, weight)]]` or equivalent structured vector
  Invariants:

  * Non-negative weights.
  * Stable tokenizer/version captured in provenance.

## LexicalBackend (BM25)

* **open(cfg)**
* **accumulate_stats(chunks)** → corpus stats (avgdl, N, df)
* **vector(text, stats)** → `[(term, weight)]`
  Invariants:

  * Uses the same tokenizer as stats.
  * Respects `k1`, `b`, and stopword policy.

> All providers raise a **structured ProviderError** (`provider`, `category`, `retryable`, `detail`) so callers can map errors to telemetry and (optionally) fallbacks.

---

# Lifecycle, ownership, and boundaries

* **Who owns concurrency?** Providers do.
  If a backend benefits from internal queues (vLLM) or HTTP inflight limits (TEI), **implement it inside the provider**. The runtime should only pass batches and respect a total `max_concurrency` hint.

* **Who owns batching?** Providers do.
  They may micro-batch within a call to `embed/encode/vector`. The runtime provides an upper bound batch; the provider can subdivide.

* **What about caching?** Provider-local.

  * *Cold cache:* model weights, tokenizers, HTTP connection pools.
  * *Hot cache:* recent text→vector map (optional, bounded by memory).
  * *Cross-run cache:* left to the writer/storage layer; providers return vectors, they don’t persist them.

* **What about text preprocessing?**
  Providers must **either** implement model-native normalization (e.g., lowercasing, unicode NFC) **or** call a shared preprocessor agreed across providers. The goal is consistent embeddings across backends when appropriate.

---

# Configuration & precedence (quick recap)

* **Selectors:** `dense.backend`, `sparse.backend`, `lexical.backend` (`none` to disable).
* **Cross-cutting:** `embedding.device|dtype|batch_size|max_concurrency|normalize_l2|offline|cache_dir`.
* **Per-backend keys:** `dense.tei.url`, `dense.qwen_vllm.model_id`, `lexical.local_bm25.k1`/`b`, `sparse.splade_st.model_id`, etc.
* **Precedence:** CLI > ENV > config file > defaults.
* **Compatibility:** legacy flags/envs map into the new keys with one-line deprecation notices.

---

# Best practices (the “10 commandments”)

1. **Lazy import heavy libs** inside `open()`; do not import vLLM, torch, transformers at module import time.
2. **Guard devices early.** If `device='cuda'` but no GPU, error with a remediation that suggests `dense.backend=sentence_transformers` or `device=cpu`.
3. **Pin revisions.** Record model id + revision + tokenizer revision in provenance. Don’t allow silent upgrades unless `allow_unpinned=true`.
4. **Normalize consistently.** Do L2 normalization for dense outputs inside the provider (default `true`) so vectors are comparable across backends.
5. **Bound memory.** Support `gpu_memory_utilization` (vLLM), `max_seq_length`, and modest micro-batches. Prefer OOM-safe defaults.
6. **Respect offline mode.** If `offline=true` and weights are missing, fail with a clear message—don’t attempt downloads.
7. **Surface real dimensions.** Provide `embedding_dim`, `vocab_source`, and `tokenizer_id` in provider metadata for manifests.
8. **Deterministic text handling.** NFC normalize; trim odd whitespace; decide and document how you treat empty strings.
9. **Tight error taxonomy.** Emit `ProviderError(category=[init|download|device|network|runtime], retryable=bool)`; never raise raw library exceptions beyond the provider boundary.
10. **Emit useful telemetry.** `provider_name`, `model_id@rev`, `device`, `dtype`, `effective_batch`, `time_open_ms/ time_embed_ms/ time_close_ms`, `fallback_used`.

---

# Advanced techniques (performance & resilience)

## Dense/vLLM (Qwen) specifics

* **Warmup** on `open()`: send a tiny batch to prime CUDA kernels and caches.
* **Queue depth & inflight requests:** bound by `embedding.max_concurrency`; reject/await when full to avoid GPU thrash.
* **Auto-batcher:** implement a simple time-boxed accumulator (e.g., 2–5 ms) to coalesce incoming requests up to `batch_size`.
* **Pinned memory & non-blocking copies:** if you’re doing manual tensor handling, use pinned CPU memory and non-blocking CUDA copies to hide latency.
* **Precision trade-offs:** prefer `bfloat16` on Ada/Blackwell if the model supports it; otherwise `float16`. Convert outputs to float32 before writing (or document writer downcast rules).
* **Multi-GPU placement:** accept `device='cuda:N'`; expose `tensor_parallelism`; avoid cross-device copies in provider.

## Dense/TEI (HTTP) specifics

* **HTTP/1.1 keep-alive or HTTP/2** with a shared connection pool.
* **Compression:** auto-gzip request bodies for large batches.
* **Adaptive inflight control:** dial `max_inflight_requests` down on 429/5xx; backoff with jitter.
* **Chunking:** if the server has a maximum batch token limit, split batches proactively.

## Sparse/SPLADE

* **Thresholding:** support `postproc.prune_below` to drop tiny activations; keep a counter in telemetry (sparsity ratio).
* **Top-k per doc (optional):** expose `topk_per_doc` only if you need strict caps for storage.

## Lexical/BM25

* **Corpus-first mode (optional):** for very large corpus embeddings, consider a one-time pass to compute global stats, then per-chunk vectors (document this mode separately).
* **Tokenizer consistency:** the stats pass and vectorization pass **must** share the exact tokenizer configuration; version it in provenance.

## Cross-provider wins

* **Adaptive batching:** collect a few micro-batches quickly; flush if queue is empty or `max_batch_latency_ms` exceeded.
* **Throughput accounting:** log `chunks/s` and `tokens/s` per provider; store percentiles (p50, p95) for batch duration.
* **Canary + shadow mode:** run a small percentage through an alternate dense backend and compare cosine deltas; gate flips with thresholds.

---

# Caching & dedup strategy (high leverage)

* **Provider-local cache (hot):** LRU of `{text_hash → vector}` bounded by memory; useful when multiple files share repeated boilerplate.
* **Cross-run cache (warm):** optional SQLite/Parquet cache keyed by `fingerprint = sha256(normalized_text + tokenizer_rev + model_id@rev + cfg_hash)`.
* **Near-duplicate suppression:** consider n-gram or MinHash sketching to skip obvious duplicates (opt-in; don’t change semantics silently).

---

# Determinism & reproducibility

* **Record everything:** `model_id@rev`, `tokenizer@rev`, `dtype`, `device`, `normalize_l2`, `max_seq_length`, and **a stable hash of the provider config** in each vector manifest.
* **No dropout.** Ensure eval mode everywhere (ST, SPLADE).
* **Unicode normalization:** choose one (e.g., NFC) and do it identically across providers.
* **Exact tokenizers:** never let tokenizer auto-upgrade; pin revisions.

---

# Observability

* **Structured logs:** per provider, emit start/stop events with config summary (excluding secrets).
* **Metrics:** counters for successes/failures; histograms for batch sizes and durations; gauges for GPU memory (sample `nvidia-smi` if available).
* **Traces:** span per batch → provider subspan; attach high-cardinality tags like `provider_name`, `model_id@rev`.

---

# Testing strategy

## Unit tests (per provider)

* **Shape & normalization:** correct dims; L2 norm ≈ 1.0 when enabled.
* **Empty input behavior:** empty string returns a stable vector (document it).
* **Error taxonomy:** bad config → `ProviderError(category=init)`; network 429 → `ProviderError(category=network, retryable=True)`.

## Integration tests

* **Parity run:** default backends produce the **same** vectors as before (within fp tolerance).
* **Throughput sanity:** small dataset runs under a threshold on reference hardware (record p50/p95).
* **Fallbacks:** deliberately break the primary dense backend → verify fallback executes and is logged.

## Property tests

* **Idempotence:** same input twice → identical vector.
* **Concatenation monotonicity (BM25):** adding tokens shouldn’t reduce document length term.

---

# Security & supply-chain notes

* **`trust_remote_code=false`** by default; flip only per-provider and log it.
* **Offline mode:** required for air-gapped runs; fail clearly if weights are missing.
* **HTTP clients:** sanitize URLs, enforce TLS verification by default, redact API keys from logs.
* **Model provenance:** store SHA256 of on-disk weights (optional) to detect silent drift.

---

# Operational playbook (rollout & maintenance)

1. **Introduce providers behind feature flags** (e.g., `embedding.providers.enabled=true` initially off for non-prod).
2. **Dogfood on small sets** with shadow mode (e.g., send 1% of dense calls to TEI and compare cosine deltas).
3. **Promote** once parity and throughput targets are met.
4. **Set SLOs:** p95 batch latency, error rate budget, throughput (chunks/s).
5. **Version providers:** semver for provider modules; bump minor on config additions, major on breaking changes.
6. **Deprecation window:** keep legacy flags as aliases for one minor release; emit deprecation logs.

---

# Anti-patterns to avoid

* **Leaking library objects** (e.g., returning torch tensors) across the boundary. Always return plain Python lists/tuples or NumPy arrays the writer expects.
* **Tight coupling to runtime pools.** Providers should *not* assume a specific thread/process model.
* **Implicit downloads** when `offline=true`.
* **Hidden state that changes results** (e.g., temperature, dropout, stochastic augmentations).
* **Mixed tokenizers** between SPLADE stats pass and encode pass.

---

# Engineer’s checklist (do this and you’re safe)

* [ ] Provider interfaces exist and are tiny (`open`, `embed/encode/vector`, `close`, `name`).
* [ ] Heavy imports only in providers; runtime imports **zero** of vLLM/ST/PySerini.
* [ ] Config precedence implemented (CLI > ENV > file > defaults) with legacy alias mapping + warnings.
* [ ] Deterministic normalization path documented and on by default.
* [ ] Structured `ProviderError` everywhere; no raw exceptions cross the boundary.
* [ ] Telemetry fields populated uniformly across providers.
* [ ] Parity tests pass (default providers produce old outputs).
* [ ] Throughput tests recorded; p50/p95 tracked in CI (optional).
* [ ] Docs updated: how to choose backends, required keys (`TEI_URL`), offline expectations.
* [ ] Fallback logic tested (primary fails → secondary used, clearly logged).

---

If you’d like, I can generate a **provider QA workbook** (CSV/Markdown) listing every config key with: default, allowed range, legacy alias, example value, and a test ID—so your agent can auto-generate validation and docs from a single source.
