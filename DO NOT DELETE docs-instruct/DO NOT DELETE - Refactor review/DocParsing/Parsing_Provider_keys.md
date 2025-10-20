Absolutely—here’s an **AI-agent-ready, narrative-only spec** for the **provider configuration keys** you’ll introduce in **PR-4**. It covers structure, precedence, concrete keys (by backend), compatibility with existing flags/env, validation rules, and telemetry fields—so the agent can wire the factory and backends without ambiguity.

---

# Guiding principles

* **Single source of truth:** One **EmbedCfg** (Pydantic v2) object feeds the **ProviderFactory**, which returns `(dense, sparse, lexical)` providers.
* **Stable defaults; zero breaking change:** Default values must reproduce current behavior (same models/params) when users don’t set anything new.
* **Clear precedence:** **CLI > ENV > config file > code defaults**.
* **Isolation:** Provider-specific keys live under their section (`dense.*`, `sparse.*`, `lexical.*`). Cross-cutting knobs live under `embedding.*`.
* **Minimal surface:** Only keys the provider truly needs. Anything about **file discovery, manifests, writers** stays outside providers.

---

# Config shape (conceptual)

* `embedding.*` → cross-cutting knobs (common across providers)
* `embedding.dense.*` → dense backend selection + per-backend settings
* `embedding.sparse.*` → SPLADE (or other) sparse settings
* `embedding.lexical.*` → BM25 settings

You’ll support env var mirrors using the prefix `DOCSTOKG_` (e.g., `DOCSTOKG_DENSE_BACKEND`), and keep legacy envs as aliases where they already exist.

---

# Precedence & compatibility

1. **CLI** flags (e.g., `--dense-backend`) override everything.
2. **ENV** (e.g., `DOCSTOKG_DENSE_BACKEND`) override config file.
3. **Config file** (TOML/YAML) override code defaults.
4. **Aliases** for backward compatibility:

   * Keep existing flags/envs (e.g., `--bm25-k1`, `--bm25-b`, Qwen batch/env knobs) as **aliases** that write into the new keys.
   * If both a legacy flag and a new key are provided, **new key wins** and you emit a one-line deprecation warning.

---

# Cross-cutting keys (apply to all providers)

| Key                         | Type / Allowed                              |          Default | Purpose / Notes                                                                                                                   |
| --------------------------- | ------------------------------------------- | ---------------: | --------------------------------------------------------------------------------------------------------------------------------- |
| `embedding.device`          | `auto` | `cpu` | `cuda` | `cuda:N`          |           `auto` | Global device hint for providers that support it. Provider may override if unsupported.                                           |
| `embedding.dtype`           | `auto` | `float32` | `float16` | `bfloat16` |           `auto` | Preferred compute/output dtype for providers that support it (dense); output to writers remains float32 unless writer down-casts. |
| `embedding.batch_size`      | int ≥ 1                                     | provider default | Global batch hint; per-provider batch keys override this.                                                                         |
| `embedding.max_concurrency` | int ≥ 1                                     | `files_parallel` | Caps **in-provider** concurrency to prevent overload (e.g., TEI inflight HTTP; vLLM queue depth).                                 |
| `embedding.normalize_l2`    | bool                                        |           `true` | Whether dense outputs are L2-normalized before writing; match current behavior.                                                   |
| `embedding.offline`         | bool                                        |          `false` | Disallow online model/downloads; providers must error if weights are not local.                                                   |
| `embedding.cache_dir`       | path                                        |     tool default | Model artifact cache root (ST weights, tokenizer, etc.).                                                                          |
| `embedding.telemetry_tags`  | map(str→str)                                |             `{}` | Extra key→value tags injected into per-provider telemetry (e.g., workload id).                                                    |

**Validation:** `device='cuda:N'` must match available GPUs; if invalid, downgrade to `cpu` with a warning unless `strict_devices=true` (optional future key).

---

# Dense provider keys

## 1) Selection (common)

| Key             | Type / Allowed                                         |                   Default | Notes                                                                                                 |
| --------------- | ------------------------------------------------------ | ------------------------: | ----------------------------------------------------------------------------------------------------- |
| `dense.backend` | `sentence_transformers` | `tei` | `qwen_vllm` | `none` | `qwen_vllm` (match today) | Setting to `none` disables dense vectors entirely (runtime still writes sparse/lexical if requested). |

**Legacy flag/env aliases (examples):**

* `--dense-backend` ↔ `DOCSTOKG_DENSE_BACKEND`
* Existing Qwen-specific flags remain as aliases (see below) and populate the `qwen_vllm.*` subtree.

---

## 2) `dense.sentence_transformers.*`

| Key                 | Type / Allowed             |                                        Default | Notes                                                       |
| ------------------- | -------------------------- | ---------------------------------------------: | ----------------------------------------------------------- |
| `model_id`          | str                        | e.g., `sentence-transformers/all-MiniLM-L6-v2` | Required if ST is selected and no legacy default exists.    |
| `revision`          | str | null                 |                                         `null` | Pin model sha/tag if needed.                                |
| `device`            | same as `embedding.device` |                                       inherits | Per-backend override.                                       |
| `dtype`             | same as `embedding.dtype`  |                                       inherits | Cast to provider-supported dtypes; bfloat16 when available. |
| `batch_size`        | int ≥ 1                    |                                       inherits | Effective embedding batch.                                  |
| `max_seq_length`    | int ≥ 16                   |                               provider default | Truncation length; align with tokenizer.                    |
| `normalize_l2`      | bool                       |                                       inherits | Final vector normalization toggle.                          |
| `trust_remote_code` | bool                       |                                        `false` | For exotic models.                                          |
| `use_memory_map`    | bool                       |                                         `true` | Memory-map large weights if supported.                      |
| `intra_op_threads`  | int≥1 | `auto`             |                                         `auto` | CPU thread hint when on CPU.                                |

**Errors & fallbacks**

* If `model_id` missing and backend selected → clear error with remediation.
* If offline & weights absent → error with “download disabled” remediation.

**Legacy**

* If you already had `--st-model` or similar, map to `dense.sentence_transformers.model_id`.

---

## 3) `dense.tei.*` (Text Embeddings Inference / HTTP)

| Key                     | Type / Allowed           |                                Default | Notes                                              |
| ----------------------- | ------------------------ | -------------------------------------: | -------------------------------------------------- |
| `url`                   | url                      |          **required** when backend=tei | Base URL of TEI server. Alias: `DOCSTOKG_TEI_URL`. |
| `api_key`               | str | null               |                                 `null` | Optional header; alias: `DOCSTOKG_TEI_API_KEY`.    |
| `timeout_s`             | float ≥ 1                |                                   `30` | HTTP request timeout.                              |
| `verify_tls`            | bool                     |                                 `true` | TLS verification.                                  |
| `compression`           | `auto` | `gzip` | `none` |                                 `auto` | Request compression policy.                        |
| `max_inflight_requests` | int ≥ 1                  | inherits (`embedding.max_concurrency`) | Client-side concurrency limit.                     |
| `batch_size`            | int ≥ 1                  |                               inherits | Batch size per request (subject to server limit).  |
| `normalize_l2`          | bool                     |                               inherits | Keep parity.                                       |
| `retry.max_attempts`    | int ≥ 0                  |                                    `5` | Network retry policy (tenacity).                   |
| `retry.backoff_s`       | float ≥ 0                |                                  `0.5` | Base backoff.                                      |
| `retry.jitter_s`        | float ≥ 0                |                                  `0.2` | Jitter.                                            |

**Errors & fallbacks**

* If `url` missing → hard error suggesting `dense.backend=sentence_transformers` or setting `TEI_URL`.
* On HTTP 429/5xx: obey retry policy; ultimately raise ProviderError.

**Legacy**

* If you have an older `DOCSTOKG_EMBEDDINGS_ENDPOINT`, keep as alias to `dense.tei.url`.

---

## 4) `dense.qwen_vllm.*` (local vLLM embedding)

| Key                      | Type / Allowed                  |                          Default | Notes                                               |
| ------------------------ | ------------------------------- | -------------------------------: | --------------------------------------------------- |
| `model_id`               | str (path or HF id)             |       current default Qwen model | Required if you altered defaults historically.      |
| `tensor_parallelism`     | int ≥ 1                         |                              `1` | TP degree.                                          |
| `gpu_memory_utilization` | 0.1–0.99                        |                            `0.9` | vLLM memory fraction.                               |
| `dtype`                  | `auto` | `float16` | `bfloat16` |                           `auto` | vLLM weight dtype.                                  |
| `max_model_len`          | int ≥ 512                       |                 provider default | Context length for embedder.                        |
| `trust_remote_code`      | bool                            |                          `false` | Mirrors vLLM setting.                               |
| `download_dir`           | path                            | inherits (`embedding.cache_dir`) | HF cache override.                                  |
| `queue_depth`            | int ≥ 1                         |      `embedding.max_concurrency` | **Internal** request queue; replaces runtime queue. |
| `batch_size`             | int ≥ 1                         |                         inherits | Per-provider batch.                                 |
| `normalize_l2`           | bool                            |                         inherits | Keep parity.                                        |
| `warmup`                 | bool                            |                           `true` | Run a minimal warmup batch at open().               |
| `max_inflight_requests`  | int ≥ 1                         |                    `queue_depth` | Back-pressure bound; keep ≤ queue depth.            |
| `device`                 | same as `embedding.device`      |                         inherits | Used only to pin GPU if multiple present.           |

**Errors & fallbacks**

* If CUDA missing and `device` not CPU → clear error explaining to switch to ST/TEI or set `device=cpu` (slow).
* If offline and weights missing → error.

**Legacy**

* Map existing Qwen/vLLM flags/envs (e.g., `--qwen-batch`, `DOCSTOKG_QWEN_MODEL_DIR`) into these keys with deprecation notes.

---

# Sparse provider keys (SPLADE)

## Selection

| Key              | Type / Allowed       |     Default |
| ---------------- | -------------------- | ----------: |
| `sparse.backend` | `splade_st` | `none` | `splade_st` |

**Legacy alias:** `--sparse-backend` ↔ `DOCSTOKG_SPARSE_BACKEND`.

## `sparse.splade_st.*`

| Key                     | Type / Allowed                              |                                                     Default | Notes                                       |
| ----------------------- | ------------------------------------------- | ----------------------------------------------------------: | ------------------------------------------- |
| `model_id`              | str                                         | `naver/splade-cocondenser-ensembledistil` (or your current) | Mirror today’s model.                       |
| `revision`              | str | null                                  |                                                      `null` | Pin exact weights if needed.                |
| `device`                | as above                                    |                                                    inherits |                                             |
| `dtype`                 | `auto` | `float32` | `float16` | `bfloat16` |                                                      `auto` | Not all models support fp16/bf16; validate. |
| `batch_size`            | int ≥ 1                                     |                                                    inherits |                                             |
| `postproc.topk_per_doc` | int ≥ 0 | `0=all`                           |                                                         `0` | Keep all non-zeros unless you want pruning. |
| `postproc.prune_below`  | float ≥ 0                                   |                                                       `0.0` | Drop tiny weights if >0.                    |
| `normalize_doclen`      | `none` | `l2`                               |                                                        `l2` | Document-length normalization scheme.       |
| `tokenizer_id`          | str | null                                  |                                                      `null` | Default = model tokenizer.                  |
| `trust_remote_code`     | bool                                        |                                                     `false` |                                             |

**Errors & fallbacks**

* If dtype not supported → coerce to `float32` with warning (unless `strict_dtypes=true` future key).

---

# Lexical provider keys (BM25)

## Selection

| Key               | Type / Allowed                          |      Default |
| ----------------- | --------------------------------------- | -----------: |
| `lexical.backend` | `local_bm25` | `pyserini_bm25` | `none` | `local_bm25` |

(You can stage `pyserini_bm25` later; keep it hidden or experimental until ready.)

## `lexical.local_bm25.*`

| Key                       | Type / Allowed                     |             Default | Notes                                                                                       |
| ------------------------- | ---------------------------------- | ------------------: | ------------------------------------------------------------------------------------------- |
| `k1`                      | float > 0                          | **current default** | Preserve present setting.                                                                   |
| `b`                       | float in [0,1]                     | **current default** |                                                                                             |
| `stopwords`               | `none` | `english` | path          |              `none` | If a path, load from file.                                                                  |
| `tokenizer`               | `simple` | `spacy_en` | `regexp:`… |            `simple` | Keep parity with current tokenizer.                                                         |
| `lowercase`               | bool                               |              `true` |                                                                                             |
| `min_df`                  | int ≥ 1                            |                 `1` | Ignore terms with df < min_df.                                                              |
| `max_df_ratio`            | 0< float ≤ 1                       |               `1.0` | Ignore extremely common terms when <1.0.                                                    |
| `accumulate_stats_policy` | `per_file` | `corpus_first`        |          `per_file` | Whether to compute stats per input file (match today) or scan corpus once (optional later). |

**Legacy**

* Map `--bm25-k1`, `--bm25-b` directly to `lexical.local_bm25.k1/b`.

---

# Output normalization & writer interface (boundary clarification)

* Providers **return** dense/sparse/lexical vectors in provider-native dtype; runtime (or writer) ensures **float32** on disk unless you explicitly enable down-cast in writer config (separate concern).
* `embedding.normalize_l2` toggles normalization **inside dense providers** to preserve parity with current flow.

---

# Error handling & fallbacks (uniform shape)

* Providers raise `ProviderError` with:

  * `provider`: backend id (e.g., `dense.qwen_vllm`)
  * `category`: `init` | `download` | `device` | `runtime` | `network`
  * `retryable`: bool (factory can decide to retry or fallback)
  * `detail`: short human message

**Optional fallback keys** (nice-to-have, can be staged later):

* `dense.fallback`: backend name or `none` (e.g., try `tei` if `qwen_vllm` fails).
* `sparse.fallback`, `lexical.fallback`: same shape.

---

# Telemetry fields (emitted per file or per batch)

Standardize the following tags across providers (attach in your existing telemetry/manifest events; do not change schema keys in this PR—add as extra fields/tags where allowed):

* `provider_name` (e.g., `dense.qwen_vllm`)
* `provider_version` (lib + model rev if available)
* `device`, `dtype`
* `batch_size_effective`, `max_inflight_requests`
* `time_open_ms`, `time_embed_ms`, `time_close_ms`
* `fallback_used` (bool)
* `error_category` (on failure)
* `normalize_l2` (bool)

---

# Mapping legacy flags/env → new keys (examples)

> The agent should implement this as a **compatibility layer** in the CLI/config loader; when both are present, **new key wins**, and log a one-liner deprecation.

| Legacy                    | Maps to                        | Notes                                         |
| ------------------------- | ------------------------------ | --------------------------------------------- |
| `--bm25-k1`               | `lexical.local_bm25.k1`        | Keep help text unchanged.                     |
| `--bm25-b`                | `lexical.local_bm25.b`         |                                               |
| `DOCSTOKG_QWEN_MODEL_DIR` | `dense.qwen_vllm.download_dir` |                                               |
| `--qwen-batch`            | `dense.qwen_vllm.batch_size`   |                                               |
| `--sparse-model`          | `sparse.splade_st.model_id`    |                                               |
| `DOCSTOKG_TEI_URL`        | `dense.tei.url`                | Keep env alias.                               |
| `--dense-backend`         | `dense.backend`                | Canonical moving forward.                     |
| `--sparse-backend`        | `sparse.backend`               |                                               |
| `--lexical-backend`       | `lexical.backend`              |                                               |
| `--device`                | `embedding.device`             | Global override unless provider-specific set. |
| `--dtype`                 | `embedding.dtype`              | Global override unless provider-specific set. |

(Extend this table in code with every existing flag/env you discover; the pattern is the same.)

---

# Validation rules & user-facing errors

* **Backend required fields:** TEI requires `url`; Qwen requires `model_id` if you don’t keep a default; ST requires `model_id`.
* **Device constraints:** If backend cannot use GPU (e.g., TEI is remote; ST on CPU requested), it must **ignore** `embedding.device` quietly and report the actual device in telemetry.
* **Batch and concurrency:** Batch ≥ 1; `max_inflight_requests ≥ 1`; `queue_depth ≥ max_inflight_requests`.
* **BM25 params:** `k1 > 0`, `0 ≤ b ≤ 1`.
* **SPLADE dtype:** Coerce unsupported dtypes to `float32` with a warning (unless strict mode later).

All validation errors include: key name, provided value, allowed values, remediation hint (e.g., “set DOCSTOKG_TEI_URL or switch dense.backend=sentence_transformers”).

---

# Implementation checklist (for the agent)

1. **Add keys to Pydantic settings** (`EmbedCfg`) with defaults exactly as listed.
2. **Implement legacy mapping**: accept old flags/envs, translate to new keys, warn on use.
3. **ProviderFactory** reads `dense.backend`, `sparse.backend`, `lexical.backend`, constructs provider instances and calls `open(cfg)` once per run.
4. **Remove queue/cache from runtime**: providers own `queue_depth`, `max_inflight_requests`, `warmup`, etc. Runtime passes only text batches and receives vectors.
5. **Populate telemetry** with the standardized fields above.
6. **Write tests**: a) config precedence, b) validation errors, c) legacy alias mapping, d) provider selection matrix (dense×sparse×lexical), e) parity run (outputs identical for defaults).

---

# Acceptance criteria

* Selecting `dense.backend=qwen_vllm`, `sparse.backend=splade_st`, `lexical.backend=local_bm25` with **no additional keys** reproduces current outputs (JSONL/Parquet) byte-for-byte or within floating-point tolerance.
* Setting `dense.backend=tei` with `dense.tei.url` works with the same chunks and writes identical vector shapes.
* `--bm25-k1/--bm25-b` still work (populate `lexical.local_bm25.*`) and print a deprecation line pointing to the new keys.
* Telemetry includes `provider_name`, `device`, `dtype`, and timing fields for each provider.
* All provider modules can run independently (unit tests) and **runtime** does not import heavy libraries directly.

---

If you want, I can turn this into a **mapping matrix** that enumerates every current CLI flag/env we detect in your repo and the exact new key it should populate, so you can generate the compatibility layer and tests automatically.
