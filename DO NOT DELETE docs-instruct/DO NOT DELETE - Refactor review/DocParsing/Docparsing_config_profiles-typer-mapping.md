Love it — here’s a **ready-to-wire Typer option table** for your CLI. It maps each flag to the **Settings key**, **type**, **default**, **ENV alias**, **help text**, and **validation notes/legacy alias**. I’ve grouped options exactly how you’ll want them to render with Typer’s `rich_help_panel`, and I’ve included the `config show|diff` and `inspect` utilities you asked for.

---

# Root app (global) — `docparse [GLOBAL OPTIONS] <command>`

| Flag(s)                                  | Settings key              |    Type | Default                    | ENV                        | Help / notes                                                                                       |
| ---------------------------------------- | ------------------------- | ------: | -------------------------- | -------------------------- | -------------------------------------------------------------------------------------------------- |
| `--profile NAME`                         | `app.profile`             |   `str` | `None`                     | `DOCSTOKG_PROFILE`         | Load `docstokg.toml`/`.yaml` `[profile.NAME]` and layer it (profile → ENV → CLI).                  |
| `--data-root PATH`                       | `app.data_root`           |  `Path` | `./Data`                   | `DOCSTOKG_DATA_ROOT`       | Root folder for `Doctags/Chunks/Vectors/Manifests`. Validate: directory may be created by writers. |
| `--manifests-root PATH`                  | `app.manifests_root`      |  `Path` | `Data/Manifests`           | `DOCSTOKG_MANIFESTS_ROOT`  | Where manifest JSONL files land.                                                                   |
| `--models-root PATH`                     | `app.models_root`         |  `Path` | `~/.cache/docstokg/models` | `DOCSTOKG_MODELS_ROOT`     | Local cache for models/tokenizers.                                                                 |
| `--log-level {DEBUG,INFO,WARNING,ERROR}` | `app.log_level`           |  `Enum` | `INFO`                     | `DOCSTOKG_LOG_LEVEL`       | Root logging level.                                                                                |
| `--log-format {console,json}`            | `app.log_format`          |  `Enum` | `console`                  | `DOCSTOKG_LOG_FORMAT`      | Pretty console or structured JSON.                                                                 |
| `-v/--verbose` (repeatable)              | (maps to) `app.log_level` | `Count` | `0` → INFO                 | —                          | `-v`=DEBUG, `-vv`=TRACE (if enabled).                                                              |
| `--metrics/--no-metrics`                 | `app.metrics_enabled`     |  `bool` | `False`                    | `DOCSTOKG_METRICS_ENABLED` | Expose Prometheus metrics on `--metrics-port`.                                                     |
| `--metrics-port PORT`                    | `app.metrics_port`        |   `int` | `9108`                     | `DOCSTOKG_METRICS_PORT`    | Guard: 1–65535.                                                                                    |
| `--tracing/--no-tracing`                 | `app.tracing_enabled`     |  `bool` | `False`                    | `DOCSTOKG_TRACING_ENABLED` | Enable OpenTelemetry export; endpoint via env.                                                     |
| `--strict-config/--no-strict-config`     | `app.strict_config`       |  `bool` | `True`                     | `DOCSTOKG_STRICT_CONFIG`   | If false, unknown/deprecated keys warn instead of error (transition aid).                          |

**Help panels**: group these under **“Global”**. Keep the runner knobs (below) in a separate panel because they’re shared across subcommands.

---

# Shared runner knobs (available on all stage commands)

| Flag(s)                                    | Settings key                |        Type | Default       | ENV                                  | Help / notes                                      |
| ------------------------------------------ | --------------------------- | ----------: | ------------- | ------------------------------------ | ------------------------------------------------- |
| `--policy {io,cpu,gpu}`                    | `runner.policy`             |      `Enum` | stage default | `DOCSTOKG_RUNNER_POLICY`             | Doctags=io, Chunk=cpu, Embed=gpu (overridable).   |
| `--workers N`                              | `runner.workers`            |   `int ≥ 1` | stage default | `DOCSTOKG_RUNNER_WORKERS`            | Max parallel files; ProcessPool uses `spawn`.     |
| `--schedule {fifo,sjf}`                    | `runner.schedule`           |      `Enum` | `fifo`        | `DOCSTOKG_RUNNER_SCHEDULE`           | SJF uses `WorkItem.cost_hint`.                    |
| `--retries N`                              | `runner.retries`            |   `int ≥ 0` | `0`           | `DOCSTOKG_RUNNER_RETRIES`            | Per-item retry attempts for **retryable** errors. |
| `--retry-backoff-s SEC`                    | `runner.retry_backoff_s`    | `float ≥ 0` | `0.5`         | `DOCSTOKG_RUNNER_RETRY_BACKOFF_S`    | Exponential backoff base + jitter.                |
| `--timeout-s SEC`                          | `runner.per_item_timeout_s` | `float ≥ 0` | `0`           | `DOCSTOKG_RUNNER_PER_ITEM_TIMEOUT_S` | 0 disables.                                       |
| `--error-budget N`                         | `runner.error_budget`       |   `int ≥ 0` | `0`           | `DOCSTOKG_RUNNER_ERROR_BUDGET`       | Stop after N failures (0 = stop on first).        |
| `--max-queue N`                            | `runner.max_queue`          |   `int ≥ 1` | `workers*2`   | `DOCSTOKG_RUNNER_MAX_QUEUE`          | Submission backpressure.                          |
| `--adaptive {off,conservative,aggressive}` | `runner.adaptive`           |      `Enum` | `off`         | `DOCSTOKG_RUNNER_ADAPTIVE`           | Auto-tune workers from p95/fail rate (optional).  |
| `--fingerprinting/--no-fingerprinting`     | `runner.fingerprinting`     |      `bool` | `True`        | `DOCSTOKG_RUNNER_FINGERPRINTING`     | Use `*.fp.json` for exact resume.                 |

**Help panel**: **“Runner (applies to this command)”**.

---

# `doctags` command (PDF/HTML → DocTags JSONL)

**I/O**

| Flag(s)                  | Settings key         |   Type | Default        | ENV                       | Notes                             |
| ------------------------ | -------------------- | -----: | -------------- | ------------------------- | --------------------------------- |
| `--input-dir PATH`       | `doctags.input_dir`  | `Path` | `Data/Raw`     | `DOCSTOKG_DOCTAGS_INPUT`  | Directory scan; accepts pdf/html. |
| `--output-dir PATH`      | `doctags.output_dir` | `Path` | `Data/Doctags` | `DOCSTOKG_DOCTAGS_OUTPUT` | Writes partitioned JSONL.         |
| `--mode {auto,pdf,html}` | `doctags.mode`       | `Enum` | `auto`         | `DOCSTOKG_DOCTAGS_MODE`   | Auto picks by extension.          |

**Engine/model**

| Flag(s)                     | Settings key                  |        Type | Default           | ENV                                  | Notes                             |
| --------------------------- | ----------------------------- | ----------: | ----------------- | ------------------------------------ | --------------------------------- |
| `--model-id NAME`           | `doctags.model_id`            |       `str` | `granite-docling` | `DOCSTOKG_DOCTAGS_MODEL`             | PDF path; html path ignores.      |
| `--vllm-wait-timeout-s SEC` | `doctags.vllm_wait_timeout_s` | `float ≥ 0` | `60`              | `DOCSTOKG_DOCTAGS_VLLM_WAIT_TIMEOUT` | Wait for auxiliary VLM readiness. |

**Workflow**

| Flag(s)                | Settings key     |   Type | Default | Notes                                  |
| ---------------------- | ---------------- | -----: | ------- | -------------------------------------- |
| `--resume/--no-resume` | `doctags.resume` | `bool` | `True`  | Skip if output+fingerprint match.      |
| `--force/--no-force`   | `doctags.force`  | `bool` | `False` | Recompute regardless (atomic replace). |

**Validation**: input dir must exist; `mode` enum; warn if both `resume` and `force` set (force wins).

---

# `chunk` command (DocTags → Chunks Parquet/JSONL)

**I/O**

| Flag(s)                    | Settings key       |   Type | Default        | ENV                         | Notes                |
| -------------------------- | ------------------ | -----: | -------------- | --------------------------- | -------------------- |
| `--in-dir PATH`            | `chunk.input_dir`  | `Path` | `Data/Doctags` | `DOCSTOKG_CHUNK_INPUT_DIR`  | Source doctags.      |
| `--out-dir PATH`           | `chunk.output_dir` | `Path` | `Data/Chunks`  | `DOCSTOKG_CHUNK_OUTPUT_DIR` | Target dataset root. |
| `--format {parquet,jsonl}` | `chunk.format`     | `Enum` | `parquet`      | `DOCSTOKG_CHUNK_FORMAT`     | PR-8 default.        |

**Chunking**

| Flag(s)                | Settings key               |        Type | Default       | Notes                                 |
| ---------------------- | -------------------------- | ----------: | ------------- | ------------------------------------- |
| `--min-tokens N`       | `chunk.min_tokens`         |   `int ≥ 1` | `120`         | Tokenizer-aware.                      |
| `--max-tokens N`       | `chunk.max_tokens`         | `int ≥ min` | `800`         |                                       |
| `--tokenizer-model ID` | `chunk.tokenizer.model_id` |       `str` | `cl100k_base` | Matches dense embedder when possible. |

**Workflow**

| Flag(s)                | Settings key   |   Type | Default | Notes |
| ---------------------- | -------------- | -----: | ------- | ----- |
| `--resume/--no-resume` | `chunk.resume` | `bool` | `True`  |       |
| `--force/--no-force`   | `chunk.force`  | `bool` | `False` |       |

---

# `embed` command (Chunks → Vectors)

**I/O & families**

| Flag(s)                           | Settings key               |   Type | Default        | ENV                             | Notes                                  |
| --------------------------------- | -------------------------- | -----: | -------------- | ------------------------------- | -------------------------------------- |
| `--chunks-dir PATH`               | `embed.input_chunks_dir`   | `Path` | `Data/Chunks`  | `DOCSTOKG_EMBED_CHUNKS_DIR`     | Parquet by default.                    |
| `--out-dir PATH`                  | `embed.output_vectors_dir` | `Path` | `Data/Vectors` | `DOCSTOKG_EMBED_OUTPUT_DIR`     | Partitioned by `family/fmt/yyyy/mm`.   |
| `--vector-format {parquet,jsonl}` | `embed.vectors.format`     | `Enum` | `parquet`      | `DOCSTOKG_EMBED_VECTOR_FORMAT`  | PR-8 default.                          |
| `--enable-dense/--no-dense`       | `embed.families.dense`     | `bool` | `True`         | `DOCSTOKG_EMBED_FAMILY_DENSE`   | Toggle families.                       |
| `--enable-sparse/--no-sparse`     | `embed.families.sparse`    | `bool` | `True`         | `DOCSTOKG_EMBED_FAMILY_SPARSE`  |                                        |
| `--enable-lexical/--no-lexical`   | `embed.families.lexical`   | `bool` | `True`         | `DOCSTOKG_EMBED_FAMILY_LEXICAL` |                                        |
| `--plan-only`                     | `embed.plan_only`          | `bool` | `False`        | —                               | Build plan & print summary; no writes. |

**Dense provider selection**

| Flag(s)                                                      | Settings key          |   Type | Default     | Notes                  |
| ------------------------------------------------------------ | --------------------- | -----: | ----------- | ---------------------- |
| `--dense-backend {qwen_vllm,tei,sentence_transformers,none}` | `embed.dense.backend` | `Enum` | `qwen_vllm` | `none` disables dense. |

**Dense: Qwen vLLM**

| Flag(s)                                | Settings key                                   |        Type | Default              | ENV                          | Notes                      |      |          |
| -------------------------------------- | ---------------------------------------------- | ----------: | -------------------- | ---------------------------- | -------------------------- | ---- | -------- |
| `--qwen-model-id ID`                   | `embed.dense.qwen_vllm.model_id`               |       `str` | `Qwen2-7B-Embedding` | `DOCSTOKG_QWEN_MODEL_ID`     |                            |      |          |
| `--qwen-batch-size N`                  | `embed.dense.qwen_vllm.batch_size`             |   `int ≥ 1` | `64`                 | `DOCSTOKG_QWEN_BATCH_SIZE`   |                            |      |          |
| `--qwen-dtype {auto,float16,bfloat16}` | `embed.dense.qwen_vllm.dtype`                  |      `Enum` | `auto`               | `DOCSTOKG_QWEN_DTYPE`        |                            |      |          |
| `--qwen-device DEV`                    | `embed.dense.qwen_vllm.device`                 |       `str` | `cuda:0`             | `DOCSTOKG_QWEN_DEVICE`       | `cpu                       | cuda | cuda:N`. |
| `--qwen-tp N`                          | `embed.dense.qwen_vllm.tensor_parallelism`     |   `int ≥ 1` | `1`                  | `DOCSTOKG_QWEN_TP`           |                            |      |          |
| `--qwen-max-model-len N`               | `embed.dense.qwen_vllm.max_model_len`          | `int ≥ 512` | `8192`               | `DOCSTOKG_QWEN_MAX_LEN`      |                            |      |          |
| `--qwen-gpu-mem-util F`                | `embed.dense.qwen_vllm.gpu_memory_utilization` |  `0.1–0.99` | `0.9`                | `DOCSTOKG_QWEN_GPU_MEM_UTIL` |                            |      |          |
| `--qwen-download-dir PATH`             | `embed.dense.qwen_vllm.download_dir`           |      `Path` | `models_root`        | `DOCSTOKG_QWEN_DIR`          |                            |      |          |
| `--qwen-warmup/--no-qwen-warmup`       | `embed.dense.qwen_vllm.warmup`                 |      `bool` | `True`               |                              | Send tiny batch at open(). |      |          |

**Dense: TEI (HTTP)**

| Flag(s)                                | Settings key                            |        Type | Default             | ENV                         | Notes                      |
| -------------------------------------- | --------------------------------------- | ----------: | ------------------- | --------------------------- | -------------------------- |
| `--tei-url URL`                        | `embed.dense.tei.url`                   |       `str` | **required**        | `DOCSTOKG_TEI_URL`          | Required when backend=tei. |
| `--tei-api-key KEY`                    | `embed.dense.tei.api_key`               |       `str` | `None`              | `DOCSTOKG_TEI_API_KEY`      | Redact in logs.            |
| `--tei-timeout-s SEC`                  | `embed.dense.tei.timeout_s`             | `float ≥ 1` | `30`                | `DOCSTOKG_TEI_TIMEOUT_S`    |                            |
| `--tei-verify-tls/--no-tei-verify-tls` | `embed.dense.tei.verify_tls`            |      `bool` | `True`              | `DOCSTOKG_TEI_VERIFY_TLS`   |                            |
| `--tei-compression {auto,gzip,none}`   | `embed.dense.tei.compression`           |      `Enum` | `auto`              | `DOCSTOKG_TEI_COMPRESSION`  |                            |
| `--tei-max-inflight N`                 | `embed.dense.tei.max_inflight_requests` |   `int ≥ 1` | `runner.workers`    | `DOCSTOKG_TEI_MAX_INFLIGHT` | Client-side limit.         |
| `--tei-batch-size N`                   | `embed.dense.tei.batch_size`            |   `int ≥ 1` | `runner.batch_size` | `DOCSTOKG_TEI_BATCH_SIZE`   | Overrides global.          |

**Dense: Sentence-Transformers**

| Flag(s)                                            | Settings key                                          |       Type | Default                  | ENV                             | Notes          |
| -------------------------------------------------- | ----------------------------------------------------- | ---------: | ------------------------ | ------------------------------- | -------------- |
| `--st-model-id ID`                                 | `embed.dense.sentence_transformers.model_id`          |      `str` | **required if selected** | `DOCSTOKG_ST_MODEL_ID`          |                |
| `--st-revision REV`                                | `embed.dense.sentence_transformers.revision`          |      `str` | `None`                   | `DOCSTOKG_ST_REVISION`          |                |
| `--st-device DEV`                                  | `embed.dense.sentence_transformers.device`            |      `str` | `auto`                   | `DOCSTOKG_ST_DEVICE`            |                |
| `--st-dtype {auto,float32,float16,bfloat16}`       | `embed.dense.sentence_transformers.dtype`             |     `Enum` | `auto`                   | `DOCSTOKG_ST_DTYPE`             |                |
| `--st-batch-size N`                                | `embed.dense.sentence_transformers.batch_size`        |  `int ≥ 1` | `64`                     | `DOCSTOKG_ST_BATCH_SIZE`        |                |
| `--st-max-seq-len N`                               | `embed.dense.sentence_transformers.max_seq_length`    | `int ≥ 16` | model default            | `DOCSTOKG_ST_MAX_SEQ_LENGTH`    |                |
| `--st-trust-remote-code/--no-st-trust-remote-code` | `embed.dense.sentence_transformers.trust_remote_code` |     `bool` | `False`                  | `DOCSTOKG_ST_TRUST_REMOTE_CODE` |                |
| `--st-use-mmap/--no-st-use-mmap`                   | `embed.dense.sentence_transformers.use_memory_map`    |     `bool` | `True`                   | `DOCSTOKG_ST_USE_MMAP`          |                |
| `--st-threads N`                                   | `embed.dense.sentence_transformers.intra_op_threads`  |  `int ≥ 1` | `auto`                   | `DOCSTOKG_ST_THREADS`           | CPU runs only. |

**Sparse: SPLADE**

| Flag(s)                                             | Settings key                                   |        Type | Default                        | ENV                                | Notes |
| --------------------------------------------------- | ---------------------------------------------- | ----------: | ------------------------------ | ---------------------------------- | ----- |
| `--splade-model-id ID`                              | `embed.sparse.splade_st.model_id`              |       `str` | `naver/splade-cocondenser-...` | `DOCSTOKG_SPLADE_MODEL_ID`         |       |
| `--splade-revision REV`                             | `embed.sparse.splade_st.revision`              |       `str` | `None`                         | `DOCSTOKG_SPLADE_REVISION`         |       |
| `--splade-device DEV`                               | `embed.sparse.splade_st.device`                |       `str` | `auto`                         | `DOCSTOKG_SPLADE_DEVICE`           |       |
| `--splade-dtype {auto,float32,float16,bfloat16}`    | `embed.sparse.splade_st.dtype`                 |      `Enum` | `auto`                         | `DOCSTOKG_SPLADE_DTYPE`            |       |
| `--splade-batch-size N`                             | `embed.sparse.splade_st.batch_size`            |   `int ≥ 1` | `64`                           | `DOCSTOKG_SPLADE_BATCH_SIZE`       |       |
| `--splade-topk N`                                   | `embed.sparse.splade_st.postproc.topk_per_doc` |   `int ≥ 0` | `0 (=all)`                     | `DOCSTOKG_SPLADE_TOPK`             |       |
| `--splade-prune-below F`                            | `embed.sparse.splade_st.postproc.prune_below`  | `float ≥ 0` | `0.0`                          | `DOCSTOKG_SPLADE_PRUNE_BELOW`      |       |
| `--splade-norm {none,l2}`                           | `embed.sparse.splade_st.normalize_doclen`      |      `Enum` | `l2`                           | `DOCSTOKG_SPLADE_NORMALIZE_DOCLEN` |       |
| `--splade-tokenizer ID`                             | `embed.sparse.splade_st.tokenizer_id`          |       `str` | model default                  | `DOCSTOKG_SPLADE_TOKENIZER_ID`     |       |
| `--splade-attn {auto,sdpa,eager,flash_attention_2}` | `embed.sparse.splade_st.attn_backend`          |      `Enum` | `auto`                         | `DOCSTOKG_SPLADE_ATTN_BACKEND`     |       |

**Lexical: BM25**

| Flag(s)                                         | Settings key                            |          Type | Default  | ENV                          | Notes                |
| ----------------------------------------------- | --------------------------------------- | ------------: | -------- | ---------------------------- | -------------------- |
| `--bm25-k1 F`                                   | `embed.lexical.local_bm25.k1`           |   `float > 0` | `1.5`    | `DOCSTOKG_BM25_K1`           | Range check.         |
| `--bm25-b F`                                    | `embed.lexical.local_bm25.b`            |   `0≤float≤1` | `0.75`   | `DOCSTOKG_BM25_B`            | Range check.         |
| `--bm25-stopwords {none,english,PATH}`          | `embed.lexical.local_bm25.stopwords`    |         `str` | `none`   | `DOCSTOKG_BM25_STOPWORDS`    | If path, must exist. |
| `--bm25-tokenizer {simple,spacy_en,regexp:...}` | `embed.lexical.local_bm25.tokenizer`    |         `str` | `simple` | `DOCSTOKG_BM25_TOKENIZER`    |                      |
| `--bm25-min-df N`                               | `embed.lexical.local_bm25.min_df`       |     `int ≥ 0` | `1`      | `DOCSTOKG_BM25_MIN_DF`       |                      |
| `--bm25-max-df-ratio F`                         | `embed.lexical.local_bm25.max_df_ratio` | `0< float ≤1` | `1.0`    | `DOCSTOKG_BM25_MAX_DF_RATIO` |                      |

**Workflow**

| Flag(s)                | Settings key   |   Type | Default | Notes                                 |
| ---------------------- | -------------- | -----: | ------- | ------------------------------------- |
| `--resume/--no-resume` | `embed.resume` | `bool` | `True`  | Skip when fingerprints+outputs match. |
| `--force/--no-force`   | `embed.force`  | `bool` | `False` | Recompute regardless.                 |

**Help panels**:

* **I/O & Families**
* **Dense Providers** (with sub-panels: Qwen vLLM / TEI / Sentence-Transformers)
* **Sparse (SPLADE)**
* **Lexical (BM25)**
* **Runner**

---

# `all` command (Doctags → Chunk → Embed)

| Flag(s)                       | Settings key       |   Type | Default  | Notes                                                          |
| ----------------------------- | ------------------ | -----: | -------- | -------------------------------------------------------------- |
| `--data-root PATH`            | `app.data_root`    | `Path` | `./Data` | Shorthand to set a common root.                                |
| `--resume/--no-resume`        | (per stage)        | `bool` | `True`   | For all stages unless overridden per-stage.                    |
| `--force/--no-force`          | (per stage)        | `bool` | `False`  | Ditto.                                                         |
| `--stop-on-fail/--keep-going` | `all.stop_on_fail` | `bool` | `True`   | Cancel downstream stages if a stage fails.                     |
| (all **Runner** knobs)        | `runner.*`         |      — | —        | Apply to each stage unless overridden in stage-specific flags. |

---

# Utilities

## `config show`

| Flag(s)                                        | Key           |   Type | Default | Notes                                         |
| ---------------------------------------------- | ------------- | -----: | ------- | --------------------------------------------- |
| `--profile NAME`                               | `app.profile` |  `str` | `None`  | Apply profile while showing.                  |
| `--stage {app,runner,doctags,chunk,embed,all}` | —             | `Enum` | `all`   | Slice output.                                 |
| `--format {yaml,json,toml,env}`                | —             | `Enum` | `yaml`  | Output format.                                |
| `--annotate-source/--no-annotate-source`       | —             | `bool` | `False` | Show per-key origin: default/profile/env/cli. |
| `--redact/--no-redact`                         | —             | `bool` | `True`  | Hide tokens/keys.                             |

## `config diff`

| Flag(s)                                        | Key   |        Type | Default         | Notes                             |                   |
| ---------------------------------------------- | ----- | ----------: | --------------- | --------------------------------- | ----------------- |
| `--lhs-profile NAME                            | none` |           — | `str`           | `none`                            | Build LHS config. |
| `--lhs-file PATH`                              | —     |      `Path` | `docstokg.toml` | Optional profile file.            |                   |
| `--lhs-env-file PATH`                          | —     |      `Path` | `None`          | Load an `.env` for LHS.           |                   |
| `--lhs-override KEY=VAL` (repeatable)          | —     | `list[str]` | `[]`            | Dot-path overrides.               |                   |
| `--rhs-profile NAME                            | none` |           — | `str`           | `gpu`                             | Build RHS config. |
| `--rhs-file PATH`                              | —     |      `Path` | `docstokg.toml` |                                   |                   |
| `--rhs-env-file PATH`                          | —     |      `Path` | `None`          |                                   |                   |
| `--rhs-override KEY=VAL` (repeatable)          | —     | `list[str]` | `[]`            |                                   |                   |
| `--stage {app,runner,doctags,chunk,embed,all}` | —     |      `Enum` | `all`           | Section to diff.                  |                   |
| `--format {unified,json,yaml,table}`           | —     |      `Enum` | `unified`       |                                   |                   |
| `--show-hash/--no-show-hash`                   | —     |      `bool` | `True`          | Print per-stage `cfg_hash` (≠/=). |                   |

## `inspect` (dataset quick scan)

| Flag(s)                                                           | Key             |        Type | Default  | Notes                                       |
| ----------------------------------------------------------------- | --------------- | ----------: | -------- | ------------------------------------------- |
| `--dataset {chunks,vectors-dense,vectors-sparse,vectors-lexical}` | —               |      `Enum` | `chunks` | Which dataset to inspect.                   |
| `--root PATH`                                                     | `app.data_root` |      `Path` | `./Data` | Dataset base.                               |
| `--columns COLS`                                                  | —               | `list[str]` | `[]`     | Optional projection.                        |
| `--filters EXPR`                                                  | —               |       `str` | `None`   | Pushdown filter (e.g., `doc_id == "…"`).    |
| `--limit N`                                                       | —               |   `int ≥ 0` | `0`      | 0 = all (may be heavy).                     |
| `--stats/--no-stats`                                              | —               |      `bool` | `True`   | Include row count, file count, total bytes. |

---

## Validation & UX guardrails (how to wire in Typer)

* Use **Enums** for every fixed choice (backends, schedule, policy, formats) for early, clear errors and shell completion.
* Use `Annotated[..., typer.Option(rich_help_panel="…")]` to group: **Global**, **Runner**, **I/O**, **Dense Providers**, **Sparse**, **Lexical**, **Advanced**.
* Add **hidden** legacy aliases (e.g., `--bm25-k1`) that set the new key and print a one-line deprecation: “use `embed.lexical.local_bm25.k1` or env `DOCSTOKG_BM25_K1`”.
* Keep **shallow** validations in option callbacks (path exists, ints ≥1). Let **Pydantic** enforce cross-field semantics (TEI requires `url`, BM25 ranges, device feasibility).
* Print the effective profile in the command header (e.g., “profile=gpu; cfg_hash.embed=9f4b…ce”).

---

## Quick acceptance tests (CliRunner snapshots)

* `docparse --help` shows the **Global** panel and subcommands.
* `docparse embed --help` shows **I/O & Families**, **Dense Providers**, **Sparse**, **Lexical**, **Runner** (in that order).
* Invalid `--bm25-b -0.1` errors immediately with: “must be in [0, 1]”.
* `docparse config show --profile gpu --stage embed --format yaml` prints merged config with correct overrides.
* `docparse config diff --lhs-profile none --rhs-profile gpu` shows changed keys and different `cfg_hash.embed`.

---

If you want, I can turn this into a **Typer signature skeleton** (just decorators + option declarations with help panels and enums) that your team can paste into the CLI module and fill in with calls to your Settings builder and runner.
