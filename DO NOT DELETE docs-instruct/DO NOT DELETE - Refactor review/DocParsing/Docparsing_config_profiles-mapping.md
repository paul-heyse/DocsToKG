Absolutely—here’s a **field-accurate legacy→new configuration mapping** for DocParsing and a **spec for a `config show|diff` Typer subcommand** you can hand to your agents. I grounded the “legacy” side in flags/env actually present in your repo and docs (citations at the end of each section).

---

# Legacy → New configuration mapping (DocParsing)

## Legend

* **Legacy**: existing CLI flag or env var users may already rely on.
* **New key**: the target `BaseSettings` path from PR-7 (with `DOCSTOKG_` env mirroring).
* **Notes**: deprecation + validation tips.

### A) Global / shared

| Legacy (CLI/ENV)                     | New key (Pydantic Settings) | Notes                                                            |      |                                       |
| ------------------------------------ | --------------------------- | ---------------------------------------------------------------- | ---- | ------------------------------------- |
| `--data-root` / `DOCSTOKG_DATA_ROOT` | `app.data_root`             | Root for `Data/*` trees.                                         |      |                                       |
| `--manifest-dir`                     | `app.manifests_root`        | Default `Data/Manifests`. Mirrors existing reader/writer paths.  |      |                                       |
| `--log-level`                        | `app.log_level`             | `DEBUG                                                           | INFO | …`; console+JSON already documented.  |
| `DOCSTOKG_LOG_DIR`                   | `app.log_dir`               | Where `docparse-*.jsonl` logs go by default.                     |      |                                       |
| `DOCSTOKG_MODEL_ROOT`                | `app.models_root`           | Base for DocTags/embedding models.                               |      |                                       |
| `DOCSTOKG_HASH_ALG`                  | `app.hash_alg`              | Default now **sha256**; legacy `sha1` supported with warning.    |      |                                       |

### B) DocTags (PDF/HTML)

| Legacy (CLI/ENV)                                             | New key                                        | Notes                                         |
| ------------------------------------------------------------ | ---------------------------------------------- | --------------------------------------------- |
| `DOCSTOKG_DOCTAGS_INPUT`                                     | `doctags.input_dir`                            | Input directory for PDF/HTML.                 |
| `DOCSTOKG_DOCTAGS_OUTPUT`                                    | `doctags.output_dir`                           | Where `Doctags/{yyyy}/{mm}/*.jsonl` land.     |
| `DOCSTOKG_DOCTAGS_MODEL`                                     | `doctags.model_id`                             | E.g., Granite-Docling.                        |
| `DOCSTOKG_DOCTAGS_WORKERS` / `--workers`                     | `doctags.workers` (overrides `runner.workers`) | PDF path uses spawn; validate `workers ≥ 1`.  |
| `DOCSTOKG_DOCTAGS_VLLM_WAIT_TIMEOUT` / `--vllm-wait-timeout` | `doctags.vllm_wait_timeout_s`                  | Seconds to wait for auxiliary VLM.            |

### C) Chunking

| Legacy (CLI/ENV)                                       | New key                                   | Notes                                     |
| ------------------------------------------------------ | ----------------------------------------- | ----------------------------------------- |
| `--in-dir` (`Data/DocTagsFiles`)                       | `chunk.input_dir`                         | Doctags source dir (jsonl).               |
| `--out-dir` (`Data/ChunkedDocTagFiles`)                | `chunk.output_dir`                        | Chunks output (Parquet default in PR-8).  |
| `--min-tokens` / `DOCSTOKG_CHUNK_MIN_TOKENS`           | `chunk.min_tokens`                        | `int ≥ 1`.                                |
| `--max-tokens` / `DOCSTOKG_CHUNK_MAX_TOKENS`           | `chunk.max_tokens`                        | `int ≥ min_tokens`.                       |
| `--tokenizer-model` / `DOCSTOKG_CHUNK_TOKENIZER_MODEL` | `chunk.tokenizer.model_id`                | Align chunk lengths with dense embedder.  |
| `--shard-count` / `--shard-index`                      | `chunk.shard.count` / `chunk.shard.index` | Deterministic sharding for big corpora.   |

### D) Embedding (dense/sparse/lexical)

**Common I/O/format**

| Legacy (CLI/ENV) | New key                                  | Notes                           |                                            |
| ---------------- | ---------------------------------------- | ------------------------------- | ------------------------------------------ |
| `--chunks-dir`   | `embed.input_chunks_dir`                 | Source chunks (jsonl/parquet).  |                                            |
| `--out-dir`      | `embed.output_vectors_dir`               | Where vectors land.             |                                            |
| `--format {jsonl | parquet}`/`DOCSTOKG_EMBED_VECTOR_FORMAT` | `embed.vectors.format`          | Parquet default in PR-8; CLI still wins.   |

**Dense (Qwen/vLLM, ST, TEI)**

| Legacy (CLI/ENV)                          | New key                              | Notes                                                                                            |          |           |
| ----------------------------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------ | -------- | --------- |
| `--batch-size-qwen`                       | `embed.dense.qwen_vllm.batch_size`   | `int ≥ 1`.                                                                                       |          |           |
| `--qwen-dtype`                            | `embed.dense.qwen_vllm.dtype`        | `float16                                                                                         | bfloat16 | auto`.    |
| `DOCSTOKG_QWEN_DIR`                       | `embed.dense.qwen_vllm.download_dir` | Model cache path.                                                                                |          |           |
| `DOCSTOKG_QWEN_DEVICE`                    | `embed.dense.qwen_vllm.device`       | `cpu                                                                                             | cuda     | cuda:N`.  |
| `DOCSTOKG_EMBED_QWEN_DIM` (validate-only) | `embed.dense.expected_dim`           | **New** field used only by validation to assert dimension; omit to accept historical artifacts.  |          |           |

**Sparse (SPLADE)**

| Legacy (CLI/ENV)                                    | New key                                  | Notes           |      |           |
| --------------------------------------------------- | ---------------------------------------- | --------------- | ---- | --------- |
| `--batch-size-splade`                               | `embed.sparse.splade_st.batch_size`      | `int ≥ 1`.      |      |           |
| `--splade-attn {auto,sdpa,eager,flash_attention_2}` | `embed.sparse.splade_st.attn_backend`    | Choice Enum.    |      |           |
| `--splade-max-active-dims`                          | `embed.sparse.splade_st.max_active_dims` | Optional cap.   |      |           |
| `DOCSTOKG_SPLADE_DIR`                               | `embed.sparse.splade_st.model_dir`       | Local weights.  |      |           |
| `DOCSTOKG_SPLADE_DEVICE`                            | `embed.sparse.splade_st.device`          | `cpu            | cuda | cuda:N`.  |

**Lexical (BM25)**

| Legacy (CLI/ENV) | New key                       | Notes         |
| ---------------- | ----------------------------- | ------------- |
| `--bm25-k1`      | `embed.lexical.local_bm25.k1` | `float > 0`.  |
| `--bm25-b`       | `embed.lexical.local_bm25.b`  | `0 ≤ b ≤ 1`.  |

> These env families and flags are explicitly cited in your README/agents docs and recent PRs (“Vector format negotiation”, model cache envs, shared flags). The mapping above preserves UX while funneling everything into one typed source of truth.

---

## Compatibility shim (one minor)

* **Rule**: When both legacy and new are provided, **new wins**, and you emit **one** deprecation line per key (e.g., “`--bm25-k1` is deprecated; use `embed.lexical.local_bm25.k1` / `DOCSTOKG_EMBED_LEXICAL_LOCAL_BM25_K1`”).
* Keep stage helpers (`from_env`, `from_args`) as thin wrappers that call the unified Settings builder, so tests/docs using those still pass while you cut over.

---

# `docparse config show` / `docparse config diff` — Typer subcommand spec

## Overview

Add a **`config`** group under your Typer root with subcommands:

* `docparse config show` — print the **effective** configuration (after profile+ENV+CLI layering), optionally restricted to one stage, in `yaml|json|toml|env`.
* `docparse config diff` — diff **two** materialized configs (e.g., `profile=gpu` vs `profile=local` + overrides), show **added/removed/changed** keys and **cfg_hash** deltas.

### Why

* Gives operators an authoritative view of **what actually runs** (with source annotations if desired).
* Makes reviews reproducible: attach a `config show --format yaml` to PRs and CI logs.

---

## `config show` (detailed)

**Signature**

```
docparse config show
  [--profile NAME]
  [--stage app|runner|doctags|chunk|embed|all]  # default: all
  [--format yaml|json|toml|env]                 # default: yaml
  [--annotate-source / --no-annotate-source]    # default: no-annotate
  [--redact / --no-redact]                      # default: redact (secrets, tokens)
```

**Behavior**

* Rebuild the effective Settings (using the same builder as root callback): **defaults → profile → ENV → CLI**.
* If `--annotate-source`, include a parallel map of **per-key origin** (`default|profile|env|cli`) gathered during the layering pass.
* If `--stage` ≠ `all`, slice the dict to that subtree.
* `--format env` emits `DOCSTOKG_*` exports (handy for `.env` pre-flight).
* Redact patterns: keys containing `token|password|secret|api_key`.

**Output examples (human)**

```yaml
app:
  data_root: /mnt/data
  log_level: INFO
runner:
  policy: gpu
  workers: 8
embed:
  vectors:
    format: parquet
  dense:
    backend: qwen_vllm
    batch_size: 64
    device: cuda:0
  lexical:
    local_bm25: { k1: 1.5, b: 0.75 }
cfg_hash:
  doctags: 3e7d…a9
  chunk:   c1db…21
  embed:   9f4b…ce
profile: gpu
```

**Acceptance**

* Shows the exact config the stages will use **before** they run.
* Produces stable YAML/JSON suitable for attachment to manifests (your `__config__` rows).

---

## `config diff` (detailed)

**Signature**

```
docparse config diff
  --lhs-profile NAME|none
  [--lhs-file PATH]             # optional profile file override
  [--lhs-env-file PATH]         # optional .env to simulate env layer
  [--lhs-override KEY=VALUE]... # repeated; dot-paths (e.g., embed.dense.backend=tei)

  --rhs-profile NAME|none
  [--rhs-file PATH]
  [--rhs-env-file PATH]
  [--rhs-override KEY=VALUE]...

  [--stage app|runner|doctags|chunk|embed|all]  # default: all
  [--format unified|json|yaml|table]            # default: unified
  [--show-hash / --no-show-hash]                # default: show-hash
```

**Behavior**

* Materialize two effective configs **independently** using the same precedence rules:

  * Start from **defaults**.
  * Layer `profile` from `--*-file` (if provided) or default `docstokg.toml`.
  * Apply `.env` from `--*-env-file` (optional).
  * Apply `--*-override KEY=VALUE` pairs (dot-path merge; e.g., `runner.workers=12`).
* Produce a **three-way diff**:

  * **changed** (key exists in both → value differs),
  * **added** (only in RHS),
  * **removed** (only in LHS).
* If `--show-hash`, compute and print `cfg_hash` by stage for LHS/RHS (and mark `≠`/`=`).
* Formats:

  * **unified**: human-readable blocks with `- old` / `+ new`.
  * **table**: columns `key | lhs | rhs | status`.
  * **json|yaml**: machine friendly.

**Output (unified)**

```
[embed] cfg_hash: 9f4b…ce  ≠  1a2c…d0
 changed: embed.dense.backend:        qwen_vllm  →  tei
 changed: embed.dense.tei.url:        (empty)    →  https://tei.intra:8080
 added:   embed.vectors.format:       parquet
 removed: embed.dense.qwen_vllm.device: cuda:0
```

**Acceptance**

* Operators can preview **exact** effects of switching profiles/overrides **before** a run.
* CI can guard drifts by snapshotting `config show` and diffing against a baseline.

---

## Implementation notes (for the agent)

* **Source annotation**: While layering, build an origin map `{dot_key: "default|profile|env|cli"}`. Use it for `--annotate-source` and to render tooltips in rich help panels.
* **Dot-path parser**: Support `KEY=VALUE` for simple literals (`int|float|bool|str`) and nested via dot (`embed.dense.backend=tei`). Overwrite scalar or merge mapping.
* **Validation**: Run Pydantic validators **after** applying overrides; present a single consolidated error (good UX).
* **Redaction**: Regex `(?i)(token|secret|password|api[_-]?key)`.

**Tests**

* **Precedence**: matrix of defaults/profile/env/override combinations for one or two keys.
* **Dot-path**: nested override merges correctly (`embed.lexical.local_bm25.k1=1.2`).
* **Hashes**: flip one key and assert `cfg_hash.embed` changes.
* **Error surfacing**: invalid `bm25.b=-0.1` yields friendly message.

---

# Rollout & UX guardrails

* Keep legacy flags/env **for one minor**; print one-line deprecations pointing to the new key every time a legacy input is used (don’t spam per item).
* Put a **“Configuration Layering & Profiles”** page in docs with a table like above, plus an ENV prefix table:

  * `DOCSTOKG_APP_*`, `DOCSTOKG_RUNNER_*`, `DOCSTOKG_DOCTAGS_*`, `DOCSTOKG_CHUNK_*`, `DOCSTOKG_EMBED_*` (and nested providers via underscores), so users can set envs without YAML. (You already list the major families in README/Agents.)
* Expose `DOCSTOKG_STRICT_CONFIG=false` to **downgrade unknown/deprecated keys to warnings** during the transition (default strict in CI).

---

## Citations (where the legacy flags/env appear)

* BM25, SPLADE, Qwen CLI flags in embedding docs/args (`--bm25-k1`, `--bm25-b`, `--batch-size-*`, `--splade-attn`, `--chunks-dir`, `--out-dir`).
* Vector format `--format parquet` and `DOCSTOKG_EMBED_VECTOR_FORMAT`; manifest `vector_format` audited.
* Shared flags (`--data-root`, `--log-level`, `--resume`, `--force`) and env families (`DOCSTOKG_DOCTAGS_*`, `DOCSTOKG_CHUNK_*`, `DOCSTOKG_EMBED_*`, caches/paths).
* Hash algorithm env (`DOCSTOKG_HASH_ALG`) and default switch to SHA-256.

---

If you’d like, I can follow up with a **ready-to-paste Typer option table** (per subcommand) and a **deprecation map** you can drop into the root callback to emit consistent warnings (including suggested replacements and links to the docs page).
