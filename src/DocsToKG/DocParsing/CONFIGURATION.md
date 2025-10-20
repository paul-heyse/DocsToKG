# DocParsing Configuration & Profiles (PR-7)

**Last updated:** October 21, 2025

## Overview

DocParsing now uses **Pydantic v2 `BaseSettings`** with a unified, type-safe configuration system. All configuration is sourced from a single source of truth with clear **precedence rules**, **typed validation**, and **profile support** for different deployment scenarios (local dev, GPU clusters, airgapped environments, etc.).

---

## Configuration Layering (Precedence)

The canonical precedence is: **CLI args > ENV vars > profile file > defaults**

Each layer completely overrides lower layers for matched keys.

```
┌─────────────────────────────────────────┐
│ CLI Arguments (--data-root, etc.)       │ ← Highest precedence
├─────────────────────────────────────────┤
│ Environment Variables (DOCSTOKG_*)      │
├─────────────────────────────────────────┤
│ Profile File (docstokg.toml [profile.*])│
├─────────────────────────────────────────┤
│ Pydantic Defaults                       │ ← Lowest precedence
└─────────────────────────────────────────┘
```

### Examples

**Scenario 1: Override profile with ENV**

```bash
# Profile says batch_size=64
DOCSTOKG_EMBED_DENSE_QWEN_VLLM_BATCH_SIZE=32 \
  docparse --profile gpu embed ...
# Result: batch_size=32 (ENV wins over profile)
```

**Scenario 2: Override everything with CLI**

```bash
docparse --profile gpu \
  --chunk-min-tokens 512 \
  chunk ...
# Result: min_tokens=512 (CLI wins)
```

**Scenario 3: Fall back through layers**

```bash
docparse embed ...
# No --profile, no DOCSTOKG_* env vars
# Result: defaults apply (min_tokens=120, max_tokens=800, etc.)
```

---

## Profiles (docstokg.toml / docstokg.yaml)

Profiles allow you to pre-configure entire pipelines for common deployment scenarios.

### Profile File Location

Place `docstokg.toml` or `docstokg.yaml` in one of these locations:

1. Project root (`./docstokg.toml`)
2. Config directory (`./config/docstokg.toml`)

The file is auto-discovered if you reference a profile name.

### Built-in Profiles

The included `docstokg.toml` provides:

- **`local`**: CPU-only development (4 workers, sentence-transformers, FIFO scheduling)
- **`gpu`**: GPU-optimized (8 workers, Qwen2-7B-Embedding, flash-attention, SJF scheduling)
- **`airgapped`**: Offline-safe (no network during inference, min workers)
- **`dev`**: Debugging (single worker, verbose logging, small batches)

### Profile Structure (TOML example)

```toml
[profile.my_profile]

[profile.my_profile.app]
data_root = "/data"
log_level = "INFO"
log_format = "json"

[profile.my_profile.runner]
policy = "gpu"
workers = 8
schedule = "sjf"

[profile.my_profile.chunk]
min_tokens = 256
max_tokens = 1024

[profile.my_profile.embed.dense]
backend = "qwen_vllm"

[profile.my_profile.embed.dense.qwen_vllm]
batch_size = 64
dtype = "bfloat16"
device = "cuda:0"

[profile.my_profile.embed.lexical.local_bm25]
k1 = 1.5
b = 0.75
```

---

## Environment Variables

All configuration can be overridden via `DOCSTOKG_*` environment variables. The mapping is:

```
Pydantic field path  →  ENV var name
app.data_root        →  DOCSTOKG_APP_DATA_ROOT
runner.workers       →  DOCSTOKG_RUNNER_WORKERS
chunk.min_tokens     →  DOCSTOKG_CHUNK_MIN_TOKENS
embed.dense.backend  →  DOCSTOKG_EMBED_DENSE_BACKEND
```

### Common ENV Variables

#### Global (app.*)

- `DOCSTOKG_APP_DATA_ROOT` — Base data directory
- `DOCSTOKG_APP_LOG_LEVEL` — `DEBUG|INFO|WARNING|ERROR`
- `DOCSTOKG_APP_PROFILE` — Profile name to load
- `DOCSTOKG_APP_METRICS_ENABLED` — `true|false`

#### Runner (runner.*)

- `DOCSTOKG_RUNNER_POLICY` — `io|cpu|gpu`
- `DOCSTOKG_RUNNER_WORKERS` — Number of parallel workers
- `DOCSTOKG_RUNNER_SCHEDULE` — `fifo|sjf`
- `DOCSTOKG_RUNNER_RETRIES` — Retry attempts

#### Chunk (chunk.*)

- `DOCSTOKG_CHUNK_MIN_TOKENS` — Minimum chunk tokens
- `DOCSTOKG_CHUNK_MAX_TOKENS` — Maximum chunk tokens
- `DOCSTOKG_CHUNK_FORMAT` — `parquet|jsonl`

#### Embed (embed.*)

- `DOCSTOKG_EMBED_VECTOR_FORMAT` — `parquet|jsonl`
- `DOCSTOKG_EMBED_DENSE_BACKEND` — `qwen_vllm|tei|sentence_transformers`
- `DOCSTOKG_EMBED_DENSE_QWEN_VLLM_BATCH_SIZE` — Qwen batch size
- `DOCSTOKG_EMBED_DENSE_QWEN_VLLM_DTYPE` — `auto|float16|bfloat16`
- `DOCSTOKG_EMBED_DENSE_TEI_URL` — TEI HTTP endpoint
- `DOCSTOKG_EMBED_DENSE_SENTENCE_TRANSFORMERS_MODEL_ID` — Hugging Face model ID
- `DOCSTOKG_EMBED_SPARSE_SPLADE_ST_BATCH_SIZE` — SPLADE batch size
- `DOCSTOKG_EMBED_LEXICAL_LOCAL_BM25_K1` — BM25 k1 parameter
- `DOCSTOKG_EMBED_LEXICAL_LOCAL_BM25_B` — BM25 b parameter

---

## CLI Options

Each stage command exposes full configuration as CLI options. Use `--help` to see them organized in **help panels** (Global, Runner, I/O, Dense, Sparse, Lexical, etc.).

### Examples

```bash
# Global options
docparse --profile local --log-level DEBUG chunk ...

# Per-stage options
docparse chunk \
  --in-dir Data/DocTags \
  --out-dir Data/Chunks \
  --min-tokens 256 \
  --max-tokens 1024

docparse embed \
  --dense-backend qwen_vllm \
  --qwen-batch-size 64 \
  --qwen-dtype bfloat16 \
  --qwen-device cuda:0 \
  --splade-batch-size 32 \
  --bm25-k1 1.5 \
  --bm25-b 0.75
```

### Runner Options (shared across all stages)

```bash
docparse [GLOBAL] <command> \
  --policy gpu                    # io|cpu|gpu
  --workers 8                     # Max parallel
  --schedule sjf                  # fifo|sjf
  --retries 2                     # Retry count
  --retry-backoff-s 1.0           # Exponential backoff base
  --timeout-s 300                 # Per-item timeout (0 = disabled)
  --error-budget 10               # Stop after N failures
  --adaptive conservative         # off|conservative|aggressive
  --fingerprinting / --no-fingerprinting  # Exact resume
```

---

## Built-in Configuration Inspection

### `config show` — View Effective Configuration

```bash
docparse config show
# Output: all effective settings (after profile+ENV+CLI layering)

docparse config show --profile gpu
# Show config if profile=gpu were applied

docparse config show --stage embed --format yaml
# Show only embed config in YAML format

docparse config show --annotate-source
# Include per-key origin (default|profile|env|cli)
```

### `config diff` — Compare Two Configurations

```bash
docparse config diff --lhs-profile local --rhs-profile gpu
# Side-by-side diff of two profiles

docparse config diff --lhs-profile none --rhs-profile gpu --show-hash
# Show config hashes (helpful for debugging changes)
```

---

## Data Contracts & Schemas

All configuration is typed via Pydantic v2 models:

- **`Settings`**: Root aggregation (app, runner, doctags, chunk, embed)
- **`AppCfg`**: Global application config
- **`RunnerCfg`**: Shared execution runner (policy, workers, retries, etc.)
- **`DocTagsCfg`**: PDF/HTML conversion (input, output, model, mode)
- **`ChunkCfg`**: Chunking stage (min/max tokens, tokenizer, format)
- **`EmbedCfg`**: Embedding stage (families, dense/sparse/lexical providers)

Models are in `src/DocsToKG/DocParsing/settings.py`.

---

## Validation

### Shallow (Typer) Validation

- Path existence (if `exists=True` on option)
- Integer/float ranges (`ge=1`, `le=65535`, etc.)
- Enum choices (typed options show available values)

### Deep (Pydantic) Validation

- **TEI URL required** when `embed.dense.backend=tei`
- **BM25 k1 > 0** and **0 ≤ b ≤ 1**
- **min_tokens ≤ max_tokens** in chunk config
- **GPU memory utilization** in range [0.1, 0.99]
- **Resume & force mutually exclusive** (force takes precedence with warning)

### Error Handling

Validation errors are caught and presented with helpful messages:

```bash
$ docparse embed --bm25-b -0.1 2>&1
Error: BM25 b must be in [0, 1], got -0.1
```

---

## Legacy Configuration (Backward Compatibility)

If you have existing code using stage-specific config helpers (e.g., `ChunkerCfg.from_env()`), those remain available but are **deprecated**. They silently delegate to the new unified `Settings` builder.

### Migration Path

**Old:**

```python
from DocsToKG.DocParsing.chunking.config import ChunkerCfg
cfg = ChunkerCfg.from_env()
```

**New (recommended):**

```python
from DocsToKG.DocParsing.app_context import build_app_context
ctx = build_app_context()
chunk_cfg = ctx.settings.chunk
```

---

## Manifest Configuration Rows

When a stage runs, it writes a `doc_id="__config__"` row to its manifest containing:

```json
{
  "doc_id": "__config__",
  "profile": "gpu",
  "cfg_hash": {
    "app": "b2738391",
    "runner": "a246eb17",
    "embed": "6d6643da"
  },
  "vector_format": "parquet",
  "status": "success",
  "attempts": 1,
  "timestamp": "2025-10-21T12:34:56Z"
}
```

This enables:

- **Reproducibility**: exact config that produced outputs
- **Change detection**: cfg_hash changes alert to recalculation needed
- **Traceability**: which profile was used for a run

---

## Configuration Testing & Validation

### Validate Configuration Without Running

```bash
docparse chunk --validate-only --in-dir Data/DocTags
# Loads config, validates, and exits (no processing)

docparse embed --validate-only --chunks-dir Data/Chunks
# Similar for embedding stage
```

### Inspect Data Without Config

```bash
docparse inspect dataset --dataset chunks --root Data
# Quick schema/stats for chunks (no full read)

docparse inspect dataset --dataset vectors-dense --limit 100
# Peek at first 100 dense vectors
```

---

## Operational Recipes

### Recipe 1: Switch from CPU to GPU

```bash
# Old: manually update many env vars
export DOCSTOKG_EMBED_WORKERS=16
export DOCSTOKG_EMBED_BATCH_SIZE_QWEN=128
export DOCSTOKG_EMBED_BATCH_SIZE_SPLADE=64
# ...

# New: one profile flag
docparse --profile gpu embed --resume
```

### Recipe 2: Debug a Stage's Configuration

```bash
docparse --profile gpu config show --stage embed --annotate-source
# Output shows every key's origin (profile, env, cli)
# Helps spot unexpected overrides
```

### Recipe 3: Dry-run with New Settings Before Commit

```bash
# Compare profiles before rolling out
docparse config diff \
  --lhs-profile gpu \
  --rhs-profile gpu \
  --rhs-override "embed.dense.qwen_vllm.batch_size=96" \
  --show-hash
# See what changes + cfg_hash delta
```

### Recipe 4: Airgapped Deployment

```bash
# Ensure no network access during inference
docparse --profile airgapped embed ...
# Profile has offline=true, minimal workers, local models only
```

---

## Troubleshooting

### "Invalid enum value"

Check that you're using the right enum value (e.g., `qwen_vllm` not `qwen`).

### "TEI URL required when backend=tei"

Set `DOCSTOKG_EMBED_DENSE_TEI_URL=http://...` or `--tei-url http://...`.

### "min_tokens (1000) must be <= max_tokens (512)"

Swap them or use `--max-tokens 1200`.

### Config not applying

Check precedence:

```bash
docparse config show --stage chunk --annotate-source
# See which layer won for each key
```

### "Configuration validation failed"

Full Pydantic error is printed. Check for:

- Typos in enum values
- Out-of-range numbers
- Missing required fields (if backend=tei, URL is required)

---

## Best Practices

1. **Use profiles** for stable configurations (e.g., `docparse --profile gpu all`)
2. **Override via ENV** for CI/CD secret handling (e.g., TEI URLs)
3. **CLI flags** for one-time tweaks or debugging
4. **Validate often** with `--validate-only` before long runs
5. **Track cfg_hashes** in manifests for reproducibility

---

## For Developers

### Adding a New Configuration Option

1. Add field to relevant `*Cfg` class in `settings.py`
2. Add `Field()` with description and validation constraints
3. Update `docstokg.toml` example profiles
4. Update CLI Typer signature in `cli.py` (when wiring is done)
5. Add to relevant test in test suite

### Custom Validators

Pydantic v2 validators use `@field_validator` or `@model_validator`:

```python
@field_validator("k1")
@classmethod
def validate_k1(cls, v: float) -> float:
    if v <= 0:
        raise ValueError(f"k1 must be > 0, got {v}")
    return v
```

### Debugging Configuration Layering

```python
from DocsToKG.DocParsing.app_context import build_app_context

ctx = build_app_context(
    profile="gpu",
    track_sources=True,  # Enable source tracking
)

# See which layer each key came from
print(ctx.source_tracking)
```

---

## References

- **Settings module**: `src/DocsToKG/DocParsing/settings.py`
- **Profile loader**: `src/DocsToKG/DocParsing/profile_loader.py`
- **App context builder**: `src/DocsToKG/DocParsing/app_context.py`
- **Profile examples**: `docstokg.toml` (in repo root)
- **CLI integration** (pending): `src/DocsToKG/DocParsing/core/cli.py`

---

## Migration Timeline

| Phase | Status | Details |
|-------|--------|---------|
| Phase 1: Core settings & profiles | ✅ Complete | Pydantic models, builders, example profiles |
| Phase 2: CLI integration | 🔄 In progress | Wire Typer callback with full signatures |
| Phase 3: Legacy shims | ⏳ Pending | Deprecate old config loaders (1 minor) |
| Phase 4: Manifest integration | ⏳ Pending | Add `__config__` rows with hash tracking |
| Phase 5: Config show/diff CLI | ⏳ Pending | Typer subcommands for introspection |
| Phase 6: Tests & docs | ⏳ Pending | Precedence matrix, validation, snapshots |
