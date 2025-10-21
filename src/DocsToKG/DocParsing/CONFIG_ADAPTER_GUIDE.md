# ConfigurationAdapter Pattern Guide

**Last Updated:** October 21, 2025
**Status:** ✅ Production Ready

---

## Overview

The **ConfigurationAdapter Pattern** bridges the modern Pydantic-based unified CLI with the existing stage runtime entry points, enabling:

- Direct configuration injection (no sys.argv re-parsing)
- Clean separation between CLI configuration and stage runtime logic
- Full backward compatibility with legacy code paths
- Testable, deterministic behavior

---

## Architecture

### Data Flow

```
┌─────────────────────────────────────────────────────────┐
│  Unified CLI (Typer)                                    │
│  • Parses command-line arguments                        │
│  • Builds AppContext (Pydantic-based)                   │
│  • Merges: CLI > ENV > profile > defaults               │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│  ConfigurationAdapter                                   │
│  • to_doctags(app_ctx, mode)  → DoctagsCfg             │
│  • to_chunk(app_ctx)          → ChunkerCfg             │
│  • to_embed(app_ctx)          → EmbedCfg               │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│  Stage Runtime (accepts config_adapter parameter)       │
│  • pdf_main(config_adapter=cfg)                         │
│  • html_main(config_adapter=cfg)                        │
│  • chunking._main_inner(config_adapter=cfg)             │
│  • embedding._main_inner(config_adapter=cfg)            │
│                                                         │
│  Falls back to sys.argv parsing if config_adapter=None │
└─────────────────────────────────────────────────────────┘
```

---

## Usage Guide

### From the Unified CLI

The adapter is used **automatically** by `cli_unified.py`:

```python
from DocsToKG.DocParsing.config_adapter import ConfigurationAdapter
from DocsToKG.DocParsing import doctags as doctags_module

# Build effective configuration from CLI args, ENV, and profiles
app_ctx: AppContext = ctx.obj

# Apply CLI overrides
if input_dir:
    app_ctx.settings.doctags.input_dir = input_dir

# Create adapted config for the stage
cfg = ConfigurationAdapter.to_doctags(app_ctx, mode="pdf")

# Call stage with direct config injection
exit_code = doctags_module.pdf_main(config_adapter=cfg)
```

### In Tests

Create a mock `AppContext` and use the adapter directly:

```python
from unittest.mock import MagicMock
from DocsToKG.DocParsing.config_adapter import ConfigurationAdapter
from DocsToKG.DocParsing.doctags import DoctagsCfg
from pathlib import Path

# Create mock context
ctx = MagicMock()
ctx.settings.app.data_root = Path("Data")
ctx.settings.app.log_level = "INFO"
ctx.settings.doctags.input_dir = Path("Data/PDFs")
ctx.settings.doctags.output_dir = Path("Data/DocTags")
ctx.settings.doctags.model_id = "granite-docling-258M"
ctx.settings.runner.workers = 4

# Create adapted config
cfg = ConfigurationAdapter.to_doctags(ctx, mode="pdf")

# Verify it's correct type
assert isinstance(cfg, DoctagsCfg)
assert cfg.mode == "pdf"
assert cfg.workers == 4
```

### In Programmatic Usage (Non-CLI)

If calling stage runtimes directly without the CLI, you have two options:

**Option 1: Use the adapter (new pattern)**

```python
# Create AppContext manually or load from config
app_ctx = build_app_context(profile="local")

# Use adapter
cfg = ConfigurationAdapter.to_chunk(app_ctx)
exit_code = chunking_runtime._main_inner(config_adapter=cfg)
```

**Option 2: Use legacy args pattern (backward compat)**

```python
# Create argparse.Namespace manually
namespace = argparse.Namespace(
    in_dir=Path("Data/DocTags"),
    out_dir=Path("Data/Chunks"),
    min_tokens=120,
    max_tokens=800,
)

# Call with args (backward compatible path)
exit_code = chunking_runtime._main_inner(args=namespace)
```

---

## Adapter Methods

### `ConfigurationAdapter.to_doctags(app_ctx, mode="pdf")`

Converts `AppContext` settings to `DoctagsCfg`.

**Parameters:**

- `app_ctx` (AppContext): Application context with merged settings
- `mode` (str): Override mode ("pdf" or "html"). Overrides `app_ctx.settings.doctags.mode`

**Returns:**

- `DoctagsCfg`: Configured doctags config instance (finalized)

**What it does:**

1. Creates a new `DoctagsCfg()` instance
2. Applies app-level settings (data_root, log_level)
3. Applies doctags-specific settings (input, output, model)
4. Applies runner settings (workers)
5. Sets the mode explicitly
6. Calls `finalize()` for normalization and validation

### `ConfigurationAdapter.to_chunk(app_ctx)`

Converts `AppContext` settings to `ChunkerCfg`.

**Parameters:**

- `app_ctx` (AppContext): Application context with merged settings

**Returns:**

- `ChunkerCfg`: Configured chunk config instance (finalized)

**What it does:**

1. Creates a new `ChunkerCfg()` instance
2. Applies app-level settings (data_root, log_level)
3. Applies chunk-specific settings (in_dir, out_dir, min/max tokens, tokenizer)
4. Applies runner settings (workers)
5. Calls `finalize()` for normalization and validation

### `ConfigurationAdapter.to_embed(app_ctx)`

Converts `AppContext` settings to `EmbedCfg`.

**Parameters:**

- `app_ctx` (AppContext): Application context with merged settings

**Returns:**

- `EmbedCfg`: Configured embedding config instance (finalized)

**What it does:**

1. Creates a new `EmbedCfg()` instance
2. Applies app-level settings (data_root, log_level)
3. Applies embed-specific settings (chunks_dir, out_dir)
4. Applies runner settings (workers as files_parallel)
5. Calls `finalize()` for normalization and validation

---

## Stage Runtime Signatures

All stage runtimes now accept an optional `config_adapter` parameter:

```python
def pdf_main(args: argparse.Namespace | None = None, config_adapter=None) -> int:
    """Convert PDFs to DocTags.

    Args:
        args: Legacy parameter for backward compat (sys.argv parsing)
        config_adapter: New parameter from ConfigurationAdapter
    """
    if config_adapter is not None:
        # NEW PATH: Use provided config directly
        cfg = config_adapter
    else:
        # LEGACY PATH: Parse sys.argv
        namespace = pdf_parse_args()
        cfg = DoctagsCfg()
        cfg.apply_args(namespace)
        cfg.finalize()

    # Rest of the logic uses cfg
```

**Key Points:**

- Both parameters have default values (None)
- Dual-path support for backward compatibility
- NEW path takes precedence (config_adapter != None)
- LEGACY path falls back to sys.argv parsing

---

## Backward Compatibility

### Legacy Calling Patterns

**Pattern 1: Direct sys.argv parsing**

```python
# Old code still works - stage parses sys.argv itself
exit_code = pdf_main()
```

**Pattern 2: Pass parsed arguments**

```python
# Old code still works - stage uses provided namespace
namespace = pdf_parse_args()
exit_code = pdf_main(args=namespace)
```

**Pattern 3: New adapter pattern (recommended)**

```python
# New code uses direct config injection
app_ctx = build_app_context()
cfg = ConfigurationAdapter.to_doctags(app_ctx, mode="pdf")
exit_code = pdf_main(config_adapter=cfg)
```

### Compatibility Matrix

| Scenario | Supported | Notes |
|----------|-----------|-------|
| CLI usage | ✅ YES | Uses adapter automatically |
| Legacy code (direct call) | ✅ YES | Falls back to sys.argv parsing |
| Legacy code (with args) | ✅ YES | Uses provided namespace |
| New adapter usage | ✅ YES | Direct injection |
| Tests with mocks | ✅ YES | Can mock config directly |
| Tests with sys.argv | ✅ YES | Legacy path still works |

---

## Testing

### Unit Tests

Use the adapter in unit tests for direct configuration injection:

```python
def test_chunking_with_adapter():
    # Create mock context
    ctx = MagicMock()
    ctx.settings.app.data_root = Path("Data")
    ctx.settings.chunk.input_dir = Path("Data/DocTags")
    ctx.settings.chunk.output_dir = Path("Data/Chunks")
    ctx.settings.chunk.min_tokens = 120
    ctx.settings.chunk.max_tokens = 800
    ctx.settings.runner.workers = 4

    # Create adapted config
    cfg = ConfigurationAdapter.to_chunk(ctx)

    # Verify it's correct
    assert isinstance(cfg, ChunkerCfg)
    assert cfg.min_tokens == 120
    assert cfg.workers == 4
```

### Smoke Tests

The repository includes smoke tests in `tests/docparsing/test_config_adapter_smoke.py` that verify:

- Adapter creates correct config instances
- Adapter applies settings correctly
- Stage entry points accept config_adapter parameter
- Backward compatibility is maintained

Run tests:

```bash
pytest tests/docparsing/test_config_adapter_smoke.py -v
```

---

## Design Decisions

### Why This Pattern?

1. **Eliminates sys.argv Re-Parsing**: Direct config injection is faster and cleaner
2. **Testable**: Mocking a config object is easier than manipulating sys.argv
3. **Backward Compatible**: Existing code continues to work unchanged
4. **Single Source of Truth**: AppContext is the authoritative config source
5. **Type-Safe**: All config is validated Pydantic models

### Why Not X?

| Alternative | Why Not |
|-------------|---------|
| Direct rewrite | Too risky, breaks compatibility |
| from_args() classmethods | Fragile, couples runtimes to argparse |
| Shared global config | Reduces testability, thread issues |
| Adapter per stage | Too much duplication (what we have is minimal) |

---

## Troubleshooting

### Config Validation Errors

**Problem:** `ValidationError` when creating config

**Solution:** Ensure AppContext has all required fields set

```python
# Check that all required settings are present
ctx.settings.app.data_root  # Must be set
ctx.settings.app.log_level  # Must be set
ctx.settings.doctags.input_dir  # Required for to_doctags()
# etc.
```

### Mode Not Applied

**Problem:** `cfg.mode` is not set to the provided mode

**Solution:** The adapter explicitly sets mode after other settings. This is correct behavior.

```python
# This will set mode to "pdf" regardless of app_ctx settings
cfg = ConfigurationAdapter.to_doctags(app_ctx, mode="pdf")
assert cfg.mode == "pdf"  # Always set to provided mode
```

### Backward Compat Not Working

**Problem:** Existing code that calls `pdf_main()` directly fails

**Solution:** Check that you didn't pass `config_adapter=something` without also setting up sys.argv

```python
# This works - legacy path
exit_code = pdf_main()  # Parses sys.argv

# This also works - adapter path
cfg = ConfigurationAdapter.to_doctags(app_ctx, mode="pdf")
exit_code = pdf_main(config_adapter=cfg)  # Uses adapter
```

---

## See Also

- `PHASE4_CONFIGURATION_ADAPTER_IMPLEMENTATION.md` — Implementation details
- `PHASE4_LONG_TERM_DESIGN_SOLUTION.md` — Architecture and design rationale
- `config_adapter.py` — Source code for the adapter
- `test_config_adapter_smoke.py` — Example tests using the adapter
