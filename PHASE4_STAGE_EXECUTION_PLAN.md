# Phase 4: Stage Execution Wiring — Implementation Plan

**Date:** October 21, 2025  
**Status:** Planning  
**Scope:** Wire CLI commands to stage implementations

---

## Architecture Overview

```
cli_unified.py commands
    ↓
Extract settings from AppContext (ctx.obj)
    ↓
Build argv list for stage
    ↓
Call stage main() function with argv
    ↓
Return exit code
```

---

## Stage Entry Points

### 1. DocTags Stage
**File:** `src/DocsToKG/DocParsing/doctags.py`

**Entry Points:**
- `pdf_main(args: argparse.Namespace | None = None) -> int` (line 1996)
- `html_main(args: argparse.Namespace | None = None) -> int` (line 2726)
- `pdf_parse_args()` / `html_parse_args()` — Builds parsers

**Behavior:**
- If `args=None`, calls `pdf_parse_args()` which parses `sys.argv[1:]`
- If `args` is provided, uses that Namespace directly
- Returns 0 on success, non-zero on failure

**CLI Flags to Map:**
- `--input-dir` → `-i/--input`
- `--output-dir` → `-o/--output`
- `--mode` → `--mode`
- `--model-id` → `--model-id`
- `--vllm-wait-timeout-s` → `--vllm-wait-timeout`
- `--resume` → `--resume`
- `--force` → `--force`
- `--workers` → `--workers` (runner override)
- `--policy` → `--policy` (runner override)

### 2. Chunk Stage
**File:** `src/DocsToKG/DocParsing/chunking/runtime.py`

**Entry Point:**
- `main(args: argparse.Namespace | SimpleNamespace | Sequence[str] | None = None) -> int` (line 1845)

**Behavior:**
- Wraps `_main_inner(args)`
- Handles `ChunkingCLIValidationError`
- Returns 0 on success, 2 on validation error, non-zero on other failures

**CLI Flags to Map:**
- `--in-dir` → `--in-dir`
- `--out-dir` → `--out-dir`
- `--format` → `--format`
- `--min-tokens` → `--min-tokens`
- `--max-tokens` → `--max-tokens`
- `--tokenizer` → `--tokenizer-model`
- `--resume` → `--resume`
- `--force` → `--force`
- `--workers` → `--workers` (runner override)
- `--policy` → `--policy` (runner override)

### 3. Embed Stage
**File:** `src/DocsToKG/DocParsing/embedding/runtime.py`

**Entry Point:**
- `main(args: argparse.Namespace | None = None) -> int` (line 2883)

**Behavior:**
- Wraps `_main_inner(args)`
- Handles `EmbeddingCLIValidationError`
- Returns 0 on success, 2 on validation error, non-zero on other failures

**CLI Flags to Map:**
- `--chunks-dir` → `--chunks-dir`
- `--out-dir` → `--out-dir`
- `--vector-format` → `--format`
- `--dense-backend` → `--dense-backend`
- Various dense/sparse/lexical options
- `--resume` → `--resume`
- `--force` → `--force`
- `--workers` → `--workers` (runner override)
- `--policy` → `--policy` (runner override)

---

## Implementation Strategy

### Step 1: Helper Function to Build Argv

Create utility function to convert Settings + CLI options → argv list:

```python
def build_stage_argv(
    settings: Settings,
    stage: Literal["doctags", "chunk", "embed"],
    cli_overrides: Dict[str, Any],
) -> List[str]:
    """Convert Settings + CLI args to argv for stage main()."""
    # Extract stage-specific config
    # Map to CLI flag names
    # Return argv list
```

### Step 2: Update CLI Commands

Replace placeholder messages with actual execution:

```python
@app.command()
def doctags(ctx: typer.Context, ...):
    app_ctx = ctx.obj
    argv = build_stage_argv(app_ctx.settings, "doctags", cli_overrides)
    exit_code = pdf_main(argv=argv)  # or html_main
    raise typer.Exit(code=exit_code)
```

### Step 3: Handle Configuration Mapping

**DocTags Settings → Argv:**
```python
if app_ctx.settings.doctags.input_dir:
    argv.append("--input")
    argv.append(str(app_ctx.settings.doctags.input_dir))
```

**Runner Settings → Argv:**
```python
if app_ctx.settings.runner.workers:
    argv.append("--workers")
    argv.append(str(app_ctx.settings.runner.workers))
```

### Step 4: Handle Resume/Force Flags

```python
if app_ctx.settings.doctags.resume:
    argv.append("--resume")
if app_ctx.settings.doctags.force:
    argv.append("--force")
```

### Step 5: Wire All Command

```python
@app.command()
def all(ctx: typer.Context, ...):
    app_ctx = ctx.obj
    
    # Run doctags
    exit_code = doctags_impl(app_ctx, ...)
    if exit_code != 0 and stop_on_fail:
        raise typer.Exit(code=exit_code)
    
    # Run chunk
    exit_code = chunk_impl(app_ctx, ...)
    if exit_code != 0 and stop_on_fail:
        raise typer.Exit(code=exit_code)
    
    # Run embed
    exit_code = embed_impl(app_ctx, ...)
    raise typer.Exit(code=exit_code)
```

---

## Implementation Checklist

- [ ] Step 1: Create `build_stage_argv()` helper
- [ ] Step 2: Wire `doctags` command
- [ ] Step 3: Wire `chunk` command
- [ ] Step 4: Wire `embed` command
- [ ] Step 5: Wire `all` command
- [ ] Step 6: Test commands execute correctly
- [ ] Step 7: Verify cfg_hashes are computed
- [ ] Step 8: Verify profiles are respected
- [ ] Step 9: Test resume/force flags
- [ ] Step 10: Update documentation

---

## Key Files to Modify

1. **`src/DocsToKG/DocParsing/cli_unified.py`**
   - Add `build_stage_argv()` helper function
   - Update `root_callback()` to build AppContext
   - Implement `doctags()`, `chunk()`, `embed()`, `all()` commands

2. **No other files need modification** (all stage implementations exist)

---

## Testing Strategy

```bash
# Test doctags
python -m DocsToKG.DocParsing.cli_unified --profile local doctags --help

# Test chunk
python -m DocsToKG.DocParsing.cli_unified --profile local chunk --help

# Test embed
python -m DocsToKG.DocParsing.cli_unified --profile local embed --help

# Test execution (small dataset)
python -m DocsToKG.DocParsing.cli_unified doctags --input Data/Samples --output Data/DocTags

# Test pipeline
python -m DocsToKG.DocParsing.cli_unified all --resume
```

---

## Success Criteria

✅ Commands execute without placeholder warnings  
✅ Manifests are created with correct cfg_hash  
✅ Profile system is respected  
✅ Resume/force flags work  
✅ All three stages can run together  
✅ Exit codes are correct  

---

**Estimated Effort:** 2-3 hours

