# PR-7: Configuration & Profiles Implementation Status

**Date:** October 21, 2025
**Status:** PHASE 1 & 2 COMPLETE (Core Settings & Profiles ✅)

---

## Summary

PR-7 implements a unified, Pydantic v2-based configuration system for DocParsing with:

- **Type-safe Settings models** (AppCfg, RunnerCfg, DocTagsCfg, ChunkCfg, EmbedCfg)
- **Profile system** (docstokg.toml with local/gpu/airgapped/dev profiles)
- **Canonical precedence** (CLI > ENV > profile > defaults)
- **Comprehensive validation** (Pydantic validators + Typer callbacks)
- **Configuration introspection** (config show/diff, source tracking, cfg_hash)
- **Backward compatibility** (legacy config loaders to be shimmed in Phase 3)

---

## Completed (✅)

### Phase 1: Core Settings Models & Builders

- ✅ **settings.py** (935 LOC): Complete Pydantic v2 settings hierarchy
  - AppCfg: Global configuration
  - RunnerCfg: Shared runner knobs
  - DocTagsCfg, ChunkCfg: Stage-specific configs
  - EmbedCfg: Embedding stage with nested provider configs (QwenVLLMCfg, TEICfg, SpladeSTCfg, LocalBM25Cfg, etc.)
  - Field validators (BM25 ranges, token limits, port ranges, GPU memory, etc.)
  - Model validators (cross-field checks like TEI URL requirement, token ordering)
  - Enums: LogLevel, LogFormat, RunnerPolicy, RunnerSchedule, RunnerAdaptive, DoctagsMode, Format, DenseBackend, AttnBackend, SpladeNorm, TeiCompression
  - Settings.compute_stage_hashes() for deterministic config hashing
  - model_dump_redacted() for safe serialization

- ✅ **profile_loader.py** (320 LOC): Profile loading & precedence layering
  - load_profile_file(): TOML/YAML profile loader
  - merge_dicts(): Deep dictionary merge
  - apply_dot_path_override(): Dot-path configuration patching
  - SettingsBuilder class: Implements canonical layering algorithm
    - add_defaults()
    - add_profile()
    - add_env_overrides()
    - add_cli_overrides()
    - build(track_sources=True) → returns (config_dict, source_tracking)

- ✅ **app_context.py** (260 LOC): AppContext & builder
  - AppContext dataclass: Holds all configs + metadata (cfg_hashes, profile, source_tracking)
  - build_app_context(**kwargs) → AppContext: Main builder orchestrating full layering
    - CLI arg mapping for all stage options
    - Profile auto-discovery (docstokg.toml, config/docstokg.toml)
    - ENV override extraction
    - Pydantic validation with helpful error messages
    - Computes per-stage cfg_hashes

### Phase 2: Example Profiles & Documentation

- ✅ **docstokg.toml** (270 LOC): Example profile file with 4 built-in profiles
  - `[profile.local]`: CPU-only development (4 workers, sentence-transformers, FIFO)
  - `[profile.gpu]`: GPU-optimized (8 workers, Qwen2-7B, SJF, flash-attention)
  - `[profile.airgapped]`: Offline-safe (no network, minimal workers)
  - `[profile.dev]`: Debugging (1 worker, verbose logging, small batches)

- ✅ **CONFIGURATION.md** (580 LOC): Comprehensive operational documentation
  - Precedence diagram and examples
  - Profile system explanation
  - ENV variable mapping table (all DOCSTOKG_* patterns)
  - CLI options overview
  - Built-in config show/diff commands (spec)
  - Data contracts & schemas
  - Validation explanation (shallow vs deep)
  - Legacy migration path
  - Manifest `__config__` row specification
  - Operational recipes (CPU→GPU, debugging, dry-runs, airgapped)
  - Troubleshooting guide
  - Developer guide (adding new options, custom validators)

### Phase 3: Comprehensive Test Suite

- ✅ **test_settings_pr7.py** (400+ LOC): 27 tests covering:
  - TestSettingsModels: 5 tests for default instantiation
  - TestValidation: 6 tests for Pydantic validators (BM25 ranges, token limits, GPU mem, TEI URL)
  - TestPrecedence: 6 tests for layering precedence matrix (8 scenarios)
  - TestProfileLoading: 3 tests for TOML loading, dict merging, builder chaining
  - TestAppContext: 5 tests for context creation, hashing, source tracking, redaction
  - TestErrorHandling: 3 tests for error messages
  - **Status**: 20/27 passing (7 test fixes pending below)

---

## Pending (Phase 2-5)

### Phase 2: CLI Integration (Root Typer Callback)

- ⏳ **Wire Typer signatures** from skeleton (core/cli.py)
  - Root `@app.callback()` with global options (--profile, --data-root, --log-level, etc.)
  - Subcommand integration (doctags, chunk, embed, all, config, inspect)
  - Help panels (Global, Runner, I/O, Dense, Sparse, Lexical, etc.)
  - Legacy flag shims (deprecated warnings for old CLI args)

### Phase 3: Legacy Shims & Deprecation (1 minor)

- ⏳ Keep old stage config loaders (ChunkerCfg.from_env(), EmbedCfg.from_args()) but delegate to new builder
- ⏳ Deprecation warnings for legacy knobs (--bm25-k1 vs embed.lexical.local_bm25.k1)
- ⏳ One-line warnings pointing to new keys

### Phase 4: Manifest Integration

- ⏳ Add `doc_id="__config__"` rows to manifests containing:
  - profile: Profile name used
  - cfg_hash: Per-stage hashes (for change detection)
  - vector_format: Parquet/JSONL for embedding stage
  - timestamp: When config was captured
- ⏳ Update doctags/chunk/embed stages to emit config rows

### Phase 5: Config Show/Diff Subcommands

- ⏳ Implement `config show` Typer subcommand
  - --profile, --stage, --format {yaml,json,toml,env}, --annotate-source, --redact
- ⏳ Implement `config diff` Typer subcommand
  - --lhs-profile, --rhs-profile, --lhs/rhs-override, --show-hash
  - Output formats: unified, json, yaml, table

### Phase 6: Test Suite Fixes & Expansion

- ⏳ Fix 7 failing tests (related to nested field syntax in test fixtures)
- ⏳ Add snapshot tests for CLI help panels
- ⏳ Test config show/diff output format stability

---

## Code Quality Metrics

| Metric | Value |
|--------|-------|
| Core settings LOC | 935 |
| Profile loader LOC | 320 |
| AppContext builder LOC | 260 |
| Documentation LOC | 580 + 270 (configs) |
| Test coverage | 27 tests (20 passing) |
| Pydantic models | 15 (AppCfg, RunnerCfg, DocTagsCfg, ChunkCfg, EmbedCfg, + nested providers) |
| Enum definitions | 8 (LogLevel, LogFormat, RunnerPolicy, RunnerSchedule, RunnerAdaptive, DoctagsMode, Format, DenseBackend, etc.) |
| Field validators | 10+ (per-field and model-level) |
| Example profiles | 4 (local, gpu, airgapped, dev) |

---

## Key Design Decisions

1. **Pydantic v2 BaseSettings** (not custom parsing)
   - Type safety & automatic coercion
   - Built-in ENV var support (DOCSTOKG_*)
   - Composable nested models for providers
   - Performance (Rust core validation)

2. **Separate builder from settings**
   - SettingsBuilder handles layering logic independently
   - Can be unit tested without Typer/CLI
   - Reusable for programmatic config construction

3. **Canonical precedence**
   - CLI > ENV > profile > defaults (enforced by builder order)
   - Source tracking optional (for debugging)
   - Hashing for change detection (cfg_hash in manifests)

4. **Provider architecture (embedding)**
   - Nested configs (DenseCfg → QwenVLLMCfg, TEICfg, SentenceTransformersCfg, etc.)
   - Backend selection via enum (DenseBackend)
   - Optional fields for non-selected backends

5. **Backward compatibility**
   - Legacy stage loaders (ChunkerCfg.from_env()) remain callable
   - One-minor deprecation window
   - No breaking changes in this PR

---

## Integration Points

Once Phase 2-5 are complete, existing code will work without change:

```python
# Old way (still works, deprecated in Phase 3)
from DocsToKG.DocParsing.chunking.config import ChunkerCfg
cfg = ChunkerCfg.from_env()

# New way (recommended, Phase 1 ready)
from DocsToKG.DocParsing.app_context import build_app_context
ctx = build_app_context(profile="gpu")
chunk_cfg = ctx.settings.chunk
```

CLI usage (Phase 2):

```bash
# Profile + overrides
docparse --profile gpu chunk --min-tokens 256

# Show effective config
docparse config show --stage chunk --format yaml

# Compare two profiles
docparse config diff --lhs-profile local --rhs-profile gpu
```

---

## Files Created/Modified

### New Files

- `src/DocsToKG/DocParsing/settings.py` (935 LOC) ✅
- `src/DocsToKG/DocParsing/profile_loader.py` (320 LOC) ✅
- `src/DocsToKG/DocParsing/app_context.py` (260 LOC) ✅
- `src/DocsToKG/DocParsing/CONFIGURATION.md` (580 LOC) ✅
- `docstokg.toml` (270 LOC) ✅
- `tests/docparsing/test_settings_pr7.py` (400+ LOC) ✅

### Modified Files

- None yet (legacy shims in Phase 3)

---

## Next Steps (for team)

### Phase 2: CLI Integration (1-2 PRs)

1. Wire Typer root callback in `core/cli.py`
2. Connect all stage commands to new AppContext
3. Implement help panels with rich formatting
4. Test with existing CLI workflows

### Phase 3: Backward Compatibility (1 PR, 1 minor release)

1. Shim stage config loaders
2. Add deprecation warnings
3. Update tests
4. Document migration path in README

### Phase 4: Manifest Integration (1 PR)

1. Emit `__config__` rows in manifests
2. Integrate cfg_hash into resume logic
3. Test reproducibility

### Phase 5: Introspection (1 PR)

1. Implement `config show` subcommand
2. Implement `config diff` subcommand
3. Add output format tests

---

## Acceptance Criteria (PR-7 Complete)

- [ ] All 27 tests pass
- [ ] CLI wired and working with `--profile` flag
- [ ] Legacy `ChunkerCfg.from_env()` etc. still work
- [ ] Manifest `__config__` rows written
- [ ] `docparse config show/diff` working
- [ ] Documentation complete and accurate
- [ ] Zero breaking changes
- [ ] Deployment risk: LOW (fully backward compatible)

---

## References

- **Settings spec:** `Docparsing_config_profiles.md` (design doc)
- **Typer skeleton:** `Docparsing_config_profiles-typer-signature-skeleton.md`
- **Mapping table:** `Docparsing_config_profiles-mapping.md` (legacy → new)
- **Typer guide:** `src/DocsToKG/DocParsing/AGENTS.md` (cursor rules)
