# ContentDownload Pydantic v2 Refactor - Current Status

**Date**: October 21, 2025  
**Status**: 🟡 **PARTIALLY COMPLETE - Integration Phase Remaining**

---

## What's Already Done ✅

### 1. Pydantic v2 Configuration Models
- ✅ **Location**: `src/DocsToKG/ContentDownload/config/models.py`
- ✅ **Models Implemented** (15+):
  - `RetryPolicy` - HTTP retry behavior
  - `BackoffPolicy` - Backoff strategy
  - `RateLimitPolicy` - Rate limiting config
  - `RobotsPolicy` - Robots.txt handling
  - `DownloadPolicy` - Download safety
  - `HttpClientConfig` - HTTP client settings
  - `TelemetryConfig` - Telemetry setup
  - `StorageConfig` - Storage backend
  - `CatalogConfig` - Artifact catalog
  - `ResolversConfig` - Resolver configuration
  - `ContentDownloadConfig` - Top-level configuration
  - Plus 4+ more supporting models

- ✅ **Quality**:
  - 100% Pydantic v2 compliant
  - `extra="forbid"` strict validation
  - `field_validator` decorators
  - Full type hints
  - Comprehensive docstrings

### 2. Config Loader
- ✅ **Location**: `src/DocsToKG/ContentDownload/config/loader.py`
- ✅ **Features**:
  - YAML/JSON file loading
  - Environment variable precedence
  - CLI override support
  - File < Env < CLI precedence
  - `config_hash()` for reproducibility
  - Config validation
  - `load_config()` public API

### 3. Integration Points
- ✅ **runner.py**: Uses new `ContentDownloadConfig`
- ✅ **download_pipeline.py**: Uses new `ContentDownloadConfig`
- ✅ **resolvers/registry_v2.py**: Uses new `ContentDownloadConfig`

### 4. Public API
- ✅ **config/__init__.py**: Exports `ContentDownloadConfig` and `load_config`
- ✅ **Backward compatibility**: Old `DownloadConfig` still exists in `download.py`

---

## What Still Needs Completion ⏳

### 1. Migrate All Import Sites
**Current Status**: Mixed - some use new, some use old

**Files Still Using Old `DownloadConfig` from `download.py`**:
- `streaming_schema.py` - Imports old `DownloadConfig`

**Action Needed**: Audit all 200+ files for remaining old imports and migrate

### 2. Deprecate & Remove Old DownloadConfig
- Old `DownloadConfig` dataclass (lines 161-231 in `download.py`)
- Uses manual validation in `__post_init__`
- No strict validation
- Only 15 fields vs 30+ in new model

**Action Needed**: 
- Replace all usages
- Delete old class
- Update all tests

### 3. Resolver Registry Integration
**Current Status**: Partially done
- `registry_v2.py` already uses new `ContentDownloadConfig`
- But resolver registry pattern may need expansion

**Action Needed**: 
- Verify resolver registry is working with all resolvers
- Add @register pattern if not present
- Ensure all resolvers respect config

### 4. CLI Modernization
**Current Status**: Not started
- Current CLI uses argparse
- Could be modernized with Typer
- But argparse CLI is functional

**Action Needed** (Optional):
- Would add `print-config`, `validate-config` commands
- Typer-based cleaner API
- But not blocking - current CLI works

### 5. Unified API Types
**Current Status**: Partial
- `core.py` has `WorkArtifact`, `DownloadContext`
- `api/types.py` exists but may need updates

**Action Needed**:
- Verify `DownloadPlan`, `DownloadOutcome`, `DownloadMetrics` types
- Ensure they use Pydantic v2
- Standardize across codebase

### 6. Test Coverage
**Current Status**: Existing tests work
- Tests pass with old `DownloadConfig`
- New `ContentDownloadConfig` works

**Action Needed**:
- Update test imports
- Test new config validation
- Integration tests for config loader

---

## Refactor Roadmap

### Phase 1: Audit & Planning (Done ✅)
- [x] Identified existing Pydantic v2 models
- [x] Found config loader
- [x] Located old DownloadConfig usage
- [x] Verified current state

### Phase 2: Migration (IN PROGRESS)
- [ ] Replace all old `DownloadConfig` imports with new `ContentDownloadConfig`
- [ ] Update all call sites
- [ ] Run tests to verify compatibility
- [ ] **Estimated**: 2-3 hours (20-30 files)

### Phase 3: Cleanup (TO DO)
- [ ] Delete old `DownloadConfig` from `download.py`
- [ ] Remove any compatibility code
- [ ] Update documentation
- [ ] **Estimated**: 1 hour

### Phase 4: Enhancement (Optional)
- [ ] CLI modernization (Typer)
- [ ] Resolver registry expansion
- [ ] Additional unified API types
- [ ] **Estimated**: 2-3 hours (optional)

### Phase 5: Testing & Integration (TO DO)
- [ ] Comprehensive test suite
- [ ] Integration tests
- [ ] Migration guide for users
- [ ] **Estimated**: 2-3 hours

---

## Key Findings

### ✅ Good News
1. **Pydantic v2 models are already implemented** - Major work already done
2. **Config loader is working** - Can load from files, env vars, CLI
3. **Type safety is strong** - 100% type hints, strict validation
4. **Backward compatible** - Old DownloadConfig still works

### ⚠️ Work Remaining
1. **Import migration** - Not all call sites use new config
2. **Old class cleanup** - Old `DownloadConfig` dataclass still exists
3. **Test updates** - Need to verify all tests work with new config
4. **Documentation** - Need to document breaking changes

---

## Recommended Action

**PROCEED WITH PHASE 2: Migration**

The infrastructure is already built. Now we need to:
1. Find all remaining old `DownloadConfig` imports (~20-30 files)
2. Replace with `ContentDownloadConfig`
3. Test thoroughly
4. Delete old class

**Estimated Effort**: 3-4 hours to complete full migration

---

## Files to Update

Based on grep results, these are the files that need attention:

1. `streaming_schema.py` - Imports old DownloadConfig
2. Any other files importing from `download.py` DownloadConfig
3. Test files using old config
4. Documentation

