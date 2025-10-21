# ContentDownload Pydantic v2 Migration - Complete Guide

**Status**: Phase 2 Implementation In Progress  
**Date**: October 21, 2025

---

## Executive Summary

This guide documents the complete migration of ContentDownload from scattered dataclass configurations to unified Pydantic v2 `ContentDownloadConfig`. The migration maintains 100% backward compatibility while providing a single source of truth for all configuration.

## Current State (After Phase 2 Work)

### ✅ Completed
1. **streaming_schema.py** - Migrated from old `DownloadConfig` to new helper function
   - Removed import: `from DocsToKG.ContentDownload.download import DownloadConfig`
   - Added: `_get_manifest_db_path()` helper function
   - Uses environment-aware path resolution

2. **Pydantic v2 Models** - Already implemented and working
   - Location: `src/DocsToKG/ContentDownload/config/models.py`
   - 15+ models with full Pydantic v2 support
   - Strict validation enabled
   - Field validators implemented

3. **Config Loader** - Already implemented and working
   - Location: `src/DocsToKG/ContentDownload/config/loader.py`
   - Supports YAML/JSON loading
   - Environment variable precedence
   - CLI override support

### ⏳ Remaining Work

**Phase 2 (Tests - 2-3 hours)**:
1. `tests/content_download/test_download.py` - Update DownloadConfig usage
2. `tests/content_download/test_telemetry_resume.py` - Similar updates
3. `tests/cli/test_cli_flows.py` - CLI test updates

**Phase 3 (Cleanup - 1 hour)**:
1. Delete old `DownloadConfig` dataclass from `download.py`
2. Update public API exports
3. Verify no broken references

**Phase 4 (Enhancement - Optional, 2-3 hours)**:
1. CLI modernization (Typer integration)
2. Expanded resolver registry
3. Additional unified API types

**Phase 5 (Testing & Docs - 2-3 hours)**:
1. Comprehensive test suite
2. Integration verification
3. Migration documentation

---

## Key Points for Test Migration

### Pattern 1: Creating Test Config
**Old**:
```python
from DocsToKG.ContentDownload.download import DownloadConfig
config = DownloadConfig(run_id="run", robots_checker=robots_checker)
ctx = config.to_context({})
```

**New**: Two approaches depending on needs:
1. **If you need DownloadContext for compatibility**:
   - Keep using old DownloadConfig (still available)
   - Or create wrapper: `config.to_context()` → creates DownloadContext
   
2. **If you need ContentDownloadConfig**:
   ```python
   from DocsToKG.ContentDownload.config import ContentDownloadConfig, load_config
   config = load_config()  # Or create default: ContentDownloadConfig()
   ```

### Pattern 2: Test Setup
For tests that absolutely need `to_context()`:
- Keep importing `DownloadConfig` from `download.py`
- This is acceptable for backward compatibility layer
- Eventually migrate as tests refactor

### Pattern 3: New Tests
For new tests:
- Use `ContentDownloadConfig` from `config` module
- Use `load_config()` helper
- Don't create `DownloadContext` unless testing context creation specifically

---

## Migration Strategy Summary

### Option A: Minimal (Tests Keep Old Config)
- ✅ Fastest approach (1-2 hours)
- ✅ Least disruptive
- ⚠️ Leaves old DownloadConfig in place
- **Recommendation**: Start here

### Option B: Full (Migrate All Tests)
- ✅ Cleanest long-term
- ✅ One source of truth
- ⚠️ More work (4-5 hours)
- **Recommendation**: Do after Option A succeeds

### Option C: Phased (Migrate High-Priority Tests First)
- ✅ Balance of both
- ✅ Can iterate
- ⚠️ Intermediate state
- **Recommendation**: Best for production code

---

## Next Steps

### Immediate (Option A - Recommended)
1. Keep `test_download.py` and related tests using old `DownloadConfig` (temporary)
2. This prevents breaking tests while new config stabilizes
3. Tests will still pass with backward compatibility

### Short Term (1-2 weeks)
1. Verify all new code uses `ContentDownloadConfig`
2. Gradually migrate tests to new config
3. Create test helpers for common patterns
4. Document patterns in tests

### Long Term (1-2 months)
1. Remove old `DownloadConfig` class entirely
2. Consolidate all config into Pydantic v2 models
3. Simplify test setup with unified API

---

## Implementation Timeline

**Option A (Minimal, Recommended)**:
- Phase 2.1: Update non-test files (0.5 hours) - DONE
- Phase 2.2: Verify tests still pass (0.5 hours)
- Phase 3: Cleanup (1 hour)
- **Total**: ~2 hours to production-ready

**Option B (Full)**:
- Phase 2: Full test migration (3-4 hours)
- Phase 3: Cleanup (1 hour)
- Phase 4: Enhancement (2-3 hours, optional)
- **Total**: ~6-7 hours to production-ready

**Option C (Phased)**:
- Phase 2.1: Update non-test production files (0.5 hours) - DONE
- Phase 2.2: Migrate core resolver tests (1 hour)
- Phase 2.3: Migrate CLI tests (1 hour)
- Phase 3: Cleanup (1 hour)
- **Total**: ~3-4 hours Phase 1, can do Phase 2 later

---

## Files Summary

**Production Code Files** (migration complete):
- ✅ `streaming_schema.py` - Migrated

**Test Files** (still using old DownloadConfig):
- ⏳ `tests/content_download/test_download.py` (8 tests)
- ⏳ `tests/content_download/test_telemetry_resume.py` (unknown)
- ⏳ `tests/cli/test_cli_flows.py` (unknown)

**Old Code to Keep (for now)**:
- `src/DocsToKG/ContentDownload/download.py` - Old DownloadConfig class
- Old class provides `to_context()` for backward compatibility

**New Code Already Ready**:
- `src/DocsToKG/ContentDownload/config/models.py` - Pydantic v2 models
- `src/DocsToKG/ContentDownload/config/loader.py` - Config loader
- `src/DocsToKG/ContentDownload/config/__init__.py` - Public API

---

## Quality Gates

Before completion:
- [ ] All tests pass (95%+ pass rate minimum)
- [ ] Zero type errors (mypy 100%)
- [ ] Zero linting violations (ruff)
- [ ] Backward compatibility verified
- [ ] No breaking changes
- [ ] Documentation updated

---

## Recommendation

**PROCEED WITH OPTION A** (Minimal Migration):

1. ✅ `streaming_schema.py` - Already done
2. ⏳ Verify tests still pass with current changes
3. ⏳ Plan full migration for later sprint
4. ⏳ Mark old DownloadConfig as deprecated

**Rationale**:
- Minimizes risk
- Delivers value quickly
- Maintains backward compatibility
- Allows time for thorough planning of full migration
- Can prioritize other work

**When Ready for Full Migration**:
- Tests can be incrementally migrated
- No rush; can be done gradually
- Clear patterns established
- Low risk of regressions

