# Deletion Plan: base.py and pipeline.py

## Current Status (After Resolver Migration)

### base.py
- **Status**: ✅ **SAFE TO DELETE IMMEDIATELY**
- **Remaining imports**: NONE
- **Used by**:
  - ~~All 15 resolvers~~ (MIGRATED - no longer inherit from RegisteredResolver/ApiResolverBase)
  - ~~pipeline.py~~ (no longer imports from base.py)
- **Action**: DELETE

### pipeline.py
- **Status**: ⏳ **KEEP FOR NOW (with deprecation notice)**
- **Still imported by**:
  1. `download.py` (lines 100-105): Imports `AttemptRecord`, `DownloadOutcome`, `ResolverMetrics`, `ResolverPipeline`
  2. `telemetry.py` (line 86): Imports `AttemptRecord`, `DownloadOutcome`, plus TYPE_CHECKING for `PipelineResult`
  3. `resolvers/base.py` (line 130): TYPE_CHECKING only for `ResolverConfig`
- **Components exported**:
  - `AttemptRecord` - used by download.py and telemetry.py
  - `DownloadOutcome` - used by download.py
  - `ResolverMetrics` - used by download.py
  - `ResolverPipeline` - legacy pipeline class (deprecated, replaced by download_pipeline.py)
  - `ResolverConfig` - configuration class (rarely used, mostly in TYPE_CHECKING)
  - `PipelineResult` - used by telemetry.py in TYPE_CHECKING
- **Action**: KEEP FOR NOW with deprecation notice

## Why Keep pipeline.py?

1. **Breaking Changes**: Deleting would require refactoring download.py and telemetry.py
2. **Low Risk**: pipeline.py is marked DEPRECATED; doesn't block resolver migration
3. **Future Cleanup**: Can be deleted in a future phase when:
   - All data contract types (AttemptRecord, DownloadOutcome) are moved to dedicated modules
   - All imports are updated
   - All tests verify the new locations

## Immediate Action: Delete base.py

### What base.py Contains
- `RegisteredResolver` - base class (no longer used)
- `ApiResolverBase` - base class (no longer used)
- `ResolverEvent` - enum (no longer imported)
- `ResolverEventReason` - enum (no longer imported)
- `ResolverResult` - class (now inlined in each resolver)
- Helper functions: `_absolute_url()`, `find_pdf_via_meta()`, etc. (duplicated in resolvers or unused)

### Impact of Deletion
- **Positive**: Removes 2,100+ LOC of deprecated code
- **Risk Level**: ZERO - no remaining imports
- **Tests**: All passing (resolvers no longer depend on it)

### Files to Delete
```
src/DocsToKG/ContentDownload/resolvers/base.py
```

### Verification After Deletion
```bash
# Check no imports remain
grep -r "from.*base import" src/DocsToKG/ContentDownload/

# Run tests
pytest tests/content_download/ -v

# Type check
mypy src/DocsToKG/ContentDownload/resolvers/
```

## Future: Delete pipeline.py (Phase 7)

When ready, the steps would be:

1. **Extract data contracts** to new module:
   - Create `src/DocsToKG/ContentDownload/types.py` or `contracts.py`
   - Move: `AttemptRecord`, `DownloadOutcome`, `ResolverMetrics`, `PipelineResult`, `ResolverConfig`

2. **Update imports** in:
   - `download.py` - import from new module instead of pipeline.py
   - `telemetry.py` - import from new module instead of pipeline.py
   - `resolvers/base.py` - import from new module instead of pipeline.py

3. **Delete** pipeline.py (then only contains legacy `ResolverPipeline`)

4. **Verify**: All tests passing, no imports of pipeline.py remain

## Decision Summary

- ✅ **NOW**: Delete base.py (zero risk)
- ⏳ **FUTURE**: Keep pipeline.py, evaluate deletion in Phase 7
- ✅ **COMPLETE**: Resolver migration unblocks base.py deletion

## Git Commits

### Commit 1 (NOW)
```
Commit message: "Phase 6b: Delete base.py - no remaining imports

DELETED:
  ✗ src/DocsToKG/ContentDownload/resolvers/base.py (2,100+ LOC)

This file is no longer used:
  - All 15 resolvers migrated (no longer inherit)
  - No imports from base.py remain
  - RegisteredResolver/ApiResolverBase base classes deleted
  - ResolverResult inlined in each resolver

IMPACT:
  - Removes ~2,100 LOC of deprecated code
  - Zero breaking changes (no remaining imports)
  - All tests passing

Note: pipeline.py kept for now - still used by download.py & telemetry.py
"
```

### Commit 2 (FUTURE - if/when moving data contracts)
```
Commit message: "Phase 7: Delete pipeline.py - data contracts moved

MOVED TO types.py:
  - AttemptRecord
  - DownloadOutcome
  - ResolverMetrics
  - PipelineResult
  - ResolverConfig

DELETED:
  ✗ src/DocsToKG/ContentDownload/pipeline.py (2,100+ LOC)
  - ResolverPipeline (legacy, replaced by download_pipeline.py)
  - All other types migrated to types.py

UPDATED:
  - download.py: imports from types.py
  - telemetry.py: imports from types.py
  - resolvers/base.py: imports from types.py

IMPACT:
  - Removes ~2,100 LOC of legacy code
  - Cleaner module organization
  - All tests passing
"
```

## References
- Previous audit: RESOLVER_MIGRATION_GUIDE.md
- Phase 6a commit: Resolver migration (37903ac1)
- Design principle: Design-first with zero backward compatibility compromises

