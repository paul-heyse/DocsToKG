# DocTags Stage Runner Integration (Gap #2)

**Date**: October 21, 2025
**Objective**: Replace legacy ProcessPoolExecutor loops in `pdf_main()` and `html_main()` with unified `run_stage()` calls
**Files Modified**: `src/DocsToKG/DocParsing/doctags.py`
**Effort**: ~6 hours
**Risk**: MEDIUM (requires corpus parity testing)

---

## Current State

### What Exists

- ✅ `_build_pdf_plan()` (lines 460–562): Creates StagePlan with WorkItems
- ✅ `_pdf_stage_worker()` (lines 565–617): Worker adapter for conversion
- ✅ `_make_pdf_stage_hooks()` (lines 620–680): Hooks for lifecycle
- ✅ `_build_html_plan()`: HTML equivalent
- ✅ `_html_stage_worker()`: HTML worker adapter
- ✅ `_make_html_stage_hooks()`: HTML hooks

### What Needs Replacing

- ❌ `pdf_main()` legacy loop (lines 2300+): ProcessPoolExecutor + tqdm loop
- ❌ `html_main()` legacy loop (lines 2985+): ProcessPoolExecutor + tqdm loop

---

## Implementation Steps

### Step 1: Import `run_stage` from core.runner

**Location**: Top of doctags.py, add to existing imports

```python
from DocsToKG.DocParsing.core import (
    # ... existing imports ...
    run_stage,  # ADD THIS
)
```

**File**: `src/DocsToKG/DocParsing/doctags.py` (around line 331)

---

### Step 2: Replace pdf_main() Legacy Loop

**Location**: `pdf_main()` after vLLM startup (around line 2300)

**Current Pattern** (REMOVE):

```python
with ProcessPoolExecutor(max_workers=workers) as ex:
    future_map = {ex.submit(pdf_convert_one, task): task for task in tasks}
    with tqdm(...) as pbar:
        for future in as_completed(future_map):
            # ... process results, write manifests ...
```

**New Pattern** (ADD):

```python
# Build stage options from config
stage_options = StageOptions(
    policy="cpu",  # ProcessPool with spawn (CPU intensive)
    workers=workers,
    per_item_timeout_s=0.0,  # No timeout by default
    retries=0,  # No retries (each PDF conversion is independent)
    retry_backoff_s=1.0,
    error_budget=0,  # Stop on first error (default)
    max_queue=0,  # No backpressure limit
    resume=cfg.resume,
    force=cfg.force,
    diagnostics_interval_s=30.0,
    dry_run=False,
)

# Get stage hooks
stage_hooks = _make_pdf_stage_hooks(
    logger=logger,
    resolved_root=resolved_root,
    resume_skipped=resume_skipped,
)

# Run unified runner
outcome = run_stage(plan, _pdf_stage_worker, stage_options, stage_hooks)

# Log summary
logger.info(
    "Conversion summary",
    extra={
        "extra_fields": {
            "ok": outcome.succeeded,
            "skip": outcome.skipped,
            "fail": outcome.failed,
            "total_wall_ms": outcome.wall_ms,
            "exec_p50_ms": outcome.exec_p50_ms,
            "exec_p95_ms": outcome.exec_p95_ms,
        }
    },
)

return 0 if outcome.failed == 0 else 1
```

**Detailed Changes**:

1. Remove lines 2300–2400 (approx) containing ProcessPoolExecutor loop
2. Build `StageOptions` from cfg attributes
3. Call `run_stage(plan, _pdf_stage_worker, stage_options, stage_hooks)`
4. Extract summary from `StageOutcome`
5. Return 0 on success, 1 on failure

---

### Step 3: Replace html_main() Legacy Loop

**Location**: `html_main()` around line 2985

**Same pattern as Step 2**:

```python
stage_options = StageOptions(
    policy="io",  # ThreadPool (I/O bound HTML parsing)
    workers=cfg.workers,
    per_item_timeout_s=0.0,
    retries=0,
    retry_backoff_s=1.0,
    error_budget=0,
    max_queue=0,
    resume=cfg.resume,
    force=cfg.force,
    diagnostics_interval_s=30.0,
    dry_run=False,
)

stage_hooks = _make_html_stage_hooks(
    logger=logger,
    resolved_root=resolved_root,
    resume_skipped=resume_skipped,
)

outcome = run_stage(plan, _html_stage_worker, stage_options, stage_hooks)

logger.info(
    "Conversion summary",
    extra={
        "extra_fields": {
            "ok": outcome.succeeded,
            "skip": outcome.skipped,
            "fail": outcome.failed,
        }
    },
)

return 0 if outcome.failed == 0 else 1
```

**Note**: HTML uses `policy="io"` (ThreadPool) instead of `"cpu"` (ProcessPool)

---

## Validation Checklist

### Pre-Implementation

- [ ] Read and understand _build_pdf_plan() and_pdf_stage_worker()
- [ ] Read and understand _build_html_plan() and_html_stage_worker()
- [ ] Understand StageOptions configuration
- [ ] Understand StageHooks lifecycle

### During Implementation

- [ ] Add run_stage import
- [ ] Replace pdf_main() loop
- [ ] Replace html_main() loop
- [ ] Verify both functions build and parse correctly
- [ ] Run mypy and ruff on file

### Post-Implementation (Testing)

- [ ] **Corpus parity test**: Run reference PDFs/HTMLs, compare manifests
- [ ] **Manifest structure**: Rows should have identical base fields
- [ ] **Error counts**: Match legacy (within ±1%)
- [ ] **Performance**: Throughput ≥ legacy
- [ ] **All unit tests passing**

---

## Corpus Parity Test Procedure

1. **Before**: Save legacy output manifests

   ```bash
   cp Data/Manifests/docparse.doctags*.manifest.jsonl /tmp/legacy_manifests/
   ```

2. **Make changes** (Step 1–3 above)

3. **Run new version**:

   ```bash
   direnv exec . python -m DocsToKG.DocParsing.core.cli doctags --mode pdf --input Data/PDFs --output Data/DocTagsFiles --force
   ```

4. **Compare manifests**:

   ```bash
   # Check error counts
   jq '.error' /tmp/legacy_manifests/docparse.doctags-*.manifest.jsonl | sort | uniq -c
   jq '.error' Data/Manifests/docparse.doctags-*.manifest.jsonl | sort | uniq -c

   # Verify schema version (should be identical)
   jq '.schema_version' /tmp/legacy_manifests/docparse.doctags-*.manifest.jsonl | sort | uniq
   jq '.schema_version' Data/Manifests/docparse.doctags-*.manifest.jsonl | sort | uniq
   ```

5. **Verify counts match**:
   - success count should match
   - skip count should match
   - error count within ±1%

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Behavioral change in conversion | Corpus parity test with detailed manifest diff |
| Loss of progress visibility | Runner has built-in progress logging every 30s |
| Timeout/retry semantics unknown | Use conservative defaults (0 timeout, 0 retries) |
| Manifest format change | New fields OK, but base fields must match |
| Escape hatch needed | Keep legacy code in separate branch if rollback required |

---

## Rollback Plan

If corpus parity test fails:

1. Revert changes to pdf_main() and html_main()
2. Keep `run_stage` import and other infrastructure
3. Investigate behavioral differences
4. Fix specific issues, then retry

---

## Files to Modify

- `src/DocsToKG/DocParsing/doctags.py`
  - Add `run_stage` import (line ~331)
  - Replace pdf_main() loop (line ~2300)
  - Replace html_main() loop (line ~2985)

---

## Success Criteria

✅ **All unit tests passing**
✅ **Corpus parity: error counts within ±1%**
✅ **Manifest base fields identical**
✅ **0 mypy/ruff violations**
✅ **Runner progress logged at 30s intervals**
