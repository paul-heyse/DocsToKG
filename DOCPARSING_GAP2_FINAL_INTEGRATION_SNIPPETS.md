# Gap #2: DocTags Runner Integration — Exact Code Snippets

**Status**: All supporting infrastructure complete (Gap #1, #3, #4, #5). Ready for manual integration.

**Files to Edit**: 1 file (`doctags.py`)

---

## STEP 1: Add Imports

**File**: `src/DocsToKG/DocParsing/doctags.py`
**Location**: Line ~339, in the `from DocsToKG.DocParsing.core import (` block

**Add these TWO imports** (in alphabetical order within the import block):

```python
    StageOptions,  # ADD THIS LINE
    # ... existing imports ...
    run_stage,  # ADD THIS LINE
```

**Full import block should look like** (excerpt):

```python
from DocsToKG.DocParsing.core import (
    DEFAULT_HTTP_TIMEOUT,
    CLIOption,
    ItemFingerprint,
    ItemOutcome,
    ResumeController,
    StageContext,
    StageError,
    StageHooks,
    StageOptions,      # NEW
    StageOutcome,
    StagePlan,
    WorkItem,
    acquire_lock,
    build_subcommand,
    derive_doc_id_and_doctags_path,
    find_free_port,
    get_http_session,
    normalize_http_timeout,
    run_stage,          # NEW
    set_spawn_or_warn,
)
```

---

## STEP 2: Replace pdf_main() Legacy Loop

**File**: `src/DocsToKG/DocParsing/doctags.py`
**Function**: `pdf_main()` (starts around line 1994)
**Location**: Around line 2300 (after vLLM startup, after manifest_log_success for `__service__`)

### FIND & REMOVE (legacy code to delete)

Find this block:

```python
            logger.info(
                "Launching workers",
                extra={
                    "extra_fields": {
                        "pdf_count": total_inputs,
                        "workers": workers,
                    }
                },
            )

            if not tasks:
                logger.info(
                    "Conversion summary",
                    extra={
                        "extra_fields": {
                            "ok": 0,
                            "skip": skip,
                            "fail": 0,
                        }
                    },
                )
                return 0

            with ProcessPoolExecutor(max_workers=workers) as ex:
                future_map = {ex.submit(pdf_convert_one, task): task for task in tasks}
                with tqdm(
                    as_completed(future_map), total=len(future_map), unit="file", desc="PDF → DocTags"
                ) as pbar:
                    for future in pbar:
                        result = future.result()
                        # ... manifest writing code ...
```

**DELETE everything from `logger.info("Launching workers"` through the end of the ProcessPoolExecutor block (approximately lines 2277–2430)**

### REPLACE WITH (new code using run_stage)

```python
            logger.info(
                "Launching workers",
                extra={
                    "extra_fields": {
                        "pdf_count": total_inputs,
                        "workers": workers,
                    }
                },
            )

            if not tasks:
                logger.info(
                    "Conversion summary",
                    extra={
                        "extra_fields": {
                            "ok": 0,
                            "skip": skip,
                            "fail": 0,
                        }
                    },
                )
                return 0

            # Build stage options from config
            stage_options = StageOptions(
                policy="cpu",  # ProcessPool with spawn (CPU intensive PDF conversion)
                workers=workers,
                per_item_timeout_s=0.0,  # No timeout by default
                retries=0,  # PDF conversion is deterministic, no retries
                retry_backoff_s=1.0,
                error_budget=0,  # Stop on first failure
                max_queue=0,  # No queue limit
                resume=cfg.resume,
                force=cfg.force,
                diagnostics_interval_s=30.0,
                dry_run=False,
            )

            # Get stage hooks for lifecycle management
            stage_hooks = _make_pdf_stage_hooks(
                logger=logger,
                resolved_root=resolved_root,
                resume_skipped=resume_skipped,
            )

            # Run unified runner
            outcome = run_stage(plan, _pdf_stage_worker, stage_options, stage_hooks)

            # Log conversion summary
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

---

## STEP 3: Replace html_main() Legacy Loop

**File**: `src/DocsToKG/DocParsing/doctags.py`
**Function**: `html_main()` (starts around line 2741)
**Location**: Around line 2985 (after vLLM startup checks, before ProcessPoolExecutor)

### FIND & REMOVE (legacy code to delete)

Find this block:

```python
            logger.info(
                "Launching workers",
                extra={
                    "extra_fields": {
                        "html_count": total_inputs,
                        "workers": cfg.workers,
                    }
                },
            )

            if not tasks:
                logger.info(
                    "Conversion summary",
                    extra={
                        "extra_fields": {
                            "ok": 0,
                            "skip": skip,
                            "fail": 0,
                        }
                    },
                )
                return 0

            with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
                futures = [ex.submit(html_convert_one, task) for task in tasks]
                for fut in tqdm(
                    as_completed(futures), total=len(futures), unit="file", desc="HTML → DocTags"
                ):
                    result = fut.result()
                    # ... manifest writing code ...
```

**DELETE everything from `logger.info("Launching workers"` through the end of the ProcessPoolExecutor block (approximately lines 2960–3050)**

### REPLACE WITH (new code using run_stage)

```python
            logger.info(
                "Launching workers",
                extra={
                    "extra_fields": {
                        "html_count": total_inputs,
                        "workers": cfg.workers,
                    }
                },
            )

            if not tasks:
                logger.info(
                    "Conversion summary",
                    extra={
                        "extra_fields": {
                            "ok": 0,
                            "skip": skip,
                            "fail": 0,
                        }
                    },
                )
                return 0

            # Build stage options from config
            stage_options = StageOptions(
                policy="io",  # ThreadPool (I/O bound HTML parsing with Docling)
                workers=cfg.workers,
                per_item_timeout_s=0.0,  # No timeout by default
                retries=0,  # HTML parsing is deterministic
                retry_backoff_s=1.0,
                error_budget=0,  # Stop on first failure
                max_queue=0,  # No queue limit
                resume=cfg.resume,
                force=cfg.force,
                diagnostics_interval_s=30.0,
                dry_run=False,
            )

            # Get stage hooks for lifecycle management
            stage_hooks = _make_html_stage_hooks(
                logger=logger,
                resolved_root=resolved_root,
                resume_skipped=resume_skipped,
            )

            # Run unified runner
            outcome = run_stage(plan, _html_stage_worker, stage_options, stage_hooks)

            # Log conversion summary
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

---

## Verification Checklist

After applying both changes:

- [ ] Imports section has both `StageOptions` and `run_stage`
- [ ] `pdf_main()` calls `run_stage(plan, _pdf_stage_worker, stage_options, stage_hooks)`
- [ ] `html_main()` calls `run_stage(plan, _html_stage_worker, stage_options, stage_hooks)`
- [ ] Both functions return `0 if outcome.failed == 0 else 1`
- [ ] Removed all ProcessPoolExecutor loops and tqdm usage
- [ ] Run type check: `mypy src/DocsToKG/DocParsing/doctags.py`
- [ ] Run linting: `ruff check src/DocsToKG/DocParsing/doctags.py`
- [ ] All tests pass: `pytest tests/docparsing/ -q`

---

## Testing After Integration

### 1. Unit Tests

```bash
pytest tests/docparsing/test_runner_semantics.py -v
```

### 2. Integration Tests

```bash
pytest tests/docparsing/ -q
```

### 3. Type Check

```bash
mypy src/DocsToKG/DocParsing/doctags.py
```

### 4. Linting

```bash
ruff check src/DocsToKG/DocParsing/doctags.py
```

### 5. Corpus Parity (Recommended)

```bash
# Save legacy manifests first
cp Data/Manifests/docparse.doctags*.manifest.jsonl /tmp/legacy_manifests/

# Run new version
direnv exec . python -m DocsToKG.DocParsing.core.cli doctags --mode pdf --input Data/PDFs --output Data/DocTagsFiles --force

# Compare manifests
jq '.status' /tmp/legacy_manifests/docparse.doctags-*.manifest.jsonl | sort | uniq -c
jq '.status' Data/Manifests/docparse.doctags-*.manifest.jsonl | sort | uniq -c
```

---

## Key Points

1. **Policy Difference**:
   - `pdf_main()` uses `policy="cpu"` (ProcessPool/spawn)
   - `html_main()` uses `policy="io"` (ThreadPool)

2. **No Timeouts by Default**:
   - `per_item_timeout_s=0.0` (unlimited)
   - Users can set `--timeout-s` from CLI if needed

3. **No Retries**:
   - PDF/HTML conversion is deterministic
   - Transient failures are rare
   - Users can enable `--retries` from CLI if needed

4. **Resume/Force Honored**:
   - Both functions pass `cfg.resume` and `cfg.force` to StageOptions
   - Runner respects fingerprints and forces as configured

5. **Hooks Integrated**:
   - Both use existing `_make_*_stage_hooks()` functions
   - Manifests written via hooks → ensures consistency

---

## Notes for Implementation

- The existing `_build_pdf_plan()`, `_pdf_stage_worker()`, and `_make_pdf_stage_hooks()` functions are already complete and correct
- Same for HTML equivalents
- No changes needed to those functions
- Only changes: imports + loop replacement in `pdf_main()` and `html_main()`

---

## Rollback Plan

If corpus parity test fails:

1. Revert the doctags.py changes
2. Keep infrastructure (manifest sink, CLI flags, tests, docs)
3. Investigate specific behavioral differences
4. Fix and retry

All infrastructure remains useful regardless of Gap #2 outcome.
