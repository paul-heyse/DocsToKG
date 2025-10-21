# DocParsing Quick Checks Results — October 21, 2025

## Summary
✅ **MOSTLY COMPLETE** — Core infrastructure is in place, but two specific manifest gaps remain.

---

## Check Results

### ✅ Check 1: Provider Provenance in Parquet Footers
**Status**: PASSING
**Evidence**:
```
src/DocsToKG/DocParsing/storage/parquet_schemas.py:153:    "docparse.provider",
src/DocsToKG/DocParsing/storage/parquet_schemas.py:215:            "docparse.provider": provider,
src/DocsToKG/DocParsing/storage/parquet_schemas.py:241:            "docparse.provider": provider,
src/DocsToKG/DocParsing/storage/parquet_schemas.py:273:            "docparse.provider": provider,
```
**Details**: Parquet footer builders (dense, sparse, lexical) properly include `docparse.provider`, `docparse.model_id`, `docparse.dtype`, `docparse.cfg_hash`, etc.

---

### ✅ Check 2: Manifest Provider Extras Infrastructure
**Status**: INFRASTRUCTURE EXISTS, but NOT YET USED IN ACTUAL SUCCESS LOGS
**Evidence**:
```
src/DocsToKG/DocParsing/embedding/runtime.py:1784-1805:
  # Extraction of provider metadata into state
  provider_metadata_extras["dense_provider_name"] = dense_id.name
  provider_metadata_extras["dense_model_id"] = dense_id.extra["model_id"]
  provider_metadata_extras["sparse_provider_name"] = sparse_id.name
  provider_metadata_extras["sparse_model_id"] = sparse_id.extra["model_id"]
  provider_metadata_extras["lexical_provider_name"] = lexical_id.name
  state["provider_metadata_extras"] = provider_metadata_extras
```
**Location**: `embedding/runtime.py:2068-2079` — Called in `after_item` hook:
```python
manifest_log_success(
    ...
    **state.get("provider_metadata_extras", {}),  # ← extras passed here
)
```

---

## ⚠️ GAPS IDENTIFIED

### GAP #1: Manifest Success Entries Missing `provider_name`, `model_id`, `vector_format`
**Severity**: MEDIUM
**File**: `src/DocsToKG/DocParsing/embedding/runtime.py:2068-2079`
**Issue**: `manifest_log_success()` is called WITH provider extras (`**state.get("provider_metadata_extras", {})`), but the **fields are not appearing in the manifest JSONL**.

**Root cause**: The `manifest_log_success()` helper is not a standard function — we need to trace where manifests are actually written.

**Autolint found**:
```json
[
  ["MANIFEST_MISSING", "embeddings", "provider_name"],
  ["MANIFEST_MISSING", "embeddings", "model_id"],
  ["MANIFEST_MISSING", "embeddings", "vector_format"]
]
```
Repeated 100+ times across recent manifest runs.

---

### GAP #2: Chunks Manifest Missing `chunks_format` Field
**Severity**: LOW (informational only)
**File**: `src/DocsToKG/DocParsing/chunking/runtime.py:1126-1137`
**Issue**: Chunk success manifest entries don't include `chunks_format` field for audit/diagnostics.

**Location in code**:
```python
manifest = {
    "input_path": str(result.input_path),
    "input_hash": result.input_hash,
    "hash_alg": hash_alg,
    "output_path": str(result.output_path),
    "schema_version": CHUNK_SCHEMA_VERSION,
    "chunk_count": result.chunk_count,
    "total_tokens": result.total_tokens,
    "parse_engine": result.parse_engine,
    "anchors_injected": result.anchors_injected,
    "sanitizer_profile": result.sanitizer_profile,
    # ← MISSING: "chunks_format": (parquet vs jsonl)
}
```

---

## ✅ Checks Passing

| Check | Result | Evidence |
|-------|--------|----------|
| Runner purity (no stage-local pools) | ✅ PASS | Executors created once in `core/runner.py:293-301` via `_create_executor()`, used by `run_stage()`. No ThreadPoolExecutor/ProcessPoolExecutor in stage implementations. |
| Fingerprints for resume | ✅ PASS | `.fp.json` created: `chunking/runtime.py:1007`, `doctags.py:511`, `embedding/runtime.py:1723` |
| Config provenance (`__config__` rows) | ✅ PASS | Written to manifests: `chunking/runtime.py:1701`, `doctags.py:2134`, `embedding/runtime.py:2548` |
| cfg_hash tracking | ✅ PASS | Computed and tracked: `embedding/runtime.py:558`, `chunking/runtime.py:947`, passed to footers and fingerprints |
| Embedding runtime is provider-only | ✅ PASS | No torch/transformers/vllm imports in `embedding/runtime.py` at module level |
| Atomic JSONL appends | ✅ PASS | `jsonl_append_iter(..., atomic=True)` wired in `io.py:110` |
| fmt=parquet partition paths | ✅ PASS | Layout properly defined: `storage/paths.py`, `storage/dataset_view.py` |

---

## What Needs Fixing

### Fix #1: Wire `manifest_log_success()` to include `vector_format` + provider metadata
**File**: Need to find where `manifest_log_success()` is defined and ensure it includes:
- `vector_format` (passed in explicitly)
- `provider_name` (dense/sparse/lexical)
- `model_id` (if available)

**Search needed**: Where is `manifest_log_success()` defined?

### Fix #2: Add `chunks_format` to chunk manifest success entries
**File**: `src/DocsToKG/DocParsing/chunking/runtime.py:1126-1137`
**Change**: Add `"chunks_format": cfg.format` to the `manifest` dict.

---

## Next Steps

1. Find `manifest_log_success()` definition (likely in telemetry or embedding runtime)
2. Verify it's being passed `vector_format` from caller
3. Check if metadata unpacking is happening correctly
4. Add `chunks_format` to chunking manifest
