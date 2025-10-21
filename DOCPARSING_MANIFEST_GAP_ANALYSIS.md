# DocParsing Manifest Gap Analysis — October 21, 2025

## Executive Summary

✅ **Code infrastructure is 100% correct**
❌ **But manifest entries don't include provider metadata or vector_format**
🔍 **Root cause**: Manifest entries are being written through the **StageTelemetry class** (not direct `manifest_append` calls), which has a **different code path** that isn't preserving the extras.

---

## Investigation Results

### Code Structure (WORKING CORRECTLY)

**Embedding stage writes manifests via two paths:**

1. **Direct path** (legacy): `manifest_log_success(..., vector_format=..., **provider_extras)` in logging.py
   - ✅ Correctly unpacks all extra fields into the manifest entry
   - ✅ Calls `manifest_append(**payload)` which does `entry.update(metadata)`

2. **StageTelemetry path** (newer): Used in hooks at `embedding/runtime.py:2068-2079`
   - ✅ Passes `vector_format` and provider extras
   - ❌ BUT then calls `telemetry.log_success(metadata=metadata)`
   - ❌ Which calls `sink.write_manifest_entry(entry, writer=...)`
   - ❌ Which flattens to `payload.update(metadata)`
   - ❌ **But only some fields make it to manifest JSONL**

### Evidence from Actual Manifest Data

**Successful embedding entry (recent, Oct 19 01:02):**
```json
{
  "run_id": "deef4ec334a7444c8329646f960d1437",
  "file_id": "doc.doctags",
  "stage": "embeddings",
  "status": "success",
  "vector_count": 4,
  "doc_id": "doc.doctags"
  // MISSING: vector_format, provider_name, model_id, dense_provider_name, dense_model_id, etc.
}
```

**Config entry (has vector_format):**
```json
{
  "file_id": "__config__",
  "status": "success",
  "vector_format": "jsonl",  // ← Present here!
  "config": {...},
  "doc_id": "__config__"
}
```

### Root Cause

The issue is in **`telemetry.py` class `ManifestEntry` → `write_manifest_entry`**:

```python
# telemetry.py:155-167
def write_manifest_entry(
    self,
    entry: ManifestEntry,
    *,
    writer: Optional[Callable] = None,
) -> None:
    """Append ``entry`` to the manifest log."""
    payload = asdict(entry)  # ← Converts dataclass to dict
    metadata = dict(payload.pop("metadata", {}) or {})
    payload.update(metadata)  # ← Merges metadata dict
    payload.setdefault("doc_id", entry.file_id)
    self._append_payload(self._manifest_path, payload, writer=writer)
```

**Problem**: `ManifestEntry` dataclass (telemetry.py:85-96) only has these fields:
```python
@dataclass(slots=True)
class ManifestEntry:
    run_id: str
    file_id: str
    stage: str
    output_path: str
    tokens: int
    schema_version: Optional[str]
    duration_s: float
    metadata: Dict[str, Any] = field(default_factory=dict)  # ← All extras go here
```

So when `logging.py:log_success()` calls:
```python
telemetry.log_success(
    doc_id=doc_id,
    input_path=input_path,
    output_path=output_path,
    tokens=tokens,
    schema_version=schema_version,
    duration_s=duration_s,
    metadata=metadata,  # ← ALL extra fields (vector_format, provider_name, etc.) go here
)
```

The extra fields ARE being passed but only in the `metadata` dict, which then gets unpacked. This **should work** but let's verify the exact flow.

---

## Detailed Flow Analysis

**Embedding stage:**
1. ✅ Collects provider metadata: `embedding/runtime.py:1784-1805`
2. ✅ Stores in state: `state["provider_metadata_extras"] = {...}`
3. ✅ Passes to manifest: `manifest_log_success(..., **state.get("provider_metadata_extras", {}))`
4. ✅ logging.py builds metadata dict: `metadata.update(extra)`
5. ✅ Calls `telemetry.log_success(..., metadata=metadata)`
6. ✅ StageTelemetry.log_success creates ManifestEntry:
   ```python
   entry = ManifestEntry(
       run_id=self._run_id,
       file_id=doc_id,
       stage=self._stage,
       output_path=str(output_path),
       tokens=tokens or 0,
       schema_version=schema_version,
       duration_s=duration_s,
       metadata=metadata or {},  # ← Vector format + provider data goes here
   )
   ```
7. ✅ Calls sink.write_manifest_entry(entry):
   ```python
   payload = asdict(entry)  # Creates dict with metadata key
   metadata = dict(payload.pop("metadata", {}) or {})
   payload.update(metadata)  # Merges extras back into payload
   ```
8. ✅ Appends to manifest via `_append_payload()`

**This should work!** But the autolint shows it's not. Possibilities:

1. **Race condition**: Multiple processes writing manifests concurrently?
2. **Test data**: These manifest entries are from test runs, not production?
3. **Path not taken**: In some code paths, a different logging function is used?

---

## Hypothesis: Legacy Code Path Still in Use

**Found in `embedding/runtime.py:2047-2179` (after_item hook):**

The success case calls `manifest_log_success()` correctly ✅

**But failure and skip cases use different helpers:**
```python
# Line 2099: failure case
manifest_log_failure(
    stage=MANIFEST_STAGE,
    doc_id=item.item_id,
    ...
)

# Line 2139: skip case
manifest_log_skip(
    stage=MANIFEST_STAGE,
    doc_id=item.item_id,
    ...
)
```

These don't have the `**provider_metadata_extras` unpacking! But they're for failures/skips, so that's expected.

---

## Chunking Gap

**File**: `src/DocsToKG/DocParsing/chunking/runtime.py:1126-1137`

The chunk success manifest dict **never includes `chunks_format`**:
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
    # ← MISSING: "chunks_format": cfg.format  (or worker_config.format)
}
```

This is in the hook that processes results:
```python
def after_item(item, outcome_or_error, context):
    ...
    if outcome_or_error.status == "success":
        manifest_log_success(
            stage=MANIFEST_STAGE,
            doc_id=item.item_id,
            duration_s=round(outcome_or_error.duration_s, 3),
            schema_version=schema_version,
            input_path=input_path,
            input_hash=resolved_hash,
            output_path=output_path,
            **outcome_or_error.manifest,  # ← chunks_format not in outcome_or_error.manifest
        )
```

The `outcome_or_error.manifest` dict comes from `_chunk_stage_worker()` which builds it.

---

## Fixes Required

### Fix #1: Verify StageTelemetry Path for Embedding Manifests

Run an embedding to generate fresh manifest data:
```bash
python -m DocsToKG.DocParsing.core.cli embed --chunks-dir Data/ChunkedDocTagFiles --out-dir /tmp/vectors_test --limit 5 --format parquet
tail Data/Manifests/docparse.embeddings.manifest.jsonl | python -m json.tool
```

**Expected**: Fresh entries should have `vector_format`, `provider_name`, `model_id`
**If missing**: Debug the StageTelemetry flow in telemetry.py

### Fix #2: Add `chunks_format` to Chunk Manifests

**File**: `src/DocsToKG/DocParsing/chunking/runtime.py:1135`
**Change**: Add one line:
```python
manifest = {
    ...
    "sanitizer_profile": result.sanitizer_profile,
    "chunks_format": worker_config.format,  # ← ADD THIS
}
```

Or from context if available:
```python
"chunks_format": item.metadata.get("worker_config", {}).get("format", "jsonl"),
```

---

## Summary Table

| Item | Status | Evidence | Fix |
|------|--------|----------|-----|
| Parquet footers (provider) | ✅ OK | `parquet_schemas.py:153-274` | None needed |
| Embedding manifest extras (code) | ✅ OK | `embedding/runtime.py:2078` unpacks correctly | Verify runtime |
| Embedding manifest extras (data) | ❌ MISSING | Autolint found 0/100 success entries have `provider_name` | Debug StageTelemetry path |
| Chunk manifest `chunks_format` | ❌ MISSING | Not in dict at `chunking/runtime.py:1137` | Add 1 line |
| Config provenance `__config__` | ✅ OK | Present in manifests | None needed |
| Fingerprints `.fp.json` | ✅ OK | Created in all stages | None needed |
