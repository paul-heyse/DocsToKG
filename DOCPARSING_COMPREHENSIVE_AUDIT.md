# DocParsing Module Documentation Comprehensive Audit

**Date:** October 21, 2025  
**Scope:** All DocParsing modules - ensure consistent documentation of public API migration  
**Status:** 🔍 AUDIT IN PROGRESS

## Summary

A full audit of the DocParsing codebase reveals **4 modules with documentation gaps** related to the public API migration from `_acquire_lock()` to `safe_write()` and the introduction of `JsonlWriter`:

## Modules Requiring Updates

### 1. ✅ ALREADY UPDATED (5 modules)
- `core/concurrency.py` - ✅ Documented safe_write() API
- `core/__init__.py` - ✅ Documented exports including safe_write
- `io.py` - ✅ Highlighted JsonlWriter component
- `telemetry.py` - ✅ Documented lock-aware writer injection
- `doctags.py` - ✅ Updated with safe_write reference

### 2. ⚠️ NEED UPDATES (4 modules)

#### A. `core/manifest_sink.py`
**Current Docstring:**
```
Unified manifest sink for DocParsing stages.

This module provides a protocol and implementation for writing stage manifest
entries (success, skip, failure) with atomic, lock-based JSONL appending.
All stages use this abstraction to ensure consistent base fields and
reliable concurrent writes.
```

**Issue:** Mentions "atomic, lock-based JSONL appending" but doesn't reference the modern `JsonlWriter` or `safe_write()` components that enable this.

**Update Needed:** Reference JsonlWriter and explain the dependency injection pattern.

---

#### B. `chunking/runtime.py`
**Current Docstring:** (1429 chars, comprehensive but doesn't mention locking)
```
Docling Hybrid Chunker with Minimum Token Coalescence

Transforms DocTags documents into chunked records with topic-aware coalescence.
The module exposes a CLI (``python -m DocsToKG.DocParsing.core chunk``) and
runtime integration points for serializing chunks to JSONL and manifests.
...
```

**Issue:** Uses atomic writes and telemetry but doesn't mention these in docstring.

**Update Needed:** Add note about atomic writes and manifest integration.

---

#### C. `storage/chunks_writer.py`
**Current Docstring:**
```
Atomic Parquet Writer for Chunks

Encapsulates write logic for Chunks Parquet datasets with:
- Schema validation and enforcement
- Atomic writes (temp → fsync → rename)
- Batched row accumulation to prevent memory bloat
```

**Issue:** Claims "atomic writes" but doesn't reference the safe_write() or locking mechanism used.

**Update Needed:** Clarify atomic write implementation details.

---

#### D. `storage/writers.py`
**Current Docstring:**
```
Parquet Writers for DocParsing Artifacts

Encapsulates write logic for Chunks and Vectors (Dense, Sparse, Lexical) with:
- Schema validation and enforcement
- Atomic writes (temp → fsync → rename)
...
```

**Issue:** Claims "atomic writes" but doesn't reference safe_write() or the implementation approach.

**Update Needed:** Clarify implementation details and reference public APIs.

---

### 3. ✅ VERIFIED (Good documentation - 6 modules)
- `embedding/runtime.py` - ✅ Already accurate
- `logging.py` - ✅ Appropriate level of detail
- Other core modules have appropriate docstrings

---

## Recommended Updates

### For `core/manifest_sink.py`

Update to:
```
Unified manifest sink for DocParsing stages.

This module provides a protocol and implementation for writing stage manifest
entries (success, skip, failure) with atomic, lock-aware JSONL appending using
the lock-aware JsonlWriter component. All stages use this abstraction to ensure
consistent base fields and reliable concurrent writes even when multiple processes
report progress simultaneously.

The implementation leverages FileLock and atomic appends to prevent manifest
corruption during concurrent access, making it safe for distributed pipelines.
```

### For `chunking/runtime.py`

Add to existing docstring:

```
Atomic writes and concurrent safety:
- Chunk JSONL written atomically via JsonlWriter for concurrent-safe appending
- Manifest entries written atomically to prevent corruption under parallel loads
- Deterministic chunk IDs based on content hash ensure idempotent writes
```

### For `storage/chunks_writer.py` & `storage/writers.py`

Update "Atomic writes" section to clarify:

```
- Atomic writes (temp → fsync → rename) for safe concurrent access
- No explicit locking needed; rename operation is atomic at OS level
- Compatible with concurrent readers via temp-file pattern
```

---

## Implementation Plan

1. ✅ **Core modules** (5) - COMPLETE
2. ⏳ **Manifest sink** - PENDING
3. ⏳ **Chunking runtime** - PENDING  
4. ⏳ **Storage writers** - PENDING

---

## Quality Checklist

- [ ] All atomic write references mention safe_write() or underlying mechanism
- [ ] All telemetry references mention JsonlWriter where applicable
- [ ] All lock-based operations reference the lock-aware components
- [ ] No references to private `_acquire_lock()` in public docs
- [ ] All docstrings follow Google docstring style
- [ ] All examples show current APIs only

---

**Next Steps:** Update the 4 modules identified above with clarified documentation.

