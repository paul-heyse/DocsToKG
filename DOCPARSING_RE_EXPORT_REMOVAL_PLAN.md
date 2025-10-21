# DocParsing Re-Export Removal Plan — Final Decision

**Date**: October 21, 2025  
**Status**: ANALYSIS COMPLETE → READY FOR IMPLEMENTATION  
**Decision**: DELETE (Explicitly Marked Legacy)

---

## Executive Summary

The re-exported symbols in `chunking/__init__.py` and `embedding/__init__.py` are **explicitly marked as legacy compatibility shims** in the code comments. They exist only for test support and add 40 lines of maintenance burden. **Recommendation: DELETE** and update tests to use direct imports.

---

## What We're Deleting

### chunking/__init__.py

**Legacy Code Block 1 (lines 18-22):**
```python
from . import runtime as _runtime  # ← Unused, only for compat injection
from .cli import CHUNK_CLI_OPTIONS, build_parser, parse_args
from .config import CHUNK_PROFILE_PRESETS, SOFT_BARRIER_MARGIN, ChunkerCfg
from .runtime import *  # noqa: F401,F403 - re-export legacy runtime symbols  ← DELETE
from .runtime import main
```

**Legacy Code Block 2 (lines 38-42):**
```python
for _name in list(globals()):  # ← Dynamic namespace pollution
    if _name.startswith("_") or _name in {"config", "cli", "runtime"}:
        continue
    if _name not in __all__:
        __all__.append(_name)
```

**Legacy Code Block 3 (lines 44-69):**
```python
# Compatibility shims for legacy imports (tests rely on these attributes existing).
_LEGACY_EXPORTS = (
    "AutoTokenizer",
    "HuggingFaceTokenizer",
    "HybridChunker",
    "ProvenanceMetadata",
    "ChunkRow",
    "get_docling_version",
    "manifest_log_failure",
    "manifest_log_success",
    "manifest_log_skip",
    "atomic_write",
    "_LOGGER",
)

for _legacy_name in _LEGACY_EXPORTS:  # ← Dynamically injects legacy symbols
    if hasattr(_runtime, _legacy_name):
        globals()[_legacy_name] = getattr(_runtime, _legacy_name)
        if _legacy_name not in __all__:
            __all__.append(_legacy_name)
    elif _legacy_name in globals():
        # Provided via DocsToKG.DocParsing.logging imports above.
        if _legacy_name not in __all__:
            __all__.append(_legacy_name)

del _legacy_name, _LEGACY_EXPORTS
del _name
```

### embedding/__init__.py

**Legacy Code Block 1 (lines 17-21):**
```python
from . import runtime as _runtime  # ← Unused, only for compat injection
from .cli import EMBED_CLI_OPTIONS, build_parser, parse_args
from .config import EMBED_PROFILE_PRESETS, EmbedCfg
from .runtime import *  # noqa: F401,F403 - re-export legacy runtime symbols  ← DELETE
from .runtime import main
```

**Legacy Code Block 2 (lines 40-44):**
```python
for _name in list(globals()):  # ← Dynamic namespace pollution
    if _name.startswith("_") or _name in {"config", "cli", "runtime"}:
        continue
    if _name not in __all__:
        __all__.append(_name)
```

**Legacy Code Block 3 (lines 46-50):**
```python
for _compat_name in ("_ensure_splade_dependencies", "_ensure_qwen_dependencies"):
    if hasattr(_runtime, _compat_name):
        globals()[_compat_name] = getattr(_runtime, _compat_name)
        if _compat_name not in __all__:
            __all__.append(_compat_name)
del _compat_name
```

---

## Exact Line Numbers to Delete

### chunking/__init__.py
- **Delete lines 18** (unused import: `from . import runtime as _runtime`)
- **Delete lines 21** (wildcard import: `from .runtime import *`)
- **Delete lines 38-42** (dynamic namespace loop)
- **Delete lines 44-69** (entire `_LEGACY_EXPORTS` block + injection loop)
- **Delete lines 69-70** (cleanup: `del _legacy_name, _LEGACY_EXPORTS` and `del _name`)

**Total to delete**: ~27 lines

### embedding/__init__.py
- **Delete line 17** (unused import: `from . import runtime as _runtime`)
- **Delete line 20** (wildcard import: `from .runtime import *`)
- **Delete lines 40-44** (dynamic namespace loop)
- **Delete lines 46-50** (compat injection loop + cleanup)
- **Delete line 53** (cleanup: `del _name`)

**Total to delete**: ~15 lines

**Grand Total**: ~42 lines of legacy code

---

## Resulting Clean Exports

### Clean chunking/__init__.py
```python
"""Chunking stage package exporting CLI, configuration, and runtime helpers."""

from DocsToKG.DocParsing.io import atomic_write
from DocsToKG.DocParsing.logging import (
    manifest_log_failure,
    manifest_log_skip,
    manifest_log_success,
)

from .cli import CHUNK_CLI_OPTIONS, build_parser, parse_args
from .config import CHUNK_PROFILE_PRESETS, SOFT_BARRIER_MARGIN, ChunkerCfg
from .runtime import main

__all__ = [
    "ChunkerCfg",
    "CHUNK_PROFILE_PRESETS",
    "SOFT_BARRIER_MARGIN",
    "CHUNK_CLI_OPTIONS",
    "build_parser",
    "parse_args",
    "main",
    "atomic_write",
    "manifest_log_failure",
    "manifest_log_skip",
    "manifest_log_success",
]
```

### Clean embedding/__init__.py
```python
"""Embedding stage namespace exposing CLI, config, and runtime adapters."""

from DocsToKG.DocParsing.formats import BM25Vector, DenseVector, SPLADEVector, VectorRow
from DocsToKG.DocParsing.logging import (
    manifest_log_failure,
    manifest_log_skip,
    manifest_log_success,
)

from .cli import EMBED_CLI_OPTIONS, build_parser, parse_args
from .config import EMBED_PROFILE_PRESETS, EmbedCfg
from .runtime import main

__all__ = [
    "EmbedCfg",
    "EMBED_PROFILE_PRESETS",
    "EMBED_CLI_OPTIONS",
    "build_parser",
    "parse_args",
    "main",
    "BM25Vector",
    "DenseVector",
    "SPLADEVector",
    "VectorRow",
    "manifest_log_failure",
    "manifest_log_skip",
    "manifest_log_success",
]
```

---

## Test Files to Update

### tests/docparsing/test_docparsing_core.py

**Location 1: Line 714**
```python
# BEFORE:
import DocsToKG.DocParsing.chunking as doc_chunking
logger = doc_chunking._LOGGER.logger

# AFTER:
from DocsToKG.DocParsing.chunking.runtime import _LOGGER
logger = _LOGGER.logger
```

**Location 2: Line 717**
```python
# BEFORE:
refs, pages = doc_chunking.extract_refs_and_pages(chunk)

# AFTER:
from DocsToKG.DocParsing.chunking.runtime import extract_refs_and_pages
refs, pages = extract_refs_and_pages(chunk)
```

**Location 3: Line 783**
```python
# BEFORE:
doc_chunking.summarize_image_metadata(chunk, "Figure caption: sample text")

# AFTER:
from DocsToKG.DocParsing.chunking.runtime import summarize_image_metadata
summarize_image_metadata(chunk, "Figure caption: sample text")
```

### tests/content_download/test_atomic_writes.py

**Location: Lines 278-279**
```python
# BEFORE:
import DocsToKG.DocParsing.chunking as chunker
import DocsToKG.DocParsing.embedding as embeddings
# Then accessing via chunker.*, embeddings.*

# AFTER:
from DocsToKG.DocParsing import chunking, embedding
# Or better: specific imports
```

---

## Implementation Steps

### Step 1: Update Test Imports
File: `tests/docparsing/test_docparsing_core.py`
- Identify all uses of `doc_chunking.*` re-exported symbols
- Replace with direct imports from `.runtime`
- Verify test still passes

File: `tests/content_download/test_atomic_writes.py`
- Identify all package-level attribute access
- Replace with direct imports
- Verify test still passes

### Step 2: Delete Legacy Code
File: `src/DocsToKG/DocParsing/chunking/__init__.py`
- Delete lines 18, 21, 38-42, 44-69, 69-70

File: `src/DocsToKG/DocParsing/embedding/__init__.py`
- Delete lines 17, 20, 40-44, 46-50, 53

### Step 3: Verify
```bash
# Run tests
pytest tests/docparsing/test_docparsing_core.py -v
pytest tests/content_download/test_atomic_writes.py -v

# Run linter
ruff check src/DocsToKG/DocParsing

# Type check
mypy src/DocsToKG/DocParsing
```

### Step 4: Commit
```bash
git add src/DocsToKG/DocParsing/chunking/__init__.py
git add src/DocsToKG/DocParsing/embedding/__init__.py
git add tests/docparsing/test_docparsing_core.py
git add tests/content_download/test_atomic_writes.py
git commit -m "DocParsing: Remove legacy re-export compatibility shims

Remove 40+ lines of legacy code explicitly marked as 'compatibility shims':
- chunking/__init__.py: Remove _runtime import + _LEGACY_EXPORTS injection
- embedding/__init__.py: Remove _runtime import + compat name injection

Update tests to use direct imports from .runtime:
- test_docparsing_core.py: Update _LOGGER, extract_refs_and_pages, etc.
- test_atomic_writes.py: Update package-level imports

Impact:
- Cleaner codebase: 40 LOC deleted
- Better clarity: Production exports only
- Better IDE support: Direct imports
- Zero production impact
- All tests passing"
```

---

## Risk Assessment

| Factor | Level | Notes |
|--------|-------|-------|
| Production Impact | NONE | Only affects tests |
| Breaking Changes | NONE | Internal only |
| Test Coverage Impact | LOW | Tests will still pass with updated imports |
| Rollback Difficulty | EASY | Revert commit if needed |
| Overall Risk | **LOW** | Mechanical changes only |

---

## Benefits

✅ **Remove 40 lines of legacy code**  
✅ **Eliminate explicitly marked compatibility shims**  
✅ **Cleaner package interface (production exports only)**  
✅ **Better IDE support (direct imports)**  
✅ **Clearer dependency tracking**  
✅ **Modern Python packaging practices**  
✅ **Zero maintenance burden**  
✅ **Align with goal: "get rid of legacy"**

---

## Status: READY FOR IMPLEMENTATION

**Decision**: DELETE  
**Effort**: ~30-45 minutes  
**Risk**: LOW  
**Impact**: POSITIVE (cleaner codebase)  
**Go/No-Go**: **✅ GO**

