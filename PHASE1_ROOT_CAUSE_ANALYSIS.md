# Phase 1 Implementation: Root Cause Analysis & Solution

## Problem Statement

When calling `extract_archive_safe()` without an explicit `max_uncompressed_bytes` parameter, the function fails with a Pydantic error:

```
PydanticUserError: `ResolvedConfig` is not fully defined; you should define `FetchSpec`, then call `ResolvedConfig.model_rebuild()`.
```

This error occurs during config initialization, not during extraction.

## Root Cause Analysis

### 1. The Forward Reference Chain

**In `settings.py` (line 614):**

```python
class ResolvedConfig(BaseModel):
    defaults: DefaultsConfig
    specs: List["FetchSpec"] = Field(default_factory=list)  # ← Forward reference
```

The `ResolvedConfig` class uses a string annotation `"FetchSpec"` because `FetchSpec` is defined in a different module (`planning.py`).

**In `planning.py`:**

```python
class FetchSpec:
    # ... definition in another module
```

### 2. Pydantic Model Lifecycle

Pydantic v2 requires explicit model rebuilding when:

- A model contains forward references (string annotations) to types defined in other modules
- The forward references haven't been resolved yet at model definition time

**Timeline:**

1. `settings.py` is imported → `ResolvedConfig` is defined with forward ref to `"FetchSpec"`
2. Pydantic marks `ResolvedConfig` as "not fully defined" because it can't find `FetchSpec`
3. `planning.py` is later imported (or not at all during early test execution)
4. When `get_default_config()` is called, it tries to instantiate `ResolvedConfig`
5. Pydantic validation fails because the model is still incomplete

### 3. Why This Affects Our New Code

When `extract_archive_safe()` is called without `max_uncompressed_bytes`:

```python
def extract_archive_safe(..., max_uncompressed_bytes: Optional[int] = None, ...):
    limit_bytes = _resolve_max_uncompressed_bytes(max_uncompressed_bytes)
    # ↓
    # _resolve_max_uncompressed_bytes() calls:
    # get_default_config().defaults.http.max_uncompressed_bytes()
```

This triggers the lazy initialization of the config cache, which fails because `ResolvedConfig` hasn't been properly built.

### 4. Why Passing `max_uncompressed_bytes` "Worked"

When we pass an explicit value:

```python
extract_archive_safe(..., max_uncompressed_bytes=100 * 1024 * 1024)
# ↓
_resolve_max_uncompressed_bytes(100 * 1024 * 1024)
# Returns immediately without calling get_default_config()
```

This bypassed the problematic config initialization entirely—not a real fix.

## The Real Solution: Model Rebuilding in settings.py

### Option 1: Rebuild After FetchSpec Import (Recommended)

Add a rebuild call at the **END** of `settings.py`:

```python
# At the very end of settings.py (after all imports are complete)

# Rebuild ResolvedConfig now that FetchSpec is available from planning module
from . import planning  # Ensure FetchSpec is imported

ResolvedConfig.model_rebuild()
```

**Why this works:**

- ✅ Ensures Pydantic can resolve `"FetchSpec"` forward reference
- ✅ Lazy import (only when settings.py is fully initialized)
- ✅ No invasive changes to code structure
- ✅ No mocking or test-specific workarounds
- ✅ Fixes the root issue, not the symptom

### Option 2: Use TYPE_CHECKING for Forward Ref

Make the import conditional:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .planning import FetchSpec
else:
    FetchSpec = "FetchSpec"  # String forward reference

class ResolvedConfig(BaseModel):
    defaults: DefaultsConfig
    specs: List[FetchSpec] = Field(default_factory=list)

# At end of file:
ResolvedConfig.model_rebuild()
```

Less clean; Option 1 is preferred.

### Option 3: Defer Import in Type Annotation

Already using `List["FetchSpec"]`—just ensure rebuild is called.

## Implementation Plan

### Step 1: Add Model Rebuild to settings.py

At the **end** of `settings.py` (after `get_default_config()` function, before or after other module-level code):

```python
# === Pydantic Model Rebuilding ===
# Ensure ResolvedConfig is fully constructed after FetchSpec is available.
# This resolves the forward reference and allows config initialization to succeed.

def _rebuild_models() -> None:
    """Rebuild Pydantic models that have forward references."""
    from . import planning  # noqa: F401  # Ensure FetchSpec is imported

    ResolvedConfig.model_rebuild()

# Call at module load time
_rebuild_models()
```

### Step 2: Revert max_uncompressed_bytes to Optional in Tests

Remove the workaround of passing explicit `max_uncompressed_bytes`:

```python
# BEFORE (workaround):
extracted = extract_archive_safe(
    archive_path,
    destination,
    extraction_policy=policy,
    max_uncompressed_bytes=100 * 1024 * 1024,  # ← Forced workaround
)

# AFTER (clean):
extracted = extract_archive_safe(
    archive_path,
    destination,
    extraction_policy=policy,
)
```

### Step 3: Verify No Side Effects

Run full test suite to ensure:

- ✅ Config initialization works in all contexts
- ✅ No circular import issues
- ✅ No performance impact
- ✅ Tests pass without workarounds

## Benefits of This Approach

| Aspect | Mocking Approach | Model Rebuild Approach |
|--------|------------------|------------------------|
| **Fixes Root Cause** | ❌ No (only hides symptom) | ✅ Yes |
| **Invasive** | ✅ Very (adds test setup) | ❌ Minimal (one function) |
| **Maintainability** | ❌ Hard (workaround scattered) | ✅ Easy (centralized fix) |
| **Production Impact** | ❌ None (test-only) | ✅ Fixes real issue |
| **Documentation** | ❌ Unclear why needed | ✅ Clear intent |
| **New Tests** | ✅ Workarounds built in | ✅ Clean API usage |

## Risk Assessment

- **Risk Level**: ⭐ Very Low
- **Circular Imports**: Mitigated by using `from . import planning` after all definitions
- **Performance**: No impact (model_rebuild is one-time at module load)
- **Breaking Changes**: None (API unchanged)

## Next Steps

1. ✅ Add `_rebuild_models()` function to `settings.py`
2. ✅ Remove explicit `max_uncompressed_bytes` from test calls
3. ✅ Run test suite to verify
4. ✅ Document the pattern for future model changes
