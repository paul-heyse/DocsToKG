# Module: gpu_patch

Compatibility shim for legacy GPU monkey patch imports.

Older services imported ``DocsToKG.HybridSearch.gpu_patch`` to force GPU-only
behaviour. The functionality now lives directly inside the core modules, but
this stub keeps the import path valid. ``apply_patch`` is a no-op retained for
backwards compatibility.

## Functions

### `apply_patch()`

Preserve legacy GPU patch API while performing no operation.

Args:
None

Returns:
None
