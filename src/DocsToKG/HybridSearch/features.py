"""
Compatibility layer for the retired ``DocsToKG.HybridSearch.features`` module.

This shim is gated behind the ``DOCSTOKG_HYBRID_FEATURES_SHIM`` environment
variable. By default the module raises an ImportError instructing callers to
switch to ``DocsToKG.HybridSearch.devtools.features``. Setting the environment
variable to a truthy value temporarily re-enables the shim (and emits a
DeprecationWarning) so teams can smoke-test outstanding imports.
"""

from __future__ import annotations

import importlib
import os
import warnings

_ALLOW_SHIM = os.getenv("DOCSTOKG_HYBRID_FEATURES_SHIM")

if not _ALLOW_SHIM or _ALLOW_SHIM.strip().lower() in {"", "0", "false", "no"}:
    raise ImportError(
        "DocsToKG.HybridSearch.features has been retired. Import from "
        "DocsToKG.HybridSearch.devtools.features instead. "
        "Set DOCSTOKG_HYBRID_FEATURES_SHIM=1 temporarily if you must rely on "
        "the legacy shim while migrating."
    )

warnings.warn(
    "DocsToKG.HybridSearch.features is deprecated; import from "
    "DocsToKG.HybridSearch.devtools.features instead.",
    DeprecationWarning,
    stacklevel=2,
)

_devtools_features = importlib.import_module(".devtools.features", package=__package__)
__all__ = getattr(
    _devtools_features,
    "__all__",
    [name for name in dir(_devtools_features) if not name.startswith("_")],
)
globals().update({name: getattr(_devtools_features, name) for name in __all__})
