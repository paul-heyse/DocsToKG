"""Deprecated feature helpers retained for backward compatibility."""

from __future__ import annotations

import warnings as _warnings

from .devtools.features import (  # noqa: F401
    FeatureGenerator,
    sliding_window,
    tokenize,
    tokenize_with_spans,
)

__all__ = ("FeatureGenerator", "sliding_window", "tokenize", "tokenize_with_spans")

_warnings.warn(
    "DocsToKG.HybridSearch.features is deprecated; "
    "import from DocsToKG.HybridSearch.devtools.features instead.",
    DeprecationWarning,
    stacklevel=2,
)
