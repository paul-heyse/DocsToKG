"""Deprecated result shaping shim maintained for backward compatibility."""

from __future__ import annotations

import warnings

from .ranking import ResultShaper

__all__ = ["ResultShaper"]

warnings.warn(
    "DocsToKG.HybridSearch.results is deprecated and will be removed in v0.6.0. "
    "Import ResultShaper from DocsToKG.HybridSearch.ranking instead.",
    DeprecationWarning,
    stacklevel=2,
)
