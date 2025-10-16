"""Deprecated schema shim maintained for backward compatibility."""

from __future__ import annotations

import warnings

from .storage import OpenSearchIndexTemplate, OpenSearchSchemaManager

__all__ = ["OpenSearchIndexTemplate", "OpenSearchSchemaManager"]

warnings.warn(
    "DocsToKG.HybridSearch.schema is deprecated and will be removed in v0.6.0. "
    "Import OpenSearch schema helpers from DocsToKG.HybridSearch.storage instead.",
    DeprecationWarning,
    stacklevel=2,
)
