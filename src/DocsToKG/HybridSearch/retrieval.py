"""Deprecated retrieval shim maintained for backward compatibility."""

from __future__ import annotations

import warnings

from .service import (
    ChannelResults,
    HybridSearchAPI,
    HybridSearchService,
    PaginationCheckResult,
    RequestValidationError,
    build_stats_snapshot,
    should_rebuild_index,
    verify_pagination,
)

__all__ = [
    "ChannelResults",
    "HybridSearchAPI",
    "HybridSearchService",
    "PaginationCheckResult",
    "RequestValidationError",
    "build_stats_snapshot",
    "should_rebuild_index",
    "verify_pagination",
]

warnings.warn(
    "DocsToKG.HybridSearch.retrieval is deprecated and will be removed in v0.6.0. "
    "Import HybridSearchService and related helpers from DocsToKG.HybridSearch.service instead.",
    DeprecationWarning,
    stacklevel=2,
)
