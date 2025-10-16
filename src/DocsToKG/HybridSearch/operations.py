"""Deprecated module providing shims for relocated service and vectorstore helpers."""

from __future__ import annotations

import warnings

from .service import (
    PaginationCheckResult,
    build_stats_snapshot,
    should_rebuild_index,
    verify_pagination,
)
from .vectorstore import restore_state, serialize_state

__all__ = [
    "PaginationCheckResult",
    "build_stats_snapshot",
    "restore_state",
    "serialize_state",
    "should_rebuild_index",
    "verify_pagination",
]

warnings.warn(
    "DocsToKG.HybridSearch.operations is deprecated and will be removed in v0.6.0. "
    "Import pagination helpers from DocsToKG.HybridSearch.service and vectorstore state "
    "utilities from DocsToKG.HybridSearch.vectorstore.",
    DeprecationWarning,
    stacklevel=2,
)
