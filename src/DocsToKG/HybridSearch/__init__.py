"""Hybrid retrieval module for DocsToKG."""
from __future__ import annotations

from .api import HybridSearchAPI
from .config import HybridSearchConfig, HybridSearchConfigManager
from .dense import FaissIndexManager
from .features import FeatureGenerator
from .fusion import ReciprocalRankFusion, apply_mmr_diversification
from .ingest import ChunkIngestionPipeline
from .observability import Observability
from .schema import OpenSearchIndexTemplate, OpenSearchSchemaManager
from .operations import (
    PaginationCheckResult,
    build_stats_snapshot,
    restore_state,
    serialize_state,
    should_rebuild_index,
    verify_pagination,
)
from .retrieval import HybridSearchService
from .types import (
    ChunkPayload,
    DocumentInput,
    HybridSearchRequest,
    HybridSearchResponse,
    HybridSearchResult,
)
from .validation import HybridSearchValidator

__all__ = [
    "ChunkIngestionPipeline",
    "HybridSearchAPI",
    "FeatureGenerator",
    "FaissIndexManager",
    "HybridSearchConfig",
    "HybridSearchConfigManager",
    "HybridSearchRequest",
    "HybridSearchResponse",
    "HybridSearchResult",
    "HybridSearchService",
    "HybridSearchValidator",
    "OpenSearchSchemaManager",
    "OpenSearchIndexTemplate",
    "ReciprocalRankFusion",
    "apply_mmr_diversification",
    "ChunkPayload",
    "DocumentInput",
    "Observability",
    "PaginationCheckResult",
    "build_stats_snapshot",
    "serialize_state",
    "restore_state",
    "verify_pagination",
    "should_rebuild_index",
]
