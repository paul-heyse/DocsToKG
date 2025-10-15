"""
Hybrid Search Module for DocsToKG

This module provides comprehensive hybrid search capabilities for DocsToKG,
combining traditional text search (BM25) with dense vector similarity search
and advanced fusion techniques for optimal retrieval performance.

The module includes components for:
- Document ingestion and chunking
- Multi-modal search (lexical + semantic)
- Result fusion and ranking
- Configuration management
- API interfaces
- Observability and monitoring

Key Features:
- Hybrid retrieval combining BM25 and dense embeddings
- GPU-accelerated vector search using FAISS
- Configurable fusion strategies (RRF, MMR)
- Real-time observability and metrics
- Scalable architecture for large document collections

Dependencies:
- faiss: Vector similarity search
- sentence-transformers: Dense embedding generation
- elasticsearch: Optional for additional search capabilities
- redis: Caching and session management

Usage:
    from docstokg.hybrid_search import HybridSearchService, HybridSearchConfig

    # Configure search service
    config = HybridSearchConfig()
    service = HybridSearchService(config)

    # Perform hybrid search
    results = service.search("machine learning algorithms")
"""

from __future__ import annotations

from .api import HybridSearchAPI
from .config import HybridSearchConfig, HybridSearchConfigManager
from .dense import FaissIndexManager
from .features import FeatureGenerator
from .fusion import ReciprocalRankFusion, apply_mmr_diversification
from .ingest import ChunkIngestionPipeline
from .observability import Observability
from .operations import (
    PaginationCheckResult,
    build_stats_snapshot,
    restore_state,
    serialize_state,
    should_rebuild_index,
    verify_pagination,
)
from .retrieval import HybridSearchService
from .schema import OpenSearchIndexTemplate, OpenSearchSchemaManager
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
