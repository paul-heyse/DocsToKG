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

import sys

from .config import HybridSearchConfig, HybridSearchConfigManager
from .features import FeatureGenerator
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
from .ranking import ReciprocalRankFusion, apply_mmr_diversification
from .service import HybridSearchAPI, HybridSearchService
from .vectorstore import FaissIndexManager
from .storage import OpenSearchIndexTemplate, OpenSearchSchemaManager
from .types import (
    ChunkPayload,
    DocumentInput,
    HybridSearchRequest,
    HybridSearchResponse,
    HybridSearchResult,
    vector_uuid_to_faiss_int,
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
    "ChunkPayload",
    "DocumentInput",
    "Observability",
    "PaginationCheckResult",
    "build_stats_snapshot",
    "serialize_state",
    "restore_state",
    "verify_pagination",
    "should_rebuild_index",
    "vector_uuid_to_faiss_int",
    "ReciprocalRankFusion",
    "apply_mmr_diversification",
]

_package = sys.modules[__name__]
sys.modules.setdefault(__name__ + ".ids", sys.modules[__name__ + ".types"])
sys.modules.setdefault(__name__ + ".schema", sys.modules[__name__ + ".storage"])
sys.modules.setdefault(__name__ + ".similarity_gpu", sys.modules[__name__ + ".vectorstore"])
