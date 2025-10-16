# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.HybridSearch",
#   "purpose": "Hybrid search public API facade",
#   "sections": []
# }
# === /NAVMAP ===

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

# --- Globals ---

__all__ = (
    "ChunkIngestionPipeline",
    "ChunkPayload",
    "DocumentInput",
    "FaissVectorStore",
    "FaissIndexManager",
    "ManagedFaissAdapter",
    "FaissRouter",
    "HybridSearchAPI",
    "HybridSearchConfig",
    "HybridSearchConfigManager",
    "HybridSearchRequest",
    "HybridSearchResponse",
    "HybridSearchResult",
    "HybridSearchService",
    "HybridSearchValidator",
    "LexicalIndex",
    "Observability",
    "PaginationCheckResult",
    "ReciprocalRankFusion",
    "ResultShaper",
    "apply_mmr_diversification",
    "build_stats_snapshot",
    "restore_state",
    "serialize_state",
    "should_rebuild_index",
    "verify_pagination",
    "vector_uuid_to_faiss_int",
)


# --- Re-exports ---

from .config import HybridSearchConfig, HybridSearchConfigManager
from .ingest import ChunkIngestionPipeline
from .interfaces import LexicalIndex
from .observability import Observability
from .ranking import ReciprocalRankFusion, ResultShaper, apply_mmr_diversification
from .router import FaissRouter
from .service import (
    HybridSearchAPI,
    HybridSearchService,
    PaginationCheckResult,
    build_stats_snapshot,
    should_rebuild_index,
    verify_pagination,
)
from .types import (
    ChunkPayload,
    DocumentInput,
    HybridSearchRequest,
    HybridSearchResponse,
    HybridSearchResult,
    vector_uuid_to_faiss_int,
)
from .validation import HybridSearchValidator
from .vectorstore import FaissIndexManager, FaissVectorStore, ManagedFaissAdapter, restore_state, serialize_state


# --- Package Setup ---

_package = sys.modules[__name__]
sys.modules.setdefault(__name__ + ".ids", sys.modules[__name__ + ".types"])
sys.modules.setdefault(__name__ + ".similarity_gpu", sys.modules[__name__ + ".vectorstore"])
