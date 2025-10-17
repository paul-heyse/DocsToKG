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
    from DocsToKG.HybridSearch import HybridSearchService
    from DocsToKG.HybridSearch.config import HybridSearchConfig

    # Configure search service
    config = HybridSearchConfig()
    service = HybridSearchService(config)

    # Perform hybrid search
    results = service.search("machine learning algorithms")
"""

from __future__ import annotations

# --- Globals ---

__all__ = (
    "AdapterStats",
    "ChunkPayload",
    "ChunkIngestionPipeline",
    "DocumentInput",
    "FaissRouter",
    "FaissVectorStore",
    "HybridSearchAPI",
    "HybridSearchConfigManager",
    "HybridSearchRequest",
    "HybridSearchResponse",
    "HybridSearchResult",
    "HybridSearchService",
    "HybridSearchValidator",
    "ManagedFaissAdapter",
    "Observability",
    "restore_state",
    "serialize_state",
)


# --- Re-exports ---

from .config import HybridSearchConfigManager
from .pipeline import ChunkIngestionPipeline, Observability
from .router import FaissRouter
from .service import HybridSearchAPI, HybridSearchService, HybridSearchValidator
from .store import (
    AdapterStats,
    FaissVectorStore,
    ManagedFaissAdapter,
    restore_state,
    serialize_state,
)
from .types import (
    ChunkPayload,
    DocumentInput,
    HybridSearchRequest,
    HybridSearchResponse,
    HybridSearchResult,
)
