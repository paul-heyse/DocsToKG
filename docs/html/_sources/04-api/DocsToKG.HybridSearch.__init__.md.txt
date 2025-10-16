# 1. Module: __init__

This reference documents the DocsToKG module ``DocsToKG.HybridSearch.__init__``.

## 1. Overview

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
