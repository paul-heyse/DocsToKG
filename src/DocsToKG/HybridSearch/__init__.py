# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.HybridSearch",
#   "purpose": "Hybrid search public API facade",
#   "sections": []
# }
# === /NAVMAP ===

"""
DocsToKG.HybridSearch exposes the ingestion, storage, and query primitives that
back the DocsToKG hybrid search reference implementation. The package couples a
custom FAISS 1.12 GPU wheel (see ``faiss-gpu-wheel-reference.md``) with lexical
retrieval, observability, and configuration helpers so agents can stand up or
extend the end-to-end retrieval pipeline without reverse engineering internals.

Core modules and how they interrelate:

- ``config`` defines the dataclasses that configure chunking, FAISS GPU indexes,
  fusion heuristics, and retrieval budgets. ``DenseIndexConfig`` maps directly
  onto FAISS constructs such as ``GpuMultipleClonerOptions``, tiling limits, and
  ``GpuDistanceParams`` toggles (FP16, cuVS) used by ``store.ManagedFaissAdapter``.
- ``pipeline`` ingests DocParsing artifacts, normalises lexical/dense features,
  and emits instrumentation. It is the canonical reference for how sparse and
  dense payloads enter the system before reaching the FAISS adapters.
- ``store`` owns the FAISS GPU lifecycle: it trains CPU indexes, migrates them
  to GPUs via ``index_cpu_to_gpu(_multiple)``, manages ``StandardGpuResources``
  pools, and offers cosine / inner-product helpers built on ``knn_gpu`` and
  ``pairwise_distance_gpu``. Snapshot/restore utilities here underpin cold-start
  and failover flows used by both ingestion and service layers.
- ``service`` orchestrates synchronous query handling, launching lexical and
  FAISS GPU searches in parallel and fusing results with Reciprocal Rank Fusion
  plus optional MMR diversification. It consumes ``HybridSearchConfig`` and the
  adapters exposed by ``store`` and ``router``.
- ``router`` maintains namespace-aware caches of managed FAISS stores, handing
  back live adapters or rehydrating snapshots on demand to keep GPU memory usage
  bounded across tenants.
- ``types`` and ``interfaces`` provide the shared contracts that ensure the
  ingestion pipeline, FAISS adapters, and query API speak the same language.
- ``devtools`` contains deterministic feature generators and an in-memory
  OpenSearch simulator that implement the ``interfaces`` protocols for testing.

Agents extending the system should start by reading the relevant module
docstrings plus the FAISS wheel reference. Together they describe the GPU
resource model, expected tensor shapes/dtypes, and failure-handling guarantees
that downstream code assumes.
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
