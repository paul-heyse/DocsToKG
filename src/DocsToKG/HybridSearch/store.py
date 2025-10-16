# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.HybridSearch.store",
#   "purpose": "Vector store orchestration, OpenSearch helpers, GPU similarity utilities, and persistence",
#   "sections": [
#     {
#       "id": "faisssearchresult",
#       "name": "FaissSearchResult",
#       "anchor": "class-faisssearchresult",
#       "kind": "class"
#     },
#     {
#       "id": "adapterstats",
#       "name": "AdapterStats",
#       "anchor": "class-adapterstats",
#       "kind": "class"
#     },
#     {
#       "id": "pendingsearch",
#       "name": "_PendingSearch",
#       "anchor": "class-pendingsearch",
#       "kind": "class"
#     },
#     {
#       "id": "searchcoalescer",
#       "name": "_SearchCoalescer",
#       "anchor": "class-searchcoalescer",
#       "kind": "class"
#     },
#     {
#       "id": "faissvectorstore",
#       "name": "FaissVectorStore",
#       "anchor": "class-faissvectorstore",
#       "kind": "class"
#     },
#     {
#       "id": "normalize-rows",
#       "name": "normalize_rows",
#       "anchor": "function-normalize-rows",
#       "kind": "function"
#     },
#     {
#       "id": "cosine-against-corpus-gpu",
#       "name": "cosine_against_corpus_gpu",
#       "anchor": "function-cosine-against-corpus-gpu",
#       "kind": "function"
#     },
#     {
#       "id": "pairwise-inner-products",
#       "name": "pairwise_inner_products",
#       "anchor": "function-pairwise-inner-products",
#       "kind": "function"
#     },
#     {
#       "id": "max-inner-product",
#       "name": "max_inner_product",
#       "anchor": "function-max-inner-product",
#       "kind": "function"
#     },
#     {
#       "id": "cosine-batch",
#       "name": "cosine_batch",
#       "anchor": "function-cosine-batch",
#       "kind": "function"
#     },
#     {
#       "id": "cosine-topk-blockwise",
#       "name": "cosine_topk_blockwise",
#       "anchor": "function-cosine-topk-blockwise",
#       "kind": "function"
#     },
#     {
#       "id": "serialize-state",
#       "name": "serialize_state",
#       "anchor": "function-serialize-state",
#       "kind": "function"
#     },
#     {
#       "id": "restore-state",
#       "name": "restore_state",
#       "anchor": "function-restore-state",
#       "kind": "function"
#     },
#     {
#       "id": "chunkregistry",
#       "name": "ChunkRegistry",
#       "anchor": "class-chunkregistry",
#       "kind": "class"
#     },
#     {
#       "id": "matches-filters",
#       "name": "matches_filters",
#       "anchor": "function-matches-filters",
#       "kind": "function"
#     },
#     {
#       "id": "getattr",
#       "name": "__getattr__",
#       "anchor": "function-getattr",
#       "kind": "function"
#     },
#     {
#       "id": "managedfaissadapter",
#       "name": "ManagedFaissAdapter",
#       "anchor": "class-managedfaissadapter",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""Unified FAISS vector store, GPU similarity utilities, and state helpers."""

from __future__ import annotations

import base64
import logging
import time
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event, RLock
from typing import Callable, Dict, Iterator, List, Mapping, Optional, Sequence

import numpy as np

from .config import DenseIndexConfig
from .interfaces import DenseVectorStore
from .opensearch_simulator import OpenSearchSimulator
from .pipeline import Observability
from .types import ChunkPayload, vector_uuid_to_faiss_int

# --- Globals ---

logger = logging.getLogger(__name__)

__all__ = (
    "FaissVectorStore",
    "ManagedFaissAdapter",
    "AdapterStats",
    "FaissSearchResult",
    "ChunkRegistry",
    "OpenSearchSimulator",
    "matches_filters",
    "cosine_against_corpus_gpu",
    "cosine_batch",
    "cosine_topk_blockwise",
    "max_inner_product",
    "normalize_rows",
    "pairwise_inner_products",
    "restore_state",
    "serialize_state",
)

try:  # pragma: no cover - exercised via integration tests
    import faiss  # type: ignore

    _FAISS_AVAILABLE = all(
        hasattr(faiss, attr)
        for attr in (
            "GpuIndexFlatIP",
            "IndexIDMap2",
            "index_cpu_to_gpu",
            "index_gpu_to_cpu",
            "StandardGpuResources",
            "pairwise_distance_gpu",
        )
    )
except Exception:  # pragma: no cover - dependency not present in test rig
    faiss = None  # type: ignore
    _FAISS_AVAILABLE = False


# --- Public Classes ---


@dataclass(slots=True)
class FaissSearchResult:
    """Dense search hit returned by FAISS.

    Attributes:
        vector_id: External identifier resolved from the FAISS internal id.
        score: Similarity score reported by FAISS (inner product).

    Examples:
        >>> FaissSearchResult(vector_id="doc-123", score=0.82)
        FaissSearchResult(vector_id='doc-123', score=0.82)
    """

    vector_id: str
    score: float


@dataclass(frozen=True)
class AdapterStats:
    """Snapshot of runtime characteristics for a managed FAISS adapter."""

    device: int
    ntotal: int
    index_description: str
    nprobe: int
    multi_gpu_mode: str
    replicated: bool
    fp16_enabled: bool
    resources: Optional["faiss.StandardGpuResources"]


class _PendingSearch:
    """Container representing a pending single-query search awaiting batching."""

    __slots__ = ("vector", "top_k", "_event", "_result", "_error")

    def __init__(self, vector: np.ndarray, top_k: int) -> None:
        self.vector = np.asarray(vector, dtype=np.float32).copy()
        self.top_k = int(top_k)
        self._event = Event()
        self._result: Optional[List[FaissSearchResult]] = None
        self._error: Optional[BaseException] = None

    def set_result(self, result: Sequence[FaissSearchResult]) -> None:
        """Fulfil the pending search with ``result`` and release any waiters."""
        self._result = list(result)
        self._event.set()

    def set_exception(self, exc: BaseException) -> None:
        """Attach ``exc`` to the pending search and release any waiters."""
        self._error = exc
        self._event.set()

    def wait(self) -> List[FaissSearchResult]:
        """Block until a result or exception is produced for this search."""
        self._event.wait()
        if self._error is not None:
            raise self._error
        if self._result is None:
            return []
        return self._result


class _SearchCoalescer:
    """Batcher that groups singleton searches into short-lived micro-batches."""

    def __init__(
        self,
        store: "FaissVectorStore",
        *,
        window_ms: float = 2.0,
        max_batch: int = 32,
    ) -> None:
        self._store = store
        self._window = max(0.0, float(window_ms)) / 1000.0
        self._lock = RLock()
        self._pending: List[_PendingSearch] = []
        self._max_batch = max(1, int(max_batch))
        self._metrics = store._observability.metrics

    def submit(self, vector: np.ndarray, top_k: int) -> List[FaissSearchResult]:
        """Coalesce a singleton search request and return its results."""
        request = _PendingSearch(vector, top_k)
        with self._lock:
            self._pending.append(request)
            is_leader = len(self._pending) == 1
        if is_leader:
            if self._window:
                time.sleep(self._window)
            batch = self._drain()
            self._execute(batch)
            return request.wait()
        return request.wait()

    def _drain(self) -> List[_PendingSearch]:
        with self._lock:
            if not self._pending:
                return []
            batch = self._pending[: self._max_batch]
            self._pending = self._pending[self._max_batch :]
        return batch

    def _execute(self, batch: List[_PendingSearch]) -> None:
        if not batch:
            return
        vectors = [self._store._ensure_dim(item.vector) for item in batch]
        k_max = max(item.top_k for item in batch)
        matrix = np.stack(vectors, dtype=np.float32)
        try:
            results = self._store._search_batch_impl(matrix, k_max)
        except Exception as exc:
            for pending in batch:
                pending.set_exception(exc)
            trailing = self._drain()
            for pending in trailing:
                pending.set_exception(exc)
            raise
        else:
            for pending, row in zip(batch, results):
                pending.set_result(row[: pending.top_k])
            if self._store._observability is not None:
                self._store._observability.metrics.observe(
                    "faiss_coalesced_batch_size", float(len(batch))
                )
            rate = 0.0 if len(batch) <= 1 else float(len(batch) - 1) / float(len(batch))
            self._metrics.set_gauge("faiss_coalescer_hit_rate", rate)
            # If additional requests arrived while executing, process them next.
            trailing = self._drain()
            if trailing:
                self._execute(trailing)


class FaissVectorStore(DenseVectorStore):
    """Manage a GPU-backed FAISS index for dense retrieval.

    The store encapsulates FAISS initialisation, GPU resource management,
    vector ingestion, deletion, and similarity search. It understands the
    custom CUDA-enabled FAISS build shipped with DocsToKG (``faiss-1.12``)
    and automatically mirrors configuration options defined in
    :class:`DenseIndexConfig`.

    Attributes:
        _dim: Dimensionality of stored vectors.
        _config: Active dense index configuration.
        _index: Underlying FAISS GPU index (possibly wrapped in ``IndexIDMap2``).
        _gpu_resources: Lazily initialised ``StandardGpuResources`` instance.
        _device: CUDA device id derived from configuration.

    Examples:
        >>> store = FaissVectorStore(dim=1536, config=DenseIndexConfig())
        >>> store.add([np.random.rand(1536)], ["chunk-1"])
        >>> results = store.search(np.random.rand(1536), top_k=5)
        >>> [hit.vector_id for hit in results]
        ['chunk-1']
    """

    @classmethod
    def create(
        cls,
        dim: int,
        config: DenseIndexConfig,
        *,
        observability: Optional[Observability] = None,
    ) -> "FaissVectorStore":
        """Factory helper matching the managed FAISS interface contracts."""

        return cls(dim=dim, config=config, observability=observability)

    def __init__(
        self,
        dim: int,
        config: DenseIndexConfig,
        *,
        observability: Optional[Observability] = None,
    ) -> None:
        """Initialise the vector store and allocate GPU resources.

        Args:
            dim: Dimensionality of dense embeddings stored in the index.
            config: Dense index configuration controlling layout and GPU flags.

        Returns:
            None

        Raises:
            RuntimeError: If the CUDA-enabled FAISS build is unavailable or no
                GPU devices are visible to the process.
        """
        if not _FAISS_AVAILABLE:
            raise RuntimeError(
                "FAISS GPU extensions are required for dense retrieval "
                "(missing GpuIndexFlatIP/index_cpu_to_gpu). Verify that faiss-gpu is installed "
                "and CUDA libraries are discoverable."
            )

        self._dim = dim
        self._config = config
        self._device = self._resolve_device(config)
        self._multi_gpu_mode = getattr(config, "multi_gpu_mode", "single")
        self._indices_32_bit = bool(getattr(config, "gpu_indices_32_bit", True))
        self._temp_memory_bytes = getattr(config, "gpu_temp_memory_bytes", None)
        self._expected_ntotal = int(getattr(config, "expected_ntotal", 0))
        self._rebuild_delete_threshold = int(getattr(config, "rebuild_delete_threshold", 10000))
        self._force_64bit_ids = bool(getattr(config, "force_64bit_ids", False))
        self._replication_enabled = bool(getattr(config, "enable_replication", True))
        self._reserve_memory_enabled = bool(getattr(config, "enable_reserve_memory", True))
        self._replicated = False
        self._gpu_resources: Optional["faiss.StandardGpuResources"] = None
        self._pinned_buffers: list[object] = []
        self._observability = observability or Observability()
        self.init_gpu()
        self._lock = RLock()
        self._index = self._create_index()
        if self._dim != dim:
            raise RuntimeError(
                f"HybridSearch initialised with dim={dim} but created index expects {self._dim}"
            )
        self._id_resolver: Optional[Callable[[int], Optional[str]]] = None
        self._remove_fallbacks = 0
        self._tombstones: set[int] = set()
        self._dirty_deletes = 0
        self._needs_rebuild = False
        self._supports_remove_ids: Optional[bool] = None
        self._last_nprobe_update = 0.0
        self._set_nprobe()
        self._search_coalescer = _SearchCoalescer(self)
        self._cpu_replica: Optional[bytes] = None
        self._cpu_replica_meta: Optional[Mapping[str, object]] = None
        self._refresh_cpu_replica()
        self._emit_gpu_state("bootstrap", level="info")

    @property
    def ntotal(self) -> int:
        """Number of vectors currently stored in the FAISS index.

        Args:
            None

        Returns:
            int: Total number of stored vectors.
        """
        return int(self._index.ntotal)

    @property
    def config(self) -> DenseIndexConfig:
        """Return the resolved dense index configuration.

        Args:
            None

        Returns:
            DenseIndexConfig: Active dense index configuration.
        """
        return self._config

    @property
    def dim(self) -> int:
        """Return the dimensionality of vectors stored in the FAISS index.

        Args:
            None

        Returns:
            int: Dimensionality of embeddings managed by the index.
        """

        return self._dim

    @property
    def gpu_resources(self) -> "faiss.StandardGpuResources | None":
        """Expose the underlying FAISS GPU resource manager.

        Args:
            None

        Returns:
            faiss.StandardGpuResources | None: GPU resource manager when initialised.
        """
        return self._gpu_resources

    def get_gpu_resources(self) -> Optional["faiss.StandardGpuResources"]:
        """Compatibility helper returning active GPU resources."""

        return self._gpu_resources

    @property
    def device(self) -> int:
        """Return the CUDA device id used for the FAISS index.

        Args:
            None

        Returns:
            int: Configured CUDA device identifier.
        """
        return int(getattr(self._config, "device", self._device))

    @property
    def adapter_stats(self) -> AdapterStats:
        """Return a read-only snapshot of adapter state."""

        index_desc = self._describe_index(getattr(self, "_index", None))
        return AdapterStats(
            device=self.device,
            ntotal=self.ntotal,
            index_description=index_desc,
            nprobe=self._current_nprobe_value(),
            multi_gpu_mode=str(self._multi_gpu_mode),
            replicated=bool(self._replicated),
            fp16_enabled=bool(getattr(self._config, "flat_use_fp16", False)),
            resources=self._gpu_resources,
        )

    def set_id_resolver(self, resolver: Callable[[int], Optional[str]]) -> None:
        """Register a callback that maps FAISS internal ids to external ids.

        Args:
            resolver: Callable receiving the FAISS integer id and returning the
                application-level identifier (or ``None`` when unresolved).

        Returns:
            None
        """
        self._id_resolver = resolver

    def train(self, vectors: Sequence[np.ndarray]) -> None:
        """Train IVF style indexes using the provided sample vectors.

        Args:
            vectors: Sequence of dense vectors used to train IVF quantizers.

        Raises:
            ValueError: If no vectors are supplied for a trainable index type.

        Returns:
            None
        """
        if not hasattr(self._index, "is_trained"):
            return
        if getattr(self._index, "is_trained"):
            return
        if not vectors:
            raise ValueError("Training vectors required for IVF indexes")
        with self._observability.trace("faiss_train", samples=str(len(vectors))):
            self._observability.metrics.increment("faiss_train_calls", amount=1.0)
            matrix = np.stack([self._ensure_dim(vec) for vec in vectors]).astype(np.float32)
            matrix = self._as_pinned(np.ascontiguousarray(matrix, dtype=np.float32))
            try:
                faiss.normalize_L2(matrix)
                nlist = int(getattr(self._config, "nlist", 1024))
                factor = max(1, int(getattr(self._config, "ivf_train_factor", 8)))
                ntrain = min(matrix.shape[0], nlist * factor)
                self._index.train(matrix[:ntrain])
            finally:
                self._release_pinned_buffers()
        self._update_gpu_metrics()
        self._refresh_cpu_replica()

    def needs_training(self) -> bool:
        """Return ``True`` when the current FAISS index still requires training.

        Args:
            None

        Returns:
            bool: ``True`` when additional training is required.
        """
        is_trained = getattr(self._index, "is_trained", True)
        return not bool(is_trained)

    def add(self, vectors: Sequence[np.ndarray], vector_ids: Sequence[str]) -> None:
        """Add vectors to the FAISS index, replacing existing ids if present.

        Args:
            vectors: Embeddings to insert into the index.
            vector_ids: Application-level identifiers paired with each vector.

        Raises:
            ValueError: When the vector and id sequences differ in length.

        Returns:
            None
        """
        original_count = len(vector_ids)
        with self._observability.trace("faiss_add", count=str(original_count)):
            if len(vectors) != len(vector_ids):
                raise ValueError("vectors and vector_ids must align")
            if not vectors:
                return
            faiss_ids = np.asarray(
                [vector_uuid_to_faiss_int(vid) for vid in vector_ids], dtype=np.int64
            )
            matrix = np.ascontiguousarray(
                np.stack([self._ensure_dim(vec) for vec in vectors]), dtype=np.float32
            )
            faiss.normalize_L2(matrix)
            dedupe_threshold = float(getattr(self._config, "ingest_dedupe_threshold", 0.0))
            dropped = 0
            if dedupe_threshold > 0.0 and self.ntotal > 0:
                try:
                    distances, indices = self._search_matrix(matrix, 1)
                except Exception:
                    logger.debug(
                        "ingest dedupe check failed; proceeding without filter", exc_info=True
                    )
                else:
                    keep_mask = []
                    for idx, score in zip(indices.ravel(), distances.ravel()):
                        keep_mask.append(idx == -1 or float(score) < dedupe_threshold)
                    if not any(keep_mask):
                        self._observability.metrics.increment(
                            "faiss_ingest_deduped", amount=float(matrix.shape[0])
                        )
                        return
                    if not all(keep_mask):
                        keep_mask_arr = np.asarray(keep_mask, dtype=bool)
                        dropped = int(np.count_nonzero(~keep_mask_arr))
                        matrix = np.ascontiguousarray(matrix[keep_mask_arr], dtype=np.float32)
                        faiss_ids = faiss_ids[keep_mask_arr]
                        self._observability.metrics.increment(
                            "faiss_ingest_deduped", amount=float(dropped)
                        )
            matrix = self._as_pinned(matrix)
            faiss.normalize_L2(matrix)
            try:
                with self._lock:
                    self._flush_pending_deletes(force=False)
                    if self._supports_remove_ids is None:
                        self._supports_remove_ids = self._probe_remove_support()
                    if self._supports_remove_ids:
                        self.remove_ids(faiss_ids, force_flush=True)
                    else:
                        existing_ids = self._lookup_existing_ids(faiss_ids)
                        if existing_ids.size:
                            self.remove_ids(existing_ids, force_flush=True)
                    base = self._index.index if hasattr(self._index, "index") else self._index
                    train_target = base
                    if hasattr(faiss, "downcast_index"):
                        try:
                            train_target = faiss.downcast_index(base)
                        except Exception:
                            train_target = base
                    if hasattr(train_target, "is_trained") and not getattr(
                        train_target, "is_trained"
                    ):
                        nlist = int(getattr(self._config, "nlist", 1024))
                        factor = max(1, int(getattr(self._config, "ivf_train_factor", 8)))
                        ntrain = min(matrix.shape[0], nlist * factor)
                        train_target.train(matrix[:ntrain])
                    self._index.add_with_ids(matrix, faiss_ids)
                    self._dirty_deletes = 0
                    self._needs_rebuild = False
            finally:
                self._release_pinned_buffers()
            self._update_gpu_metrics()
            inserted = matrix.shape[0]
            if inserted:
                self._observability.metrics.increment("faiss_add_vectors", amount=float(inserted))
                self._refresh_cpu_replica()

    def remove(self, vector_ids: Sequence[str]) -> None:
        """Remove vectors from the index using application-level identifiers.

        Args:
            vector_ids: Sequence of vector ids scheduled for deletion.

        Returns:
            None
        """
        if not vector_ids:
            return
        count = len(vector_ids)
        ids = np.array([vector_uuid_to_faiss_int(vid) for vid in vector_ids], dtype=np.int64)
        with self._observability.trace("faiss_remove", count=str(count)):
            self._observability.metrics.increment("faiss_remove_vectors", amount=float(count))
            with self._lock:
                self.remove_ids(ids, force_flush=True)
        self._update_gpu_metrics()
        self._refresh_cpu_replica()

    def remove_ids(self, ids: np.ndarray, *, force_flush: bool = False) -> int:
        """Remove vectors using FAISS internal ids.

        Args:
            ids: Array of FAISS integer ids to delete.
            force_flush: Force immediate rebuild when tombstones accumulate.

        Returns:
            Number of ids scheduled for removal.
        """
        with self._lock:
            ids64 = np.asarray(ids, dtype=np.int64)
            if ids64.size == 0:
                return 0
        self._remove_ids(ids64)
        self._flush_pending_deletes(force=force_flush)
        self._update_gpu_metrics()
        return int(ids64.size)

    def _current_index_ids(self) -> np.ndarray:
        # Robustly extract id_map across wrappers / GPU clones.
        if self._index is None or self._index.ntotal == 0:
            return np.empty(0, dtype=np.int64)
        try:
            id_map = getattr(self._index, "id_map", None)
            if id_map is not None:
                return np.asarray(faiss.vector_to_array(id_map), dtype=np.int64)
        except Exception:
            pass
        try:
            cpu = self._to_cpu(self._index)
            id_map_cpu = getattr(cpu, "id_map", None)
            if id_map_cpu is not None:
                return np.asarray(faiss.vector_to_array(id_map_cpu), dtype=np.int64)
        except Exception:
            logger.debug("Unable to enumerate FAISS id_map; returning empty set", exc_info=True)
        return np.empty(0, dtype=np.int64)

    def _lookup_existing_ids(self, candidate_ids: np.ndarray) -> np.ndarray:
        if candidate_ids.size == 0 or self._index.ntotal == 0:
            return np.empty(0, dtype=np.int64)
        current_ids = self._current_index_ids()
        mask = np.isin(current_ids, candidate_ids)
        return current_ids[mask]

    def search(self, query: np.ndarray, top_k: int) -> List[FaissSearchResult]:
        """Search the index for the ``top_k`` nearest neighbours of ``query``."""

        vector = self._ensure_dim(query)
        with self._observability.trace("faiss_search", top_k=str(top_k)):
            if self._search_coalescer is not None:
                return self._search_coalescer.submit(vector, top_k)
            results = self._search_batch_impl(vector.reshape(1, -1), top_k)
            return results[0] if results else []

    def search_many(self, queries: np.ndarray, top_k: int) -> List[List[FaissSearchResult]]:
        """Search the index for multiple queries in a single FAISS call."""

        return self.search_batch(queries, top_k)

    def search_batch(self, queries: np.ndarray, top_k: int) -> List[List[FaissSearchResult]]:
        """Alias for ``search_many`` to support explicit batch workloads."""

        matrix = np.atleast_2d(np.ascontiguousarray(queries, dtype=np.float32))
        if matrix.shape[1] != self._dim:
            raise ValueError(f"Query dimensionality {matrix.shape[1]} != index dim {self._dim}")
        if matrix.shape[0] == 1 and self._search_coalescer is not None:
            hits = self._search_coalescer.submit(matrix[0], top_k)
            return [hits]
        return self._search_batch_impl(matrix, top_k)

    def _search_batch_impl(self, matrix: np.ndarray, top_k: int) -> List[List[FaissSearchResult]]:
        matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        if matrix.shape[1] != self._dim:
            raise ValueError(f"Query dimensionality {matrix.shape[1]} != index dim {self._dim}")
        matrix = self._as_pinned(matrix)
        faiss.normalize_L2(matrix)
        batch = matrix.shape[0]
        try:
            with self._observability.trace(
                "faiss_search_batch", batch=str(batch), top_k=str(top_k)
            ):
                self._observability.metrics.increment("faiss_search_queries", amount=float(batch))
                distances, indices = self._search_matrix(matrix, top_k)
            results = self._resolve_search_results(distances, indices)
            total_results = float(sum(len(row) for row in results))
            self._observability.metrics.observe("faiss_search_results", total_results)
            return results
        finally:
            self._release_pinned_buffers()

    def range_search(
        self,
        query: np.ndarray,
        min_score: float,
        *,
        limit: Optional[int] = None,
    ) -> List[FaissSearchResult]:
        """Return all vectors scoring above ``min_score`` for ``query``."""

        vector = self._ensure_dim(query)
        matrix = np.ascontiguousarray(vector.reshape(1, -1), dtype=np.float32)
        matrix = self._as_pinned(matrix)
        faiss.normalize_L2(matrix)
        threshold = float(min_score)
        results: List[FaissSearchResult] = []
        try:
            with self._observability.trace(
                "faiss_range_search",
                threshold=f"{threshold:.6f}",
                limit="*" if limit is None else str(int(limit)),
            ):
                self._observability.metrics.increment("faiss_range_queries", amount=1.0)
                pairs: List[tuple[float, int]]
                try:
                    with self._lock:
                        self._flush_pending_deletes(force=False)
                        self._set_nprobe()
                        if not hasattr(self._index, "range_search"):
                            raise AttributeError("range_search unsupported by index")
                        lims, distances, labels = self._index.range_search(matrix, threshold)
                except Exception:
                    fallback_k = limit or max(32, min(self.ntotal, 2048))
                    distances, indices = self._search_matrix(matrix, int(fallback_k))
                    pairs = [
                        (float(score), int(internal_id))
                        for score, internal_id in zip(distances[0], indices[0])
                        if internal_id != -1 and float(score) >= threshold
                    ]
                else:
                    start = int(lims[0])
                    end = int(lims[1])
                    pairs = [
                        (float(distances[i]), int(labels[i]))
                        for i in range(start, end)
                        if labels[i] != -1 and float(distances[i]) >= threshold
                    ]
                scored: List[tuple[float, str]] = []
                for score, internal_id in pairs:
                    vector_id = self._resolve_vector_id(int(internal_id))
                    if vector_id is None:
                        continue
                    scored.append((score, vector_id))
                scored.sort(key=lambda item: (-item[0], item[1]))
                if limit is not None:
                    scored = scored[: int(limit)]
                for score, vector_id in scored:
                    results.append(FaissSearchResult(vector_id=vector_id, score=float(score)))
                self._observability.metrics.observe("faiss_range_results", float(len(results)))
            return results
        finally:
            self._release_pinned_buffers()

    def _search_matrix(self, matrix: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            self._flush_pending_deletes(force=False)
            self._set_nprobe()
            return self._index.search(matrix, int(top_k))

    def _as_pinned(self, array: np.ndarray, *, threshold_bytes: int = 1_048_576) -> np.ndarray:
        """Return a pinned-memory view of ``array`` when practical."""

        if not bool(getattr(self._config, "use_pinned_memory", True)):
            return array
        if array.nbytes < threshold_bytes:
            return array
        try:
            import torch
        except Exception:
            return array
        if not torch.cuda.is_available():
            return array
        dtype_map = {
            np.float32: torch.float32,
            np.float16: torch.float16,
            np.int32: torch.int32,
            np.int64: torch.int64,
        }
        torch_dtype = dtype_map.get(array.dtype.type)
        if torch_dtype is None:
            return array
        tensor = torch.empty(tuple(array.shape), dtype=torch_dtype, pin_memory=True)
        source = torch.as_tensor(array, dtype=torch_dtype)
        tensor.copy_(source)
        self._pinned_buffers.append(tensor)
        return tensor.numpy()

    def _release_pinned_buffers(self) -> None:
        if self._pinned_buffers:
            self._pinned_buffers.clear()

    def _refresh_cpu_replica(self) -> None:
        if getattr(self._config, "persist_mode", "cpu_bytes") == "disabled":
            return
        try:
            payload = self.serialize()
        except Exception:  # pragma: no cover - best effort
            logger.debug("Unable to refresh CPU replica", exc_info=True)
            return
        self._cpu_replica = payload
        self._cpu_replica_meta = self.snapshot_meta()

    def promote_cpu_replica(self) -> bool:
        """Promote the cached CPU replica back onto the GPU index."""

        replica = self._cpu_replica
        if replica is None:
            return False
        meta = self._cpu_replica_meta
        try:
            self.restore(replica, meta=meta)
        except Exception:  # pragma: no cover - recovery path
            self._observability.logger.exception(
                "faiss-promote-cpu-replica",
                extra={"event": {"status": "failed"}},
            )
            return False
        self._cpu_replica = replica
        self._cpu_replica_meta = meta
        self._observability.logger.info(
            "faiss-promote-cpu-replica",
            extra={"event": {"status": "success"}},
        )
        return True

    def _current_nprobe_value(self) -> int:
        index = getattr(self, "_index", None)
        candidates = []
        if index is not None:
            candidates.append(index)
            base = getattr(index, "index", None)
            if base is not None:
                candidates.append(base)
        for candidate in candidates:
            value = getattr(candidate, "nprobe", None)
            if value is not None:
                try:
                    return int(value)
                except Exception:
                    continue
        return int(getattr(self._config, "nprobe", 0))

    def _update_gpu_metrics(self) -> None:
        stats = self.adapter_stats
        self._observability.metrics.set_gauge("faiss_ntotal", float(stats.ntotal))
        self._observability.metrics.set_gauge("faiss_nprobe_effective", float(stats.nprobe))
        resources = stats.resources
        if resources is None or not hasattr(resources, "getMemoryInfo"):
            return
        try:
            free, total = resources.getMemoryInfo(int(stats.device))
        except Exception:
            logger.debug("Unable to fetch GPU memory info", exc_info=True)
            return
        self._observability.metrics.set_gauge("faiss_gpu_mem_free_bytes", float(free))
        self._observability.metrics.set_gauge("faiss_gpu_mem_total_bytes", float(total))
        self._observability.metrics.set_gauge("faiss_gpu_bytes", float(max(0.0, total - free)))

    def _emit_gpu_state(self, action: str, *, level: str = "debug") -> None:
        self._update_gpu_metrics()
        stats = self.adapter_stats
        payload = {
            "action": action,
            "device": stats.device,
            "index": stats.index_description,
            "replicated": stats.replicated,
            "multi_gpu_mode": stats.multi_gpu_mode,
            "fp16_enabled": stats.fp16_enabled,
            "indices_32_bit": bool(self._indices_32_bit and not self._force_64bit_ids),
            "nprobe": stats.nprobe,
            "ntotal": stats.ntotal,
        }
        log_fn = (
            self._observability.logger.info if level == "info" else self._observability.logger.debug
        )
        log_fn("faiss-gpu-state", extra={"event": payload})

    def _describe_index(self, index: Optional["faiss.Index"]) -> str:
        if index is None:
            return "uninitialised"
        try:
            return faiss.describe_index(index)
        except Exception:
            return type(index).__name__

    def _resolve_search_results(
        self, distances: np.ndarray, indices: np.ndarray
    ) -> List[List[FaissSearchResult]]:
        results: List[List[FaissSearchResult]] = []
        for row_scores, row_ids in zip(distances, indices):
            row: List[FaissSearchResult] = []
            for score, internal_id in zip(row_scores, row_ids):
                if internal_id == -1:
                    continue
                vector_id = self._resolve_vector_id(int(internal_id))
                if vector_id is None:
                    continue
                row.append(FaissSearchResult(vector_id=vector_id, score=float(score)))
            results.append(row)
        return results

    def serialize(self) -> bytes:
        """Return a CPU-serialised representation of the FAISS index.

        Args:
            None

        Returns:
            Byte payload produced by :func:`faiss.serialize_index`.

        Raises:
            RuntimeError: If the index has not been initialised.
        """
        with self._observability.trace("faiss_serialize"):
            with self._lock:
                if self._index is None:
                    raise RuntimeError("index is empty")
                cpu_index = self._to_cpu(self._index)
                blob = faiss.serialize_index(cpu_index)
                payload = bytes(blob)
            self._observability.metrics.increment("faiss_serialize_calls", amount=1.0)
            return payload

    def save(self, path: str) -> None:
        """Persist the FAISS index to ``path`` when persistence is enabled.

        Args:
            path: Filesystem destination for the serialised index.

        Returns:
            None

        Raises:
            RuntimeError: If the index has not been initialised.
        """
        with self._observability.trace("faiss_save", path=str(path)):
            if self._index is None:
                raise RuntimeError("index is empty")
            if getattr(self._config, "persist_mode", "cpu_bytes") == "disabled":
                logger.info("faiss-save-skip", extra={"event": {"reason": "persist_mode=disabled"}})
                return
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(self.serialize())
            self._observability.metrics.increment("faiss_save_calls", amount=1.0)

    @classmethod
    def load(
        cls,
        path: str,
        config: DenseIndexConfig,
        dim: int,
        *,
        observability: Optional[Observability] = None,
    ) -> "FaissVectorStore":
        """Restore a vector store from disk."""

        obs = observability or Observability()
        with obs.trace("faiss_load", path=str(path)):
            blob = Path(path).read_bytes()
            manager = cls(dim=dim, config=config, observability=obs)
            manager.restore(blob)
            manager._observability.metrics.increment("faiss_load_calls", amount=1.0)
        return manager

    def restore(
        self,
        payload: bytes,
        *,
        meta: Optional[Mapping[str, object]] = None,
    ) -> None:
        """Load an index from ``payload`` and promote it to the GPU.

        Args:
            payload: Bytes produced by :meth:`serialize`.
            meta: Optional snapshot metadata for validation.

        Raises:
            ValueError: If the payload is empty.

        Returns:
            None
        """
        with self._observability.trace("faiss_restore"):
            if not payload:
                raise ValueError("Empty FAISS payload")
            if meta:
                self._validate_snapshot_meta(meta)
            with self._lock:
                cpu_index = faiss.deserialize_index(np.frombuffer(payload, dtype=np.uint8))
                self._index = self._maybe_to_gpu(cpu_index)
                self._tombstones.clear()
                self._dirty_deletes = 0
                self._needs_rebuild = False
                self._set_nprobe()
            self._observability.metrics.increment("faiss_restore_calls", amount=1.0)
        if getattr(self._config, "persist_mode", "cpu_bytes") != "disabled":
            self._cpu_replica = bytes(payload)
            self._cpu_replica_meta = self.snapshot_meta()
        self._emit_gpu_state("restore", level="info")

    def set_nprobe(
        self,
        nprobe: int,
        *,
        clamp_min: int = 1,
        clamp_max: Optional[int] = None,
    ) -> int:
        """Update the active ``nprobe`` value and propagate it to the index."""

        if clamp_min <= 0:
            raise ValueError("clamp_min must be positive")
        target = max(clamp_min, int(nprobe))
        if clamp_max is not None:
            if clamp_max < clamp_min:
                raise ValueError("clamp_max must be >= clamp_min")
            target = min(target, int(clamp_max))
        current = int(getattr(self._config, "nprobe", target))
        if target == current:
            return target
        now = time.time()
        if self._last_nprobe_update and now - self._last_nprobe_update < 60.0:
            self._observability.metrics.increment("faiss_nprobe_rate_limited", amount=1.0)
            self._observability.logger.debug(
                "faiss-nprobe-rate-limit",
                extra={"event": {"previous": current, "requested": target}},
            )
            return current
        logger.info(
            "faiss-nprobe-update",
            extra={
                "event": {"previous": current, "current": target, "index": self._config.index_type}
            },
        )
        self._config = replace(self._config, nprobe=target)
        with self._observability.trace("faiss_set_nprobe", nprobe=str(target)):
            with self._lock:
                self._set_nprobe()
                self._last_nprobe_update = now
        self._observability.metrics.set_gauge("faiss_nprobe", float(target))
        self._emit_gpu_state("set_nprobe")
        return target

    def stats(self) -> dict[str, float | str]:
        """Return diagnostic metrics describing the active FAISS index.

        Args:
            None

        Returns:
            Mapping of human-readable metric names to values (counts or strings).
        """
        base = getattr(self._index, "index", None) or self._index
        if hasattr(faiss, "downcast_index"):
            try:
                base = faiss.downcast_index(base)
            except Exception:  # pragma: no cover - best effort introspection
                pass
        stats: dict[str, float | str] = {
            "ntotal": float(self.ntotal),
            "index_type": self._config.index_type,
            "nlist": float(getattr(self._config, "nlist", 0)),
            "nprobe": float(getattr(self._config, "nprobe", 0)),
            "pq_m": float(getattr(self._config, "pq_m", 0)),
            "pq_bits": float(getattr(self._config, "pq_bits", 0)),
            "ingest_dedupe_threshold": float(getattr(self._config, "ingest_dedupe_threshold", 0.0)),
            "gpu_remove_fallbacks": float(self._remove_fallbacks),
            "multi_gpu_mode": self._multi_gpu_mode,
            "gpu_indices_32_bit": bool(self._indices_32_bit and not self._force_64bit_ids),
            "gpu_device": str(getattr(self._config, "device", 0)),
            "device": str(getattr(self._config, "device", 0)),
            "fp16_enabled": bool(getattr(self._config, "flat_use_fp16", False)),
            "cpu_replica_cached": bool(self._cpu_replica is not None),
            "cpu_replica_bytes": float(len(self._cpu_replica)) if self._cpu_replica else 0.0,
        }
        if self._temp_memory_bytes is not None:
            stats["gpu_temp_memory_bytes"] = float(self._temp_memory_bytes)
        try:
            stats["gpu_base"] = "GpuIndex" in type(base).__name__
            device = self._detect_device(self._index)
            if device is not None:
                stats["gpu_device"] = str(device)
                stats["device"] = str(device)
                resources = self._gpu_resources
                if resources is not None and hasattr(resources, "getMemoryInfo"):
                    free, total = resources.getMemoryInfo(device)
                    stats["gpu_mem_free"] = float(free)
                    stats["gpu_mem_total"] = float(total)
            stats["dirty_deletes"] = float(getattr(self, "_dirty_deletes", 0))
        except Exception:  # pragma: no cover - diagnostic only
            logger.debug("Unable to gather complete FAISS stats", exc_info=True)
        return stats

    def snapshot_meta(self) -> dict[str, object]:
        """Return metadata describing the configuration backing the index."""

        return {
            "dim": int(self._dim),
            "index_type": str(self._config.index_type),
            "nlist": int(getattr(self._config, "nlist", 0)),
            "pq_m": int(getattr(self._config, "pq_m", 0)),
            "pq_bits": int(getattr(self._config, "pq_bits", 0)),
            "nprobe": int(getattr(self._config, "nprobe", 0)),
            "gpu_indices_32_bit": bool(self._indices_32_bit and not self._force_64bit_ids),
            "force_64bit_ids": bool(self._force_64bit_ids),
            "flat_use_fp16": bool(getattr(self._config, "flat_use_fp16", False)),
            "multi_gpu_mode": str(self._multi_gpu_mode),
            "device": int(getattr(self._config, "device", 0)),
        }

    def rebuild_needed(self) -> bool:
        """Return ``True`` when tombstones require a full FAISS rebuild.

        Args:
            None

        Returns:
            bool: ``True`` when a rebuild should be triggered.
        """
        if self._needs_rebuild:
            return True
        if self._rebuild_delete_threshold <= 0:
            return False
        return self._dirty_deletes >= self._rebuild_delete_threshold

    def rebuild_if_needed(self) -> bool:
        """Trigger a rebuild when tombstones exceed thresholds.

        Returns:
            bool: ``True`` if a rebuild was executed.
        """

        with self._observability.trace("faiss_rebuild_check"):
            with self._lock:
                if not self.rebuild_needed():
                    return False
                self._rebuild_index()
                self._set_nprobe()
                self._needs_rebuild = False
            self._observability.metrics.increment("faiss_rebuilds", amount=1.0)
            self._refresh_cpu_replica()
            return True

    def init_gpu(self) -> None:
        """Initialise FAISS GPU resources for the configured CUDA device.

        Args:
            None

        Raises:
            RuntimeError: When GPU support is unavailable or the requested
                device id is invalid.

        Returns:
            None
        """
        if self._gpu_resources is not None:
            return
        if not hasattr(faiss, "get_num_gpus"):
            raise RuntimeError(
                "HybridSearch requires a CUDA-enabled faiss build exposing get_num_gpus(). "
                "Install faiss-gpu>=1.9 with CUDA support."
            )
        available = int(faiss.get_num_gpus())
        if available <= 0:
            raise RuntimeError(
                "HybridSearch requires at least one visible CUDA device. "
                "Verify GPU drivers and CUDA visibility."
            )
        device = int(self.device)
        if device < 0 or device >= available:
            raise RuntimeError(
                f"HybridSearch configured for CUDA device {device}, but only {available} GPU(s) are available. "
                "Update DenseIndexConfig.device or adjust CUDA visibility."
            )
        try:
            resources = faiss.StandardGpuResources()
            if self._temp_memory_bytes is not None and hasattr(resources, "setTempMemory"):
                resources.setTempMemory(self._temp_memory_bytes)
            self._gpu_resources = resources
        except Exception as exc:  # pragma: no cover - GPU-specific failure
            raise RuntimeError(
                "HybridSearch failed to initialise FAISS GPU resources. "
                "Check CUDA driver installation and faiss-gpu compatibility."
            ) from exc

    def distribute_to_all_gpus(self, index: "faiss.Index", *, shard: bool = False) -> "faiss.Index":
        """Clone ``index`` across available GPUs when the build supports it.

        Args:
            index: FAISS index to replicate or shard.
            shard: When ``True`` attempt sharded replication if supported.

        Returns:
            FAISS index after attempting replication/sharding.

        Raises:
            RuntimeError: If sharding is requested but unsupported by the
                linked FAISS build.
        """
        self._replicated = False
        if not self._replication_enabled:
            return index
        if not hasattr(faiss, "index_cpu_to_all_gpus") or not hasattr(faiss, "index_cpu_to_gpu"):
            return index
        if shard and hasattr(faiss, "index_cpu_to_all_gpus_knn"):
            return faiss.index_cpu_to_all_gpus(index, co=None, cloner_options=None, shard=True)
        if shard and hasattr(faiss, "index_cpu_to_gpu_multiple"):
            raise RuntimeError("Sharded multi-GPU not supported by current faiss build")
        if shard:
            return index
        if self._replicated:
            return index
        try:
            cloned = faiss.index_gpu_to_cpu(index)
            opts = faiss.GpuMultipleClonerOptions()
            opts.shard = False
            opts.useFloat16 = bool(getattr(self._config, "flat_use_fp16", False))
            multi = faiss.index_cpu_to_all_gpus(cloned, opts)
            self._replicated = True
            return multi
        except Exception:  # pragma: no cover - GPU-specific failure
            logger.warning("Unable to replicate FAISS index across GPUs", exc_info=True)
            return index

    def _maybe_to_gpu(self, index: "faiss.Index") -> "faiss.Index":
        self.init_gpu()
        if self._gpu_resources is None:
            raise RuntimeError(
                "HybridSearch could not allocate FAISS GPU resources; GPU-backed indexes are mandatory."
            )
        device = int(self.device)
        try:
            if hasattr(faiss, "GpuClonerOptions"):
                co = faiss.GpuClonerOptions()
                co.device = device
                co.verbose = True
                co.allowCpuCoarseQuantizer = False
                if (
                    not self._force_64bit_ids
                    and self._indices_32_bit
                    and hasattr(faiss, "INDICES_32_BIT")
                ):
                    co.indicesOptions = faiss.INDICES_32_BIT
                promoted = faiss.index_cpu_to_gpu(self._gpu_resources, device, index, co)
                return (
                    self.distribute_to_all_gpus(promoted, shard=self._multi_gpu_mode == "shard")
                    if self._multi_gpu_mode in ("replicate", "shard")
                    else promoted
                )
            cloned = faiss.index_cpu_to_gpu(self._gpu_resources, device, index)
            return (
                self.distribute_to_all_gpus(cloned, shard=self._multi_gpu_mode == "shard")
                if self._multi_gpu_mode in ("replicate", "shard")
                else cloned
            )
        except Exception as exc:  # pragma: no cover - hardware specific failure
            raise RuntimeError(
                "Failed to promote FAISS index to GPU "
                f"(index type={type(index).__name__}, device={device}): {exc}"
            ) from exc

    def _maybe_reserve_memory(self, index: "faiss.Index") -> None:
        expected = self._expected_ntotal
        if expected <= 0:
            return
        if not self._reserve_memory_enabled:
            return
        base = index.index if hasattr(index, "index") else index
        if hasattr(faiss, "downcast_index"):
            try:
                base = faiss.downcast_index(base)
            except Exception:  # pragma: no cover - best effort
                pass
        if hasattr(base, "reserveMemory"):
            try:  # pragma: no cover - optimisation path
                base.reserveMemory(int(expected))
            except Exception:
                logger.debug(
                    "Unable to reserve FAISS GPU memory",
                    extra={"event": {"expected_ntotal": expected}},
                    exc_info=True,
                )

    def _to_cpu(self, index: "faiss.Index") -> "faiss.Index":
        if not hasattr(faiss, "index_gpu_to_cpu"):
            raise RuntimeError(
                "FAISS index_gpu_to_cpu is unavailable; install faiss-gpu>=1.7.4 with GPU support."
            )
        try:
            return faiss.index_gpu_to_cpu(index)
        except Exception as exc:  # pragma: no cover - hardware specific failure
            raise RuntimeError(
                f"Unable to transfer FAISS index from GPU to CPU for serialization: {exc}"
            ) from exc

    def _set_nprobe(self) -> None:
        index = getattr(self, "_index", None)
        if index is None or not self._config.index_type.startswith("ivf"):
            return
        nprobe = int(self._config.nprobe)
        try:
            if hasattr(faiss, "GpuParameterSpace"):
                gps = faiss.GpuParameterSpace()
                gps.set_index_parameter(index, "nprobe", nprobe)
                self._log_index_configuration(index)
                return
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("Unable to set nprobe via GpuParameterSpace", exc_info=True)
        base = getattr(index, "index", None) or index
        try:
            if hasattr(base, "nprobe"):
                base.nprobe = nprobe
            elif hasattr(index, "nprobe"):
                index.nprobe = nprobe
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("Unable to set nprobe attribute on FAISS index", exc_info=True)
        self._log_index_configuration(index)

    def _log_index_configuration(self, index: "faiss.Index") -> None:
        try:
            desc = faiss.describe_index(index)
        except Exception:  # pragma: no cover - best effort
            desc = type(index).__name__
        logger.debug(
            "faiss-index-config",
            extra={"event": {"nprobe": getattr(self._config, "nprobe", 0), "index": desc}},
        )

    def _create_index(self) -> "faiss.Index":
        self.init_gpu()
        metric = faiss.METRIC_INNER_PRODUCT
        index_type = self._config.index_type

        dev = int(self.device)
        if index_type == "flat":
            cfg = faiss.GpuIndexFlatConfig() if hasattr(faiss, "GpuIndexFlatConfig") else None
            if cfg is not None:
                cfg.device = dev
                if hasattr(cfg, "useFloat16"):
                    cfg.useFloat16 = bool(getattr(self._config, "flat_use_fp16", False))
                if hasattr(cfg, "indicesOptions"):
                    cfg.indicesOptions = (
                        faiss.INDICES_32_BIT
                        if self._indices_32_bit and not self._force_64bit_ids
                        else 0
                    )
                index = faiss.GpuIndexFlatIP(self._gpu_resources, self._dim, cfg)
            else:
                index = faiss.GpuIndexFlatIP(
                    self._gpu_resources, self._dim, faiss.METRIC_INNER_PRODUCT
                )
            self._maybe_reserve_memory(index)
            return faiss.IndexIDMap2(index)

        if index_type == "ivf_flat":
            if not hasattr(faiss, "GpuIndexIVFFlat"):
                raise RuntimeError("faiss build missing GpuIndexIVFFlat support")
            cfg = faiss.GpuIndexIVFFlatConfig()
            cfg.device = dev
            if hasattr(cfg, "indicesOptions"):
                cfg.indicesOptions = (
                    faiss.INDICES_32_BIT
                    if self._indices_32_bit and not self._force_64bit_ids
                    else 0
                )
            index = faiss.GpuIndexIVFFlat(
                self._gpu_resources, self._dim, int(self._config.nlist), metric, cfg
            )
            self._maybe_reserve_memory(index)
            return faiss.IndexIDMap2(index)
        if index_type == "ivf_pq":
            if not hasattr(faiss, "GpuIndexIVFPQ"):
                raise RuntimeError("faiss build missing GpuIndexIVFPQ support")
            cfg = faiss.GpuIndexIVFPQConfig()
            cfg.device = dev
            if hasattr(cfg, "usePrecomputedTables"):
                cfg.usePrecomputedTables = bool(
                    getattr(self._config, "ivfpq_use_precomputed", True)
                )
            if hasattr(cfg, "useFloat16LookupTables"):
                cfg.useFloat16LookupTables = bool(getattr(self._config, "ivfpq_float16_lut", True))
            if hasattr(cfg, "interleavedLayout"):
                cfg.interleavedLayout = bool(getattr(self._config, "interleaved_layout", True))
            if hasattr(cfg, "indicesOptions"):
                cfg.indicesOptions = (
                    faiss.INDICES_32_BIT
                    if self._indices_32_bit and not self._force_64bit_ids
                    else 0
                )
            index = faiss.GpuIndexIVFPQ(
                self._gpu_resources,
                self._dim,
                int(self._config.nlist),
                int(self._config.pq_m),
                int(self._config.pq_bits),
                metric,
                cfg,
            )
            self._maybe_reserve_memory(index)
            return faiss.IndexIDMap2(index)

        raise RuntimeError(f"Unsupported or unimplemented FAISS index type: {index_type}")

    def _probe_remove_support(self) -> bool:
        selector = faiss.IDSelectorBatch(np.empty(0, dtype=np.int64))
        try:
            self._index.remove_ids(selector)
        except RuntimeError as exc:
            if "remove_ids not implemented" in str(exc).lower():
                return False
            raise
        return True

    def _remove_ids(self, ids: np.ndarray) -> None:
        if ids.size == 0:
            return
        selector = faiss.IDSelectorBatch(ids.astype(np.int64))
        try:
            removed = int(self._index.remove_ids(selector))
            if self._supports_remove_ids is None:
                self._supports_remove_ids = True
        except RuntimeError as exc:
            if "remove_ids not implemented" not in str(exc).lower():
                raise
            self._supports_remove_ids = False
            logger.warning(
                "FAISS remove_ids not implemented on GPU index; scheduling rebuild.",
                extra={"event": {"ntotal": self.ntotal, "error": str(exc)}},
            )
            remaining = ids
        else:
            if removed >= int(ids.size):
                return
            current = self._current_index_ids()
            remaining = ids[np.isin(ids, current)]
            if remaining.size == 0:
                return
        self._remove_fallbacks += 1
        self._tombstones.update(int(v) for v in remaining.tolist())
        self._dirty_deletes += int(remaining.size)
        threshold = int(self._rebuild_delete_threshold)
        if threshold > 0 and self._dirty_deletes >= threshold:
            self._rebuild_index()
            self._tombstones.clear()
            self._dirty_deletes = 0
            self._needs_rebuild = False
        else:
            self._needs_rebuild = True

    def _flush_pending_deletes(self, *, force: bool) -> None:
        if not self._tombstones:
            return
        if not force and not self.rebuild_needed():
            return
        threshold = int(self._rebuild_delete_threshold)
        if force or threshold <= 0 or self._dirty_deletes >= threshold:
            self._rebuild_index()
            self._tombstones.clear()
            self._dirty_deletes = 0
            self._needs_rebuild = False

    def _rebuild_index(self) -> None:
        old_index = self._index
        base = old_index.index if hasattr(old_index, "index") else old_index
        current_ids = self._current_index_ids()
        if current_ids.size == 0:
            self._index = self._create_index()
            self._set_nprobe()
            return
        if self._tombstones:
            tombstone_array = np.fromiter(self._tombstones, dtype=np.int64)
            keep_mask = ~np.isin(current_ids, tombstone_array)
        else:
            keep_mask = np.ones_like(current_ids, dtype=bool)
        keep_positions = np.nonzero(keep_mask)[0].astype(np.int64)
        survivor_ids = current_ids[keep_mask]
        if base.ntotal != len(current_ids):  # pragma: no cover - defensive guard
            raise RuntimeError("FAISS id_map out of sync with underlying index")
        keys = np.ascontiguousarray(keep_positions, dtype=np.int64)
        vectors = (
            base.reconstruct_batch(keys)
            if keys.size
            else np.empty((0, self._dim), dtype=np.float32)
        )
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self._index = self._create_index()
        self._set_nprobe()
        if vectors.size:
            base_new = self._index.index if hasattr(self._index, "index") else self._index
            train_target = base_new
            if hasattr(faiss, "downcast_index"):
                try:
                    train_target = faiss.downcast_index(base_new)
                except Exception:
                    train_target = base_new
            if hasattr(train_target, "is_trained") and not getattr(train_target, "is_trained"):
                nlist = int(getattr(self._config, "nlist", 1024))
                factor = max(1, int(getattr(self._config, "ivf_train_factor", 8)))
                ntrain = min(vectors.shape[0], nlist * factor)
                train_target.train(vectors[:ntrain])
            self._index.add_with_ids(vectors, survivor_ids.astype(np.int64))

    def _resolve_vector_id(self, internal_id: int) -> Optional[str]:
        if self._id_resolver is None:
            return str(internal_id)
        try:
            return self._id_resolver(internal_id)
        except Exception:  # pragma: no cover - best effort
            logger.debug("Failed to resolve FAISS vector id", exc_info=True)
            return None

    def _ensure_dim(self, vector: np.ndarray) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError("vector must be 1-dimensional")
        if arr.size != self._dim:
            raise ValueError(f"vector dimension mismatch: expected {self._dim}, got {arr.size}")
        return arr

    def _validate_snapshot_meta(self, meta: Mapping[str, object]) -> None:
        expected = self.snapshot_meta()
        mismatches: dict[str, tuple[object, object]] = {}
        for key, expected_value in expected.items():
            if key not in meta:
                continue
            raw_value = meta[key]
            try:
                if isinstance(expected_value, bool):
                    actual_value = bool(raw_value)
                elif isinstance(expected_value, int) and not isinstance(expected_value, bool):
                    actual_value = int(raw_value)
                elif isinstance(expected_value, float):
                    actual_value = float(raw_value)
                else:
                    actual_value = str(raw_value)
            except Exception:
                actual_value = raw_value
            if actual_value != expected_value:
                mismatches[key] = (expected_value, actual_value)
        if mismatches:
            raise RuntimeError(f"FAISS snapshot meta mismatch: {mismatches}")

    def _detect_device(self, index: "faiss.Index") -> Optional[int]:
        try:
            if hasattr(index, "getDevice"):
                return int(index.getDevice())
            base = getattr(index, "index", None)
            if base is not None and hasattr(base, "getDevice"):
                return int(base.getDevice())
        except Exception:  # pragma: no cover - best effort
            logger.debug("Failed to detect FAISS device", exc_info=True)
        return None

    def _resolve_device(self, config: DenseIndexConfig) -> int:
        return int(getattr(config, "device", 0))


# --- Public Functions ---


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise each row of ``matrix`` in-place.

    Args:
        matrix: Contiguous float32 matrix whose rows represent vectors.

    Returns:
        Normalised matrix (same object as ``matrix``).

    Raises:
        TypeError: If ``matrix`` is not a contiguous float32 ``ndarray``.
    """
    if matrix.dtype != np.float32 or not matrix.flags.c_contiguous:
        raise TypeError("normalize_rows expects a contiguous float32 array")
    if not hasattr(faiss, "normalize_L2"):
        raise RuntimeError(
            "FAISS normalize_L2 unavailable; ensure faiss-gpu is installed with CUDA support."
        )
    faiss.normalize_L2(matrix)
    return matrix


def cosine_against_corpus_gpu(
    query: np.ndarray,
    corpus: np.ndarray,
    *,
    device: int = 0,
    resources: Optional["faiss.StandardGpuResources"] = None,
) -> np.ndarray:
    """Compute cosine similarity between ``query`` and each vector in ``corpus`` on GPU.

    Args:
        query: Query vector or matrix (``N x D``) to compare.
        corpus: Matrix representing the corpus (``M x D``).
        device: CUDA device id used for the computation.
        resources: FAISS GPU resources to execute the kernel.

    Returns:
        numpy.ndarray: Matrix of cosine similarities with shape ``(N, M)``.

    Raises:
        RuntimeError: If GPU resources are unavailable.
        ValueError: When the query and corpus dimensionality mismatch.
    """
    if resources is None:
        raise RuntimeError("FAISS GPU resources are required for cosine comparisons")
    if query.ndim == 1:
        query = query.reshape(1, -1)
    if query.shape[1] != corpus.shape[1]:
        raise ValueError("Query and corpus dimensionality must match")
    sims = cosine_batch(query, corpus, device=device, resources=resources)
    return np.asarray(sims, dtype=np.float32)


def pairwise_inner_products(
    a: np.ndarray,
    b: Optional[np.ndarray] = None,
    *,
    device: int = 0,
    resources: Optional["faiss.StandardGpuResources"] = None,
) -> np.ndarray:
    """Compute cosine similarities between two corpora on GPU.

    Args:
        a: First matrix of vectors.
        b: Optional second matrix; when omitted ``a`` is used (symmetrical case).
        device: CUDA device id to run the computation.
        resources: FAISS GPU resources backing the computation.

    Returns:
        numpy.ndarray: Pairwise similarity matrix with rows from ``a`` and columns from ``b``.

    Raises:
        RuntimeError: If GPU resources are unavailable.
        ValueError: When input matrices have mismatched dimensionality.
    """
    if resources is None:
        raise RuntimeError("FAISS GPU resources are required for cosine comparisons")
    if a.size == 0:
        if b is None:
            return np.zeros((0, 0), dtype=np.float32)
        return np.zeros((0, b.shape[0]), dtype=np.float32)
    if b is None:
        b = a
    if a.shape[1] != b.shape[1]:
        raise ValueError("Input matrices must share the same dimensionality")
    sims = cosine_batch(a, b, device=device, resources=resources)
    return np.asarray(sims, dtype=np.float32)


def max_inner_product(
    target: np.ndarray,
    corpus: np.ndarray,
    *,
    device: int = 0,
    resources: Optional["faiss.StandardGpuResources"] = None,
) -> float:
    """Return the maximum cosine similarity between ``target`` and ``corpus``.

    Args:
        target: Query vector whose maximum similarity is desired.
        corpus: Corpus matrix used for comparison.
        device: CUDA device id for FAISS computations.
        resources: FAISS GPU resources needed for similarity search.

    Returns:
        float: Maximum cosine similarity value.

    Raises:
        RuntimeError: If GPU resources are unavailable.
    """
    if resources is None:
        raise RuntimeError("FAISS GPU resources are required for cosine comparisons")
    if corpus.size == 0:
        return float("-inf")
    sims = cosine_against_corpus_gpu(target, corpus, device=device, resources=resources)
    return float(np.max(sims))


def cosine_batch(
    q: np.ndarray,
    C: np.ndarray,
    *,
    device: int,
    resources: "faiss.StandardGpuResources",
) -> np.ndarray:
    """Helper that normalises and computes cosine similarities on GPU.

    Args:
        q: Query matrix (``N x D``).
        C: Corpus matrix (``M x D``).
        device: CUDA device id used for computation.
        resources: FAISS GPU resources backing the kernel.

    Returns:
        numpy.ndarray: Pairwise cosine similarities with shape ``(N, M)``.
    """
    if q.ndim == 1:
        q = q.reshape(1, -1)
    q = np.ascontiguousarray(q, dtype=np.float32)
    C = np.ascontiguousarray(C, dtype=np.float32)
    faiss.normalize_L2(q)
    faiss.normalize_L2(C)
    return faiss.pairwise_distance_gpu(
        resources,
        q,
        C,
        metric=faiss.METRIC_INNER_PRODUCT,
        device=int(device),
    )


def cosine_topk_blockwise(
    q: np.ndarray,
    C: np.ndarray,
    *,
    k: int,
    device: int,
    resources: "faiss.StandardGpuResources",
    block_rows: int = 65_536,
    use_fp16: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Top-K cosine similarities between ``q`` and ``C`` using GPU tiling.

    The helper avoids materialising the full ``(N × M)`` similarity matrix by
    iterating over ``C`` in row blocks and maintaining a running Top-K per query
    row. Inputs are copied and normalised inside the routine so callers retain
    ownership of their buffers.

    Args:
        q: Query vector or matrix (``N × D``).
        C: Corpus matrix (``M × D``).
        k: Number of neighbours to return per query row.
        device: CUDA device ordinal used for FAISS kernels.
        resources: FAISS GPU resources backing ``pairwise_distance_gpu``.
        block_rows: Number of corpus rows processed per iteration.

    Returns:
        Tuple ``(scores, indices)`` where each has shape ``(N × K)``. Scores are
        sorted in descending order for every query row and indices reference rows
        within ``C``.
    """

    if resources is None:
        raise RuntimeError("FAISS GPU resources are required for cosine comparisons")
    if block_rows <= 0:
        raise ValueError("block_rows must be positive")

    if q.ndim == 1:
        q = q.reshape(1, -1)
    q = np.ascontiguousarray(q, dtype=np.float32).copy()
    C = np.ascontiguousarray(C, dtype=np.float32)

    if q.shape[1] != C.shape[1]:
        raise ValueError("q and C must have same dimensionality")
    if C.shape[0] == 0:
        empty_scores = np.empty((q.shape[0], 0), dtype=np.float32)
        empty_indices = np.empty((q.shape[0], 0), dtype=np.int64)
        return empty_scores, empty_indices

    if k <= 0:
        raise ValueError("k must be positive")
    k = min(k, C.shape[0])

    faiss.normalize_L2(q)
    q_view = q.astype(np.float16, copy=False) if use_fp16 else q

    N, M = q.shape[0], C.shape[0]
    best_scores = np.full((N, k), -np.inf, dtype=np.float32)
    best_index = np.full((N, k), -1, dtype=np.int64)

    start = 0
    while start < M:
        end = min(M, start + block_rows)
        block = np.array(C[start:end], dtype=np.float32, copy=True)
        faiss.normalize_L2(block)
        block_view = block.astype(np.float16, copy=False) if use_fp16 else block
        sims = faiss.pairwise_distance_gpu(
            resources,
            q_view,
            block_view,
            metric=faiss.METRIC_INNER_PRODUCT,
            device=int(device),
        )

        block_idx = np.arange(start, end, dtype=np.int64)
        cand_scores = np.concatenate([best_scores, sims], axis=1)
        cand_index = np.concatenate(
            [best_index, np.broadcast_to(block_idx, (N, block_idx.size))], axis=1
        )
        select = np.argpartition(cand_scores, -k, axis=1)[:, -k:]
        best_scores = np.take_along_axis(cand_scores, select, axis=1)
        best_index = np.take_along_axis(cand_index, select, axis=1)
        start = end

    order = np.argsort(best_scores, axis=1)[:, ::-1]
    best_scores = np.take_along_axis(best_scores, order, axis=1)
    best_index = np.take_along_axis(best_index, order, axis=1)
    return best_scores, best_index


def serialize_state(faiss_index: FaissVectorStore, registry: "ChunkRegistry") -> dict[str, object]:
    """Serialize the vector store and chunk registry to a JSON-safe payload.

    Args:
        faiss_index: Vector store whose state should be captured.
        registry: Chunk registry providing vector identifier mappings.

    Returns:
        dict[str, object]: Dictionary containing FAISS bytes, registry ids, and snapshot metadata.
    """
    faiss_bytes = faiss_index.serialize()
    encoded = base64.b64encode(faiss_bytes).decode("ascii")
    return {
        "faiss": encoded,
        "vector_ids": registry.vector_ids(),
        "meta": faiss_index.snapshot_meta(),
    }


def restore_state(
    faiss_index: FaissVectorStore,
    payload: dict[str, object],
    *,
    allow_legacy: bool = False,
) -> None:
    """Restore the vector store from a payload produced by :func:`serialize_state`.

    Args:
        faiss_index: Vector store receiving the restored state.
        payload: Mapping with ``faiss`` (base64) and registry vector ids.
        allow_legacy: Permit payloads missing ``meta`` (emits a warning).

    Returns:
        None

    Raises:
        ValueError: If the payload is missing the FAISS byte stream.
    """
    encoded = payload.get("faiss")
    if not isinstance(encoded, str):
        raise ValueError("Missing FAISS payload")
    meta = payload.get("meta")
    if not isinstance(meta, Mapping):
        if not allow_legacy:
            raise ValueError("FAISS snapshot payload missing 'meta'")
        logger.warning("restore_state: legacy payload without 'meta'; defaults will be applied")
        faiss_index.restore(base64.b64decode(encoded.encode("ascii")), meta=None)
        return
    faiss_index.restore(base64.b64decode(encoded.encode("ascii")), meta=dict(meta))


class ChunkRegistry:
    """Durable mapping of vector identifiers to chunk payloads."""

    def __init__(self) -> None:
        self._chunks: Dict[str, ChunkPayload] = {}
        self._bridge: Dict[int, str] = {}

    def upsert(self, chunks: Sequence[ChunkPayload]) -> None:
        """Insert or update registry entries for ``chunks``."""

        for chunk in chunks:
            self._chunks[chunk.vector_id] = chunk
            self._bridge[vector_uuid_to_faiss_int(chunk.vector_id)] = chunk.vector_id

    def delete(self, vector_ids: Sequence[str]) -> None:
        """Remove registry entries for the supplied vector identifiers."""

        for vector_id in vector_ids:
            self._chunks.pop(vector_id, None)
            self._bridge.pop(vector_uuid_to_faiss_int(vector_id), None)

    def get(self, vector_id: str) -> Optional[ChunkPayload]:
        """Return the chunk payload for ``vector_id`` when available."""

        return self._chunks.get(vector_id)

    def bulk_get(self, vector_ids: Sequence[str]) -> List[ChunkPayload]:
        """Return chunk payloads for identifiers present in the registry."""

        return [self._chunks[vid] for vid in vector_ids if vid in self._chunks]

    def resolve_faiss_id(self, internal_id: int) -> Optional[str]:
        """Translate a FAISS integer id back to the original vector identifier."""

        return self._bridge.get(internal_id)

    def all(self) -> List[ChunkPayload]:
        """Return all cached chunk payloads."""

        return list(self._chunks.values())

    def iter_all(self) -> Iterator[ChunkPayload]:
        """Yield chunk payloads without materialising the full list."""

        return iter(self._chunks.values())

    def count(self) -> int:
        """Return the number of chunks tracked by the registry."""

        return len(self._chunks)

    def vector_ids(self) -> List[str]:
        """Return all vector identifiers in insertion order."""

        return list(self._chunks.keys())


def matches_filters(chunk: ChunkPayload, filters: Mapping[str, object]) -> bool:
    """Check whether ``chunk`` satisfies the provided OpenSearch-style filters."""

    for key, expected in filters.items():
        if key == "namespace":
            if chunk.namespace != expected:
                return False
            continue

        value = chunk.metadata.get(key)
        if isinstance(expected, list):
            if isinstance(value, list):
                if not any(item in value for item in expected):
                    return False
            else:
                if value not in expected:
                    return False
        else:
            if value != expected:
                return False
    return True


def __getattr__(name: str):
    if name == "FaissIndexManager":
        warnings.warn(
            "FaissIndexManager is deprecated; import FaissVectorStore instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return FaissVectorStore
    raise AttributeError(name)


class ManagedFaissAdapter(DenseVectorStore):
    """Restrictive wrapper exposing only the managed DenseVectorStore surface."""

    def __init__(self, inner: FaissVectorStore) -> None:
        self._inner = inner

    def search(self, query: np.ndarray, top_k: int) -> List[FaissSearchResult]:
        """Delegate single-query search to the managed store.

        Args:
            query: Query embedding to search against the index.
            top_k: Number of nearest neighbours to return.

        Returns:
            Ranked FAISS search results.
        """
        return self._inner.search(query, top_k)

    def search_many(self, queries: np.ndarray, top_k: int) -> List[List[FaissSearchResult]]:
        """Execute vector search for multiple queries in a batch.

        Args:
            queries: Matrix of query embeddings.
            top_k: Number of nearest neighbours per query.

        Returns:
            Per-query lists of FAISS search results.
        """
        return self._inner.search_many(queries, top_k)

    def search_batch(self, queries: np.ndarray, top_k: int) -> List[List[FaissSearchResult]]:
        """Alias for :meth:`search_many` retaining legacy naming.

        Args:
            queries: Matrix of query embeddings.
            top_k: Number of nearest neighbours per query.

        Returns:
            Per-query lists of FAISS search results.
        """
        return self._inner.search_batch(queries, top_k)

    def add(self, vectors: Sequence[np.ndarray], vector_ids: Sequence[str]) -> None:
        """Insert vectors and identifiers into the managed index.

        Args:
            vectors: Embedding vectors to index.
            vector_ids: Identifiers associated with ``vectors``.
        """
        self._inner.add(vectors, vector_ids)

    def set_nprobe(
        self,
        nprobe: int,
        *,
        clamp_min: int = 1,
        clamp_max: Optional[int] = None,
    ) -> int:
        """Tune ``nprobe`` while clamping to the managed safe range."""

        return self._inner.set_nprobe(nprobe, clamp_min=clamp_min, clamp_max=clamp_max)

    def remove(self, vector_ids: Sequence[str]) -> None:
        """Remove vectors corresponding to ``vector_ids`` from the index."""

        self._inner.remove(vector_ids)

    def set_id_resolver(self, resolver: Callable[[int], Optional[str]]) -> None:
        """Configure identifier resolver on the inner store.

        Args:
            resolver: Callable mapping FAISS ids to external identifiers.
        """
        self._inner.set_id_resolver(resolver)

    def range_search(
        self,
        query: np.ndarray,
        min_score: float,
        *,
        limit: Optional[int] = None,
    ) -> List[FaissSearchResult]:
        """Delegate range search to the managed store."""

        return list(self._inner.range_search(query, min_score, limit=limit))

    def needs_training(self) -> bool:
        """Return ``True`` when the underlying index requires training."""

        return self._inner.needs_training()

    def train(self, vectors: Sequence[np.ndarray]) -> None:
        """Train the managed index using ``vectors``."""

        self._inner.train(vectors)

    @property
    def device(self) -> int:
        """Return the CUDA device identifier for the managed index."""

        return self._inner.device

    @property
    def adapter_stats(self) -> AdapterStats:
        """Return a read-only snapshot of adapter state."""

        return self._inner.adapter_stats

    @property
    def ntotal(self) -> int:
        """Return the number of vectors stored in the managed index."""

        return self._inner.ntotal

    def rebuild_if_needed(self) -> bool:
        """Trigger inner index rebuild when required by FAISS heuristics."""

        return self._inner.rebuild_if_needed()

    def stats(self) -> Mapping[str, float | str]:
        """Expose diagnostic statistics from the managed store."""

        return self._inner.stats()

    def get_gpu_resources(self) -> Optional["faiss.StandardGpuResources"]:
        """Return GPU resources backing the managed index (if available)."""

        getter = getattr(self._inner, "get_gpu_resources", None)
        if callable(getter):
            return getter()
        adapter_stats = getattr(self._inner, "adapter_stats", None)
        if adapter_stats is None:
            return None
        return adapter_stats.resources
