# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.HybridSearch.vectorstore",
#   "purpose": "FAISS vector store orchestration, GPU similarity utilities, and persistence helpers",
#   "sections": [
#     {
#       "id": "faisssearchresult",
#       "name": "FaissSearchResult",
#       "anchor": "class-faisssearchresult",
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
#     }
#   ]
# }
# === /NAVMAP ===

"""Unified FAISS vector store, GPU similarity utilities, and state helpers."""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Callable, List, Optional, Sequence

import numpy as np

from .config import DenseIndexConfig
from .interfaces import DenseVectorStore
from .types import vector_uuid_to_faiss_int

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .storage import ChunkRegistry

# --- Globals ---

logger = logging.getLogger(__name__)

__all__ = (
    "FaissVectorStore",
    "FaissSearchResult",
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

    def __init__(self, dim: int, config: DenseIndexConfig) -> None:
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
        self._replicated = False
        self._gpu_resources: Optional["faiss.StandardGpuResources"] = None
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
        self._set_nprobe()

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

    @property
    def device(self) -> int:
        """Return the CUDA device id used for the FAISS index.

        Args:
            None

        Returns:
            int: Configured CUDA device identifier.
        """
        return int(getattr(self._config, "device", self._device))

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
        matrix = np.stack([self._ensure_dim(vec) for vec in vectors]).astype(np.float32)
        faiss.normalize_L2(matrix)
        nlist = int(getattr(self._config, "nlist", 1024))
        factor = max(1, int(getattr(self._config, "ivf_train_factor", 8)))
        ntrain = min(matrix.shape[0], nlist * factor)
        self._index.train(matrix[:ntrain])

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
        with self._lock:
            self._flush_pending_deletes(force=False)
            if len(vectors) != len(vector_ids):
                raise ValueError("vectors and vector_ids must align")
            if not vectors:
                return
            faiss_ids = np.array(
                [vector_uuid_to_faiss_int(vid) for vid in vector_ids], dtype=np.int64
            )
            if self._supports_remove_ids is None:
                self._supports_remove_ids = self._probe_remove_support()
            if self._supports_remove_ids:
                self.remove_ids(faiss_ids, force_flush=True)
            else:
                existing_ids = self._lookup_existing_ids(faiss_ids)
                if existing_ids.size:
                    self.remove_ids(existing_ids, force_flush=True)
            matrix = np.ascontiguousarray(
                np.stack([self._ensure_dim(vec) for vec in vectors]), dtype=np.float32
            )
            faiss.normalize_L2(matrix)
            base = self._index.index if hasattr(self._index, "index") else self._index
            train_target = base
            if hasattr(faiss, "downcast_index"):
                try:
                    train_target = faiss.downcast_index(base)
                except Exception:
                    train_target = base
        if hasattr(train_target, "is_trained") and not getattr(train_target, "is_trained"):
            nlist = int(getattr(self._config, "nlist", 1024))
            factor = max(1, int(getattr(self._config, "ivf_train_factor", 8)))
            ntrain = min(matrix.shape[0], nlist * factor)
            train_target.train(matrix[:ntrain])
            self._index.add_with_ids(matrix, faiss_ids)
            self._dirty_deletes = 0
            self._needs_rebuild = False

    def remove(self, vector_ids: Sequence[str]) -> None:
        """Remove vectors from the index using application-level identifiers.

        Args:
            vector_ids: Sequence of vector ids scheduled for deletion.

        Returns:
            None
        """
        if not vector_ids:
            return
        ids = np.array([vector_uuid_to_faiss_int(vid) for vid in vector_ids], dtype=np.int64)
        with self._lock:
            self.remove_ids(ids, force_flush=True)

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
        """Search the index for the ``top_k`` nearest neighbours of ``query``.

        Args:
            query: Dense query vector with dimensionality ``self._dim``.
            top_k: Maximum number of nearest neighbours to return.

        Returns:
            Ranked list of :class:`FaissSearchResult` objects.
        """
        with self._lock:
            self._flush_pending_deletes(force=False)
            query_matrix = np.ascontiguousarray(
                self._ensure_dim(query).reshape(1, -1), dtype=np.float32
            )
            faiss.normalize_L2(query_matrix)
            self._set_nprobe()
            scores, ids = self._index.search(query_matrix, top_k)
            results: List[FaissSearchResult] = []
            for score, internal_id in zip(scores[0], ids[0]):
                if internal_id == -1:
                    continue
                vector_id = self._resolve_vector_id(int(internal_id))
                if vector_id is None:
                    continue
                results.append(FaissSearchResult(vector_id=vector_id, score=float(score)))
            return results

    def search_many(self, queries: np.ndarray, top_k: int) -> List[List[FaissSearchResult]]:
        """Search the index for multiple queries in a single FAISS call."""

        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        with self._lock:
            self._flush_pending_deletes(force=False)
            matrix = np.ascontiguousarray(queries, dtype=np.float32)
            if matrix.shape[1] != self._dim:
                raise ValueError(f"Query dimensionality {matrix.shape[1]} != index dim {self._dim}")
            faiss.normalize_L2(matrix)
            self._set_nprobe()
            scores, ids = self._index.search(matrix, top_k)
        batched: List[List[FaissSearchResult]] = []
        for row_scores, row_ids in zip(scores, ids):
            row_results: List[FaissSearchResult] = []
            for score, internal_id in zip(row_scores, row_ids):
                if internal_id == -1:
                    continue
                vector_id = self._resolve_vector_id(int(internal_id))
                if vector_id is None:
                    continue
                row_results.append(FaissSearchResult(vector_id=vector_id, score=float(score)))
            batched.append(row_results)
        return batched

    def search_batch(self, queries: np.ndarray, top_k: int) -> List[List[FaissSearchResult]]:
        """Alias for ``search_many`` to support batch query workflows."""

        return self.search_many(queries, top_k)

    def serialize(self) -> bytes:
        """Return a CPU-serialised representation of the FAISS index.

        Args:
            None

        Returns:
            Byte payload produced by :func:`faiss.serialize_index`.

        Raises:
            RuntimeError: If the index has not been initialised.
        """
        with self._lock:
            if self._index is None:
                raise RuntimeError("index is empty")
            cpu_index = self._to_cpu(self._index)
            blob = faiss.serialize_index(cpu_index)
            return bytes(blob)

    def save(self, path: str) -> None:
        """Persist the FAISS index to ``path`` when persistence is enabled.

        Args:
            path: Filesystem destination for the serialised index.

        Returns:
            None

        Raises:
            RuntimeError: If the index has not been initialised.
        """
        if self._index is None:
            raise RuntimeError("index is empty")
        if getattr(self._config, "persist_mode", "cpu_bytes") == "disabled":
            logger.info("faiss-save-skip", extra={"event": {"reason": "persist_mode=disabled"}})
            return
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.serialize())

    @classmethod
    def load(cls, path: str, config: DenseIndexConfig, dim: int) -> "FaissVectorStore":
        """Restore a vector store from disk.

        Args:
            path: Filesystem path containing a previously saved index payload.
            config: Dense index configuration to apply to the reloaded store.
            dim: Dimensionality of vectors stored in the index.

        Returns:
            Fresh :class:`FaissVectorStore` instance initialised from ``path``.

        Raises:
            OSError: If ``path`` cannot be read from disk.
        """
        blob = Path(path).read_bytes()
        manager = cls(dim=dim, config=config)
        manager.restore(blob)
        return manager

    def restore(self, payload: bytes) -> None:
        """Load an index from ``payload`` and promote it to the GPU.

        Args:
            payload: Bytes produced by :meth:`serialize`.

        Raises:
            ValueError: If the payload is empty.

        Returns:
            None
        """
        if not payload:
            raise ValueError("Empty FAISS payload")
        with self._lock:
            cpu_index = faiss.deserialize_index(np.frombuffer(payload, dtype=np.uint8))
            self._index = self._maybe_to_gpu(cpu_index)
            self._tombstones.clear()
            self._dirty_deletes = 0
            self._needs_rebuild = False
            self._set_nprobe()

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
            "gpu_remove_fallbacks": float(self._remove_fallbacks),
            "multi_gpu_mode": self._multi_gpu_mode,
            "gpu_indices_32_bit": bool(self._indices_32_bit and not self._force_64bit_ids),
            "gpu_device": str(getattr(self._config, "device", 0)),
            "device": str(getattr(self._config, "device", 0)),
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
            opts.useFloat16 = getattr(self._config, "flat_use_fp16", False)
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
                cfg.useFloat16LookupTables = bool(
                    getattr(self._config, "ivfpq_float16_lut", True)
                )
            if hasattr(cfg, "interleavedLayout"):
                cfg.interleavedLayout = bool(getattr(self._config, "interleaved_layout", True))
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

    N, M = q.shape[0], C.shape[0]
    best_scores = np.full((N, k), -np.inf, dtype=np.float32)
    best_index = np.full((N, k), -1, dtype=np.int64)

    start = 0
    while start < M:
        end = min(M, start + block_rows)
        block = np.array(C[start:end], dtype=np.float32, copy=True)
        faiss.normalize_L2(block)
        sims = faiss.pairwise_distance_gpu(
            resources,
            q,
            block,
            metric=faiss.METRIC_INNER_PRODUCT,
            device=int(device),
        )

        block_idx = np.arange(start, end, dtype=np.int64)
        cand_scores = np.concatenate([best_scores, sims], axis=1)
        cand_index = np.concatenate(
            [best_index, np.broadcast_to(block_idx, (N, block_idx.size))], axis=1
        )
        select = np.argpartition(cand_scores, -k, axis=1)[:, -k:]
        rows = np.arange(N)[:, None]
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
        dict[str, object]: Dictionary containing base64-encoded FAISS bytes and registry ids.
    """
    faiss_bytes = faiss_index.serialize()
    encoded = base64.b64encode(faiss_bytes).decode("ascii")
    return {
        "faiss": encoded,
        "vector_ids": registry.vector_ids(),
    }


def restore_state(faiss_index: FaissVectorStore, payload: dict[str, object]) -> None:
    """Restore the vector store from a payload produced by :func:`serialize_state`.

    Args:
        faiss_index: Vector store receiving the restored state.
        payload: Mapping with ``faiss`` (base64) and registry vector ids.

    Returns:
        None

    Raises:
        ValueError: If the payload is missing the FAISS byte stream.
    """
    encoded = payload.get("faiss")
    if not isinstance(encoded, str):
        raise ValueError("Missing FAISS payload")
    faiss_index.restore(base64.b64decode(encoded.encode("ascii")))
