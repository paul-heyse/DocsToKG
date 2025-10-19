# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.HybridSearch.store",
#   "purpose": "Vector store orchestration, OpenSearch helpers, GPU similarity utilities, and persistence",
#   "sections": [
#     {
#       "id": "vector-uuid-to-faiss-int",
#       "name": "_vector_uuid_to_faiss_int",
#       "anchor": "function-vector-uuid-to-faiss-int",
#       "kind": "function"
#     },
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
import uuid
from dataclasses import asdict, dataclass, replace
import inspect
from pathlib import Path
from threading import Event, RLock
from typing import Callable, ClassVar, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .config import DenseIndexConfig
from .devtools.opensearch_simulator import OpenSearchSimulator
from .interfaces import DenseVectorStore
from .pipeline import Observability
from .types import ChunkPayload

# --- Globals ---

logger = logging.getLogger(__name__)

__all__ = (
    "FaissVectorStore",
    "ManagedFaissAdapter",
    "AdapterStats",
    "FaissSearchResult",
    "ChunkRegistry",
    "OpenSearchSimulator",
    "cosine_against_corpus_gpu",
    "cosine_batch",
    "cosine_topk_blockwise",
    "resolve_cuvs_state",
    "max_inner_product",
    "normalize_rows",
    "pairwise_inner_products",
    "restore_state",
    "serialize_state",
)

_MASK_63_BITS = (1 << 63) - 1

_COSINE_TOPK_DEFAULT_BLOCK_ROWS = 65_536
_COSINE_TOPK_AUTO_BLOCK_ROWS_SENTINEL = -1
_COSINE_TOPK_AUTO_MEM_FRACTION = 0.5


def _vector_uuid_to_faiss_int(vector_id: str) -> int:
    """Translate a vector UUID into a FAISS-compatible 63-bit integer."""

    return uuid.UUID(vector_id).int & _MASK_63_BITS


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


# --- Helpers ---


def resolve_cuvs_state(requested: Optional[bool]) -> tuple[bool, bool, Optional[bool]]:
    """Determine whether cuVS kernels should be enabled for FAISS helpers."""

    if not _FAISS_AVAILABLE:
        return False, False, None
    knn_runner = getattr(faiss, "knn_gpu", None)
    if knn_runner is None:
        return False, False, None

    reported_available: Optional[bool]
    should_use = getattr(faiss, "should_use_cuvs", None)
    if callable(should_use):
        try:
            reported_available = bool(should_use())
        except Exception:  # pragma: no cover - defensive best effort
            reported_available = None
    else:
        reported_available = None

    if requested is None:
        enabled = bool(reported_available)
    else:
        enabled = bool(requested)

    enabled = bool(knn_runner) and enabled
    available = bool(knn_runner) and (
        bool(reported_available) if reported_available is not None else False
    )
    return enabled, available, reported_available


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
    cuvs_enabled: bool
    cuvs_available: bool
    cuvs_reported: Optional[bool]
    cuvs_requested: Optional[bool]
    cuvs_applied: Optional[bool]


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
        while batch:
            vectors = [self._store._ensure_dim(item.vector) for item in batch]
            k_max = max(item.top_k for item in batch)
            matrix = np.stack(vectors, dtype=np.float32)
            try:
                results = self._store._search_batch_impl(matrix, k_max)
            except Exception as exc:
                while True:
                    for pending in batch:
                        pending.set_exception(exc)
                    batch = self._drain()
                    if not batch:
                        break
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
                batch = self._drain()


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

    _RUNTIME_MUTABLE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"nprobe", "ingest_dedupe_threshold"}
    )

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
        self._gpu_use_default_null_stream_all_devices = bool(
            getattr(config, "gpu_use_default_null_stream_all_devices", False)
            or getattr(config, "gpu_default_null_stream_all_devices", False)
        )
        self._gpu_use_default_null_stream = bool(
            getattr(config, "gpu_use_default_null_stream", False)
            or getattr(config, "gpu_default_null_stream", False)
        )
        self._expected_ntotal = int(getattr(config, "expected_ntotal", 0))
        self._rebuild_delete_threshold = int(getattr(config, "rebuild_delete_threshold", 10000))
        self._force_64bit_ids = bool(getattr(config, "force_64bit_ids", False))
        self._force_remove_ids_fallback = bool(getattr(config, "force_remove_ids_fallback", False))
        self._replication_enabled = bool(getattr(config, "enable_replication", True))
        raw_replication_ids = getattr(config, "replication_gpu_ids", None)
        self._has_explicit_replication_ids = raw_replication_ids is not None
        if raw_replication_ids is None:
            self._replication_gpu_ids: Optional[Tuple[int, ...]] = None
        else:
            normalised_ids: list[int] = []
            for raw_id in raw_replication_ids:
                try:
                    candidate = int(raw_id)
                except (TypeError, ValueError):
                    logger.debug(
                        "Ignoring invalid replication gpu id", extra={"event": {"value": raw_id}}
                    )
                    continue
                normalised_ids.append(candidate)
            self._replication_gpu_ids = tuple(normalised_ids)
        self._reserve_memory_enabled = bool(getattr(config, "enable_reserve_memory", True))
        self._replicated = False
        self._gpu_resources: Optional["faiss.StandardGpuResources"] = None
        self._replica_gpu_resources: list["faiss.StandardGpuResources"] = []
        self._pinned_buffers: list[object] = []
        self._observability = observability or Observability()
        self._last_applied_cuvs: Optional[bool] = None
        self.init_gpu()
        self._lock = RLock()
        self._index = self._create_index()
        self._index = self._maybe_distribute_multi_gpu(self._index)
        if self._dim != dim:
            raise RuntimeError(
                f"HybridSearch initialised with dim={dim} but created index expects {self._dim}"
            )
        self._id_resolver: Optional[Callable[[int], Optional[str]]] = None
        self._remove_fallbacks = 0
        self._rebuilds = 0
        self._tombstones: set[int] = set()
        self._dirty_deletes = 0
        self._needs_rebuild = False
        self._supports_remove_ids: Optional[bool] = None
        self._last_nprobe_update = 0.0
        self._last_applied_nprobe: Optional[int] = None
        self._last_applied_nprobe_monotonic = 0.0
        self._set_nprobe()
        self._search_coalescer = _SearchCoalescer(self)
        self._cpu_replica: Optional[bytes] = None
        self._cpu_replica_meta: Optional[Mapping[str, object]] = None
        self._snapshot_refresh_interval = max(
            0.0, float(getattr(self._config, "snapshot_refresh_interval_seconds", 0.0))
        )
        self._snapshot_refresh_writes = max(
            0, int(getattr(self._config, "snapshot_refresh_writes", 0))
        )
        self._last_snapshot_refresh = 0.0
        self._writes_since_snapshot = 0
        self._snapshot_lock = RLock()
        self._refresh_cpu_replica(reason="bootstrap", forced=True)
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

    def set_config(self, new_config: DenseIndexConfig) -> None:
        """Apply runtime-safe configuration updates."""

        if not isinstance(new_config, DenseIndexConfig):
            raise TypeError("new_config must be a DenseIndexConfig instance")
        current = asdict(self._config)
        updated = asdict(new_config)
        diffs: Dict[str, tuple[object, object]] = {}
        for key, old_value in current.items():
            new_value = updated.get(key, old_value)
            if old_value != new_value:
                diffs[key] = (old_value, new_value)
        if not diffs:
            return
        disallowed = [name for name in diffs if name not in self._RUNTIME_MUTABLE_FIELDS]
        if disallowed:
            raise ValueError(
                "Runtime config updates are restricted; attempted changes to disallowed fields: "
                + ", ".join(sorted(disallowed))
            )
        changes = {name: change[1] for name, change in diffs.items()}
        self._config = replace(self._config, **changes)
        self._observability.logger.info(
            "faiss-config-update",
            extra={
                "event": {
                    "changes": {
                        name: {"old": change[0], "new": change[1]} for name, change in diffs.items()
                    }
                }
            },
        )
        if "nprobe" in changes:
            with self._observability.trace(
                "faiss_set_config_nprobe", nprobe=str(changes["nprobe"])
            ):
                with self._lock:
                    self._reset_nprobe_cache()
                    self._set_nprobe()
                    self._last_nprobe_update = time.time()
            self._observability.metrics.set_gauge("faiss_nprobe", float(changes["nprobe"]))
        if "ingest_dedupe_threshold" in changes:
            self._observability.metrics.set_gauge(
                "faiss_ingest_dedupe_threshold", float(changes["ingest_dedupe_threshold"])
            )
        self._emit_gpu_state("set_config")

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
        config_device = getattr(self._config, "device", None)
        if config_device is not None:
            return int(config_device)
        return int(getattr(self, "_device", 0))

    @property
    def adapter_stats(self) -> AdapterStats:
        """Return a read-only snapshot of adapter state."""

        index_desc = self._describe_index(getattr(self, "_index", None))
        requested_cuvs = getattr(self._config, "use_cuvs", None)
        cuvs_enabled, cuvs_available, cuvs_reported = resolve_cuvs_state(requested_cuvs)
        return AdapterStats(
            device=self.device,
            ntotal=self.ntotal,
            index_description=index_desc,
            nprobe=self._current_nprobe_value(),
            multi_gpu_mode=str(self._multi_gpu_mode),
            replicated=bool(self._replicated),
            fp16_enabled=bool(getattr(self._config, "flat_use_fp16", False)),
            resources=self._gpu_resources,
            cuvs_enabled=cuvs_enabled,
            cuvs_available=cuvs_available,
            cuvs_reported=cuvs_reported,
            cuvs_requested=requested_cuvs,
            cuvs_applied=self._last_applied_cuvs,
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
        self.flush_snapshot(reason="train")

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
                [_vector_uuid_to_faiss_int(vid) for vid in vector_ids], dtype=np.int64
            )
            matrix = self._coerce_batch(
                np.stack([self._ensure_dim(vec) for vec in vectors]), normalize=False
            )
            dedupe_threshold = float(getattr(self._config, "ingest_dedupe_threshold", 0.0))
            dropped = 0
            if dedupe_threshold > 0.0 and self.ntotal > 0:
                dedupe_matrix = np.ascontiguousarray(matrix.copy(), dtype=np.float32)
                try:
                    faiss.normalize_L2(dedupe_matrix)
                    distances, indices = self._search_matrix(dedupe_matrix, 1)
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
                        self.remove_ids(faiss_ids, force_flush=True, reason="add_dedupe")
                    else:
                        existing_ids = self._lookup_existing_ids(faiss_ids)
                        if existing_ids.size:
                            self.remove_ids(existing_ids, force_flush=True, reason="add_cleanup")
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
                self._maybe_refresh_snapshot(writes_delta=int(inserted), reason="add")

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
        ids = np.array([_vector_uuid_to_faiss_int(vid) for vid in vector_ids], dtype=np.int64)
        with self._observability.trace("faiss_remove", count=str(count)):
            self._observability.metrics.increment("faiss_remove_vectors", amount=float(count))
            with self._lock:
                self.remove_ids(ids, force_flush=True, reason="remove")
        self._update_gpu_metrics()

    def remove_ids(
        self, ids: np.ndarray, *, force_flush: bool = False, reason: str = "remove"
    ) -> int:
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
            count = int(ids64.size)
            if count:
                self._maybe_refresh_snapshot(writes_delta=count, reason=reason)
            return count

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

        matrix = self._coerce_query(query)
        vector = matrix[0]
        with self._observability.trace("faiss_search", top_k=str(top_k)):
            if self._search_coalescer is not None:
                return self._search_coalescer.submit(vector, top_k)
            results = self._search_batch_impl(matrix, top_k)
            return results[0] if results else []

    def search_many(self, queries: np.ndarray, top_k: int) -> List[List[FaissSearchResult]]:
        """Search the index for multiple queries in a single FAISS call."""

        return self.search_batch(queries, top_k)

    def search_batch(self, queries: np.ndarray, top_k: int) -> List[List[FaissSearchResult]]:
        """Alias for ``search_many`` to support explicit batch workloads."""

        matrix = self._coerce_batch(queries)
        if matrix.shape[0] == 1 and self._search_coalescer is not None:
            hits = self._search_coalescer.submit(matrix[0], top_k)
            return [hits]
        return self._search_batch_impl(matrix, top_k)

    def _search_batch_impl(self, matrix: np.ndarray, top_k: int) -> List[List[FaissSearchResult]]:
        matrix = self._coerce_batch(matrix, normalize=False)
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

        matrix = self._coerce_query(query)
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
                    pairs = []
                    for i in range(start, end):
                        if labels[i] == -1:
                            continue
                        score = float(distances[i])
                        if score < threshold:
                            continue
                        pairs.append((score, int(labels[i])))
                        if limit is not None and len(pairs) >= int(limit):
                            break
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

    def flush_snapshot(self, *, reason: str = "flush") -> None:
        """Force a snapshot refresh bypassing throttle safeguards."""

        self._refresh_cpu_replica(reason=reason, forced=True)

    def _maybe_refresh_snapshot(self, *, writes_delta: int = 0, reason: str) -> None:
        if getattr(self._config, "persist_mode", "cpu_bytes") == "disabled":
            return
        now = time.time()
        interval = self._snapshot_refresh_interval
        write_threshold = self._snapshot_refresh_writes
        with self._snapshot_lock:
            if writes_delta > 0:
                self._writes_since_snapshot += int(writes_delta)
            writes_since = self._writes_since_snapshot
            last_refresh = self._last_snapshot_refresh
        if last_refresh == 0.0:
            self._refresh_cpu_replica(reason=reason, forced=False)
            return
        interval_ready = interval > 0.0 and now - last_refresh >= interval
        writes_ready = write_threshold > 0 and writes_since >= write_threshold
        should_refresh = False
        if interval <= 0.0 and write_threshold <= 0:
            should_refresh = True
        else:
            should_refresh = interval_ready or writes_ready
        if should_refresh:
            self._refresh_cpu_replica(reason=reason, forced=False)
            return
        age = max(0.0, now - last_refresh)
        self._observability.metrics.set_gauge("faiss_snapshot_age_seconds", float(age))
        self._observability.metrics.increment(
            "faiss_snapshot_refresh_skipped",
            amount=1.0,
            reason=reason,
        )

    def _refresh_cpu_replica(self, *, reason: str, forced: bool = False) -> None:
        if getattr(self._config, "persist_mode", "cpu_bytes") == "disabled":
            return
        try:
            payload = self.serialize()
        except Exception:  # pragma: no cover - best effort
            logger.debug("Unable to refresh CPU replica", exc_info=True)
            return
        meta = self.snapshot_meta()
        with self._snapshot_lock:
            self._cpu_replica = payload
            self._cpu_replica_meta = meta
            self._last_snapshot_refresh = time.time()
            self._writes_since_snapshot = 0
        self._observability.metrics.increment(
            "faiss_snapshot_refresh_total",
            amount=1.0,
            reason=reason,
            forced=str(bool(forced)),
        )
        self._observability.metrics.set_gauge("faiss_snapshot_age_seconds", 0.0)
        self._observability.logger.info(
            "faiss-snapshot-refresh",
            extra={
                "event": {
                    "reason": reason,
                    "forced": bool(forced),
                    "bytes": int(len(payload)),
                }
            },
        )

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
        self._observability.metrics.set_gauge(
            "faiss_cuvs_enabled", 1.0 if stats.cuvs_enabled else 0.0
        )
        self._observability.metrics.set_gauge(
            "faiss_cuvs_available", 1.0 if stats.cuvs_available else 0.0
        )
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
            "cuvs_enabled": stats.cuvs_enabled,
            "cuvs_available": stats.cuvs_available,
            "cuvs_requested": stats.cuvs_requested,
            "cuvs_reported": stats.cuvs_reported,
            "cuvs_applied": stats.cuvs_applied,
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
                self._reset_nprobe_cache()
                self._set_nprobe()
            self._observability.metrics.increment("faiss_restore_calls", amount=1.0)
        if getattr(self._config, "persist_mode", "cpu_bytes") != "disabled":
            snapshot_meta = self.snapshot_meta()
            with self._snapshot_lock:
                self._cpu_replica = bytes(payload)
                self._cpu_replica_meta = snapshot_meta
                self._last_snapshot_refresh = time.time()
                self._writes_since_snapshot = 0
            self._observability.metrics.set_gauge("faiss_snapshot_age_seconds", 0.0)
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
                self._reset_nprobe_cache()
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
            "total_rebuilds": float(self._rebuilds),
            "multi_gpu_mode": self._multi_gpu_mode,
            "gpu_indices_32_bit": bool(self._indices_32_bit and not self._force_64bit_ids),
            "gpu_device": str(getattr(self._config, "device", 0)),
            "device": str(getattr(self._config, "device", 0)),
            "fp16_enabled": bool(getattr(self._config, "flat_use_fp16", False)),
            "cpu_replica_cached": bool(self._cpu_replica is not None),
            "cpu_replica_bytes": float(len(self._cpu_replica)) if self._cpu_replica else 0.0,
            "snapshot_refresh_interval_seconds": float(self._snapshot_refresh_interval),
            "snapshot_refresh_writes": float(self._snapshot_refresh_writes),
            "snapshot_writes_since_refresh": float(self._writes_since_snapshot),
            "snapshot_last_refresh_epoch": float(self._last_snapshot_refresh),
        }
        cuvs_enabled, cuvs_available, cuvs_reported = resolve_cuvs_state(
            getattr(self._config, "use_cuvs", None)
        )
        stats["cuvs_enabled"] = cuvs_enabled
        stats["cuvs_available"] = cuvs_available
        stats["cuvs_reported_available"] = (
            bool(cuvs_reported) if cuvs_reported is not None else False
        )
        stats["cuvs_requested"] = getattr(self._config, "use_cuvs", None)
        stats["cuvs_applied"] = self._last_applied_cuvs
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
            "use_cuvs": getattr(self._config, "use_cuvs", None),
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
            self.flush_snapshot(reason="rebuild")
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
            resources = self._create_gpu_resources(device=device)
            self._gpu_resources = resources
            if self._gpu_resources is not None:
                self._configure_gpu_resource(self._gpu_resources, device=device)
                self._record_gpu_resource_configuration(device=device)
        except Exception as exc:  # pragma: no cover - GPU-specific failure
            raise RuntimeError(
                "HybridSearch failed to initialise FAISS GPU resources. "
                "Check CUDA driver installation and faiss-gpu compatibility."
            ) from exc

    def _resolve_replication_targets(self, available: int) -> Tuple[int, ...]:
        """Return the filtered GPU ids that should participate in replication."""

        if not self._replication_gpu_ids:
            return tuple()
        allowed: list[int] = []
        seen: set[int] = set()
        skipped: list[int] = []
        for gpu_id in self._replication_gpu_ids:
            if gpu_id in seen:
                continue
            seen.add(gpu_id)
            if 0 <= gpu_id < available:
                allowed.append(gpu_id)
            else:
                skipped.append(gpu_id)
        if skipped:
            logger.info(
                "Skipping unavailable GPUs during FAISS replication",
                extra={
                    "event": {
                        "requested_gpu_ids": tuple(self._replication_gpu_ids),
                        "allowed_gpu_ids": tuple(allowed),
                        "skipped_gpu_ids": tuple(skipped),
                    }
                },
            )
        return tuple(allowed)

    def _create_gpu_resources(
        self, *, device: Optional[int] = None
    ) -> "faiss.StandardGpuResources":
        """Instantiate ``StandardGpuResources`` for ``device`` without additional tweaks."""

        return faiss.StandardGpuResources()

    def _configure_gpu_resource(
        self, resource: "faiss.StandardGpuResources", *, device: Optional[int] = None
    ) -> None:
        """Apply configured knobs to a FAISS GPU resource manager."""

        temp_memory = getattr(self, "_temp_memory_bytes", None)
        if temp_memory is not None and hasattr(resource, "setTempMemory"):
            try:
                resource.setTempMemory(temp_memory)
            except Exception:  # pragma: no cover - best effort guard
                logger.debug("Unable to apply GPU temp memory cap", exc_info=True)

        # Configure pinned memory for faster H2D/D2H transfers
        if hasattr(self, "_config") and self._config is not None:
            pinned_memory_bytes = getattr(self._config, "gpu_pinned_memory_bytes", None)
            if pinned_memory_bytes is not None and hasattr(resource, "setPinnedMemory"):
                try:
                    resource.setPinnedMemory(pinned_memory_bytes)
                except Exception:  # pragma: no cover - best effort guard
                    logger.debug("Unable to apply GPU pinned memory configuration", exc_info=True)

        use_null_all = getattr(self, "_gpu_use_default_null_stream_all_devices", False)
        if use_null_all:
            method = getattr(resource, "setDefaultNullStreamAllDevices", None)
            if callable(method):
                try:  # pragma: no cover - hardware/driver dependent
                    method()
                except Exception:
                    logger.debug(
                        "Unable to enable default CUDA null stream across devices",
                        exc_info=True,
                    )
            return

        use_null = getattr(self, "_gpu_use_default_null_stream", False)
        if not use_null:
            return

        method = getattr(resource, "setDefaultNullStream", None)
        if not callable(method):
            return

        tried_device = False
        if device is not None:
            try:  # pragma: no cover - GPU binding specific
                method(int(device))
                tried_device = True
            except TypeError:
                tried_device = False
            except Exception:
                logger.debug("Unable to bind default CUDA null stream for device", exc_info=True)
                tried_device = True

        if tried_device:
            return

        try:  # pragma: no cover - GPU binding specific
            method()
        except TypeError:
            if device is not None:
                try:
                    method(int(device))
                except Exception:
                    logger.debug("Unable to fall back when binding CUDA null stream", exc_info=True)
        except Exception:
            logger.debug(
                "Unable to bind default CUDA null stream without explicit device",
                exc_info=True,
            )

    def _record_gpu_resource_configuration(self, *, device: Optional[int] = None) -> None:
        """Emit observability breadcrumbs for configured GPU resource settings."""

        observability = getattr(self, "_observability", None)
        if observability is None:
            return

        temp_memory = getattr(self, "_temp_memory_bytes", None)
        if temp_memory is not None:
            observability.metrics.set_gauge("faiss_gpu_temp_memory_bytes", float(temp_memory))
        observability.metrics.set_gauge(
            "faiss_gpu_default_null_stream",
            1.0 if getattr(self, "_gpu_use_default_null_stream", False) else 0.0,
        )
        observability.metrics.set_gauge(
            "faiss_gpu_default_null_stream_all_devices",
            1.0 if getattr(self, "_gpu_use_default_null_stream_all_devices", False) else 0.0,
        )
        observability.logger.info(
            "faiss-gpu-resource-configured",
            extra={
                "event": {
                    "device": None if device is None else int(device),
                    "temp_memory_bytes": temp_memory,
                    "default_null_stream": bool(
                        getattr(self, "_gpu_use_default_null_stream", False)
                    ),
                    "default_null_stream_all_devices": bool(
                        getattr(self, "_gpu_use_default_null_stream_all_devices", False)
                    ),
                }
            },
        )

    def _requires_gpu_resource_customization(self) -> bool:
        """Return whether replica resources need to be retained for custom settings."""

        return (
            getattr(self, "_temp_memory_bytes", None) is not None
            or getattr(self, "_gpu_use_default_null_stream", False)
            or getattr(self, "_gpu_use_default_null_stream_all_devices", False)
        )

    def _configure_gpu_cloner_options(self, options: Optional[object]) -> None:
        """Apply DenseIndexConfig-aware flags to FAISS GPU cloner options."""

        if options is None:
            return

        config_obj = getattr(self, "_config", None)
        fp16_enabled = bool(getattr(config_obj, "flat_use_fp16", False))
        force_64bit_ids = bool(
            getattr(self, "_force_64bit_ids", getattr(config_obj, "force_64bit_ids", False))
        )
        indices_32_enabled = bool(
            getattr(self, "_indices_32_bit", getattr(config_obj, "gpu_indices_32_bit", False))
        )
        indices_flag = 0
        if not force_64bit_ids and indices_32_enabled and hasattr(faiss, "INDICES_32_BIT"):
            indices_flag = getattr(faiss, "INDICES_32_BIT")

        if hasattr(options, "indicesOptions"):
            try:
                setattr(options, "indicesOptions", indices_flag)
            except Exception:  # pragma: no cover - defensive best effort
                logger.debug("Unable to configure indicesOptions on cloner", exc_info=True)

        if hasattr(options, "useFloat16"):
            try:
                setattr(options, "useFloat16", fp16_enabled)
            except Exception:  # pragma: no cover - defensive best effort
                logger.debug("Unable to configure useFloat16 on cloner", exc_info=True)

        if hasattr(options, "useFloat16CoarseQuantizer"):
            try:  # pragma: no cover - rarely exposed attribute
                setattr(options, "useFloat16CoarseQuantizer", fp16_enabled)
            except Exception:
                logger.debug(
                    "Unable to configure useFloat16CoarseQuantizer on cloner", exc_info=True
                )

        if hasattr(options, "useFloat16LookupTables"):
            try:  # pragma: no cover - rarely exposed attribute
                setattr(options, "useFloat16LookupTables", fp16_enabled)
            except Exception:
                logger.debug("Unable to configure useFloat16LookupTables on cloner", exc_info=True)

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
        explicit_targets_configured = bool(self._has_explicit_replication_ids)
        if explicit_targets_configured and not hasattr(faiss, "index_cpu_to_gpus_list"):
            logger.warning(
                "FAISS build missing index_cpu_to_gpus_list; cannot honour explicit replication gpu ids"
            )
            self._observability.metrics.increment(
                "faiss_gpu_explicit_target_unavailable",
                reason="missing_index_cpu_to_gpus_list",
            )
            self._observability.logger.warning(
                "faiss-explicit-gpu-targets-unavailable",
                extra={
                    "event": {
                        "component": "faiss",
                        "action": "explicit_gpu_targets_unavailable",
                        "reason": "missing_index_cpu_to_gpus_list",
                        "requested_gpu_ids": tuple(self._replication_gpu_ids or ()),
                    }
                },
            )
            return index
        if not explicit_targets_configured and not hasattr(faiss, "index_cpu_to_all_gpus"):
            if shard:
                raise RuntimeError("Sharded multi-GPU not supported by current faiss build")
            return index
        if shard and not hasattr(faiss, "GpuMultipleClonerOptions"):
            raise RuntimeError("Sharded multi-GPU not supported by current faiss build")

        available_gpus = 0
        if hasattr(faiss, "get_num_gpus"):
            try:
                available_gpus = int(faiss.get_num_gpus())
            except Exception:  # pragma: no cover - best effort guard
                available_gpus = 0
        if available_gpus <= 1:
            return index
        if self._replicated:
            return index

        target_gpus: Tuple[int, ...] = tuple()
        if explicit_targets_configured:
            target_gpus = self._resolve_replication_targets(available_gpus)
            if len(target_gpus) <= 1:
                if self._replication_gpu_ids:
                    logger.info(
                        "Insufficient GPU targets after filtering explicit replication ids",
                        extra={
                            "event": {
                                "requested_gpu_ids": tuple(self._replication_gpu_ids),
                                "filtered_gpu_ids": target_gpus,
                            }
                        },
                    )
                self._observability.metrics.increment(
                    "faiss_gpu_explicit_target_unavailable",
                    reason="insufficient_filtered_targets",
                )
                self._observability.logger.info(
                    "faiss-explicit-gpu-targets-partially-unavailable",
                    extra={
                        "event": {
                            "component": "faiss",
                            "action": "explicit_gpu_targets_filtered",
                            "requested_gpu_ids": tuple(self._replication_gpu_ids or ()),
                            "filtered_gpu_ids": target_gpus,
                        }
                    },
                )
                return index

        if explicit_targets_configured:
            gpu_ids: List[int] = list(target_gpus)
        else:
            gpu_ids = list(range(available_gpus))
        gpu_count = len(gpu_ids)

        original_ids = (
            tuple(self._replication_gpu_ids or ()) if explicit_targets_configured else tuple()
        )
        filtered_targets = bool(original_ids) and len(gpu_ids) < len(original_ids)

        try:
            base_index = index
            if hasattr(faiss, "index_gpu_to_cpu"):
                try:
                    base_index = faiss.index_gpu_to_cpu(index)
                except Exception:  # pragma: no cover - gracefully fall back to provided index
                    base_index = index

            cloner_options: "faiss.GpuMultipleClonerOptions | None" = None
            if hasattr(faiss, "GpuMultipleClonerOptions"):
                cloner_options = faiss.GpuMultipleClonerOptions()
                cloner_options.shard = bool(shard)
                if shard and hasattr(cloner_options, "common_ivf_quantizer"):
                    cloner_options.common_ivf_quantizer = True

                self._configure_gpu_cloner_options(cloner_options)
                self._apply_cloner_reservation(cloner_options, gpu_ids=gpu_ids)

            gpu_ids: List[int]
            if explicit_targets_configured:
                gpu_ids = list(target_gpus)
            else:
                gpu_ids = list(range(available_gpus))
            gpu_count = len(gpu_ids)

            resources_supported = hasattr(faiss, "GpuResourcesVector")
            resources_vector: "faiss.GpuResourcesVector | None" = None
            if resources_supported and not filtered_targets:
                try:
                    resources_vector = faiss.GpuResourcesVector()
                except Exception:  # pragma: no cover - fall back to legacy path
                    resources_supported = False
                    resources_vector = None

            self._replica_gpu_resources = []
            primary_resource = getattr(self, "_gpu_resources", None)
            multi: "faiss.Index"
            retain_manual_resources = (
                resources_vector is not None or self._requires_gpu_resource_customization()
            )

            for gpu_id in gpu_ids:
                resource: "faiss.StandardGpuResources"
                if primary_resource is not None and int(self.device) == gpu_id:
                    resource = primary_resource
                else:
                    resource = self._create_gpu_resources(device=gpu_id)
                    if retain_manual_resources:
                        self._replica_gpu_resources.append(resource)
                self._configure_gpu_resource(resource, device=gpu_id)
                if resources_vector is not None:
                    resources_vector.push_back(resource)

            if resources_vector is not None:
                if hasattr(faiss, "index_cpu_to_gpu_multiple") and not filtered_targets:
                    try:
                        multi = faiss.index_cpu_to_gpu_multiple(
                            resources_vector, gpu_ids, base_index, cloner_options
                        )
                        self._replicated = True
                        return multi
                    except TypeError:
                        pass

                if hasattr(faiss, "index_cpu_to_gpus_list"):
                    multi = faiss.index_cpu_to_gpus_list(
                        base_index,
                        gpus=gpu_ids,
                        co=cloner_options,
                        resources=resources_vector,
                    )
                    self._observability.metrics.increment(
                        "faiss_gpu_manual_resource_path", amount=1.0
                    )
                    self._observability.logger.info(
                        "faiss-manual-resource-path-engaged",
                        extra={
                            "event": {
                                "component": "faiss",
                                "action": "manual_resource_replication",
                                "gpu_ids": tuple(gpu_ids),
                            }
                        },
                    )
                    self._replicated = True
                    return multi

            if explicit_targets_configured and hasattr(faiss, "index_cpu_to_gpus_list"):
                if not gpu_ids:
                    logger.error("index_cpu_to_gpus_list fallback requires at least one GPU id")
                    raise AssertionError(
                        "gpu_ids must not be empty when invoking manual GPU replication"
                    )
                multi = faiss.index_cpu_to_gpus_list(
                    base_index,
                    gpus=gpu_ids,
                    co=cloner_options,
                )
                self._observability.metrics.increment("faiss_gpu_manual_resource_path", amount=1.0)
                self._observability.logger.info(
                    "faiss-manual-resource-path-engaged",
                    extra={
                        "event": {
                            "component": "faiss",
                            "action": "manual_resource_replication",
                            "gpu_ids": tuple(gpu_ids),
                        }
                    },
                )
                self._replicated = True
                return multi

            if hasattr(faiss, "index_cpu_to_all_gpus"):
                multi = faiss.index_cpu_to_all_gpus(
                    base_index,
                    co=cloner_options,
                    ngpu=gpu_count,
                )
                self._replicated = True
                return multi
        except RuntimeError:
            raise
        except Exception:  # pragma: no cover - GPU-specific failure
            logger.warning("Unable to replicate FAISS index across GPUs", exc_info=True)
            return index

    def _maybe_distribute_multi_gpu(self, index: "faiss.Index") -> "faiss.Index":
        """Conditionally replicate or shard ``index`` based on configuration."""

        if self._multi_gpu_mode not in ("replicate", "shard"):
            self._replicated = False
            self._apply_use_cuvs_parameter(index)
            return index
        if not self._replication_enabled:
            self._replicated = False
            self._apply_use_cuvs_parameter(index)
            return index
        shard = self._multi_gpu_mode == "shard"
        distributed = self.distribute_to_all_gpus(index, shard=shard)
        self._apply_use_cuvs_parameter(distributed)
        return distributed

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
                self._configure_gpu_cloner_options(co)
                self._apply_cloner_reservation(co)
                promoted = faiss.index_cpu_to_gpu(self._gpu_resources, device, index, co)
                return self._maybe_distribute_multi_gpu(promoted)
            cloned = faiss.index_cpu_to_gpu(self._gpu_resources, device, index)
            return self._maybe_distribute_multi_gpu(cloned)
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

    def _apply_cloner_reservation(
        self,
        cloner_options: object | None,
        *,
        gpu_ids: Optional[Sequence[int]] = None,
    ) -> None:
        """Populate FAISS cloner reservation knobs when ``expected_ntotal`` is set."""

        if cloner_options is None:
            return
        expected = getattr(self, "_expected_ntotal", 0)
        if expected <= 0:
            return
        reserve = int(expected)
        participant_count = len(gpu_ids) if gpu_ids else 0
        if self._multi_gpu_mode == "shard":
            participants = max(participant_count, 1)
            reserve = max(1, (reserve + participants - 1) // participants)

        applied = False
        try:
            if hasattr(cloner_options, "reserveVecs"):
                setattr(cloner_options, "reserveVecs", reserve)
                applied = True
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("Unable to set reserveVecs on FAISS cloner options", exc_info=True)

        per_device_applied = False
        if gpu_ids:
            try:
                if hasattr(cloner_options, "eachReserveVecs"):
                    vector_factory = (
                        getattr(faiss, "IntVector", None) if faiss is not None else None
                    )
                    if vector_factory is not None:
                        reserve_vector = vector_factory()
                        for _gpu in gpu_ids:
                            reserve_vector.push_back(reserve)
                    else:
                        reserve_vector = [reserve for _ in gpu_ids]
                    setattr(cloner_options, "eachReserveVecs", reserve_vector)
                    per_device_applied = True
            except Exception:  # pragma: no cover - defensive guard
                logger.debug(
                    "Unable to configure per-device reserve vector on FAISS cloner options",
                    exc_info=True,
                )

            if not per_device_applied and hasattr(cloner_options, "setReserveVecs"):
                try:  # pragma: no cover - defensive guard
                    cloner_options.setReserveVecs([reserve for _ in gpu_ids])
                    per_device_applied = True
                except Exception:
                    logger.debug(
                        "Unable to invoke setReserveVecs on FAISS cloner options", exc_info=True
                    )

        if applied or per_device_applied:
            observer = getattr(self, "_observability", None)
            structured_logger = (
                getattr(observer, "logger", logger) if observer is not None else logger
            )
            event: Dict[str, object] = {
                "component": "faiss",
                "action": "gpu_cloner_reserve",
                "reserve_vecs": reserve,
            }
            if gpu_ids is not None:
                event["gpu_ids"] = tuple(int(gpu) for gpu in gpu_ids)
            structured_logger.info("faiss-gpu-cloner-reserve", extra={"event": event})

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

    def _reset_nprobe_cache(self) -> None:
        """Invalidate cached nprobe state applied to the active FAISS index."""

        self._last_applied_nprobe = None
        self._last_applied_nprobe_monotonic = 0.0

    def _set_nprobe(self) -> None:
        index = getattr(self, "_index", None)
        if index is None or not self._config.index_type.startswith("ivf"):
            self._reset_nprobe_cache()
            return
        nprobe = int(self._config.nprobe)
        if self._last_applied_nprobe == nprobe and self._last_applied_nprobe_monotonic > 0.0:
            return
        applied = False
        try:
            if hasattr(faiss, "GpuParameterSpace"):
                gps = faiss.GpuParameterSpace()
                if hasattr(gps, "initialize"):
                    gps.initialize(index)
                gps.set_index_parameter(index, "nprobe", nprobe)
                applied = True
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("Unable to set nprobe via GpuParameterSpace", exc_info=True)
        if not applied:
            base = getattr(index, "index", None) or index
            try:
                if hasattr(base, "nprobe"):
                    base.nprobe = nprobe
                    applied = True
                elif hasattr(index, "nprobe"):
                    index.nprobe = nprobe
                    applied = True
            except Exception:  # pragma: no cover - defensive guard
                logger.debug("Unable to set nprobe attribute on FAISS index", exc_info=True)
        if applied:
            self._last_applied_nprobe = nprobe
            self._last_applied_nprobe_monotonic = time.monotonic()
        else:
            self._reset_nprobe_cache()
        self._apply_use_cuvs_parameter(index)
        self._log_index_configuration(index)

    def _apply_use_cuvs_parameter(self, index: Optional["faiss.Index"]) -> None:
        """Propagate the cuVS toggle to ``index`` and any GPU replicas."""

        if index is None or faiss is None:
            self._last_applied_cuvs = None
            return
        requested = getattr(self._config, "use_cuvs", None)
        cuvs_enabled, cuvs_available, _ = resolve_cuvs_state(requested)
        if not cuvs_available and requested is False:
            # Respect explicit opt-out even when FAISS reports no availability.
            cuvs_enabled = False
        if not hasattr(faiss, "GpuParameterSpace"):
            self._last_applied_cuvs = cuvs_enabled if requested is not None else None
            return
        try:
            gps = faiss.GpuParameterSpace()
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("Unable to construct FAISS GpuParameterSpace", exc_info=True)
            self._last_applied_cuvs = None
            return

        targets = list(self._iter_gpu_index_variants(index))
        applied = False
        for target in targets:
            try:
                if hasattr(gps, "initialize"):
                    gps.initialize(target)
            except Exception:  # pragma: no cover - defensive guard
                logger.debug("Failed to initialise GpuParameterSpace for index", exc_info=True)
            try:
                gps.set_index_parameter(target, "use_cuvs", bool(cuvs_enabled))
                applied = True
            except Exception:  # pragma: no cover - defensive guard
                logger.debug("Unable to set use_cuvs on FAISS index", exc_info=True)

        if applied and hasattr(gps, "get_index_parameter") and targets:
            try:  # pragma: no cover - best effort confirmation
                applied_value = bool(gps.get_index_parameter(targets[0], "use_cuvs"))
            except Exception:
                applied_value = bool(cuvs_enabled)
        elif applied:
            applied_value = bool(cuvs_enabled)
        else:
            applied_value = None

        self._last_applied_cuvs = applied_value

    def _iter_gpu_index_variants(self, root: "faiss.Index") -> list["faiss.Index"]:
        """Return FAISS index variants associated with ``root``.

        This walks nested wrappers (e.g. IndexIDMap2, replicas, shards) so
        parameter updates (``use_cuvs``) propagate to each GPU replica.
        """

        stack: list["faiss.Index"] = [root]
        seen: set[int] = set()
        collected: list["faiss.Index"] = []
        while stack:
            current = stack.pop()
            if current is None:
                continue
            ident = id(current)
            if ident in seen:
                continue
            seen.add(ident)
            collected.append(current)

            nested = getattr(current, "index", None)
            if nested is not None and nested is not current:
                stack.append(nested)

            if hasattr(faiss, "downcast_index"):
                try:
                    downcast = faiss.downcast_index(current)
                except Exception:  # pragma: no cover - defensive guard
                    downcast = None
                if downcast is not None and downcast is not current:
                    stack.append(downcast)

            for accessor_name in ("at", "index_at"):
                accessor = getattr(current, accessor_name, None)
                if accessor is None:
                    continue
                for offset in range(0, 64):
                    try:
                        candidate = accessor(offset)
                    except Exception:
                        break
                    if candidate is None:
                        break
                    stack.append(candidate)

        return collected

    def _log_index_configuration(self, index: "faiss.Index") -> None:
        try:
            desc = faiss.describe_index(index)
        except Exception:  # pragma: no cover - best effort
            desc = type(index).__name__
        logger.debug(
            "faiss-index-config",
            extra={
                "event": {
                    "nprobe": getattr(self._config, "nprobe", 0),
                    "index": desc,
                    "use_cuvs": self._last_applied_cuvs,
                }
            },
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
        if self._force_remove_ids_fallback:
            return False
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
        if self._force_remove_ids_fallback:
            self._supports_remove_ids = False
            remaining = ids
        else:
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
        self._rebuilds += 1
        old_index = self._index
        base = old_index.index if hasattr(old_index, "index") else old_index
        current_ids = self._current_index_ids()
        if current_ids.size == 0:
            self._index = self._create_index()
            self._index = self._maybe_distribute_multi_gpu(self._index)
            self._reset_nprobe_cache()
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
        self._index = self._maybe_distribute_multi_gpu(self._index)
        self._reset_nprobe_cache()
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

    def _coerce_batch(self, xb: np.ndarray, *, normalize: bool = True) -> np.ndarray:
        array = np.asarray(xb, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] != self._dim:
            raise ValueError(f"bad batch shape {array.shape}, expected (*,{self._dim})")
        array = np.ascontiguousarray(array, dtype=np.float32)
        array = array.copy()
        if normalize:
            normalize_rows(array)
        return array

    def _coerce_query(self, x: np.ndarray) -> np.ndarray:
        matrix = self._coerce_batch(x)
        if matrix.shape[0] != 1:
            raise ValueError("single-query operations require exactly one vector")
        return matrix

    def _validate_snapshot_meta(self, meta: Mapping[str, object]) -> None:
        expected = self.snapshot_meta()
        mismatches: dict[str, tuple[object, object]] = {}
        for key, expected_value in expected.items():
            if key not in meta:
                continue
            raw_value = meta[key]
            try:
                if expected_value is None:
                    actual_value = None if raw_value in (None, "None") else raw_value
                elif isinstance(expected_value, bool):
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
    pairwise_fn: Optional[Callable[..., np.ndarray]] = None,
) -> np.ndarray:
    """Compute cosine similarity between ``query`` and each vector in ``corpus`` on GPU.

    Args:
        query: Query vector or matrix (``N x D``) to compare.
        corpus: Matrix representing the corpus (``M x D``).
        device: CUDA device id used for the computation.
        resources: FAISS GPU resources to execute the kernel.
        pairwise_fn: Optional kernel override for ``faiss.pairwise_distance_gpu``.

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
    sims = cosine_batch(
        query,
        corpus,
        device=device,
        resources=resources,
        pairwise_fn=pairwise_fn,
    )
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
    pairwise_fn: Optional[Callable[..., np.ndarray]] = None,
) -> np.ndarray:
    """Helper that normalises and computes cosine similarities on GPU.

    Args:
        q: Query matrix (``N x D``).
        C: Corpus matrix (``M x D``).
        device: CUDA device id used for computation.
        resources: FAISS GPU resources backing the kernel.
        pairwise_fn: Optional kernel override for ``faiss.pairwise_distance_gpu``.

    Returns:
        numpy.ndarray: Pairwise cosine similarities with shape ``(N, M)``.
    """
    q = np.array(q, dtype=np.float32, copy=True, order="C")
    if q.ndim == 1:
        q = q.reshape(1, -1)

    C = np.array(C, dtype=np.float32, copy=True, order="C")
    faiss.normalize_L2(q)
    faiss.normalize_L2(C)
    kernel = pairwise_fn or faiss.pairwise_distance_gpu
    return kernel(
        resources,
        q,
        C,
        metric=faiss.METRIC_INNER_PRODUCT,
        device=int(device),
    )


def _auto_block_rows(
    resources: "faiss.StandardGpuResources",
    device: int,
    dim: int,
) -> tuple[int, Optional[dict[str, int]]]:
    """Estimate a safe block row count from GPU memory metrics."""

    get_memory_info = getattr(resources, "getMemoryInfo", None)
    if not callable(get_memory_info):
        return _COSINE_TOPK_DEFAULT_BLOCK_ROWS, None
    try:
        free_bytes, total_bytes = get_memory_info(int(device))
    except Exception:  # pragma: no cover - defensive guard
        logger.debug("cosine_topk_blockwise: unable to read GPU memory info", exc_info=True)
        return _COSINE_TOPK_DEFAULT_BLOCK_ROWS, None

    row_width = max(np.dtype(np.float32).itemsize * max(dim, 1), 1)
    permitted = max(
        int((free_bytes * _COSINE_TOPK_AUTO_MEM_FRACTION) // row_width),
        1,
    )
    effective = min(permitted, _COSINE_TOPK_DEFAULT_BLOCK_ROWS)
    return effective, {"free_bytes": int(free_bytes), "total_bytes": int(total_bytes)}
def _build_distance_params(
    *,
    use_fp16: bool,
    cuvs_enabled: bool,
) -> Optional[object]:
    """Construct a ``GpuDistanceParams`` instance when half precision is requested."""

    if not use_fp16:
        return None

    params_factory = getattr(faiss, "GpuDistanceParams", None)
    if params_factory is None:
        return None

    try:
        params = params_factory()
    except Exception:
        return None

    try:
        dtype_constant = getattr(faiss, "DistanceDataType_F16", None)
    except Exception:  # pragma: no cover - defensive guard for unexpected faiss builds
        dtype_constant = None

    if dtype_constant is not None:
        try:
            params.xType = dtype_constant
            params.yType = dtype_constant
        except Exception:
            pass

    if hasattr(params, "use_cuvs"):
        try:
            params.use_cuvs = bool(cuvs_enabled)
        except Exception:
            pass

    return params


def cosine_topk_blockwise(
    q: np.ndarray,
    C: np.ndarray,
    *,
    k: int,
    device: int,
    resources: "faiss.StandardGpuResources",
    block_rows: int = _COSINE_TOPK_AUTO_BLOCK_ROWS_SENTINEL,
    use_fp16: bool = False,
    use_cuvs: Optional[bool] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Top-K cosine similarities between ``q`` and ``C`` using GPU tiling.

    The helper avoids materialising the full ``(N x M)`` similarity matrix by
    iterating over ``C`` in row blocks and maintaining a running Top-K per query
    row. Inputs are copied and normalised inside the routine so callers retain
    ownership of their buffers. When ``block_rows`` is left at the sentinel value
    (``-1``), the helper inspects ``resources.getMemoryInfo`` to pick a block
    size that fits comfortably within the currently free GPU memory.

    Args:
        q: Query vector or matrix (``N x D``).
        C: Corpus matrix (``M x D``).
        k: Number of neighbours to return per query row.
        device: CUDA device ordinal used for FAISS kernels.
        resources: FAISS GPU resources backing ``pairwise_distance_gpu``.
        block_rows: Number of corpus rows processed per iteration. Provide a
            positive value to override the automatic sizing. The default
            sentinel (``-1``) enables adaptive sizing derived from GPU memory
            telemetry when available.
        use_fp16: Enable float16 compute for pairwise distance kernels.
        use_cuvs: Optional override forcing cuVS acceleration when supported
            (``True``) or disabling it (``False``). ``None`` defers to
            :func:`faiss.should_use_cuvs` when available.

    Returns:
        Tuple ``(scores, indices)`` where each has shape ``(N x K)``. Scores are
        sorted in descending order for every query row and indices reference rows
        within ``C``.
    """

    if resources is None:
        raise RuntimeError("FAISS GPU resources are required for cosine comparisons")
    requested_block_rows = block_rows
    memory_snapshot: Optional[dict[str, int]] = None
    if block_rows == _COSINE_TOPK_AUTO_BLOCK_ROWS_SENTINEL:
        block_rows, memory_snapshot = _auto_block_rows(resources, device, C.shape[1])
    if block_rows <= 0:
        raise ValueError("block_rows must be positive")

    log_payload: dict[str, object] = {
        "action": "cosine_topk_blockwise",
        "block_rows": int(block_rows),
        "requested_block_rows": int(requested_block_rows),
        "auto_block_rows": requested_block_rows == _COSINE_TOPK_AUTO_BLOCK_ROWS_SENTINEL,
        "dim": int(C.shape[1]),
        "device": int(device),
    }
    if memory_snapshot is not None:
        log_payload.update(memory_snapshot)
    logger.info("cosine-topk-block-config", extra={"event": log_payload})

    if q.ndim == 1:
        q = q.reshape(1, -1)
    q = np.array(q, dtype=np.float32, copy=True, order="C")
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

    cuvs_enabled, _, _ = resolve_cuvs_state(use_cuvs)
    distance_params = _build_distance_params(
        use_fp16=use_fp16,
        cuvs_enabled=cuvs_enabled,
    )

    knn_runner = getattr(faiss, "knn_gpu", None)
    if knn_runner is not None:
        try:
            corpus_copy = np.array(C, dtype=np.float32, copy=True, order="C")
            faiss.normalize_L2(corpus_copy)
            corpus_view = (
                corpus_copy.astype(np.float16, copy=False)
                if use_fp16
                else corpus_copy
            )
            row_bytes = np.dtype(np.float32).itemsize * C.shape[1]
            vector_limit = int(block_rows) * row_bytes
            query_rows = max(int(block_rows), q.shape[0])
            query_limit = (
                query_rows * np.dtype(np.float32).itemsize * q.shape[1]
            )
            extra_kwargs: dict[str, object] = {}
            if distance_params is not None:
                extra_kwargs["params"] = distance_params
            elif cuvs_enabled or use_cuvs is not None:
                extra_kwargs["use_cuvs"] = cuvs_enabled
            distances, indices = knn_runner(
                resources,
                q_view,
                corpus_view,
                k,
                metric=faiss.METRIC_INNER_PRODUCT,
                device=int(device),
                vectorsMemoryLimit=vector_limit,
                queriesMemoryLimit=query_limit,
                **extra_kwargs,
            )
        except TypeError:
            if extra_kwargs.pop("params", None) is not None:
                try:
                    distances, indices = knn_runner(
                        resources,
                        q_view,
                        corpus_view,
                        k,
                        metric=faiss.METRIC_INNER_PRODUCT,
                        device=int(device),
                        vectorsMemoryLimit=vector_limit,
                        queriesMemoryLimit=query_limit,
                        **extra_kwargs,
                    )
                except Exception:
                    distances = indices = None
            else:
                distances = indices = None
        except Exception:
            distances = indices = None
        else:
            if distances is not None and indices is not None:
                return (
                    np.asarray(distances, dtype=np.float32),
                    np.asarray(indices, dtype=np.int64),
                )

    N, M = q.shape[0], C.shape[0]
    best_scores = np.full((N, k), -np.inf, dtype=np.float32)
    best_index = np.full((N, k), -1, dtype=np.int64)

    start = 0
    while start < M:
        end = min(M, start + block_rows)
        block = np.array(C[start:end], dtype=np.float32, copy=True)
        faiss.normalize_L2(block)
        block_view = block.astype(np.float16, copy=False) if use_fp16 else block
        pairwise_kwargs: dict[str, object] = {
            "metric": faiss.METRIC_INNER_PRODUCT,
            "device": int(device),
        }
        if distance_params is not None:
            pairwise_kwargs["params"] = distance_params

        try:
            sims = faiss.pairwise_distance_gpu(
                resources,
                q_view,
                block_view,
                **pairwise_kwargs,
            )
        except TypeError:
            if pairwise_kwargs.pop("params", None) is not None:
                sims = faiss.pairwise_distance_gpu(
                    resources,
                    q_view,
                    block_view,
                    metric=faiss.METRIC_INNER_PRODUCT,
                    device=int(device),
                )
            else:
                raise

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

    @staticmethod
    def to_faiss_id(vector_id: str) -> int:
        """Return the FAISS integer identifier for ``vector_id``."""

        return _vector_uuid_to_faiss_int(vector_id)

    def upsert(self, chunks: Sequence[ChunkPayload]) -> None:
        """Insert or update registry entries for ``chunks``."""

        for chunk in chunks:
            self._chunks[chunk.vector_id] = chunk
            self._bridge[self.to_faiss_id(chunk.vector_id)] = chunk.vector_id

    def delete(self, vector_ids: Sequence[str]) -> None:
        """Remove registry entries for the supplied vector identifiers."""

        for vector_id in vector_ids:
            self._chunks.pop(vector_id, None)
            self._bridge.pop(self.to_faiss_id(vector_id), None)

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
        if any(isinstance(vec_id, (int, np.integer)) for vec_id in vector_ids):
            raise TypeError(
                "ManagedFaissAdapter.add() expects UUID string identifiers; "
                "use ChunkRegistry for FAISS id bridging"
            )
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

    def serialize(self) -> bytes:
        """Serialize the managed FAISS index to bytes."""

        return self._inner.serialize()

    def restore(
        self,
        payload: bytes,
        *,
        meta: Optional[Mapping[str, object]] = None,
    ) -> None:
        """Restore the managed FAISS index from ``payload``."""

        restore_fn = getattr(self._inner, "restore")
        if meta is None:
            restore_fn(payload)
            return

        try:
            signature = inspect.signature(restore_fn)
        except (TypeError, ValueError):
            signature = None

        if signature is not None:
            for parameter in signature.parameters.values():
                if parameter.name == "meta" and parameter.kind in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                ):
                    restore_fn(payload, meta=meta)
                    return

        # Fallback for stores that do not accept snapshot metadata.
        restore_fn(payload)

    def save(self, path: str) -> None:
        """Persist the managed FAISS index to ``path``."""

        self._inner.save(path)

    @classmethod
    def load(
        cls,
        path: str,
        config: DenseIndexConfig,
        dim: int,
        *,
        observability: Optional[Observability] = None,
    ) -> "ManagedFaissAdapter":
        """Load a managed FAISS adapter from ``path``."""

        inner = FaissVectorStore.load(
            path, config=config, dim=dim, observability=observability
        )
        return cls(inner)

    def needs_training(self) -> bool:
        """Return ``True`` when the underlying index requires training."""

        return self._inner.needs_training()

    def train(self, vectors: Sequence[np.ndarray]) -> None:
        """Train the managed index using ``vectors``."""

        self._inner.train(vectors)

    @property
    def config(self) -> DenseIndexConfig:
        """Return the dense index configuration for the managed store."""

        return self._inner.config

    @property
    def dim(self) -> int:
        """Return the embedding dimensionality exposed by the inner store."""

        return self._inner.dim

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

    def flush_snapshot(self, *, reason: str = "flush") -> None:
        """Forward snapshot flush requests to the managed store."""

        self._inner.flush_snapshot(reason=reason)

    def snapshot_meta(self) -> Mapping[str, object]:
        """Return snapshot metadata exposed by the managed store."""

        meta_getter = getattr(self._inner, "snapshot_meta", None)
        if not callable(meta_getter):
            return {}
        meta = meta_getter()
        if isinstance(meta, Mapping):
            return dict(meta)
        return {}

    def get_gpu_resources(self) -> Optional["faiss.StandardGpuResources"]:
        """Return GPU resources backing the managed index (if available)."""

        getter = getattr(self._inner, "get_gpu_resources", None)
        if callable(getter):
            return getter()
        adapter_stats = getattr(self._inner, "adapter_stats", None)
        if adapter_stats is None:
            return None
        return adapter_stats.resources
