"""GPU-only FAISS index management for dense retrieval."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np

from .config import DenseIndexConfig
from .ids import vector_uuid_to_faiss_int

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised indirectly via integration tests
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


@dataclass(slots=True)
class FaissSearchResult:
    """Dense search hit returned by FAISS.

    Attributes:
        vector_id: Identifier of the matched vector.
        score: Cosine similarity score returned by FAISS.

    Examples:
        >>> result = FaissSearchResult(vector_id="vec-1", score=0.92)
        >>> result.vector_id
        'vec-1'
    """

    vector_id: str
    score: float


class FaissIndexManager:
    """Manage lifecycle of a GPU-resident FAISS index.

    Attributes:
        _dim: Dimensionality of vectors stored in the index.
        _config: DenseIndexConfig controlling index behaviour.
        _device: GPU device identifier used for FAISS operations.
        _gpu_resources: FAISS GPU resources allocated for the index.

    Examples:
        >>> manager = FaissIndexManager(dim=128, config=DenseIndexConfig())  # doctest: +SKIP
        >>> manager.ntotal  # doctest: +SKIP
        0
    """

    def __init__(self, dim: int, config: DenseIndexConfig) -> None:
        """Create a GPU-resident FAISS index manager for the given dimensionality.

        Args:
            dim: Dimensionality of the dense vectors managed by the index.
            config: Configuration object governing index construction and tuning.

        Returns:
            None

        Raises:
            RuntimeError: If FAISS GPU extensions are unavailable.
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
        self._index = self._create_index()
        self._id_resolver: Optional[Callable[[int], Optional[str]]] = None
        self._remove_fallbacks = 0
        self._tombstones: set[int] = set()
        self._dirty_deletes = 0
        self._needs_rebuild = False
        self._set_nprobe()

    @property
    def ntotal(self) -> int:
        """Number of vectors currently stored in the FAISS index.

        Args:
            None

        Returns:
            Count of vectors indexed in FAISS.
        """
        return int(self._index.ntotal)

    @property
    def config(self) -> DenseIndexConfig:
        """Return the dense index configuration in use.

        Args:
            None

        Returns:
            DenseIndexConfig associated with this manager.
        """
        return self._config

    @property
    def gpu_resources(self) -> "faiss.StandardGpuResources | None":
        """Return the FAISS GPU resources backing the index.

        Args:
            None

        Returns:
            `faiss.StandardGpuResources` instance when GPU execution is enabled, otherwise ``None``.
        """
        return self._gpu_resources

    @property
    def device(self) -> int:
        """Return the GPU device identifier assigned to the index manager.

        Args:
            None

        Returns:
            Integer CUDA device ordinal used for FAISS kernels.
        """
        return int(getattr(self._config, "device", self._device))

    def set_id_resolver(self, resolver: Callable[[int], Optional[str]]) -> None:
        """Register a callback translating FAISS internal IDs to vector UUIDs.

        Args:
            resolver: Callable mapping FAISS internal IDs to vector UUIDs.

        Returns:
            None
        """
        self._id_resolver = resolver

    def train(self, vectors: Sequence[np.ndarray]) -> None:
        """Train IVF-style indexes with the provided vectors when required.

        Args:
            vectors: Sequence of vectors used for training.

        Returns:
            None

        Raises:
            ValueError: If training vectors are required but not provided.
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
        oversample = max(1, int(getattr(self._config, "oversample", 2)))
        ntrain = min(matrix.shape[0], nlist * oversample)
        self._index.train(matrix[:ntrain])

    def needs_training(self) -> bool:
        """Return True when the underlying index still requires training.

        Args:
            None

        Returns:
            Boolean indicating if training is required.
        """
        is_trained = getattr(self._index, "is_trained", True)
        return not bool(is_trained)

    def add(self, vectors: Sequence[np.ndarray], vector_ids: Sequence[str]) -> None:
        """Add vectors to the index, replacing existing entries when necessary.

        Args:
            vectors: Sequence of embedding vectors to add.
            vector_ids: Corresponding vector identifiers.

        Returns:
            None

        Raises:
            ValueError: If the lengths of `vectors` and `vector_ids` differ.
        """
        # Only rebuild when the configured threshold is reached to avoid
        # triggering tombstone rebuilds in the query path.
        self._flush_pending_deletes(force=False)
        if len(vectors) != len(vector_ids):
            raise ValueError("vectors and vector_ids must align")
        if not vectors:
            return
        faiss_ids = np.array([vector_uuid_to_faiss_int(vid) for vid in vector_ids], dtype=np.int64)
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
            oversample = max(1, int(getattr(self._config, "oversample", 2)))
            ntrain = min(matrix.shape[0], nlist * oversample)
            train_target.train(matrix[:ntrain])
        self._index.add_with_ids(matrix, faiss_ids)
        self._dirty_deletes = 0
        self._needs_rebuild = False

    def remove(self, vector_ids: Sequence[str]) -> None:
        """Remove vectors from FAISS and the in-memory cache by vector UUID.

        Args:
            vector_ids: Identifiers of vectors to remove.

        Returns:
            None
        """
        if not vector_ids:
            return
        ids = np.array([vector_uuid_to_faiss_int(vid) for vid in vector_ids], dtype=np.int64)
        self.remove_ids(ids, force_flush=True)

    def remove_ids(self, ids: np.ndarray, *, force_flush: bool = False) -> int:
        """Remove vectors by FAISS integer identifiers with optional batching.

        Args:
            ids: Array of FAISS internal identifiers to remove.
            force_flush: When True, rebuild immediately instead of batching.

        Returns:
            Number of identifiers scheduled for removal.
        """

        ids64 = np.asarray(ids, dtype=np.int64)
        if ids64.size == 0:
            return 0
        self._remove_ids(ids64)
        self._flush_pending_deletes(force=force_flush)
        return int(ids64.size)

    def _current_index_ids(self) -> np.ndarray:
        """Return the FAISS internal identifiers currently stored in the index.

        Args:
            None

        Returns:
            NumPy array containing the FAISS internal IDs present in the index.
        """

        if self._index is None or self._index.ntotal == 0:
            return np.empty(0, dtype=np.int64)
        return np.asarray(faiss.vector_to_array(self._index.id_map), dtype=np.int64)

    def _lookup_existing_ids(self, candidate_ids: np.ndarray) -> np.ndarray:
        """Identify which of the supplied FAISS IDs already exist in the index.

        Args:
            candidate_ids: Array of FAISS internal IDs to test for membership.

        Returns:
            NumPy array containing the subset of ``candidate_ids`` currently stored.
        """

        if candidate_ids.size == 0 or self._index.ntotal == 0:
            return np.empty(0, dtype=np.int64)
        current_ids = self._current_index_ids()
        mask = np.isin(current_ids, candidate_ids)
        return current_ids[mask]

    def search(self, query: np.ndarray, top_k: int) -> List[FaissSearchResult]:
        """Execute a cosine-similarity search returning the best `top_k` results.

        Args:
            query: Query vector to search against the index.
            top_k: Maximum number of nearest neighbours to return.

        Returns:
            List of `FaissSearchResult` objects ordered by score.
        """
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

    def serialize(self) -> bytes:
        """Serialize the FAISS index to bytes.

        Args:
            None

        Returns:
            Byte string containing the serialized FAISS index.

        Raises:
            RuntimeError: If the index has not been initialised.
        """

        if self._index is None:
            raise RuntimeError("index is empty")
        cpu_index = self._to_cpu(self._index)
        blob = faiss.serialize_index(cpu_index)
        return bytes(blob)

    def save(self, path: str) -> None:
        """Persist the FAISS index to disk without CPU fallbacks.

        Args:
            path: Destination filepath for the serialized FAISS index.

        Returns:
            None

        Raises:
            RuntimeError: If the index has not been initialised.
        """

        if self._index is None:
            raise RuntimeError("index is empty")
        # NOTE: FAISS disk writer APIs are intentionally NOT used because the GPU wheel
        # lacks writer hooks for GPU-backed IndexIDMap2 layouts. Byte serialization is canonical.
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.serialize())

    @classmethod
    def load(cls, path: str, config: DenseIndexConfig, dim: int) -> "FaissIndexManager":
        """Load a FAISS index from disk and ensure it resides on GPU.

        Args:
            path: Filesystem path pointing to a serialized FAISS index.
            config: Dense index configuration used to rebuild runtime properties.
            dim: Dimensionality of the vectors contained in the index.

        Returns:
            Instance of ``FaissIndexManager`` with GPU state initialised.

        Raises:
            RuntimeError: If the FAISS index cannot be read or promoted to GPU memory.
        """

        blob = Path(path).read_bytes()
        manager = cls(dim=dim, config=config)
        manager.restore(blob)
        return manager

    def restore(self, payload: bytes) -> None:
        """Restore FAISS state from bytes produced by `serialize`.

        Args:
            payload: Bytes previously produced by `serialize`.

        Returns:
            None

        Raises:
            ValueError: If the payload is invalid or incompatible.
        """
        if not payload:
            raise ValueError("Empty FAISS payload")
        cpu_index = faiss.deserialize_index(np.frombuffer(payload, dtype=np.uint8))
        self._index = self._maybe_to_gpu(cpu_index)
        self._tombstones.clear()
        self._dirty_deletes = 0
        self._needs_rebuild = False
        self._set_nprobe()

    def stats(self) -> dict[str, float | str]:
        """Expose diagnostic metrics for monitoring.

        Args:
            None

        Returns:
            Dictionary containing index configuration and diagnostics.
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

    def _create_index(self) -> "faiss.Index":
        """Construct a GPU-native FAISS index based on the configured index_type.

        Args:
            None

        Returns:
            Initialised ``faiss.Index`` residing on the configured GPU.

        Raises:
            RuntimeError: If GPU resources have not been initialised before invocation.
        """

        self.init_gpu()
        if self._gpu_resources is None:
            raise RuntimeError("GPU resources must be initialised before index creation")
        metric = faiss.METRIC_INNER_PRODUCT
        dev = int(self.device)
        index_type = self._config.index_type

        if index_type == "flat":
            cfg = faiss.GpuIndexFlatConfig() if hasattr(faiss, "GpuIndexFlatConfig") else None
            if cfg is not None:
                cfg.device = dev
                if hasattr(cfg, "useFloat16"):
                    cfg.useFloat16 = bool(getattr(self._config, "flat_use_fp16", False))
                base = faiss.GpuIndexFlatIP(self._gpu_resources, self._dim, cfg)
            else:
                base = faiss.GpuIndexFlatIP(self._gpu_resources, self._dim)
            index: "faiss.Index" = faiss.IndexIDMap2(base)
        elif index_type == "ivf_flat":
            q_cfg = faiss.GpuIndexFlatConfig() if hasattr(faiss, "GpuIndexFlatConfig") else None
            if q_cfg is not None:
                q_cfg.device = dev
                if hasattr(q_cfg, "useFloat16"):
                    q_cfg.useFloat16 = bool(getattr(self._config, "flat_use_fp16", False))
                quantizer = faiss.GpuIndexFlatIP(self._gpu_resources, self._dim, q_cfg)
            else:
                quantizer = faiss.GpuIndexFlatIP(self._gpu_resources, self._dim)
            cfg = faiss.GpuIndexIVFFlatConfig() if hasattr(faiss, "GpuIndexIVFFlatConfig") else None
            if cfg is not None:
                cfg.device = dev
                if hasattr(cfg, "interleavedLayout"):
                    cfg.interleavedLayout = bool(getattr(self._config, "interleaved_layout", True))
                base = faiss.GpuIndexIVFFlat(
                    self._gpu_resources,
                    quantizer,
                    self._dim,
                    int(self._config.nlist),
                    metric,
                    cfg,
                )
            else:
                base = faiss.GpuIndexIVFFlat(
                    self._gpu_resources,
                    quantizer,
                    self._dim,
                    int(self._config.nlist),
                    metric,
                )
            base.nprobe = int(self._config.nprobe)
            setattr(base, "_quantizer_ref", quantizer)
            index = faiss.IndexIDMap2(base)
        elif index_type == "ivf_pq":
            cfg = faiss.GpuIndexIVFPQConfig() if hasattr(faiss, "GpuIndexIVFPQConfig") else None
            if cfg is not None:
                cfg.device = dev
                if hasattr(cfg, "interleavedLayout"):
                    cfg.interleavedLayout = bool(getattr(self._config, "interleaved_layout", True))
                if hasattr(cfg, "usePrecomputedTables"):
                    cfg.usePrecomputedTables = bool(
                        getattr(self._config, "ivfpq_use_precomputed", True)
                    )
                if hasattr(cfg, "useFloat16LookupTables"):
                    cfg.useFloat16LookupTables = bool(
                        getattr(self._config, "ivfpq_float16_lut", True)
                    )
                base = faiss.GpuIndexIVFPQ(
                    self._gpu_resources,
                    self._dim,
                    int(self._config.nlist),
                    int(self._config.pq_m),
                    int(self._config.pq_bits),
                    metric,
                    cfg,
                )
            else:
                base = faiss.GpuIndexIVFPQ(
                    self._gpu_resources,
                    self._dim,
                    int(self._config.nlist),
                    int(self._config.pq_m),
                    int(self._config.pq_bits),
                    metric,
                )
            base.nprobe = int(self._config.nprobe)
            index = faiss.IndexIDMap2(base)
        else:
            raise ValueError(f"Unsupported index_type: {index_type}")

        self._maybe_reserve_memory(index)
        if self._multi_gpu_mode == "replicate":
            index = self.replicate_to_all_gpus(index)

        return index

    def replicate_to_all_gpus(self, index: "faiss.Index | None" = None) -> "faiss.Index":
        """Replicate a FAISS index across every visible GPU device.

        Args:
            index: Optional FAISS index to clone; defaults to the actively managed index.

        Returns:
            FAISS index handle representing the replicated multi-GPU index.

        Raises:
            RuntimeError: If multi-GPU replication is unsupported or no index is available.
        """
        if not hasattr(faiss, "index_cpu_to_all_gpus"):
            raise RuntimeError("FAISS build does not support multi-GPU replication")
        target = index or getattr(self, "_index", None)
        if target is None:
            raise RuntimeError("No FAISS index available to replicate")
        base = getattr(target, "index", None) or target
        if "GpuIndex" in type(base).__name__ and hasattr(faiss, "index_gpu_to_cpu"):
            cpu = faiss.index_gpu_to_cpu(target)
        else:
            cpu = target
        co = (
            faiss.GpuMultipleClonerOptions() if hasattr(faiss, "GpuMultipleClonerOptions") else None
        )
        if co is not None:
            co.shard = False
            co.verbose = True
            co.allowCpuCoarseQuantizer = False
            if (
                not self._force_64bit_ids
                and self._indices_32_bit
                and hasattr(faiss, "INDICES_32_BIT")
            ):
                co.indicesOptions = faiss.INDICES_32_BIT
        replicated = faiss.index_cpu_to_all_gpus(cpu, co=co)
        self._replicated = True
        return replicated

    def _maybe_to_gpu(self, index: "faiss.Index") -> "faiss.Index":
        """Promote a CPU index to GPU while enforcing strict cloning guarantees.

        Args:
            index: CPU-resident FAISS index to migrate onto the configured GPU(s).

        Returns:
            GPU-backed FAISS index, optionally replicated across devices.

        Raises:
            RuntimeError: If GPU resources are unavailable or promotion fails.
        """

        self.init_gpu()
        if self._gpu_resources is None:
            raise RuntimeError("GPU resources are not initialised")
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
                return (
                    self.replicate_to_all_gpus(
                        faiss.index_cpu_to_gpu(self._gpu_resources, device, index, co)
                    )
                    if self._multi_gpu_mode == "replicate"
                    else faiss.index_cpu_to_gpu(self._gpu_resources, device, index, co)
                )
            cloned = faiss.index_cpu_to_gpu(self._gpu_resources, device, index)
            return (
                self.replicate_to_all_gpus(cloned)
                if self._multi_gpu_mode == "replicate"
                else cloned
            )
        except Exception as exc:  # pragma: no cover - hardware specific failure
            raise RuntimeError(
                "Failed to promote FAISS index to GPU "
                f"(index type={type(index).__name__}, device={device}): {exc}"
            ) from exc

    def _maybe_reserve_memory(self, index: "faiss.Index") -> None:
        """Reserve GPU memory for the expected corpus size when supported.

        Args:
            index: Target FAISS index whose underlying storage may be extended.

        Returns:
            None
        """

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
        """Clone a GPU index back to CPU memory for persistence operations.

        Args:
            index: GPU-resident FAISS index to convert to CPU representation.

        Returns:
            CPU-backed FAISS index ready for serialization.

        Raises:
            RuntimeError: If FAISS lacks GPU→CPU conversion support.
        """

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
        """Apply IVF search breadth configuration to the current index.

        Args:
            None

        Returns:
            None
        """

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
        """Emit structured logging that captures the live index configuration.

        Args:
            index: FAISS index whose configuration should be summarised.

        Returns:
            None
        """

        if index is None:
            return
        device = self._detect_device(index)
        event = {
            "index_type": self._config.index_type,
            "nlist": int(self._config.nlist),
            "nprobe": int(self._config.nprobe),
            "pq_m": int(self._config.pq_m),
            "pq_bits": int(self._config.pq_bits),
            "device": device,
        }
        logger.info("faiss-index-config", extra={"event": event})

    def _resolve_vector_id(self, internal_id: int) -> Optional[str]:
        """Translate a FAISS internal identifier to the original vector UUID.

        Args:
            internal_id: FAISS-managed integer identifier produced during search.

        Returns:
            Matching vector UUID when available, otherwise ``None``.
        """

        if self._id_resolver is None:
            return None
        try:
            return self._id_resolver(internal_id)
        except Exception:  # pragma: no cover - resolver is user supplied
            logger.debug("id-resolver failure", exc_info=True)
            return None

    def _detect_device(self, index: "faiss.Index") -> Optional[int]:
        """Determine the CUDA device hosting the provided index, if any.

        Args:
            index: FAISS index whose device affinity is being inspected.

        Returns:
            CUDA device ordinal when discoverable, otherwise ``None``.
        """

        base = index.index if hasattr(index, "index") else index
        candidate = base
        if hasattr(faiss, "downcast_index"):
            try:
                candidate = faiss.downcast_index(base)
            except Exception:  # pragma: no cover - best-effort downcast
                candidate = base
        if hasattr(candidate, "getDevice"):
            try:
                return int(candidate.getDevice())
            except Exception:  # pragma: no cover - defensive guard
                return None
        return None

    def _resolve_device(self, config: DenseIndexConfig) -> int:
        """Determine the target GPU device, honouring runtime overrides.

        Args:
            config: Dense index configuration supplying the default device.

        Returns:
            int: GPU device identifier to use for FAISS operations.
        """

        return int(getattr(config, "device", 0))

    def init_gpu(self) -> None:
        """Initialise FAISS GPU resources in line with configuration settings.

        Args:
            None

        Returns:
            None

        Raises:
            RuntimeError: If FAISS lacks GPU support or an unsuitable device is requested.
        """
        if self._gpu_resources is not None:
            return
        if not hasattr(faiss, "StandardGpuResources"):
            raise RuntimeError(
                "FAISS GPU resources are unavailable; ensure faiss-gpu is installed with CUDA support."
            )
        if hasattr(faiss, "get_num_gpus"):
            try:
                num_gpus = int(faiss.get_num_gpus())
            except Exception as exc:  # pragma: no cover - hardware query failure
                raise RuntimeError("Unable to enumerate FAISS GPUs") from exc
            device = int(self.device)
            if num_gpus <= device:
                raise RuntimeError(
                    f"Requested GPU device {device} but only {num_gpus} device(s) visible"
                )
        try:
            resources = faiss.StandardGpuResources()
        except Exception as exc:  # pragma: no cover - hardware specific failure
            raise RuntimeError(
                f"Unable to initialise FAISS GPU resources (StandardGpuResources): {exc}"
            ) from exc
        if self._temp_memory_bytes:
            try:
                resources.setTempMemory(int(self._temp_memory_bytes))
            except Exception:  # pragma: no cover - advisory only
                logger.debug(
                    "Unable to set FAISS GPU temp memory",
                    extra={"event": {"requested_bytes": self._temp_memory_bytes}},
                    exc_info=True,
                )
        self._gpu_resources = resources

    def _ensure_dim(self, vector: np.ndarray) -> np.ndarray:
        """Validate that the provided embedding matches the configured dimensionality.

        Args:
            vector: Vector whose shape should align with the configured dimension.

        Returns:
            NumPy array validated to match ``self._dim``.

        Raises:
            ValueError: If the vector dimensionality differs from ``self._dim``.
        """

        if vector.shape != (self._dim,):
            raise ValueError(f"Expected embedding dimension {self._dim}, got {vector.shape}")
        return vector

    def _flush_pending_deletes(self, *, force: bool = False) -> None:
        """Rebuild the index when tombstone thresholds or manual flush requests demand it.

        Args:
            force: When ``True``, rebuild even if thresholds have not been reached.

        Returns:
            None
        """

        if not self._needs_rebuild and not force:
            return
        if not self._tombstones:
            self._dirty_deletes = 0
            self._needs_rebuild = False
            return
        threshold = int(self._rebuild_delete_threshold)
        if force or threshold <= 0 or self._dirty_deletes >= threshold:
            self._rebuild_index()
            self._tombstones.clear()
            self._dirty_deletes = 0
            self._needs_rebuild = False

    def _remove_ids(self, ids: np.ndarray) -> None:
        """Attempt to delete FAISS IDs directly, falling back to tombstones when required.

        Args:
            ids: Array of FAISS internal identifiers targeted for removal.

        Returns:
            None
        """

        if ids.size == 0:
            return
        selector = faiss.IDSelectorBatch(ids.astype(np.int64))
        try:
            removed = int(self._index.remove_ids(selector))
        except RuntimeError as exc:
            if "remove_ids not implemented" not in str(exc).lower():
                raise
            logger.warning(
                "FAISS remove_ids not implemented on GPU index; scheduling rebuild.",
                extra={"event": {"ntotal": self.ntotal, "remove_ids_error": str(exc)}},
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

    def _rebuild_index(self) -> None:
        """Recreate the GPU index from live vectors after tombstones accumulate.

        Args:
            None

        Returns:
            None
        """

        # Recreate the FAISS index on GPU using cached vectors when direct removal is unsupported.
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
                oversample = max(1, int(getattr(self._config, "oversample", 2)))
                ntrain = min(vectors.shape[0], nlist * oversample)
                train_target.train(vectors[:ntrain])
            self._index.add_with_ids(vectors, survivor_ids.astype(np.int64))
