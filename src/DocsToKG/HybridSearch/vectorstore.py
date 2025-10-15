"""Unified FAISS vector store and similarity utilities."""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional, Sequence

import numpy as np

from .config import DenseIndexConfig
from .types import vector_uuid_to_faiss_int

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .storage import ChunkRegistry

logger = logging.getLogger(__name__)

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


@dataclass(slots=True)
class FaissSearchResult:
    """Dense search hit returned by FAISS."""

    vector_id: str
    score: float


class FaissVectorStore:
    """Manage lifecycle of a GPU-resident FAISS index and related utilities."""

    def __init__(self, dim: int, config: DenseIndexConfig) -> None:
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
        self._supports_remove_ids: Optional[bool] = None
        self._set_nprobe()

    @property
    def ntotal(self) -> int:
        return int(self._index.ntotal)

    @property
    def config(self) -> DenseIndexConfig:
        return self._config

    @property
    def gpu_resources(self) -> "faiss.StandardGpuResources | None":
        return self._gpu_resources

    @property
    def device(self) -> int:
        return int(getattr(self._config, "device", self._device))

    def set_id_resolver(self, resolver: Callable[[int], Optional[str]]) -> None:
        self._id_resolver = resolver

    def train(self, vectors: Sequence[np.ndarray]) -> None:
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
        is_trained = getattr(self._index, "is_trained", True)
        return not bool(is_trained)

    def add(self, vectors: Sequence[np.ndarray], vector_ids: Sequence[str]) -> None:
        self._flush_pending_deletes(force=False)
        if len(vectors) != len(vector_ids):
            raise ValueError("vectors and vector_ids must align")
        if not vectors:
            return
        faiss_ids = np.array([vector_uuid_to_faiss_int(vid) for vid in vector_ids], dtype=np.int64)
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
            oversample = max(1, int(getattr(self._config, "oversample", 2)))
            ntrain = min(matrix.shape[0], nlist * oversample)
            train_target.train(matrix[:ntrain])
        self._index.add_with_ids(matrix, faiss_ids)
        self._dirty_deletes = 0
        self._needs_rebuild = False

    def remove(self, vector_ids: Sequence[str]) -> None:
        if not vector_ids:
            return
        ids = np.array([vector_uuid_to_faiss_int(vid) for vid in vector_ids], dtype=np.int64)
        self.remove_ids(ids, force_flush=True)

    def remove_ids(self, ids: np.ndarray, *, force_flush: bool = False) -> int:
        ids64 = np.asarray(ids, dtype=np.int64)
        if ids64.size == 0:
            return 0
        self._remove_ids(ids64)
        self._flush_pending_deletes(force=force_flush)
        return int(ids64.size)

    def _current_index_ids(self) -> np.ndarray:
        if self._index is None or self._index.ntotal == 0:
            return np.empty(0, dtype=np.int64)
        return np.asarray(faiss.vector_to_array(self._index.id_map), dtype=np.int64)

    def _lookup_existing_ids(self, candidate_ids: np.ndarray) -> np.ndarray:
        if candidate_ids.size == 0 or self._index.ntotal == 0:
            return np.empty(0, dtype=np.int64)
        current_ids = self._current_index_ids()
        mask = np.isin(current_ids, candidate_ids)
        return current_ids[mask]

    def search(self, query: np.ndarray, top_k: int) -> List[FaissSearchResult]:
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
        if self._index is None:
            raise RuntimeError("index is empty")
        cpu_index = self._to_cpu(self._index)
        blob = faiss.serialize_index(cpu_index)
        return bytes(blob)

    def save(self, path: str) -> None:
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
        blob = Path(path).read_bytes()
        manager = cls(dim=dim, config=config)
        manager.restore(blob)
        return manager

    def restore(self, payload: bytes) -> None:
        if not payload:
            raise ValueError("Empty FAISS payload")
        cpu_index = faiss.deserialize_index(np.frombuffer(payload, dtype=np.uint8))
        self._index = self._maybe_to_gpu(cpu_index)
        self._tombstones.clear()
        self._dirty_deletes = 0
        self._needs_rebuild = False
        self._set_nprobe()

    def stats(self) -> dict[str, float | str]:
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
        if self._needs_rebuild:
            return True
        if self._rebuild_delete_threshold <= 0:
            return False
        return self._dirty_deletes >= self._rebuild_delete_threshold

    def init_gpu(self) -> None:
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
                        faiss.INDICES_32_BIT if self._indices_32_bit and not self._force_64bit_ids else 0
                    )
                index = faiss.GpuIndexFlatIP(self._gpu_resources, self._dim, cfg)
            else:
                index = faiss.GpuIndexFlatIP(self._gpu_resources, self._dim, faiss.METRIC_INNER_PRODUCT)
            self._maybe_reserve_memory(index)
            return faiss.IndexIDMap2(index)

        quantizer = faiss.IndexFlatIP(self._dim)
        if index_type == "ivf_flat":
            base = faiss.IndexIVFFlat(quantizer, self._dim, int(self._config.nlist), metric)
        elif index_type == "ivf_pq":
            base = faiss.IndexIVFPQ(
                quantizer,
                self._dim,
                int(self._config.nlist),
                int(self._config.pq_m),
                int(self._config.pq_bits),
            )
        else:
            raise RuntimeError(f"Unsupported index type: {index_type}")
        base.nprobe = int(self._config.nprobe)
        index = faiss.IndexIDMap2(base)
        gpu_index = self._maybe_to_gpu(index)
        self._maybe_reserve_memory(gpu_index)
        return gpu_index

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
                oversample = max(1, int(getattr(self._config, "oversample", 2)))
                ntrain = min(vectors.shape[0], nlist * oversample)
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


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.dtype != np.float32 or not matrix.flags.c_contiguous:
        raise TypeError("normalize_rows expects a contiguous float32 array")
    if hasattr(faiss, "normalize_L2"):
        faiss.normalize_L2(matrix)
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    matrix /= norms
    return matrix


def cosine_against_corpus_gpu(
    query: np.ndarray,
    corpus: np.ndarray,
    *,
    device: int = 0,
    resources: Optional["faiss.StandardGpuResources"] = None,
) -> np.ndarray:
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
    if q.ndim == 1:
        q = q.reshape(1, -1)
    q = np.ascontiguousarray(q, dtype=np.float32).copy()
    C = np.ascontiguousarray(C, dtype=np.float32).copy()
    faiss.normalize_L2(q)
    faiss.normalize_L2(C)
    return faiss.pairwise_distance_gpu(
        resources,
        q,
        C,
        metric=faiss.METRIC_INNER_PRODUCT,
        device=int(device),
    )


def serialize_state(faiss_index: FaissVectorStore, registry: "ChunkRegistry") -> dict[str, object]:
    faiss_bytes = faiss_index.serialize()
    encoded = base64.b64encode(faiss_bytes).decode("ascii")
    return {
        "faiss": encoded,
        "vector_ids": registry.vector_ids(),
    }


def restore_state(faiss_index: FaissVectorStore, payload: dict[str, object]) -> None:
    encoded = payload.get("faiss")
    if not isinstance(encoded, str):
        raise ValueError("Missing FAISS payload")
    faiss_index.restore(base64.b64decode(encoded.encode("ascii")))


# Backwards compatibility alias for legacy imports.
FaissIndexManager = FaissVectorStore
