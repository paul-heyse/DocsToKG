"""GPU-only FAISS index management for dense retrieval."""

from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .config import DenseIndexConfig
from .faiss_gpu import GPUOpts, gpu_index_factory, maybe_clone_to_gpu
from .ids import vector_uuid_to_faiss_int

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised indirectly via integration tests
    import faiss  # type: ignore

    _FAISS_AVAILABLE = all(
        hasattr(faiss, attr)
        for attr in (
            "GpuIndexFlatIP",
            "IndexFlatIP",
            "IndexIDMap2",
            "index_factory",
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
        self._replicated = False
        self._gpu_opts = GPUOpts(
            device=self._device,
            ivfpq_use_precomputed=bool(getattr(config, "ivfpq_use_precomputed", True)),
            ivfpq_float16_lut=bool(getattr(config, "ivfpq_float16_lut", True)),
        )
        self._gpu_resources = self._init_gpu_resources()
        self._index = self._create_index()
        self._vectors: Dict[str, np.ndarray] = {}
        self._id_resolver: Optional[Callable[[int], Optional[str]]] = None
        self._use_native = True  # Backwards compat for older tests expecting this flag
        self._remove_fallbacks = 0
        self._apply_search_parameters(self._index)

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
        if len(vectors) != len(vector_ids):
            raise ValueError("vectors and vector_ids must align")
        if not vectors:
            return
        existing_ids = [vector_id for vector_id in vector_ids if vector_id in self._vectors]
        if existing_ids:
            for vector_id in existing_ids:
                self._vectors.pop(vector_id, None)
            ids = np.array([vector_uuid_to_faiss_int(vid) for vid in existing_ids], dtype=np.int64)
            self._remove_ids(ids)
        matrix = np.ascontiguousarray(
            np.stack([self._ensure_dim(vec) for vec in vectors]), dtype=np.float32
        )
        faiss.normalize_L2(matrix)
        if hasattr(self._index, "is_trained") and not getattr(self._index, "is_trained"):
            nlist = int(getattr(self._config, "nlist", 1024))
            oversample = max(1, int(getattr(self._config, "oversample", 2)))
            ntrain = min(matrix.shape[0], nlist * oversample)
            self._index.train(matrix[:ntrain])
        ids = np.array([vector_uuid_to_faiss_int(vid) for vid in vector_ids], dtype=np.int64)
        self._index.add_with_ids(matrix, ids)
        for row, vector_id in zip(matrix, vector_ids):
            self._vectors[vector_id] = row.copy()

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
        for vector_id in vector_ids:
            self._vectors.pop(vector_id, None)
        self._remove_ids(ids)

    def search(self, query: np.ndarray, top_k: int) -> List[FaissSearchResult]:
        """Execute a cosine-similarity search returning the best `top_k` results.

        Args:
            query: Query vector to search against the index.
            top_k: Maximum number of nearest neighbours to return.

        Returns:
            List of `FaissSearchResult` objects ordered by score.
        """
        query_matrix = self._ensure_dim(query).reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_matrix)
        if self._config.index_type.startswith("ivf"):
            nprobe = int(self._config.nprobe)
            nprobe_set = False
            if hasattr(faiss, "GpuParameterSpace"):
                try:
                    faiss.GpuParameterSpace().set_index_parameter(self._index, "nprobe", nprobe)
                    nprobe_set = True
                except Exception:  # pragma: no cover - defensive guard
                    logger.debug("GpuParameterSpace failed to set nprobe", exc_info=True)
            if not nprobe_set and hasattr(faiss, "ParameterSpace"):
                try:
                    faiss.ParameterSpace().set_index_parameter(self._index, "nprobe", nprobe)
                    nprobe_set = True
                except Exception:  # pragma: no cover - defensive guard
                    logger.debug("ParameterSpace failed to set nprobe", exc_info=True)
            if not nprobe_set:
                target = self._index
                if hasattr(target, "index"):
                    target = target.index  # IndexIDMap2 wraps the GPU index
                if hasattr(faiss, "downcast_index"):
                    try:  # pragma: no cover - best-effort GPU cast
                        target = faiss.downcast_index(target)
                    except Exception:
                        target = target
                try:
                    if hasattr(target, "nprobe"):
                        target.nprobe = nprobe
                    elif hasattr(self._index, "nprobe"):
                        self._index.nprobe = nprobe
                except Exception:  # pragma: no cover - defensive guard for exotic indexes
                    logger.debug("Unable to set nprobe on FAISS index during search", exc_info=True)
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
        """Serialize the FAISS index and cached vectors for persistence.

        Args:
            None

        Returns:
            Bytes object containing serialized index and vector cache.
        """
        cpu_index = self._to_cpu(self._index)
        index_bytes = faiss.serialize_index(cpu_index)
        payload = {
            "mode": "native",
            "index": base64.b64encode(index_bytes).decode("ascii"),
            "vectors": {
                vector_id: base64.b64encode(vector.tobytes()).decode("ascii")
                for vector_id, vector in self._vectors.items()
            },
        }
        return json.dumps(payload).encode("utf-8")

    def restore(self, payload: bytes) -> None:
        """Restore FAISS state from bytes produced by `serialize`.

        Args:
            payload: Bytes previously produced by `serialize`.

        Returns:
            None

        Raises:
            ValueError: If the payload is invalid or incompatible.
        """
        try:
            data = json.loads(payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("Invalid FAISS payload") from exc

        if data.get("mode") != "native":
            raise ValueError("Unsupported FAISS payload mode")

        encoded = data.get("index")
        if not isinstance(encoded, str):
            raise ValueError("Invalid FAISS payload")

        index_bytes = base64.b64decode(encoded.encode("ascii"))
        cpu_index = faiss.deserialize_index(np.frombuffer(index_bytes, dtype=np.uint8))
        self._index = self._maybe_to_gpu(cpu_index)
        self._apply_search_parameters(self._index)
        vectors_blob = data.get("vectors", {})
        restored: Dict[str, np.ndarray] = {}
        if isinstance(vectors_blob, dict):
            for vector_id, encoded in vectors_blob.items():
                try:
                    raw = base64.b64decode(str(encoded))
                except Exception as exc:
                    raise ValueError("Invalid FAISS vector payload") from exc
                vector = np.frombuffer(raw, dtype=np.float32)
                if vector.size != self._dim:
                    raise ValueError("Invalid FAISS vector dimension in payload")
                restored[vector_id] = vector.copy()
        self._vectors = restored

    def stats(self) -> Dict[str, float | str]:
        """Expose diagnostic metrics for monitoring.

        Args:
            None

        Returns:
            Dictionary containing index configuration and diagnostics.
        """
        device = self._detect_device(self._index)
        stats: Dict[str, float | str] = {
            "ntotal": float(self.ntotal),
            "index_type": self._config.index_type,
            "nlist": float(getattr(self._config, "nlist", 0)),
            "nprobe": float(getattr(self._config, "nprobe", 0)),
            "pq_m": float(getattr(self._config, "pq_m", 0)),
            "pq_bits": float(getattr(self._config, "pq_bits", 0)),
            "device": "*" if device is None else str(device),
            "gpu_remove_fallbacks": float(self._remove_fallbacks),
            "multi_gpu_mode": self._multi_gpu_mode,
            "gpu_indices_32_bit": str(self._indices_32_bit).lower(),
        }
        if self._temp_memory_bytes is not None:
            stats["gpu_temp_memory_bytes"] = float(self._temp_memory_bytes)
        if device is not None:
            stats["gpu_device"] = str(device)
            resources = self._gpu_resources
            if resources is not None and hasattr(resources, "getMemoryInfo"):
                try:
                    free, total = resources.getMemoryInfo(device)
                    stats["gpu_mem_free"] = float(free)
                    stats["gpu_mem_total"] = float(total)
                except Exception:  # pragma: no cover - not all builds expose memory info
                    logger.debug("FAISS GPU memory info unavailable", exc_info=True)
        return stats

    def _create_index(self) -> "faiss.Index":
        if self._multi_gpu_mode == "replicate":
            metric = faiss.METRIC_INNER_PRODUCT
            spec_map = {
                "flat": "Flat",
                "ivf_flat": f"IVF{int(self._config.nlist)},Flat",
                "ivf_pq": f"IVF{int(self._config.nlist)},PQ{int(self._config.pq_m)}x{int(self._config.pq_bits)}",
            }
            try:
                spec = spec_map[self._config.index_type]
            except KeyError as exc:
                raise ValueError(f"Unsupported FAISS index type: {self._config.index_type}") from exc
            cpu_index = faiss.index_factory(self._dim, spec, metric)
            mapped = faiss.IndexIDMap2(cpu_index)
            return self._maybe_to_gpu(mapped)

        try:
            return gpu_index_factory(
                self._dim,
                index_type=self._config.index_type,
                nlist=int(self._config.nlist),
                nprobe=int(self._config.nprobe),
                pq_m=int(self._config.pq_m),
                pq_bits=int(self._config.pq_bits),
                resources=self._gpu_resources,
                opts=self._gpu_opts,
            )
        except ValueError as exc:
            raise ValueError(f"Unsupported FAISS index type: {self._config.index_type}") from exc

    def _maybe_to_gpu(self, index: "faiss.Index") -> "faiss.Index":
        if self._multi_gpu_mode == "replicate":
            if not hasattr(faiss, "index_cpu_to_all_gpus"):
                raise RuntimeError(
                    "FAISS missing index_cpu_to_all_gpus; multi_gpu_mode='replicate' unavailable"
                )
            try:
                cloner = faiss.GpuMultipleClonerOptions() if hasattr(faiss, "GpuMultipleClonerOptions") else None
                if cloner is not None:
                    cloner.shard = False
                    cloner.verbose = True
                    cloner.allowCpuCoarseQuantizer = False
                    if hasattr(cloner, "indicesOptions") and self._indices_32_bit and hasattr(
                        faiss, "INDICES_32_BIT"
                    ):
                        cloner.indicesOptions = faiss.INDICES_32_BIT
                    if hasattr(cloner, "usePrecomputed"):
                        cloner.usePrecomputed = bool(
                            getattr(self._config, "ivfpq_use_precomputed", True)
                            if self._config.index_type == "ivf_pq"
                            else False
                        )
                    if hasattr(cloner, "useFloat16"):
                        cloner.useFloat16 = bool(
                            getattr(self._config, "ivfpq_float16_lut", True)
                            if self._config.index_type == "ivf_pq"
                            else False
                        )
                    if hasattr(cloner, "useFloat16CoarseQuantizer"):
                        cloner.useFloat16CoarseQuantizer = False
                gpu_index = faiss.index_cpu_to_all_gpus(index, co=cloner)
            except Exception as exc:  # pragma: no cover - hardware specific failure
                raise RuntimeError(
                    "Failed to replicate FAISS index across GPUs "
                    f"(index type={type(index).__name__}): {exc}"
                ) from exc
            logger.info("FAISS index replicated across available GPUs")
            self._replicated = True
            return gpu_index

        if self._gpu_resources is None:
            raise RuntimeError("GPU resources are not initialised")
        try:
            gpu_index = maybe_clone_to_gpu(
                index,
                device=self._device,
                resources=self._gpu_resources,
                indices_32_bits=self._indices_32_bit,
            )
        except Exception as exc:  # pragma: no cover - hardware specific failure
            raise RuntimeError(
                "Failed to promote FAISS index to GPU "
                f"(index type={type(index).__name__}, device={self._device}): {exc}"
            ) from exc
        logger.info("FAISS index promoted to GPU")
        return gpu_index

    def _to_cpu(self, index: "faiss.Index") -> "faiss.Index":
        if hasattr(faiss, "index_gpu_to_cpu"):
            try:
                return faiss.index_gpu_to_cpu(index)
            except Exception as exc:  # pragma: no cover - hardware specific failure
                raise RuntimeError(
                    f"Unable to transfer FAISS index from GPU to CPU for serialization: {exc}"
                ) from exc
        raise RuntimeError(
            "FAISS index_gpu_to_cpu is unavailable; install faiss-gpu>=1.7.4 with GPU support."
        )

    def _apply_search_parameters(self, index: "faiss.Index | None") -> None:
        if index is None:
            return

        if self._config.index_type.startswith("ivf"):
            nprobe = int(self._config.nprobe)
            base = index.index if hasattr(index, "index") else index
            if hasattr(base, "nprobe"):
                try:
                    base.nprobe = nprobe
                    self._log_index_configuration(index)
                    return
                except Exception:  # pragma: no cover - parameter guard
                    logger.debug("Unable to set nprobe directly on FAISS index", exc_info=True)
            handled = False
            if hasattr(faiss, "GpuParameterSpace"):
                try:
                    faiss.GpuParameterSpace().set_index_parameter(index, "nprobe", nprobe)
                    handled = True
                except Exception:  # pragma: no cover - GPU parameter guard
                    logger.debug("Unable to set nprobe via GpuParameterSpace", exc_info=True)
                    handled = False
            if not handled and hasattr(faiss, "ParameterSpace"):
                try:
                    space = faiss.ParameterSpace()
                    space.set_index_parameter(index, "nprobe", nprobe)
                    handled = True
                except Exception:  # pragma: no cover - parameter guard
                    logger.debug("Unable to set nprobe via ParameterSpace", exc_info=True)
                    handled = False
            if not handled:
                raise RuntimeError(
                    "Unable to configure FAISS nprobe parameter on IVF index; "
                    "confirm faiss-gpu is fully installed and the index type supports nprobe."
                )

        self._log_index_configuration(index)

    def _log_index_configuration(self, index: "faiss.Index") -> None:
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
        if self._id_resolver is None:
            return None
        try:
            return self._id_resolver(internal_id)
        except Exception:  # pragma: no cover - resolver is user supplied
            logger.debug("id-resolver failure", exc_info=True)
            return None

    def _detect_device(self, index: "faiss.Index") -> Optional[int]:
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

        env_value = os.getenv("HYBRIDSEARCH_FAISS_DEVICE")
        if env_value is not None:
            try:
                return int(env_value)
            except ValueError:
                logger.warning(
                    "Invalid HYBRIDSEARCH_FAISS_DEVICE value '%s'; falling back to config",
                    env_value,
                )
        return int(getattr(config, "device", 0))

    def _init_gpu_resources(self) -> "faiss.StandardGpuResources":
        if not hasattr(faiss, "StandardGpuResources"):
            raise RuntimeError(
                "FAISS GPU resources are unavailable; ensure faiss-gpu is installed with CUDA support."
            )
        if hasattr(faiss, "get_num_gpus"):
            try:
                num_gpus = int(faiss.get_num_gpus())
            except Exception as exc:  # pragma: no cover - hardware query failure
                raise RuntimeError("Unable to enumerate FAISS GPUs") from exc
            if num_gpus <= self._device:
                raise RuntimeError(
                    f"Requested GPU device {self._device} but only {num_gpus} device(s) visible"
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
        return resources

    def _ensure_dim(self, vector: np.ndarray) -> np.ndarray:
        if vector.shape != (self._dim,):
            raise ValueError(f"Expected embedding dimension {self._dim}, got {vector.shape}")
        return vector

    def _remove_ids(self, ids: np.ndarray) -> None:
        if ids.size == 0:
            return
        selector = faiss.IDSelectorBatch(ids.astype(np.int64))
        try:
            self._index.remove_ids(selector)
        except RuntimeError as exc:
            if "remove_ids not implemented" not in str(exc).lower():
                raise
            logger.warning(
                "FAISS remove_ids not implemented on GPU index; rebuilding index from cached vectors.",
                extra={"event": {"ntotal": self.ntotal, "remove_ids_error": str(exc)}},
            )
            self._remove_fallbacks += 1
            self._rebuild_index()

    def _rebuild_index(self) -> None:
        # Recreate the FAISS index on GPU using cached vectors when direct removal is unsupported.
        vector_items = list(self._vectors.items())
        self._index = self._create_index()
        self._apply_search_parameters(self._index)
        if not vector_items:
            return
        vector_ids = [item[0] for item in vector_items]
        matrix = np.stack([item[1] for item in vector_items]).astype(np.float32, copy=False)
        faiss.normalize_L2(matrix)
        if hasattr(self._index, "is_trained") and not getattr(self._index, "is_trained"):
            nlist = int(getattr(self._config, "nlist", 1024))
            oversample = max(1, int(getattr(self._config, "oversample", 2)))
            ntrain = min(matrix.shape[0], nlist * oversample)
            self._index.train(matrix[:ntrain])
        ids = np.array([vector_uuid_to_faiss_int(vid) for vid in vector_ids], dtype=np.int64)
        self._index.add_with_ids(matrix, ids)
