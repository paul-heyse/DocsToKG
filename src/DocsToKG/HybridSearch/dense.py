"""GPU-first FAISS index management for dense retrieval."""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

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
    """Dense search hit returned by FAISS."""

    vector_id: str
    score: float


class FaissIndexManager:
    """Manage lifecycle of a GPU-resident FAISS index."""

    def __init__(self, dim: int, config: DenseIndexConfig) -> None:
        if not _FAISS_AVAILABLE:
            raise RuntimeError(
                "FAISS GPU extensions are required for dense retrieval "
                "(missing GpuIndexFlatIP/index_cpu_to_gpu). Verify that faiss-gpu is installed "
                "and CUDA libraries are discoverable."
            )

        self._dim = dim
        self._config = config
        self._gpu_resources = self._init_gpu_resources()
        self._index = self._create_index()
        self._vectors: Dict[str, np.ndarray] = {}
        self._id_resolver: Optional[Callable[[int], Optional[str]]] = None
        self._use_native = True
        self._remove_fallbacks = 0
        self._apply_search_parameters(self._index)

    @property
    def ntotal(self) -> int:
        return int(self._index.ntotal)

    @property
    def config(self) -> DenseIndexConfig:
        return self._config

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
        self._index.train(matrix)

    def needs_training(self) -> bool:
        is_trained = getattr(self._index, "is_trained", True)
        return not bool(is_trained)

    def add(self, vectors: Sequence[np.ndarray], vector_ids: Sequence[str]) -> None:
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
        matrix = np.stack([self._ensure_dim(vec) for vec in vectors]).astype(np.float32)
        faiss.normalize_L2(matrix)
        ids = np.array([vector_uuid_to_faiss_int(vid) for vid in vector_ids], dtype=np.int64)
        self._index.add_with_ids(matrix, ids)
        for row, vector_id in zip(matrix, vector_ids):
            self._vectors[vector_id] = row.copy()

    def remove(self, vector_ids: Sequence[str]) -> None:
        if not vector_ids:
            return
        ids = np.array([vector_uuid_to_faiss_int(vid) for vid in vector_ids], dtype=np.int64)
        for vector_id in vector_ids:
            self._vectors.pop(vector_id, None)
        self._remove_ids(ids)

    def search(self, query: np.ndarray, top_k: int) -> List[FaissSearchResult]:
        query_matrix = self._ensure_dim(query).reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_matrix)
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
        device: Optional[int] = None
        if hasattr(self._index, "getDevice"):
            try:
                device = int(self._index.getDevice())
            except Exception:  # pragma: no cover - defensive logging guard
                device = None
        return {
            "ntotal": float(self.ntotal),
            "index_type": self._config.index_type,
            "nprobe": float(self._config.nprobe),
            "device": "*" if device is None else str(device),
            "gpu_remove_fallbacks": float(self._remove_fallbacks),
        }

    def _create_index(self) -> "faiss.Index":
        metric = faiss.METRIC_INNER_PRODUCT
        index_type = self._config.index_type

        if index_type == "flat":
            if self._gpu_resources is None:
                raise RuntimeError("GPU resources must be initialised before index creation")
            base = faiss.GpuIndexFlatIP(self._gpu_resources, self._dim)
            return faiss.IndexIDMap2(base)

        spec_map = {
            "flat": "Flat",
            "ivf_flat": f"IVF{self._config.nlist},Flat",
            "ivf_pq": f"IVF{self._config.nlist},PQ{self._config.pq_m}x{self._config.pq_bits}",
        }
        try:
            spec = spec_map[index_type]
        except KeyError as exc:
            raise ValueError(f"Unsupported FAISS index type: {index_type}") from exc

        cpu = faiss.index_factory(self._dim, spec, metric)
        mapped = faiss.IndexIDMap2(cpu)
        return self._maybe_to_gpu(mapped)

    def _maybe_to_gpu(self, index: "faiss.Index") -> "faiss.Index":
        if self._gpu_resources is None:
            raise RuntimeError("GPU resources are not initialised")
        try:
            gpu_index = faiss.index_cpu_to_gpu(self._gpu_resources, 0, index)
        except Exception as exc:  # pragma: no cover - hardware specific failure
            raise RuntimeError(
                f"Failed to promote FAISS index to GPU (index type={type(index).__name__}): {exc}"
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
            handled = False
            if hasattr(faiss, "GpuParameterSpace"):
                try:
                    faiss.GpuParameterSpace().set_index_parameter(index, "nprobe", nprobe)
                    handled = True
                except Exception as exc:  # pragma: no cover - GPU parameter guard
                    logger.debug("Unable to set nprobe via GpuParameterSpace", exc_info=True)
                    handled = False
            if not handled and hasattr(faiss, "ParameterSpace"):
                try:
                    space = faiss.ParameterSpace()
                    space.set_index_parameter(index, "nprobe", nprobe)
                    handled = True
                except Exception as exc:  # pragma: no cover - parameter guard
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
        device: Optional[int] = None
        if hasattr(index, "getDevice"):
            try:
                device = int(index.getDevice())
            except Exception:  # pragma: no cover - defensive logging guard
                device = None
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

    def _init_gpu_resources(self) -> "faiss.StandardGpuResources":
        if not hasattr(faiss, "StandardGpuResources"):
            raise RuntimeError(
                "FAISS GPU resources are unavailable; ensure faiss-gpu is installed with CUDA support."
            )
        try:
            resources = faiss.StandardGpuResources()
        except Exception as exc:  # pragma: no cover - hardware specific failure
            raise RuntimeError(
                f"Unable to initialise FAISS GPU resources (StandardGpuResources): {exc}"
            ) from exc
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
            self._index.train(matrix)
        ids = np.array([vector_uuid_to_faiss_int(vid) for vid in vector_ids], dtype=np.int64)
        self._index.add_with_ids(matrix, ids)
