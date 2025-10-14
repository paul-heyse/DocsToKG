"""FAISS index management with GPU-aware fallbacks."""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from .config import DenseIndexConfig

logger = logging.getLogger(__name__)

try:  # pragma: no cover - import tested indirectly
    import faiss  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError("faiss module is required for dense retrieval") from exc


@dataclass(slots=True)
class FaissSearchResult:
    """Dense search hit returned by FAISS."""

    vector_id: str
    score: float


class FaissIndexManager:
    """Manage lifecycle of a FAISS index with optional GPU acceleration."""

    def __init__(self, dim: int, config: DenseIndexConfig) -> None:
        self._dim = dim
        self._config = config
        self._gpu_resources = self._init_gpu_resources()
        self._index = self._create_index()
        self._id_lookup: Dict[int, str] = {}

    @property
    def ntotal(self) -> int:
        return int(self._index.ntotal)

    @property
    def config(self) -> DenseIndexConfig:
        return self._config

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
        if not hasattr(self._index, "is_trained"):
            return False
        return not bool(getattr(self._index, "is_trained"))

    def add(self, vectors: Sequence[np.ndarray], vector_ids: Sequence[str]) -> None:
        if len(vectors) != len(vector_ids):
            raise ValueError("vectors and vector_ids must align")
        if not vectors:
            return
        matrix = np.stack([self._ensure_dim(vec) for vec in vectors]).astype(np.float32)
        faiss.normalize_L2(matrix)
        ids = np.array([self._uuid_to_int64(vid) for vid in vector_ids], dtype=np.int64)
        self._remove_ids(ids)
        self._index.add_with_ids(matrix, ids)
        for internal_id, vector_id in zip(ids, vector_ids):
            self._id_lookup[int(internal_id)] = vector_id

    def remove(self, vector_ids: Sequence[str]) -> None:
        if not vector_ids:
            return
        ids = np.array([self._uuid_to_int64(vid) for vid in vector_ids], dtype=np.int64)
        self._remove_ids(ids)
        for internal_id in ids:
            self._id_lookup.pop(int(internal_id), None)

    def search(self, query: np.ndarray, top_k: int) -> List[FaissSearchResult]:
        query_matrix = self._ensure_dim(query).reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_matrix)
        scores, ids = self._index.search(query_matrix, top_k)
        results: List[FaissSearchResult] = []
        for score, internal_id in zip(scores[0], ids[0]):
            if internal_id == -1:
                continue
            vector_id = self._id_lookup.get(int(internal_id))
            if vector_id is None:
                continue
            results.append(FaissSearchResult(vector_id=vector_id, score=float(score)))
        return results

    def serialize(self) -> bytes:
        cpu_index = self._to_cpu(self._index)
        return faiss.serialize_index(cpu_index)

    def restore(self, payload: bytes) -> None:
        cpu_index = faiss.deserialize_index(payload)
        self._index = self._maybe_to_gpu(cpu_index)

    def stats(self) -> Dict[str, float | str]:
        return {
            "ntotal": float(self.ntotal),
            "index_type": self._config.index_type,
        }

    def _create_index(self) -> "faiss.Index":
        if self._config.index_type == "flat":
            base_index = faiss.IndexFlatIP(self._dim)
        elif self._config.index_type == "ivf_flat":
            quantizer = faiss.IndexFlatIP(self._dim)
            base_index = faiss.IndexIVFFlat(
                quantizer,
                self._dim,
                self._config.nlist,
                faiss.METRIC_INNER_PRODUCT,
            )
        else:
            quantizer = faiss.IndexFlatIP(self._dim)
            base_index = faiss.IndexIVFPQ(
                quantizer,
                self._dim,
                self._config.nlist,
                self._config.pq_m,
                self._config.pq_bits,
            )
        index = faiss.IndexIDMap2(base_index)
        return self._maybe_to_gpu(index)

    def _maybe_to_gpu(self, index: "faiss.Index") -> "faiss.Index":
        if self._gpu_resources is None:
            return index
        try:
            gpu_index = faiss.index_cpu_to_gpu(self._gpu_resources, 0, index)
            logger.info("FAISS index promoted to GPU")
            return gpu_index
        except Exception as exc:  # pragma: no cover - GPU promotion path
            logger.warning("Failed to promote FAISS index to GPU: %s", exc)
            return index

    def _to_cpu(self, index: "faiss.Index") -> "faiss.Index":
        if hasattr(faiss, "index_gpu_to_cpu"):
            try:
                return faiss.index_gpu_to_cpu(index)
            except Exception:  # pragma: no cover - best effort fallback
                return index
        return index

    def _init_gpu_resources(self) -> "faiss.StandardGpuResources | None":
        if not hasattr(faiss, "StandardGpuResources"):
            return None
        try:
            resources = faiss.StandardGpuResources()
            return resources
        except Exception as exc:  # pragma: no cover - GPU setup only when available
            logger.warning("GPU resources unavailable, using CPU FAISS: %s", exc)
            return None

    def _ensure_dim(self, vector: np.ndarray) -> np.ndarray:
        if vector.shape != (self._dim,):
            raise ValueError(f"Expected embedding dimension {self._dim}, got {vector.shape}")
        return vector

    def _uuid_to_int64(self, value: str) -> int:
        return uuid.UUID(value).int & ((1 << 63) - 1)

    def _remove_ids(self, ids: np.ndarray) -> None:
        if ids.size == 0:
            return
        id_array = ids.astype(np.int64)
        try:
            selector = faiss.IDSelectorArray(id_array.size, faiss.swig_ptr(id_array))
        except AttributeError:
            selector = faiss.IDSelectorBatch(id_array)
        self._index.remove_ids(selector)

