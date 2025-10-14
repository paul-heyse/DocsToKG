"""FAISS index management with GPU-aware fallbacks."""
from __future__ import annotations

import base64
import io
import json
import logging
import uuid
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from .config import DenseIndexConfig

logger = logging.getLogger(__name__)

try:  # pragma: no cover - import tested indirectly
    import faiss  # type: ignore

    _FAISS_AVAILABLE = all(
        hasattr(faiss, attr) for attr in ("IndexFlatIP", "IndexIDMap2", "IDSelectorArray")
    )
except Exception:  # pragma: no cover - environment without GPU/FAISS deps
    faiss = None  # type: ignore
    _FAISS_AVAILABLE = False


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
        self._use_native = _FAISS_AVAILABLE
        self._gpu_resources = self._init_gpu_resources() if self._use_native else None
        self._index = self._create_index() if self._use_native else None
        self._id_lookup: Dict[int, str] = {}
        self._vectors: Dict[str, np.ndarray] = {}
        self._remove_fallbacks = 0

    @property
    def ntotal(self) -> int:
        if self._use_native and self._index is not None:
            return int(self._index.ntotal)
        return len(self._vectors)

    @property
    def config(self) -> DenseIndexConfig:
        return self._config

    def train(self, vectors: Sequence[np.ndarray]) -> None:
        if not self._use_native or self._index is None:
            return
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
        if not self._use_native or self._index is None:
            return False
        return not bool(getattr(self._index, "is_trained"))

    def add(self, vectors: Sequence[np.ndarray], vector_ids: Sequence[str]) -> None:
        if len(vectors) != len(vector_ids):
            raise ValueError("vectors and vector_ids must align")
        if not vectors:
            return
        matrix = np.stack([self._ensure_dim(vec) for vec in vectors]).astype(np.float32)
        faiss.normalize_L2(matrix) if self._use_native else self._normalize(matrix)
        if self._use_native and self._index is not None:
            ids = np.array([self._uuid_to_int64(vid) for vid in vector_ids], dtype=np.int64)
            self._remove_ids(ids)
            self._index.add_with_ids(matrix, ids)
            for internal_id, vector_id in zip(ids, vector_ids):
                self._id_lookup[int(internal_id)] = vector_id
        else:
            for vector, vector_id in zip(matrix, vector_ids):
                self._vectors[vector_id] = vector.copy()

    def remove(self, vector_ids: Sequence[str]) -> None:
        if not vector_ids:
            return
        if self._use_native and self._index is not None:
            ids = np.array([self._uuid_to_int64(vid) for vid in vector_ids], dtype=np.int64)
            self._remove_ids(ids)
            for internal_id in ids:
                self._id_lookup.pop(int(internal_id), None)
        else:
            for vector_id in vector_ids:
                self._vectors.pop(vector_id, None)

    def search(self, query: np.ndarray, top_k: int) -> List[FaissSearchResult]:
        query_matrix = self._ensure_dim(query).reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_matrix) if self._use_native else self._normalize(query_matrix)
        if self._use_native and self._index is not None:
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
        results: List[FaissSearchResult] = []
        if not self._vectors:
            return results
        query_vec = query_matrix[0]
        all_items = [
            (vector_id, float(np.dot(query_vec, stored)))
            for vector_id, stored in self._vectors.items()
        ]
        all_items.sort(key=lambda item: item[1], reverse=True)
        for vector_id, score in all_items[:top_k]:
            results.append(FaissSearchResult(vector_id=vector_id, score=score))
        return results

    def serialize(self) -> bytes:
        if self._use_native and self._index is not None:
            cpu_index = self._to_cpu(self._index)
            index_bytes = faiss.serialize_index(cpu_index)
            payload = {
                "mode": "native",
                "index": base64.b64encode(index_bytes).decode("ascii"),
                "id_lookup": {str(internal_id): vector_id for internal_id, vector_id in self._id_lookup.items()},
            }
            return json.dumps(payload).encode("utf-8")
        buffer = io.BytesIO()
        payload = {vector_id: vector.tolist() for vector_id, vector in self._vectors.items()}
        buffer.write(json.dumps(payload).encode("utf-8"))
        return buffer.getvalue()

    def restore(self, payload: bytes) -> None:
        if self._use_native:
            try:
                data = json.loads(payload.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                cpu_index = faiss.deserialize_index(np.frombuffer(payload, dtype=np.uint8))
                self._index = self._maybe_to_gpu(cpu_index)
                self._id_lookup = {}
                return
            if data.get("mode") == "native":
                encoded = data.get("index")
                if not isinstance(encoded, str):
                    raise ValueError("Invalid FAISS payload")
                index_bytes = base64.b64decode(encoded.encode("ascii"))
                cpu_index = faiss.deserialize_index(np.frombuffer(index_bytes, dtype=np.uint8))
                self._index = self._maybe_to_gpu(cpu_index)
                raw_lookup = data.get("id_lookup", {})
                self._id_lookup = {int(key): str(value) for key, value in raw_lookup.items()}
            else:
                cpu_index = faiss.deserialize_index(np.frombuffer(payload, dtype=np.uint8))
                self._index = self._maybe_to_gpu(cpu_index)
                self._id_lookup = {}
        else:
            data = json.loads(payload.decode("utf-8"))
            self._vectors = {
                vector_id: np.array(values, dtype=np.float32)
                for vector_id, values in data.items()
            }

    def stats(self) -> Dict[str, float | str]:
        return {
            "ntotal": float(self.ntotal),
            "index_type": self._config.index_type,
            "gpu_remove_fallbacks": float(self._remove_fallbacks),
        }

    def _create_index(self) -> "faiss.Index":
        if not self._use_native:
            return None
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
        if not self._use_native or self._gpu_resources is None:
            return index
        try:
            gpu_index = faiss.index_cpu_to_gpu(self._gpu_resources, 0, index)
            logger.info("FAISS index promoted to GPU")
            return gpu_index
        except Exception as exc:  # pragma: no cover - GPU promotion path
            logger.warning("Failed to promote FAISS index to GPU: %s", exc)
            return index

    def _to_cpu(self, index: "faiss.Index") -> "faiss.Index":
        if not self._use_native:
            return index
        if hasattr(faiss, "index_gpu_to_cpu"):
            try:
                return faiss.index_gpu_to_cpu(index)
            except Exception:  # pragma: no cover - best effort fallback
                return index
        return index

    def _init_gpu_resources(self) -> "faiss.StandardGpuResources | None":
        if not self._use_native or not hasattr(faiss, "StandardGpuResources"):
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
        if not self._use_native or self._index is None or ids.size == 0:
            return
        id_array = ids.astype(np.int64)
        try:
            selector = faiss.IDSelectorArray(id_array.size, faiss.swig_ptr(id_array))
        except AttributeError:
            selector = faiss.IDSelectorBatch(id_array)
        try:
            self._index.remove_ids(selector)
        except RuntimeError as exc:
            message = str(exc).lower()
            if "remove_ids not implemented" not in message:
                raise
            logger.warning("FAISS remove_ids not implemented on GPU index, falling back to CPU")
            self._remove_fallbacks += 1
            cpu_index = self._to_cpu(self._index)
            cpu_index.remove_ids(selector)
            self._index = self._maybe_to_gpu(cpu_index)
        finally:
            for internal_id in id_array:
                self._id_lookup.pop(int(internal_id), None)

    def _normalize(self, matrix: np.ndarray) -> None:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        matrix /= norms
