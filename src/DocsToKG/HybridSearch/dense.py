"""
FAISS index management with GPU-aware fallbacks.

This module provides comprehensive FAISS index management for DocsToKG,
including GPU acceleration support, automatic fallbacks, and memory-efficient
vector storage and retrieval operations.

The module supports multiple FAISS index types and automatically handles
GPU availability with graceful CPU fallbacks for environments without
GPU support.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .config import DenseIndexConfig
from .ids import vector_uuid_to_faiss_int

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
    """Dense search hit returned by FAISS.

    This class represents a single result from FAISS vector similarity search,
    containing the vector identifier and similarity score.

    Attributes:
        vector_id: Unique identifier for the vector in the FAISS index
        score: Similarity score (lower values indicate higher similarity for distance metrics)

    Examples:
        >>> result = FaissSearchResult(
        ...     vector_id="vec_123",
        ...     score=0.15  # L2 distance
        ... )
    """

    vector_id: str
    score: float


class FaissIndexManager:
    """Manage lifecycle of a FAISS index with optional GPU acceleration.

    This class provides comprehensive FAISS index management including:
    - Automatic GPU detection and resource management
    - Multiple index type support (Flat, IVF, PQ)
    - Memory-efficient vector storage and retrieval
    - Graceful fallbacks for GPU-unavailable environments
    - Index persistence and restoration capabilities

    The manager automatically selects the best available index type and
    acceleration method based on the provided configuration and available
    hardware resources.

    Attributes:
        _dim: Vector dimensionality
        _config: Dense index configuration parameters
        _use_native: Whether FAISS native acceleration is available
        _gpu_resources: GPU resources for acceleration (if available)
        _index: FAISS index instance
        _vectors: In-memory vector storage for fallback operations
        _remove_fallbacks: Counter for fallback operations

    Examples:
        >>> config = DenseIndexConfig(index_type="ivf_pq")
        >>> manager = FaissIndexManager(dimension=768, config=config)
        >>> manager.add_vectors([vector1, vector2], ["id1", "id2"])
        >>> results = manager.search(query_vector, k=10)
    """

    def __init__(self, dim: int, config: DenseIndexConfig) -> None:
        """Initialize FAISS index manager with configuration.

        Sets up the FAISS index manager with the specified configuration,
        automatically detecting GPU availability and creating appropriate
        index structures.

        Args:
            dim: Vector dimensionality for the index
            config: Dense index configuration parameters

        Raises:
            ValueError: If configuration parameters are invalid
        """
        self._dim = dim
        self._config = config
        self._use_native = _FAISS_AVAILABLE
        self._gpu_resources = self._init_gpu_resources() if self._use_native else None
        self._index = self._create_index() if self._use_native else None
        self._vectors: Dict[str, np.ndarray] = {}
        self._remove_fallbacks = 0
        self._id_resolver: Optional[Callable[[int], Optional[str]]] = None

        if self._use_native and self._index is not None:
            self._apply_search_parameters(self._index)

    @property
    def ntotal(self) -> int:
        """Get total number of vectors in the index.

        Returns the total count of vectors currently stored in the index,
        using native FAISS counting when available or fallback storage.

        Returns:
            Total number of vectors in the index
        """
        if self._use_native and self._index is not None:
            return int(self._index.ntotal)
        return len(self._vectors)

    @property
    def config(self) -> DenseIndexConfig:
        """Get the current index configuration.

        Returns:
            Current DenseIndexConfig instance
        """
        return self._config

    def set_id_resolver(self, resolver: Callable[[int], Optional[str]]) -> None:
        """Attach a resolver used to translate FAISS int IDs to vector IDs."""

        self._id_resolver = resolver

    def train(self, vectors: Sequence[np.ndarray]) -> None:
        """Train the FAISS index with sample vectors.

        This method trains IVF-based indexes with representative data
        to optimize search performance. Flat indexes don't require training.

        Args:
            vectors: Sample vectors for index training

        Raises:
            ValueError: If training vectors are required but not provided
        """
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
        """Check if the index requires training before adding vectors.

        Returns:
            True if index needs training, False otherwise
        """
        if not self._use_native or self._index is None:
            return False
        return not bool(getattr(self._index, "is_trained"))

    def add(self, vectors: Sequence[np.ndarray], vector_ids: Sequence[str]) -> None:
        """Add vectors to the FAISS index with associated IDs.

        This method adds new vectors to the index, optionally using GPU
        acceleration when available and falling back to CPU storage when needed.

        Args:
            vectors: Sequence of vector arrays to add
            vector_ids: Corresponding vector identifiers

        Raises:
            ValueError: If vectors and vector_ids lengths don't match
        """
        if len(vectors) != len(vector_ids):
            raise ValueError("vectors and vector_ids must align")
        if not vectors:
            return
        matrix = np.stack([self._ensure_dim(vec) for vec in vectors]).astype(np.float32)
        faiss.normalize_L2(matrix) if self._use_native else self._normalize(matrix)
        if self._use_native and self._index is not None:
            ids = np.array([vector_uuid_to_faiss_int(vid) for vid in vector_ids], dtype=np.int64)
            self._remove_ids(ids)
            self._index.add_with_ids(matrix, ids)
        else:
            for vector, vector_id in zip(matrix, vector_ids):
                self._vectors[vector_id] = vector.copy()

    def remove(self, vector_ids: Sequence[str]) -> None:
        """Remove vectors from the index by their IDs.

        This method removes specified vectors from the index, updating
        both the FAISS index and the internal ID lookup mappings.

        Args:
            vector_ids: Sequence of vector IDs to remove

        Note:
            FAISS doesn't support direct vector removal, so this method
            marks vectors as removed in the lookup table
        """
        if not vector_ids:
            return
        if self._use_native and self._index is not None:
            ids = np.array([vector_uuid_to_faiss_int(vid) for vid in vector_ids], dtype=np.int64)
            self._remove_ids(ids)
        else:
            for vector_id in vector_ids:
                self._vectors.pop(vector_id, None)

    def search(self, query: np.ndarray, top_k: int) -> List[FaissSearchResult]:
        """Search for similar vectors in the index.

        This method performs similarity search using either FAISS native
        acceleration or CPU-based fallback, returning the most similar
        vectors with their similarity scores.

        Args:
            query: Query vector for similarity search
            top_k: Maximum number of results to return

        Returns:
            List of FaissSearchResult objects with vector IDs and scores

        Examples:
            >>> results = manager.search(query_vector, k=10)
            >>> for result in results:
            ...     print(f"Vector {result.vector_id}: score {result.score}")
        """
        query_matrix = self._ensure_dim(query).reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_matrix) if self._use_native else self._normalize(query_matrix)
        if self._use_native and self._index is not None:
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
        """Serialize the index state for persistence.

        This method converts the index to a byte representation that can be
        stored and later restored, supporting both native FAISS and fallback
        storage modes.

        Returns:
            Serialized index data as bytes

        Raises:
            RuntimeError: If serialization fails
        """
        if self._use_native and self._index is not None:
            cpu_index = self._to_cpu(self._index)
            index_bytes = faiss.serialize_index(cpu_index)
            payload = {
                "mode": "native",
                "index": base64.b64encode(index_bytes).decode("ascii"),
            }
            return json.dumps(payload).encode("utf-8")
        buffer = io.BytesIO()
        payload = {vector_id: vector.tolist() for vector_id, vector in self._vectors.items()}
        buffer.write(json.dumps(payload).encode("utf-8"))
        return buffer.getvalue()

    def restore(self, payload: bytes) -> None:
        """Restore index state from serialized data.

        This method reconstructs the index from previously serialized data,
        supporting both native FAISS and fallback storage restoration.

        Args:
            payload: Serialized index data from serialize() method

        Raises:
            ValueError: If payload format is invalid or incompatible
            RuntimeError: If restoration fails
        """
        if self._use_native:
            try:
                data = json.loads(payload.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                cpu_index = faiss.deserialize_index(np.frombuffer(payload, dtype=np.uint8))
                self._index = self._maybe_to_gpu(cpu_index)
                self._apply_search_parameters(self._index)
                return
            if data.get("mode") == "native":
                encoded = data.get("index")
                if not isinstance(encoded, str):
                    raise ValueError("Invalid FAISS payload")
                index_bytes = base64.b64decode(encoded.encode("ascii"))
                cpu_index = faiss.deserialize_index(np.frombuffer(index_bytes, dtype=np.uint8))
                self._index = self._maybe_to_gpu(cpu_index)
                self._apply_search_parameters(self._index)
            else:
                cpu_index = faiss.deserialize_index(np.frombuffer(payload, dtype=np.uint8))
                self._index = self._maybe_to_gpu(cpu_index)
                self._apply_search_parameters(self._index)
        else:
            data = json.loads(payload.decode("utf-8"))
            self._vectors = {
                vector_id: np.array(values, dtype=np.float32) for vector_id, values in data.items()
            }

    def stats(self) -> Dict[str, float | str]:
        return {
            "ntotal": float(self.ntotal),
            "index_type": self._config.index_type,
            "gpu_remove_fallbacks": float(self._remove_fallbacks),
            "nprobe": float(self._config.nprobe) if self._use_native else 0.0,
        }

    def _create_index(self) -> "faiss.Index":
        if not self._use_native:
            return None

        metric = faiss.METRIC_INNER_PRODUCT
        index_type = self._config.index_type

        if (
            self._gpu_resources is not None
            and index_type == "flat"
            and hasattr(faiss, "GpuIndexFlatIP")
        ):
            base = faiss.GpuIndexFlatIP(self._gpu_resources, self._dim)
            return faiss.IndexIDMap2(base)

        spec_map = {
            "flat": "Flat",
            "ivf_flat": f"IVF{self._config.nlist},Flat",
            "ivf_pq": f"IVF{self._config.nlist},PQ{self._config.pq_m}x{self._config.pq_bits}",
        }
        try:
            spec = spec_map[index_type]
        except KeyError as exc:  # pragma: no cover - config validation elsewhere
            raise ValueError(f"Unsupported FAISS index type: {index_type}") from exc

        cpu_index = faiss.index_factory(self._dim, spec, metric)
        mapped = faiss.IndexIDMap2(cpu_index)
        return self._maybe_to_gpu(mapped)

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

    def _apply_search_parameters(self, index: "faiss.Index | None") -> None:
        if not self._use_native or index is None:
            return

        if self._config.index_type.startswith("ivf"):
            nprobe = int(self._config.nprobe)
            applied = False
            if self._gpu_resources is not None and hasattr(faiss, "GpuParameterSpace"):
                try:
                    faiss.GpuParameterSpace().set_index_parameter(index, "nprobe", nprobe)
                    applied = True
                except Exception:  # pragma: no cover - GPU parameter guard
                    applied = False
            if not applied and hasattr(faiss, "ParameterSpace"):
                try:
                    space = faiss.ParameterSpace()
                    space.set_index_parameter(index, "nprobe", nprobe)
                except Exception:  # pragma: no cover - CPU parameter guard
                    logger.debug("Unable to set nprobe via ParameterSpace", exc_info=True)

        self._log_index_configuration(index)

    def _log_index_configuration(self, index: "faiss.Index") -> None:
        if not self._use_native or index is None:
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
        except Exception:  # pragma: no cover - resolver failures should not crash search
            logger.debug("id-resolver failure", exc_info=True)
            return None

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

    def _remove_ids(self, ids: np.ndarray) -> None:
        if not self._use_native or self._index is None or ids.size == 0:
            return

        id_array = ids.astype(np.int64)
        selector = faiss.IDSelectorBatch(id_array)
        try:
            self._index.remove_ids(selector)
        except RuntimeError as exc:
            if "remove_ids not implemented" not in str(exc).lower():
                raise
            logger.warning("FAISS remove_ids not implemented on GPU index, falling back to CPU")
            self._remove_fallbacks += 1
            cpu_index = self._to_cpu(self._index)
            cpu_index.remove_ids(selector)
            self._index = self._maybe_to_gpu(cpu_index)
            self._apply_search_parameters(self._index)

    def _normalize(self, matrix: np.ndarray) -> None:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        matrix /= norms
