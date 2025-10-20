import types
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from DocsToKG.HybridSearch.pipeline import ChunkIngestionPipeline
from DocsToKG.HybridSearch.store import ChunkRegistry
from DocsToKG.HybridSearch.types import ChunkFeatures, ChunkPayload


class DummyLexicalIndex:
    def bulk_upsert(self, _: Sequence[ChunkPayload]) -> None:  # pragma: no cover - not used in test
        return None

    def bulk_delete(self, _: Sequence[str]) -> None:  # pragma: no cover - not used in test
        return None


class DummyFaissStore:
    def __init__(self) -> None:
        self.config = types.SimpleNamespace(nlist=1, ivf_train_factor=1)
        self.dim = 3
        self._needs_training = True
        self._resolver: Optional[Callable[[int], Optional[str]]] = None
        self._vectors: Dict[str, np.ndarray] = {}
        self.trained_vectors: Optional[List[np.ndarray]] = None

    def set_id_resolver(self, resolver: Callable[[int], Optional[str]]) -> None:
        self._resolver = resolver

    def needs_training(self) -> bool:
        return self._needs_training

    def train(self, vectors: Sequence[np.ndarray]) -> None:
        self.trained_vectors = [np.asarray(vector, dtype=np.float32) for vector in vectors]
        self._needs_training = False

    def reconstruct_batch(self, vector_ids: Sequence[str]) -> np.ndarray:
        rows = [np.asarray(self._vectors[vector_id], dtype=np.float32) for vector_id in vector_ids]
        return np.vstack(rows) if rows else np.empty((0, self.dim), dtype=np.float32)

    def add(self, vectors: Sequence[np.ndarray], vector_ids: Sequence[str]) -> None:  # pragma: no cover - helper only
        for vector_id, vector in zip(vector_ids, vectors):
            self._vectors[vector_id] = np.asarray(vector, dtype=np.float32)

    def remove(self, _: Sequence[str]) -> None:  # pragma: no cover - helper only
        return None


def test_training_sample_combines_registry_and_new_chunks() -> None:
    registry = ChunkRegistry()
    faiss = DummyFaissStore()
    opensearch = DummyLexicalIndex()

    existing_embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    existing_chunk = ChunkPayload(
        doc_id="doc-1",
        chunk_id="chunk-1",
        vector_id="00000000-0000-0000-0000-000000000001",
        namespace="ns",
        text="existing",
        metadata={},
        features=ChunkFeatures({}, {}, existing_embedding),
        token_count=0,
        source_chunk_idxs=(),
        doc_items_refs=(),
    )
    faiss._vectors = {existing_chunk.vector_id: existing_embedding}
    registry.upsert([existing_chunk])

    pipeline = ChunkIngestionPipeline(
        faiss_index=faiss,
        opensearch=opensearch,
        registry=registry,
        training_sample_rng_factory=lambda: np.random.default_rng(0),
    )

    new_embedding = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    new_chunk = ChunkPayload(
        doc_id="doc-2",
        chunk_id="chunk-2",
        vector_id="00000000-0000-0000-0000-000000000002",
        namespace="ns",
        text="new",
        metadata={},
        features=ChunkFeatures({}, {}, new_embedding),
        token_count=0,
        source_chunk_idxs=(),
        doc_items_refs=(),
    )

    pipeline._prepare_faiss([new_chunk])

    assert faiss.trained_vectors is not None
    assert any(np.allclose(vector, existing_embedding) for vector in faiss.trained_vectors)
    assert any(np.allclose(vector, new_embedding) for vector in faiss.trained_vectors)
