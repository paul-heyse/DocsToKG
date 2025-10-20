from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Dict, Iterable, Tuple

import numpy as np

from DocsToKG.HybridSearch.store import ChunkRegistry
from DocsToKG.HybridSearch.types import ChunkFeatures, ChunkPayload


class _DummyDenseStore:
    dim = 3

    def reconstruct_batch(self, vector_ids: Iterable[str]) -> np.ndarray:
        rows = []
        for vector_id in vector_ids:
            try:
                seed = int(vector_id.split("-")[-1])
            except ValueError:
                seed = 0
            rows.append(np.full(self.dim, float(seed), dtype=np.float32))
        if not rows:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.vstack(rows).astype(np.float32, copy=False)


def _make_chunk(doc_id: str, namespace: str, vector_id: str) -> ChunkPayload:
    return ChunkPayload(
        doc_id=doc_id,
        chunk_id=f"{vector_id}-chunk",
        vector_id=vector_id,
        namespace=namespace,
        text="payload",
        metadata={},
        features=ChunkFeatures({}, {}, np.zeros(3, dtype=np.float32)),
        token_count=0,
        source_chunk_idxs=[],
        doc_items_refs=[],
    )


def test_chunk_registry_thread_safe_access() -> None:
    registry = ChunkRegistry()
    registry.attach_dense_store(_DummyDenseStore())

    docs = [f"doc-{idx}" for idx in range(4)]
    namespace = "ns"

    active_ids = deque[str]()
    id_to_scope: Dict[str, Tuple[str, str]] = {}
    scope_to_ids: Dict[Tuple[str, str], set[str]] = defaultdict(set)
    tracking_lock = threading.Lock()
    writer_finished = threading.Event()

    def writer() -> None:
        for idx in range(200):
            doc = docs[idx % len(docs)]
            vector_id = f"vec-{idx}"
            chunk = _make_chunk(doc, namespace, vector_id)
            registry.upsert([chunk])
            with tracking_lock:
                active_ids.append(vector_id)
                id_to_scope[vector_id] = (doc, namespace)
                scope_to_ids[(doc, namespace)].add(vector_id)
            if idx % 4 == 0:
                victim = None
                with tracking_lock:
                    if active_ids:
                        victim = active_ids.popleft()
                if victim is not None:
                    registry.delete([victim])
                    with tracking_lock:
                        scope = id_to_scope.pop(victim, None)
                        if scope is not None:
                            scope_to_ids[scope].discard(victim)
                            if not scope_to_ids[scope]:
                                scope_to_ids.pop(scope)
            time.sleep(0.001)
        writer_finished.set()

    def reader() -> None:
        cache: Dict[str, np.ndarray] = {}
        while not writer_finished.is_set():
            for doc in docs:
                ids = list(registry.vector_ids_for(doc, namespace))
                if ids:
                    registry.bulk_get(ids)
                    registry.resolve_embeddings(ids, cache=cache)
            list(registry.iter_all())
            registry.all()
            registry.count()
            time.sleep(0.001)

    reader_thread = threading.Thread(target=reader)
    writer_thread = threading.Thread(target=writer)

    reader_thread.start()
    writer_thread.start()

    writer_thread.join()
    reader_thread.join()

    with tracking_lock:
        remaining_ids = list(active_ids)
        expected_scopes = {scope: set(ids) for scope, ids in scope_to_ids.items()}

    assert sorted(registry.vector_ids()) == sorted(remaining_ids)
    assert registry.count() == len(remaining_ids)

    for doc in docs:
        expected = expected_scopes.get((doc, namespace), set())
        actual = set(registry.vector_ids_for(doc, namespace))
        assert actual == expected
