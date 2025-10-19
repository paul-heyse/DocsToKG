"""Unit tests covering the FaissRouter snapshot restore workflow."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from DocsToKG.HybridSearch.config import DenseIndexConfig
from DocsToKG.HybridSearch.router import FaissRouter


class DummyDenseStore:
    """Lightweight in-memory stand-in satisfying ``DenseVectorStore``."""

    def __init__(self, namespace: str) -> None:
        self.namespace = namespace
        self._dim = 3
        self._device = -1
        self._config = DenseIndexConfig()
        self._resolver: Optional[Callable[[int], Optional[str]]] = None
        self._vectors: dict[str, float] = {}

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def ntotal(self) -> int:
        return len(self._vectors)

    @property
    def device(self) -> int:
        return self._device

    @property
    def config(self) -> DenseIndexConfig:
        return self._config

    @property
    def adapter_stats(self) -> SimpleNamespace:
        return SimpleNamespace(
            device=self._device,
            ntotal=self.ntotal,
            index_description="dummy",
            nprobe=0,
            multi_gpu_mode="single",
            replicated=False,
            fp16_enabled=False,
            resources=None,
        )

    def add(self, vectors: Sequence[np.ndarray], vector_ids: Sequence[str]) -> None:
        for vector_id in vector_ids:
            self._vectors[vector_id] = float(len(self._vectors))

    def remove(self, vector_ids: Sequence[str]) -> None:
        for vector_id in vector_ids:
            self._vectors.pop(vector_id, None)

    def search(self, query: np.ndarray, top_k: int):  # pragma: no cover - unused stub
        return []

    def search_many(self, queries: np.ndarray, top_k: int):  # pragma: no cover - unused stub
        return []

    def search_batch(self, queries: np.ndarray, top_k: int):  # pragma: no cover - unused stub
        return []

    def range_search(
        self, query: np.ndarray, min_score: float, *, limit: Optional[int] = None
    ):  # pragma: no cover - unused stub
        return []

    def serialize(self) -> bytes:
        payload = json.dumps({"ids": sorted(self._vectors.keys())})
        return payload.encode("utf-8")

    def restore(self, payload: bytes) -> None:
        data = json.loads(payload.decode("utf-8"))
        ids = data.get("ids", [])
        self._vectors = {vector_id: float(index) for index, vector_id in enumerate(ids)}

    def stats(self) -> Mapping[str, float | str]:
        return {"ntotal": float(self.ntotal)}

    def set_id_resolver(self, resolver: Callable[[int], Optional[str]]) -> None:
        self._resolver = resolver


def test_restore_all_rehydrates_multiple_namespaces() -> None:
    """Ensure per-namespace restores hydrate every serialized store."""

    router = FaissRouter(
        per_namespace=True,
        default_store=DummyDenseStore("__default__"),
        factory=lambda namespace: DummyDenseStore(namespace),
    )
    for namespace in ("alpha", "beta"):
        store = router.get(namespace)
        store.add([np.zeros(3, dtype=np.float32)], [f"{namespace}-vector"])

    payloads = router.serialize_all()

    restored_router = FaissRouter(
        per_namespace=True,
        default_store=DummyDenseStore("__default__"),
        factory=lambda namespace: DummyDenseStore(namespace),
    )
    restored_router._snapshots["alpha"] = (b"legacy", None)
    before_restore = time.time()

    restored_router.restore_all(payloads)

    for namespace in ("alpha", "beta"):
        assert namespace in restored_router._stores
        restored_store = restored_router._stores[namespace]
        assert restored_store.ntotal == 1
        assert restored_router._last_used[namespace] >= before_restore
        assert namespace not in restored_router._snapshots
