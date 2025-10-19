"""Unit tests covering the FaissRouter snapshot restore workflow."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from DocsToKG.HybridSearch.config import DenseIndexConfig
from DocsToKG.HybridSearch.router import FaissRouter
from DocsToKG.HybridSearch.store import ManagedFaissAdapter


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

    def restore(
        self, payload: bytes, *, meta: Optional[Mapping[str, object]] = None
    ) -> None:
        data = json.loads(payload.decode("utf-8"))
        ids = data.get("ids", [])
        self._vectors = {vector_id: float(index) for index, vector_id in enumerate(ids)}

    def stats(self) -> Mapping[str, float | str]:
        return {"ntotal": float(self.ntotal)}

    def set_id_resolver(self, resolver: Callable[[int], Optional[str]]) -> None:
        self._resolver = resolver


class RecordingFaissStore:
    """Minimal FAISS-like store capturing restore metadata for assertions."""

    def __init__(self, namespace: str) -> None:
        self.namespace = namespace
        self._dim = 3
        self._device = -1
        self._vectors: list[str] = []
        self._resolver: Optional[Callable[[int], Optional[str]]] = None
        self.last_restore_meta: Optional[Mapping[str, object]] = None
        self.last_snapshot_meta: Optional[Mapping[str, object]] = None

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
    def adapter_stats(self) -> SimpleNamespace:
        return SimpleNamespace(
            device=self._device,
            ntotal=self.ntotal,
            index_description="recording",
            nprobe=0,
            multi_gpu_mode="single",
            replicated=False,
            fp16_enabled=False,
            resources=None,
        )

    def add(self, vectors: Sequence[np.ndarray], vector_ids: Sequence[str]) -> None:
        for vector_id in vector_ids:
            if vector_id not in self._vectors:
                self._vectors.append(vector_id)

    def remove(self, vector_ids: Sequence[str]) -> None:
        for vector_id in vector_ids:
            if vector_id in self._vectors:
                self._vectors.remove(vector_id)

    def serialize(self) -> bytes:
        payload = json.dumps({"ids": list(self._vectors), "namespace": self.namespace})
        return payload.encode("utf-8")

    def restore(self, payload: bytes, *, meta: Optional[Mapping[str, object]] = None) -> None:
        data = json.loads(payload.decode("utf-8"))
        ids = data.get("ids", [])
        self._vectors = [str(vector_id) for vector_id in ids]
        self.last_restore_meta = meta

    def snapshot_meta(self) -> Mapping[str, object]:
        return {"namespace": self.namespace, "dim": self._dim}

    def stats(self) -> Mapping[str, float | str]:
        return {"ntotal": float(self.ntotal)}

    def set_id_resolver(self, resolver: Callable[[int], Optional[str]]) -> None:
        self._resolver = resolver

    def rebuild_if_needed(self) -> bool:
        return False


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


def test_managed_adapter_restores_with_snapshot_metadata() -> None:
    """Managed adapters should serialize/restore with metadata passthrough."""

    router = FaissRouter(
        per_namespace=True,
        default_store=ManagedFaissAdapter(RecordingFaissStore("__default__")),
        factory=lambda namespace: ManagedFaissAdapter(RecordingFaissStore(namespace)),
    )
    managed_store = router.get("alpha")
    managed_store.add([np.zeros(3, dtype=np.float32)], ["alpha-vector"])

    payloads = router.serialize_all()
    snapshot_meta = {"namespace": "alpha", "marker": "router-test"}
    router._snapshots["alpha"] = (payloads["alpha"]["faiss"], snapshot_meta)
    del router._stores["alpha"]

    restored_store = router.get("alpha")
    inner_store = restored_store._inner  # type: ignore[attr-defined]
    assert isinstance(inner_store, RecordingFaissStore)
    assert inner_store.last_restore_meta == snapshot_meta
    assert inner_store._vectors == ["alpha-vector"]


def test_serialize_and_restore_roundtrip_carries_metadata() -> None:
    """Router serialization should retain metadata for restore_all."""

    router = FaissRouter(
        per_namespace=True,
        default_store=ManagedFaissAdapter(RecordingFaissStore("__default__")),
        factory=lambda namespace: ManagedFaissAdapter(RecordingFaissStore(namespace)),
    )
    store = router.get("alpha")
    store.add([np.zeros(3, dtype=np.float32)], ["alpha-vector"])

    payloads = router.serialize_all()
    alpha_payload = payloads["alpha"]
    assert isinstance(alpha_payload["meta"], Mapping)

    router._stores.pop("alpha")
    router.restore_all(payloads)

    restored_store = router._stores["alpha"]
    inner_store = restored_store._inner  # type: ignore[attr-defined]
    assert isinstance(inner_store, RecordingFaissStore)
    assert inner_store.last_restore_meta == alpha_payload["meta"]
    assert inner_store._vectors == ["alpha-vector"]
