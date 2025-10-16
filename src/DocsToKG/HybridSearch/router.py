"""Namespace router utilities for hybrid search vector stores."""

from __future__ import annotations

from threading import RLock
from typing import Callable, Dict, Mapping, Optional, Sequence

from .interfaces import DenseVectorStore

DEFAULT_NAMESPACE = "__default__"


class FaissRouter:
    """Lightweight namespace-aware router for managed FAISS instances."""

    def __init__(
        self,
        *,
        per_namespace: bool,
        default_store: DenseVectorStore,
        factory: Optional[Callable[[str], DenseVectorStore]] = None,
    ) -> None:
        if per_namespace and factory is None:
            raise ValueError("factory is required when per_namespace=True")
        self._per_namespace = per_namespace
        self._default_store = default_store
        self._factory = factory
        self._stores: Dict[str, DenseVectorStore] = {}
        if per_namespace:
            self._stores[DEFAULT_NAMESPACE] = default_store
        self._lock = RLock()
        self._resolver: Optional[Callable[[int], Optional[str]]] = None

    @property
    def per_namespace(self) -> bool:
        """Return ``True`` when a dedicated store is provisioned per namespace."""

        return self._per_namespace

    @property
    def default_store(self) -> DenseVectorStore:
        """Return the default store used when namespaces are disabled."""

        return self._default_store

    def get(self, namespace: Optional[str]) -> DenseVectorStore:
        """Return the store serving ``namespace`` (creating one if necessary)."""

        if not self._per_namespace:
            return self._default_store
        key = namespace or DEFAULT_NAMESPACE
        with self._lock:
            store = self._stores.get(key)
            if store is None:
                store = self._factory(key)  # type: ignore[arg-type]
                if self._resolver:
                    store.set_id_resolver(self._resolver)
                self._stores[key] = store
            return store

    def stats(self) -> Dict[str, object]:
        """Return stats for all managed stores (namespaced and aggregate)."""

        if not self._per_namespace:
            snapshots = {DEFAULT_NAMESPACE: self._default_store.stats()}
        else:
            with self._lock:
                snapshots = {ns: store.stats() for ns, store in self._stores.items()}
        aggregate = self._aggregate_stats(snapshots)
        return {"namespaces": snapshots, "aggregate": aggregate}

    @staticmethod
    def _aggregate_stats(namespaced: Mapping[str, Mapping[str, object]]) -> Dict[str, float]:
        totals: Dict[str, float] = {}
        for stats in namespaced.values():
            if not isinstance(stats, Mapping):
                continue
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    totals[key] = totals.get(key, 0.0) + float(value)
        return totals

    def serialize_all(self) -> Dict[str, bytes]:
        """Serialize every managed store (namespace -> payload)."""

        payloads: Dict[str, bytes] = {}
        if not self._per_namespace:
            payloads[DEFAULT_NAMESPACE] = self._default_store.serialize()
            return payloads
        with self._lock:
            for namespace, store in self._stores.items():
                payloads[namespace] = store.serialize()
        return payloads

    def restore_all(self, payloads: Mapping[str, bytes]) -> None:
        """Restore stores from serialized payloads."""

        if not self._per_namespace:
            blob = payloads.get(DEFAULT_NAMESPACE)
            if blob is not None:
                self._default_store.restore(blob)
            return
        with self._lock:
            for namespace, blob in payloads.items():
                store = self._stores.get(namespace)
                if store is None:
                    if self._factory is None:
                        continue
                    store = self._factory(namespace)
                    if self._resolver:
                        store.set_id_resolver(self._resolver)
                self._stores[namespace] = store
                store.restore(blob)

    def rebuild_if_needed(self) -> bool:
        """Attempt to rebuild all managed stores; returns True if any rebuild occurs."""

        rebuilt = False
        if not self._per_namespace:
            return self._default_store.rebuild_if_needed() or rebuilt
        with self._lock:
            for store in self._stores.values():
                rebuilt = store.rebuild_if_needed() or rebuilt
        return rebuilt

    def set_resolver(self, resolver: Callable[[int], Optional[str]]) -> None:
        """Register a resolver applied to existing and future stores."""

        self._resolver = resolver
        if not self._per_namespace:
            self._default_store.set_id_resolver(resolver)
            return
        with self._lock:
            for store in self._stores.values():
                store.set_id_resolver(resolver)

    def run_maintenance(
        self,
        *,
        training_sampler: Optional[Callable[[str, DenseVectorStore], Sequence]] = None,
    ) -> Dict[str, Dict[str, bool]]:
        """Run optional training and rebuild checks across managed stores.

        Args:
            training_sampler: Callable receiving `(namespace, store)` and returning
                representative vectors for training when the store reports that it
                requires training. If omitted, training is skipped.

        Returns:
            Mapping of namespace to maintenance actions performed.
        """

        results: Dict[str, Dict[str, bool]] = {}
        if not self._per_namespace:
            namespace = DEFAULT_NAMESPACE
            store = self._default_store
            results[namespace] = self._maintain_store(namespace, store, training_sampler)
            return results

        with self._lock:
            for namespace, store in self._stores.items():
                results[namespace] = self._maintain_store(namespace, store, training_sampler)
        return results

    def _maintain_store(
        self,
        namespace: str,
        store: DenseVectorStore,
        training_sampler: Optional[Callable[[str, DenseVectorStore], Sequence]],
    ) -> Dict[str, bool]:
        actions = {"trained": False, "rebuilt": False}
        if training_sampler is not None and store.needs_training():
            vectors = training_sampler(namespace, store)
            if vectors:
                store.train(vectors)
                actions["trained"] = True
        if store.rebuild_if_needed():
            actions["rebuilt"] = True
        return actions
