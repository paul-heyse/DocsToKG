"""Namespace-aware coordination of managed FAISS stores and their snapshots.

``FaissRouter`` encapsulates the namespace strategy described in the package
README: a deployment may run a single shared FAISS instance or dedicate a GPU
index per tenant. The router hides the book-keeping required to make that work:

- It keeps a thread-safe map of namespace → ``DenseVectorStore`` instances,
  instantiating new stores on demand via a factory that typically produces
  ``ManagedFaissAdapter`` objects already wired for ``StandardGpuResources`` and
  ``ChunkRegistry`` access.
- When a namespace is evicted to reclaim GPU memory, the router caches the
  serialized FAISS bytes (and optional snapshot metadata) so that a subsequent
  request can call ``restore`` without blocking process startup.
- ``stats`` and ``serialize_all`` expose aggregated metrics and payloads for
  observability and persistence. These methods merge live stores with any
  cached snapshots so operators can gauge GPU memory utilisation, last-access
  timestamps, and serialized footprint before deciding which namespaces to trim.
- Optional ``set_id_resolver`` propagation lets downstream adapters resolve
  FAISS integer ids back to vector ids—mirroring how ``ManagedFaissAdapter``
  annotates search results.

Agents modifying namespace behaviour should adjust this router alongside the
service layer so eviction policy, snapshot metadata, and restoration semantics
stay aligned.
"""

from __future__ import annotations

import inspect
import logging
import time
from threading import RLock
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple, Union, cast

from .interfaces import DenseVectorStore

DEFAULT_NAMESPACE = "__default__"
logger = logging.getLogger(__name__)


class _StoreMap(dict):
    """Dictionary that ignores deletions for missing namespaces."""

    def __delitem__(self, key: object) -> None:  # type: ignore[override]
        if key in self:
            super().__delitem__(key)


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
        self._stores: Dict[str, DenseVectorStore] = _StoreMap()
        if per_namespace:
            self._stores[DEFAULT_NAMESPACE] = default_store
        self._lock = RLock()
        self._resolver: Optional[Callable[[int], Optional[str]]] = None
        now = time.time()
        self._last_used: Dict[str, float] = {DEFAULT_NAMESPACE: now}
        self._snapshots: Dict[str, Tuple[bytes, Optional[Mapping[str, object]]]] = {}

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
            self._last_used[DEFAULT_NAMESPACE] = time.time()
            return self._default_store
        key = namespace or DEFAULT_NAMESPACE
        with self._lock:
            store = self._stores.get(key)
            if store is None:
                store = self._factory(key)  # type: ignore[arg-type]
                if self._resolver:
                    store.set_id_resolver(self._resolver)
                snapshot = self._snapshots.pop(key, None)
                if snapshot is not None:
                    payload, meta = snapshot
                    try:
                        store.restore(payload, meta=meta)
                    except Exception:
                        logger.exception(
                            "faiss-router-restore-failed",
                            extra={"event": {"namespace": key}},
                        )
                self._stores[key] = store
            self._last_used[key] = time.time()
            return store

    def stats(self) -> Dict[str, object]:
        """Return stats for all managed stores (namespaced and aggregate)."""

        if not self._per_namespace:
            stats = dict(self._default_store.stats())
            stats["last_used_ts"] = self._last_used.get(DEFAULT_NAMESPACE, 0.0)
            snapshots = {DEFAULT_NAMESPACE: stats}
        else:
            with self._lock:
                snapshots = {}
                for ns, store in self._stores.items():
                    payload = dict(store.stats())
                    payload["last_used_ts"] = self._last_used.get(ns, 0.0)
                    payload["evicted"] = False
                    snapshots[ns] = payload
                for ns, snapshot in self._snapshots.items():
                    payload, _ = snapshot
                    snapshots.setdefault(
                        ns,
                        {
                            "evicted": True,
                            "serialized_bytes": float(len(payload)),
                            "last_used_ts": self._last_used.get(ns, 0.0),
                        },
                    )
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

    def serialize_all(self) -> Dict[str, Dict[str, object]]:
        """Serialize every managed store including snapshot metadata."""

        class SnapshotEntry(dict):
            """Mapping that exposes ``payload`` while aliasing ``faiss`` to the same bytes."""

            def __getitem__(self, key: object) -> object:  # type: ignore[override]
                if key == "faiss":
                    return super().__getitem__("payload")
                return super().__getitem__(key)

            def get(self, key: object, default: object = None) -> object:  # type: ignore[override]
                if key == "faiss":
                    return super().get("payload", default)
                return super().get(key, default)

        def build_entry(payload: bytes, meta: Optional[Mapping[str, object]]) -> Dict[str, object]:
            """Package FAISS payload bytes and optional metadata for persistence."""

            entry = SnapshotEntry()
            entry["payload"] = bytes(payload)
            entry["meta"] = dict(meta) if isinstance(meta, Mapping) else None
            return entry

        def collect(store: DenseVectorStore) -> Tuple[bytes, Optional[Mapping[str, object]]]:
            """Extract serialized payload and snapshot metadata from ``store``.

            Args:
                store: Vector store to snapshot.

            Returns:
                Tuple[bytes, Optional[Mapping[str, object]]]: Serialized FAISS bytes and metadata.
            """
            payload = store.serialize()
            meta_getter = getattr(store, "snapshot_meta", None)
            meta: Optional[Mapping[str, object]] = None
            if callable(meta_getter):
                raw_meta = meta_getter()
                if isinstance(raw_meta, Mapping):
                    meta = dict(raw_meta)
            return payload, meta

        payloads: Dict[str, Dict[str, object]] = {}
        if not self._per_namespace:
            payload, meta = collect(self._default_store)
            payloads[DEFAULT_NAMESPACE] = build_entry(payload, meta)
            return payloads
        with self._lock:
            for namespace, store in self._stores.items():
                payload, meta = collect(store)
                payloads[namespace] = build_entry(payload, meta)
            for namespace, snapshot in self._snapshots.items():
                payload, meta = snapshot
                payloads.setdefault(namespace, build_entry(payload, meta))
        return payloads

    @staticmethod
    def _serialize_with_meta(store: DenseVectorStore) -> Dict[str, object]:
        payload = store.serialize()
        meta_getter = getattr(store, "snapshot_meta", None)
        meta = meta_getter() if callable(meta_getter) else None
        return {"payload": payload, "meta": meta}

    def iter_stores(self) -> Sequence[Tuple[str, DenseVectorStore]]:
        """Return a snapshot of managed stores keyed by namespace."""

        if not self._per_namespace:
            return [(DEFAULT_NAMESPACE, self._default_store)]
        with self._lock:
            items = list(self._stores.items())
        return items

    def restore_all(self, payloads: Mapping[str, object]) -> None:
        """Restore stores from serialized payloads and metadata."""

        def coerce_entry(
            entry: object,
        ) -> Tuple[Optional[bytes], Optional[Mapping[str, object]]]:
            """Normalise stored payloads into raw bytes and metadata mapping.

            Args:
                entry: Persisted snapshot entry in legacy or current format.

            Returns:
                Tuple[Optional[bytes], Optional[Mapping[str, object]]]: Normalised payload and metadata.
            """
            if isinstance(entry, (bytes, bytearray, memoryview)):
                return bytes(entry), None
            if isinstance(entry, Mapping):
                blob = entry.get("payload")
                if not isinstance(blob, (bytes, bytearray, memoryview)):
                    blob = entry.get("faiss")
                if isinstance(blob, (bytes, bytearray, memoryview)):
                    meta_obj = entry.get("meta")
                    meta = meta_obj if isinstance(meta_obj, Mapping) else None
                    return bytes(blob), meta
            return None, None

        def restore_store(
            store: DenseVectorStore,
            blob: bytes,
            meta: Optional[Mapping[str, object]],
        ) -> None:
            """Restore a store from serialized payload and optional metadata.

            Args:
                store: Vector store instance to restore.
                blob: Serialized FAISS bytes.
                meta: Supplemental metadata to pass to ``restore`` when supported.
            """
            restore_fn = getattr(store, "restore")
            if meta is None:
                restore_fn(blob)
                return
            try:
                signature = inspect.signature(restore_fn)
            except (TypeError, ValueError):
                signature = None
            if signature is not None:
                for parameter in signature.parameters.values():
                    if parameter.name == "meta" and parameter.kind in (
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    ):
                        restore_fn(blob, meta=meta)
                        return
            restore_fn(blob)

        if not self._per_namespace:
            entry = payloads.get(DEFAULT_NAMESPACE)
            if entry is not None:
                blob, meta = coerce_entry(entry)
                if blob is not None:
                    restore_store(self._default_store, blob, meta)
            return

        with self._lock:
            for namespace, packed in payloads.items():
                store = self._stores.get(namespace)
                if store is None:
                    if self._factory is None:
                        continue
                    store = self._factory(namespace)
                    if self._resolver:
                        store.set_id_resolver(self._resolver)
                    self._stores[namespace] = store
                elif self._resolver:
                    store.set_id_resolver(self._resolver)
                payload, meta = self._extract_payload_and_meta(packed)
                try:
                    store.restore(payload, meta=meta)
                finally:
                    self._last_used[namespace] = time.time()
                    self._snapshots.pop(namespace, None)

    @staticmethod
    def _extract_payload_and_meta(
        packed: Union[bytes, Mapping[str, object]],
    ) -> Tuple[bytes, Optional[Mapping[str, object]]]:
        if isinstance(packed, (bytes, bytearray, memoryview)):
            return (bytes(packed), None)
        payload_obj = cast(Optional[bytes], packed.get("payload"))
        if payload_obj is None:
            raise ValueError("Serialized namespace payload is missing 'payload' bytes")
        meta_obj = packed.get("meta")
        meta = cast(Optional[Mapping[str, object]], meta_obj)
        return payload_obj, meta

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

    def set_id_resolver(self, resolver: Callable[[int], Optional[str]]) -> None:
        """Alias for :meth:`set_resolver` to improve readability."""

        self.set_resolver(resolver)

    def evict_idle(self, *, max_idle_seconds: int, skip_default: bool = True) -> int:
        """Serialize and evict stores idle longer than ``max_idle_seconds``."""

        if max_idle_seconds <= 0:
            raise ValueError("max_idle_seconds must be positive")
        if not self._per_namespace:
            return 0
        evicted = 0
        now = time.time()
        with self._lock:
            for namespace, store in list(self._stores.items()):
                if skip_default and namespace == DEFAULT_NAMESPACE:
                    continue
                last_used = self._last_used.get(namespace, 0.0)
                if last_used and now - last_used <= max_idle_seconds:
                    continue
                try:
                    payload = store.serialize()
                    meta_getter = getattr(store, "snapshot_meta", None)
                    meta = meta_getter() if callable(meta_getter) else None
                    self._snapshots[namespace] = (payload, meta)
                except Exception:
                    logger.exception(
                        "faiss-router-serialize-failed",
                        extra={"event": {"namespace": namespace}},
                    )
                    continue
                closer = getattr(store, "close", None)
                if callable(closer):
                    try:
                        closer()
                    except Exception:
                        logger.debug(
                            "faiss-router-close-failed",
                            extra={"event": {"namespace": namespace}},
                            exc_info=True,
                        )
                del self._stores[namespace]
                evicted += 1
        return evicted

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
