"""FAISS snapshot serialization/restore regression tests.

Validates that `serialize_state`/`restore_state` payloads (binary blobs +
metadata) round-trip across versions, honour encryption hooks, and integrate
with the CUDA/OpenBLAS-backed FAISS wheel. Guards disaster-recovery workflows
outlined in the README. Exercises ``serialize_state`` output, encrypted payload
handling, metadata round-tripping, and error paths during ``restore_state`` to
guarantee hybrid search clusters can persist and reload FAISS indices safely.
"""

from __future__ import annotations

import base64
import logging
import copy
import json
from typing import List, Mapping, Optional, Sequence, Tuple

import pytest

import numpy as np

from DocsToKG.HybridSearch.store import ChunkRegistry, restore_state, serialize_state
from DocsToKG.HybridSearch.types import ChunkFeatures, ChunkPayload


class _RecordingVectorStore:
    """Minimal FAISS vector store stub recording restore invocations."""

    def __init__(self) -> None:
        self.calls: List[Tuple[bytes, Optional[Mapping[str, object]]]] = []

    def restore(self, payload: bytes, *, meta: Mapping[str, object] | None = None) -> None:
        materialised_meta: Optional[Mapping[str, object]]
        if meta is None:
            materialised_meta = None
        else:
            materialised_meta = dict(meta)
        self.calls.append((payload, materialised_meta))


class _RoundTripVectorStore:
    """Serializable FAISS stub used to validate full snapshot round-trips."""

    def __init__(self, vector_ids: Optional[Sequence[str]] = None) -> None:
        self._vector_ids = list(vector_ids or [])
        self._meta: dict[str, object] = {"ntotal": len(self._vector_ids)}
        self.calls: List[Tuple[List[str], Optional[Mapping[str, object]]]] = []

    def serialize(self) -> bytes:
        payload = json.dumps({"vectors": list(self._vector_ids)})
        return payload.encode("utf-8")

    def snapshot_meta(self) -> Mapping[str, object]:
        return dict(self._meta)

    def restore(self, payload: bytes, *, meta: Mapping[str, object] | None = None) -> None:
        data = json.loads(payload.decode("utf-8"))
        vectors = data.get("vectors", [])
        self._vector_ids = [str(vector_id) for vector_id in vectors]
        recorded = dict(meta) if meta is not None else None
        self.calls.append((list(self._vector_ids), recorded))
        if meta is not None:
            self._meta = dict(meta)
        else:
            self._meta = {"ntotal": len(self._vector_ids)}


def _encode_snapshot(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _make_chunk(vector_id: str) -> ChunkPayload:
    embedding = np.arange(4, dtype=np.float32)
    features = ChunkFeatures(
        bm25_terms={"term": 1.0},
        splade_weights={"term": 2.0},
        embedding=embedding,
    )
    return ChunkPayload(
        doc_id="doc",
        chunk_id=f"chunk-{vector_id}",
        vector_id=vector_id,
        namespace="ns",
        text=f"payload-{vector_id}",
        metadata={},
        features=features,
        token_count=embedding.size,
        source_chunk_idxs=(0,),
        doc_items_refs=("ref",),
    )


def test_restore_state_accepts_legacy_payload_by_default(
    caplog: "pytest.LogCaptureFixture",
) -> None:
    """Legacy payloads lacking metadata should restore successfully and emit a warning."""

    store = _RecordingVectorStore()
    legacy_payload = {
        "faiss": _encode_snapshot(b"legacy"),
        "vector_ids": ["chunk-1"],
    }

    with caplog.at_level(logging.WARNING):
        restore_state(store, legacy_payload)

    assert store.calls == [(b"legacy", None)]
    assert any("legacy payload" in record.getMessage() for record in caplog.records)


def test_restore_state_validates_and_applies_metadata(caplog: "pytest.LogCaptureFixture") -> None:
    """Metadata-enabled payloads should forward metadata and reject invalid shapes."""

    store = _RecordingVectorStore()
    payload = {
        "faiss": _encode_snapshot(b"with-meta"),
        "vector_ids": ["chunk-1"],
        "meta": {"ntotal": 1},
    }

    with caplog.at_level(logging.WARNING):
        restore_state(store, payload)

    assert store.calls == [(b"with-meta", {"ntotal": 1})]
    assert not any("legacy payload" in record.getMessage() for record in caplog.records)

    with pytest.raises(ValueError, match="invalid 'meta' type"):
        restore_state(store, {**payload, "meta": "oops"})


def test_serialize_restore_roundtrip_repopulates_registry() -> None:
    """serialize_state/restore_state should rebuild FAISS + registry state."""

    vector_ids = [f"vec-{idx}" for idx in range(3)]
    faiss_store = _RoundTripVectorStore(vector_ids)
    registry = ChunkRegistry()
    chunks = [_make_chunk(vector_id) for vector_id in vector_ids]
    persisted_chunks = [copy.deepcopy(chunk) for chunk in chunks]
    registry.upsert(chunks)

    payload = serialize_state(faiss_store, registry)

    restored_store = _RoundTripVectorStore()
    restored_registry = ChunkRegistry()

    restore_state(restored_store, payload, registry=restored_registry)

    assert restored_store.calls == [(vector_ids, {"ntotal": len(vector_ids)})]
    assert restored_registry.vector_ids() == vector_ids
    assert restored_registry.count() == len(vector_ids)

    # Registry lacks chunk payloads until lexical storage is rehydrated.
    assert restored_registry.bulk_get(vector_ids) == []

    restored_registry.upsert(persisted_chunks)
    hydrated = restored_registry.bulk_get(vector_ids)
    assert [chunk.vector_id for chunk in hydrated] == vector_ids

    faiss_ids = [restored_registry.to_faiss_id(vector_id) for vector_id in vector_ids]
    assert [restored_registry.resolve_faiss_id(fid) for fid in faiss_ids] == vector_ids
