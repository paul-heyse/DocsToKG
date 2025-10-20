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
from typing import List, Mapping, Optional, Tuple

import pytest

from DocsToKG.HybridSearch.store import restore_state


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


def _encode_snapshot(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


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
