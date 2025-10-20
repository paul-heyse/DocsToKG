"""Unit tests for guarding streaming vector cache growth during ingestion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Tuple

import pytest

from DocsToKG.HybridSearch import ChunkIngestionPipeline, DocumentInput, Observability
from DocsToKG.HybridSearch.pipeline import IngestError


class _StubFaiss:
    """Minimal FAISS stub exposing the attributes used by the pipeline."""

    dim = 2

    def set_id_resolver(self, _: Callable[[str], int]) -> None:  # pragma: no cover - trivial
        return None


class _StubRegistry:
    """Registry stub supplying the resolver bridge required by the pipeline."""

    def resolve_faiss_id(self, _: str) -> int:  # pragma: no cover - trivial mapping
        return 0


class _StubLexical:
    """Lexical index stub compatible with the ingestion constructor."""

    pass


def _write_jsonl(path: Path, entries: List[dict]) -> None:
    payload = "\n".join(json.dumps(entry) for entry in entries) + "\n"
    path.write_text(payload, encoding="utf-8")


def _misordered_document(tmp_path: Path) -> DocumentInput:
    vector_ids = ["vec-0", "vec-1", "vec-2"]
    chunk_entries = [
        {
            "uuid": vector_ids[2],
            "chunk_id": 0,
            "text": "misordered chunk",
            "num_tokens": 1,
            "source_chunk_idxs": [0],
            "doc_items_refs": [],
        },
        {
            "uuid": vector_ids[0],
            "chunk_id": 1,
            "text": "second chunk",
            "num_tokens": 1,
            "source_chunk_idxs": [1],
            "doc_items_refs": [],
        },
        {
            "uuid": vector_ids[1],
            "chunk_id": 2,
            "text": "third chunk",
            "num_tokens": 1,
            "source_chunk_idxs": [2],
            "doc_items_refs": [],
        },
    ]
    vector_entries = [
        {
            "UUID": vector_ids[0],
            "BM25": {"terms": [], "weights": []},
            "SPLADEv3": {"tokens": [], "weights": []},
            "Qwen3-4B": {"vector": [0.0, 0.0], "dimension": 2},
            "model_metadata": {},
        },
        {
            "UUID": vector_ids[1],
            "BM25": {"terms": [], "weights": []},
            "SPLADEv3": {"tokens": [], "weights": []},
            "Qwen3-4B": {"vector": [1.0, 1.0], "dimension": 2},
            "model_metadata": {},
        },
        {
            "UUID": vector_ids[2],
            "BM25": {"terms": [], "weights": []},
            "SPLADEv3": {"tokens": [], "weights": []},
            "Qwen3-4B": {"vector": [2.0, 2.0], "dimension": 2},
            "model_metadata": {},
        },
    ]

    chunk_dir = tmp_path / "chunks"
    vector_dir = tmp_path / "vectors"
    chunk_dir.mkdir()
    vector_dir.mkdir()

    chunk_path = chunk_dir / "doc.chunks.jsonl"
    vector_path = vector_dir / "doc.vectors.jsonl"
    _write_jsonl(chunk_path, chunk_entries)
    _write_jsonl(vector_path, vector_entries)

    return DocumentInput(
        doc_id="doc",
        namespace="ns",
        chunk_path=chunk_path,
        vector_path=vector_path,
        metadata={},
    )


def test_streaming_vector_cache_guard_raises(tmp_path: Path) -> None:
    """Shuffled vector artifacts should trigger the safety guard."""

    document = _misordered_document(tmp_path)

    stats: List[Tuple[int, str]] = []

    pipeline = ChunkIngestionPipeline(
        faiss_index=_StubFaiss(),
        opensearch=_StubLexical(),
        registry=_StubRegistry(),
        observability=Observability(),
        vector_cache_limit=1,
        vector_cache_stats_hook=lambda size, doc: stats.append((size, doc.doc_id)),
    )

    with pytest.raises(IngestError, match="Vector cache grew beyond the configured safety limit") as excinfo:
        list(pipeline._load_precomputed_chunks(document))

    assert "sorted consistently" in str(excinfo.value)
    assert stats, "Expected the stats hook to be invoked"
    assert any(size > 1 for size, doc_id in stats if doc_id == "doc")


def test_vector_cache_guard_zero_limit_fails_fast(tmp_path: Path) -> None:
    """A zero cache limit should fail immediately when drift is detected."""

    document = _misordered_document(tmp_path)

    stats: List[Tuple[int, str]] = []

    pipeline = ChunkIngestionPipeline(
        faiss_index=_StubFaiss(),
        opensearch=_StubLexical(),
        registry=_StubRegistry(),
        observability=Observability(),
        vector_cache_limit=0,
        vector_cache_stats_hook=lambda size, doc: stats.append((size, doc.doc_id)),
    )

    with pytest.raises(IngestError, match="Vector cache grew beyond the configured safety limit"):
        list(pipeline._load_precomputed_chunks(document))

    assert stats == [(1, "doc")]
