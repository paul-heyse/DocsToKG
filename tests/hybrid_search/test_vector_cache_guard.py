"""Unit tests for guarding streaming vector cache growth during ingestion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, List, Tuple

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


def _write_jsonl_iter(path: Path, entries: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry))
            handle.write("\n")


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


def _ordered_document(
    tmp_path: Path,
    *,
    count: int,
    drop_last_vector: bool = False,
    doc_id: str = "ordered",
) -> DocumentInput:
    chunk_dir = tmp_path / f"chunks-{doc_id}"
    vector_dir = tmp_path / f"vectors-{doc_id}"
    chunk_dir.mkdir()
    vector_dir.mkdir()

    chunk_path = chunk_dir / f"{doc_id}.chunks.jsonl"
    vector_path = vector_dir / f"{doc_id}.vectors.jsonl"

    def chunk_entries() -> Iterable[dict]:
        for idx in range(count):
            yield {
                "uuid": f"vec-{idx}",
                "chunk_id": idx,
                "text": f"chunk {idx}",
                "num_tokens": 1,
                "source_chunk_idxs": [idx],
                "doc_items_refs": [],
            }

    def vector_entries() -> Iterable[dict]:
        upper = count - 1 if drop_last_vector and count else count
        for idx in range(upper):
            yield {
                "UUID": f"vec-{idx}",
                "BM25": {"terms": [], "weights": []},
                "SPLADEv3": {"tokens": [], "weights": []},
                "Qwen3-4B": {"vector": [float(idx % 3), float((idx + 1) % 3)], "dimension": 2},
                "model_metadata": {},
            }

    _write_jsonl_iter(chunk_path, chunk_entries())
    _write_jsonl_iter(vector_path, vector_entries())

    return DocumentInput(
        doc_id=doc_id,
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

    with pytest.raises(
        IngestError, match="Vector cache grew beyond the configured safety limit"
    ) as excinfo:
        list(pipeline._load_precomputed_chunks(document))

    assert "sorted consistently" in str(excinfo.value)
    assert stats, "Expected the stats hook to be invoked"
    assert any(size > 1 for size, doc_id in stats if doc_id == "doc")


def test_vector_cache_guard_zero_limit_fails_fast(tmp_path: Path) -> None:
    """A zero cache limit should fail immediately when drift is detected."""

    document = _misordered_document(tmp_path)

    pipeline = ChunkIngestionPipeline(
        faiss_index=_StubFaiss(),
        opensearch=_StubLexical(),
        registry=_StubRegistry(),
        observability=Observability(),
        vector_cache_limit=0,
    )

    with pytest.raises(IngestError, match="Vector cache grew beyond the configured safety limit"):
        list(pipeline._load_precomputed_chunks(document))


def test_vector_cache_stats_hook_tracks_cache_shrink(tmp_path: Path) -> None:
    """Vector cache observers should see both growth and shrink events."""

    chunk_dir = tmp_path / "chunks"
    vector_dir = tmp_path / "vectors"
    chunk_dir.mkdir()
    vector_dir.mkdir()

    vector_ids = ["vec-0", "vec-1"]
    chunk_entries = [
        {
            "uuid": vector_ids[0],
            "chunk_id": 0,
            "text": "first chunk",
            "num_tokens": 1,
            "source_chunk_idxs": [0],
            "doc_items_refs": [],
        },
        {
            "uuid": vector_ids[1],
            "chunk_id": 1,
            "text": "second chunk",
            "num_tokens": 1,
            "source_chunk_idxs": [1],
            "doc_items_refs": [],
        },
    ]
    vector_entries = [
        {
            "UUID": vector_ids[1],
            "BM25": {"terms": [], "weights": []},
            "SPLADEv3": {"tokens": [], "weights": []},
            "Qwen3-4B": {"vector": [0.0, 0.0], "dimension": 2},
            "model_metadata": {},
        },
        {
            "UUID": vector_ids[0],
            "BM25": {"terms": [], "weights": []},
            "SPLADEv3": {"tokens": [], "weights": []},
            "Qwen3-4B": {"vector": [1.0, 1.0], "dimension": 2},
            "model_metadata": {},
        },
    ]

    chunk_path = chunk_dir / "doc.chunks.jsonl"
    vector_path = vector_dir / "doc.vectors.jsonl"
    _write_jsonl(chunk_path, chunk_entries)
    _write_jsonl(vector_path, vector_entries)

    document = DocumentInput(
        doc_id="doc",
        namespace="ns",
        chunk_path=chunk_path,
        vector_path=vector_path,
        metadata={},
    )

    stats: List[Tuple[int, str]] = []

    guarded = ChunkIngestionPipeline(
        faiss_index=_StubFaiss(),
        opensearch=_StubLexical(),
        registry=_StubRegistry(),
        observability=Observability(),
        vector_cache_limit=0,
        vector_cache_stats_hook=lambda size, doc: stats.append((size, doc.doc_id)),
    )

    with pytest.raises(IngestError, match="Vector cache grew beyond the configured safety limit"):
        list(guarded._load_precomputed_chunks(document))

    assert stats and stats[-1] == (1, "doc")

    stats.clear()
    pipeline = ChunkIngestionPipeline(
        faiss_index=_StubFaiss(),
        opensearch=_StubLexical(),
        registry=_StubRegistry(),
        observability=Observability(),
        vector_cache_limit=10,
        vector_cache_stats_hook=lambda size, doc: stats.append((size, doc.doc_id)),
    )

    loaded = list(pipeline._load_precomputed_chunks(document))

    assert len(loaded) == 2
    sizes = [size for size, doc_id in stats if doc_id == "doc"]
    assert any(size > 0 for size in sizes), "Expected a cache growth event"
    assert sizes[-1] == 0, "Expected cache to be empty after loading"
    assert any(
        earlier > later for earlier, later in zip(sizes, sizes[1:])
    ), "Expected at least one cache shrink event"


def test_missing_vectors_detected_during_streaming(tmp_path: Path) -> None:
    """Missing vectors should raise even when chunk JSONL is streamed lazily."""

    document = _ordered_document(tmp_path, count=3, drop_last_vector=True, doc_id="missing")

    pipeline = ChunkIngestionPipeline(
        faiss_index=_StubFaiss(),
        opensearch=_StubLexical(),
        registry=_StubRegistry(),
        observability=Observability(),
        vector_cache_limit=4,
    )

    with pytest.raises(IngestError, match="Missing vector entries for chunk UUIDs: vec-2"):
        pipeline._load_precomputed_chunks(document)


def test_large_jsonl_ingest_bounded_memory(tmp_path: Path) -> None:
    """Ingesting large JSONL artifacts should keep the vector cache bounded."""

    document = _ordered_document(tmp_path, count=5_000, doc_id="large")
    stats: List[Tuple[int, str]] = []
    pipeline = ChunkIngestionPipeline(
        faiss_index=_StubFaiss(),
        opensearch=_StubLexical(),
        registry=_StubRegistry(),
        observability=Observability(),
        vector_cache_limit=16,
        vector_cache_stats_hook=lambda size, doc: stats.append((size, doc.doc_id)),
    )

    import tracemalloc

    tracemalloc.start()
    try:
        payloads = pipeline._load_precomputed_chunks(document)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert len(payloads) == 5_000
    max_cache = max((size for size, doc_id in stats if doc_id == "large"), default=0)
    assert max_cache <= 1
    # Ensure the ingestion stays well within a 100 MiB working set while streaming.
    assert peak < 100 * 1024 * 1024
