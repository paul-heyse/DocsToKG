"""Tests covering vector/chunk alignment validation during ingestion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pyarrow as pa
import pyarrow.parquet as pq
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


def _write_jsonl(path: Path, entries: list[dict]) -> None:
    payload = "\n".join(json.dumps(entry) for entry in entries) + "\n"
    path.write_text(payload, encoding="utf-8")


def _write_parquet(path: Path, entries: list[dict]) -> None:
    normalized: list[dict] = []
    for entry in entries:
        clone = dict(entry)
        metadata = clone.get("model_metadata")
        if isinstance(metadata, dict) and not metadata:
            clone["model_metadata"] = {"__hybrid_dummy__": None}
        normalized.append(clone)
    table = pa.Table.from_pylist(normalized)
    pq.write_table(table, path)


def _pipeline() -> ChunkIngestionPipeline:
    return ChunkIngestionPipeline(
        faiss_index=_StubFaiss(),
        opensearch=_StubLexical(),
        registry=_StubRegistry(),
        observability=Observability(),
    )


def _document(chunk_path: Path, vector_path: Path) -> DocumentInput:
    return DocumentInput(
        doc_id="doc",
        namespace="ns",
        chunk_path=chunk_path,
        vector_path=vector_path,
        metadata={},
    )


def test_extra_vector_entries_jsonl_raise(tmp_path: Path) -> None:
    """JSONL vectors with extra UUIDs should raise an ingest error."""

    chunk_entries = [
        {
            "uuid": "vec-0",
            "chunk_id": 0,
            "text": "chunk-0",
            "num_tokens": 1,
            "source_chunk_idxs": [0],
            "doc_items_refs": [],
        },
        {
            "uuid": "vec-1",
            "chunk_id": 1,
            "text": "chunk-1",
            "num_tokens": 1,
            "source_chunk_idxs": [1],
            "doc_items_refs": [],
        },
    ]

    vector_entries = [
        {
            "UUID": "vec-0",
            "BM25": {"terms": [], "weights": []},
            "SPLADEv3": {"tokens": [], "weights": []},
            "Qwen3-4B": {"vector": [0.0, 0.0], "dimension": 2},
            "model_metadata": {},
        },
        {
            "UUID": "vec-1",
            "BM25": {"terms": [], "weights": []},
            "SPLADEv3": {"tokens": [], "weights": []},
            "Qwen3-4B": {"vector": [1.0, 1.0], "dimension": 2},
            "model_metadata": {},
        },
        {
            "UUID": "vec-extra",
            "BM25": {"terms": [], "weights": []},
            "SPLADEv3": {"tokens": [], "weights": []},
            "Qwen3-4B": {"vector": [2.0, 2.0], "dimension": 2},
            "model_metadata": {},
        },
    ]

    chunk_path = tmp_path / "doc.chunks.jsonl"
    vector_path = tmp_path / "doc.vectors.jsonl"
    _write_jsonl(chunk_path, chunk_entries)
    _write_jsonl(vector_path, vector_entries)

    pipeline = _pipeline()
    document = _document(chunk_path, vector_path)

    with pytest.raises(IngestError, match="Found vector entries without matching chunks") as excinfo:
        pipeline._load_precomputed_chunks(document)

    assert "vec-extra" in str(excinfo.value)


def test_extra_vector_entries_parquet_raise(tmp_path: Path) -> None:
    """Parquet vectors with extra UUIDs should raise an ingest error."""

    chunk_entries = [
        {
            "uuid": "vec-0",
            "chunk_id": 0,
            "text": "chunk-0",
            "num_tokens": 1,
            "source_chunk_idxs": [0],
            "doc_items_refs": [],
        },
        {
            "uuid": "vec-1",
            "chunk_id": 1,
            "text": "chunk-1",
            "num_tokens": 1,
            "source_chunk_idxs": [1],
            "doc_items_refs": [],
        },
    ]

    vector_entries = [
        {
            "UUID": "vec-0",
            "BM25": {"terms": [], "weights": []},
            "SPLADEv3": {"tokens": [], "weights": []},
            "Qwen3-4B": {"vector": [0.0, 0.0], "dimension": 2},
            "model_metadata": {},
        },
        {
            "UUID": "vec-1",
            "BM25": {"terms": [], "weights": []},
            "SPLADEv3": {"tokens": [], "weights": []},
            "Qwen3-4B": {"vector": [1.0, 1.0], "dimension": 2},
            "model_metadata": {},
        },
        {
            "UUID": "vec-extra",
            "BM25": {"terms": [], "weights": []},
            "SPLADEv3": {"tokens": [], "weights": []},
            "Qwen3-4B": {"vector": [2.0, 2.0], "dimension": 2},
            "model_metadata": {},
        },
    ]

    chunk_path = tmp_path / "doc.chunks.jsonl"
    vector_path = tmp_path / "doc.vectors.parquet"
    _write_jsonl(chunk_path, chunk_entries)
    _write_parquet(vector_path, vector_entries)

    pipeline = _pipeline()
    document = _document(chunk_path, vector_path)

    with pytest.raises(IngestError, match="Found vector entries without matching chunks") as excinfo:
        pipeline._load_precomputed_chunks(document)

    assert "vec-extra" in str(excinfo.value)
