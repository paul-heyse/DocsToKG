"""Tests for embedding fallback behaviour when parquet writes fail."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from DocsToKG.DocParsing.core.runner import WorkItem
from DocsToKG.DocParsing.embedding.backends.base import ProviderIdentity
from DocsToKG.DocParsing.embedding.runtime import (
    VectorWriterError,
    _embedding_stage_worker,
    _set_embed_worker_state,
    _vector_output_path_for_format,
)
from DocsToKG.DocParsing.logging import get_logger


class _DummyBundle:
    """Minimal provider bundle returning identity metadata."""

    def __init__(self) -> None:
        identity = ProviderIdentity(name="dummy", version="0.0.0")
        self._identities = {
            "dense": identity,
            "sparse": identity,
            "lexical": identity,
        }
        self.dense = object()
        self.sparse = object()
        self.lexical = object()
        self.context = SimpleNamespace(batch_hint=None)

    def identities(self) -> dict[str, ProviderIdentity]:
        return self._identities


class _DummyValidator:
    """Validator stub used for fallback testing."""

    zero_nnz_chunks: list[str] = []
    top_n: int = 0

    def validate(self, *args, **kwargs) -> None:  # pragma: no cover - stub
        return None

    def report(self, _logger) -> None:  # pragma: no cover - stub
        return None


@pytest.fixture(autouse=True)
def _clear_worker_state() -> None:
    """Ensure worker state is reset before and after each test."""

    _set_embed_worker_state({})
    yield
    _set_embed_worker_state({})


def test_embedding_worker_falls_back_to_jsonl(monkeypatch, tmp_path: Path) -> None:
    """Embedding worker should re-run with JSONL when parquet writing fails."""

    chunk_path = tmp_path / "Chunks" / "doc.chunks.jsonl"
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_path.write_text("[]\n", encoding="utf-8")

    parquet_path = tmp_path / "Vectors" / "family=dense" / "fmt=parquet" / "2024" / "10" / "doc.vectors.parquet"
    metadata = {
        "chunk_path": str(chunk_path),
        "output_path": str(parquet_path),
        "input_hash": "",
        "input_relpath": "Chunks/doc.chunks.jsonl",
        "output_relpath": "Vectors/family=dense/fmt=parquet/2024/10/doc.vectors.parquet",
        "fingerprint_path": str(parquet_path.with_suffix(".parquet.fp.json")),
        "vector_format": "parquet",
    }

    work_item = WorkItem(
        item_id="doc.doctags",
        inputs={"chunk": chunk_path},
        outputs={"vectors": parquet_path},
        cfg_hash="cfg",
        metadata=metadata,
    )

    bundle = _DummyBundle()
    cfg = SimpleNamespace(
        sparse_splade_st_attn_backend=None,
        sparsity_warn_threshold_pct=0.0,
    )
    stats = SimpleNamespace(avgdl=0.0, N=0)
    validator = _DummyValidator()
    logger = get_logger("docparse.tests", base_fields={"stage": "embed"})

    _set_embed_worker_state(
        {
            "bundle": bundle,
            "cfg": cfg,
            "stats": stats,
            "validator": validator,
            "logger": logger,
            "vector_format": "parquet",
            "resolved_root": tmp_path,
            "cfg_hash": "cfg",
            "stub_vectors": False,
            "stub_counters": None,
        }
    )

    fallback_calls: list[str] = []

    def fake_process_chunk_file_vectors(*args, vector_format: str = "parquet", **kwargs):
        fallback_calls.append(vector_format)
        output_path = Path(args[1])
        if vector_format == "parquet":
            raise VectorWriterError("parquet", output_path, RuntimeError("boom"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump({"vector_format": vector_format}, handle)
        return 2, [1, 1], [1.0, 1.0]

    monkeypatch.setattr(
        "DocsToKG.DocParsing.embedding.runtime.process_chunk_file_vectors",
        fake_process_chunk_file_vectors,
    )

    item = work_item.materialize()
    outcome = _embedding_stage_worker(item)

    assert outcome.status == "success"
    assert outcome.manifest["vector_format"] == "jsonl"
    assert outcome.manifest["vector_format_fallback_from"] == "parquet"
    assert outcome.result["vector_format"] == "jsonl"
    assert item.metadata["output_path"].endswith(".vectors.jsonl")

    jsonl_path = Path(item.metadata["output_path"])
    assert jsonl_path.exists()
    assert not parquet_path.exists()

    # The fallback helper should compute the same path
    assert jsonl_path == _vector_output_path_for_format(parquet_path, "jsonl")
    assert fallback_calls == ["parquet", "jsonl"]
