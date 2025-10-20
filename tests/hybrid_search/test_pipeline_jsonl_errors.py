"""Regression tests for JSONL ingestion failures in the pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from DocsToKG.HybridSearch import ChunkIngestionPipeline, Observability
from DocsToKG.HybridSearch.pipeline import IngestError


class _StubFaiss:
    """Minimal FAISS stub exposing the attributes used by the pipeline."""

    dim = 1

    def set_id_resolver(self, _: object) -> None:  # pragma: no cover - interface stub
        return None


class _StubRegistry:
    """Registry stub supplying the resolver bridge required by the pipeline."""

    def resolve_faiss_id(self, _: str) -> int:  # pragma: no cover - deterministic stub
        return 0


class _StubLexical:
    """Lexical index stub compatible with the ingestion constructor."""

    pass


def _pipeline() -> ChunkIngestionPipeline:
    return ChunkIngestionPipeline(
        faiss_index=_StubFaiss(),
        opensearch=_StubLexical(),
        registry=_StubRegistry(),
        observability=Observability(),
    )


def _write(path: Path) -> Path:
    payload = '{"good": true}\n{"bad": }\n'
    path.write_text(payload, encoding="utf-8")
    return path


def test_iter_jsonl_wraps_json_decode_error(tmp_path: Path) -> None:
    """Malformed JSON lines should surface as `IngestError` with context."""

    jsonl_path = _write(tmp_path / "broken.jsonl")
    pipeline = _pipeline()

    with pytest.raises(IngestError) as excinfo:
        list(pipeline._iter_jsonl(jsonl_path))

    message = str(excinfo.value)
    assert excinfo.value.__class__ is IngestError
    assert "broken.jsonl" in message
    assert "line 2" in message
    assert "Expecting value" in message
