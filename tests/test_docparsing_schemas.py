"""Tests for DocParsing Pydantic schemas."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from DocsToKG.DocParsing import schemas


def make_chunk_row(**overrides):
    base = {
        "doc_id": "doc",
        "source_path": "path",
        "chunk_id": 0,
        "source_chunk_idxs": [0],
        "num_tokens": 10,
        "text": "hello",
    }
    base.update(overrides)
    return base


def make_vector_row(**overrides):
    base = {
        "UUID": "uuid",
        "BM25": {"terms": ["a"], "weights": [1.0], "avgdl": 1.0, "N": 1},
        "SPLADEv3": {"tokens": ["a"], "weights": [0.5]},
        "Qwen3-4B": {"model_id": "model", "vector": [0.1, 0.2], "dimension": 2},
    }
    base.update(overrides)
    return base


def test_chunk_row_valid() -> None:
    parsed = schemas.validate_chunk_row(make_chunk_row())
    assert parsed.doc_id == "doc"
    assert parsed.schema_version == schemas.CHUNK_SCHEMA_VERSION


def test_chunk_row_missing_field() -> None:
    with pytest.raises(ValueError):
        schemas.validate_chunk_row({"doc_id": "doc"})


def test_chunk_row_invalid_num_tokens() -> None:
    with pytest.raises(ValueError):
        schemas.validate_chunk_row(make_chunk_row(num_tokens=0))
    with pytest.raises(ValueError):
        schemas.validate_chunk_row(make_chunk_row(num_tokens=200_000))


def test_chunk_row_invalid_page_numbers() -> None:
    with pytest.raises(ValueError):
        schemas.validate_chunk_row(make_chunk_row(page_nos=[0, 1]))


def test_provenance_invalid_engine() -> None:
    with pytest.raises(ValueError):
        schemas.ProvenanceMetadata(parse_engine="bad", docling_version="1")


def test_vector_row_valid() -> None:
    parsed = schemas.validate_vector_row(make_vector_row())
    assert parsed.UUID == "uuid"
    assert parsed.schema_version == schemas.VECTOR_SCHEMA_VERSION


def test_vector_row_mismatched_terms() -> None:
    data = make_vector_row(BM25={"terms": ["a"], "weights": [1.0, 2.0], "avgdl": 1.0, "N": 1})
    with pytest.raises(ValueError):
        schemas.validate_vector_row(data)


def test_vector_row_negative_weights() -> None:
    data = make_vector_row(SPLADEv3={"tokens": ["a"], "weights": [-0.1]})
    with pytest.raises(ValueError):
        schemas.validate_vector_row(data)


def test_dense_vector_dimension_mismatch() -> None:
    data = make_vector_row(Qwen3_4B={"model_id": "model", "vector": [0.1], "dimension": 2})
    with pytest.raises(ValueError):
        schemas.validate_vector_row(data)


def test_get_docling_version(monkeypatch: pytest.MonkeyPatch) -> None:
    module = SimpleNamespace(__version__="1.2.3")
    monkeypatch.setitem(sys.modules, "docling", module)
    assert schemas.get_docling_version() == "1.2.3"
    monkeypatch.delitem(sys.modules, "docling", raising=False)
    assert schemas.get_docling_version() == "unknown"


def test_validate_schema_version() -> None:
    assert (
        schemas.validate_schema_version("docparse/1.1.0", schemas.COMPATIBLE_CHUNK_VERSIONS)
        == "docparse/1.1.0"
    )
    with pytest.raises(ValueError):
        schemas.validate_schema_version("other", schemas.COMPATIBLE_CHUNK_VERSIONS)
    with pytest.raises(ValueError):
        schemas.validate_schema_version(None, schemas.COMPATIBLE_CHUNK_VERSIONS)


def test_chunk_row_invalid_schema_version() -> None:
    with pytest.raises(ValueError):
        schemas.validate_chunk_row(make_chunk_row(schema_version="docparse/0.9.0"))


def test_vector_row_invalid_schema_version() -> None:
    with pytest.raises(ValueError):
        schemas.validate_vector_row(make_vector_row(schema_version="embeddings/0.9.0"))
