"""Core DocParsing utility and schema validation tests."""

from __future__ import annotations

import hashlib
import json
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from DocsToKG.DocParsing import _common, schemas
from DocsToKG.DocParsing.schemas import (
    CHUNK_SCHEMA_VERSION,
    VECTOR_SCHEMA_VERSION,
    validate_chunk_row,
    validate_vector_row,
)


# ---------------------------------------------------------------------------
# _common utility tests


@pytest.fixture(autouse=True)
def _reset_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Force utilities to operate inside a temporary data root."""

    data_root = tmp_path / "Data"
    data_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("DOCSTOKG_DATA_ROOT", str(data_root))
    yield
    monkeypatch.delenv("DOCSTOKG_DATA_ROOT", raising=False)


def test_detect_data_root_env(tmp_path: Path) -> None:
    data = tmp_path / "Data"
    data.mkdir(exist_ok=True)
    root = _common.detect_data_root()
    assert root == data.resolve()


def test_detect_data_root_scan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DOCSTOKG_DATA_ROOT", raising=False)
    project = tmp_path / "workspace" / "DocsToKG" / "src"
    target = tmp_path / "workspace" / "DocsToKG" / "Data"
    project.mkdir(parents=True)
    (target / "DocTagsFiles").mkdir(parents=True)
    monkeypatch.chdir(project)
    resolved = _common.detect_data_root()
    assert resolved == target.resolve()


def test_detect_data_root_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DOCSTOKG_DATA_ROOT", raising=False)
    start = tmp_path / "no_data_here"
    start.mkdir()
    resolved = _common.detect_data_root(start)
    assert resolved == (start / "Data").resolve()


def test_data_directories_created(tmp_path: Path) -> None:
    expected = {
        _common.data_doctags(): "DocTagsFiles",
        _common.data_chunks(): "ChunkedDocTagFiles",
        _common.data_vectors(): "Vectors",
        _common.data_manifests(): "Manifests",
        _common.data_pdfs(): "PDFs",
        _common.data_html(): "HTML",
    }
    for path, folder in expected.items():
        assert folder in str(path)
        assert path.exists()


def test_get_logger_idempotent() -> None:
    logger1 = _common.get_logger("docparse-test")
    logger2 = _common.get_logger("docparse-test")
    assert logger1 is logger2


def test_find_free_port_basic() -> None:
    port = _common.find_free_port(start=9000, span=4)
    assert isinstance(port, int)


def test_find_free_port_fallback() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    busy_port = sock.getsockname()[1]
    try:
        result = _common.find_free_port(start=busy_port, span=1)
        assert result != busy_port
    finally:
        sock.close()


def test_atomic_write_success(tmp_path: Path) -> None:
    target = tmp_path / "file.json"
    with _common.atomic_write(target) as handle:
        handle.write("data")
    assert target.read_text(encoding="utf-8") == "data"


def test_atomic_write_failure(tmp_path: Path) -> None:
    target = tmp_path / "file.json"
    with pytest.raises(RuntimeError):
        with _common.atomic_write(target) as handle:
            handle.write("data")
            raise RuntimeError("boom")
    assert not target.exists()
    assert not target.with_suffix(".json.tmp").exists()


def test_iter_doctags(tmp_path: Path) -> None:
    doctags_dir = _common.data_doctags()
    target = Path(doctags_dir)
    files = [target / "a.doctags", target / "b.doctag"]
    for file in files:
        file.write_text("content", encoding="utf-8")
    results = list(_common.iter_doctags(target))
    assert results == sorted(file.resolve() for file in files)


def test_iter_chunks(tmp_path: Path) -> None:
    chunks_dir = _common.data_chunks()
    good = chunks_dir / "doc.chunks.jsonl"
    other = chunks_dir / "doc.jsonl"
    good.write_text("{}\n", encoding="utf-8")
    other.write_text("{}\n", encoding="utf-8")
    results = list(_common.iter_chunks(chunks_dir))
    assert results == [good.resolve()]


def test_jsonl_load_and_save(tmp_path: Path) -> None:
    target = tmp_path / "example.jsonl"
    _common.jsonl_save(target, [{"a": 1}, {"b": 2}])
    rows = _common.jsonl_load(target)
    assert rows == [{"a": 1}, {"b": 2}]


def test_jsonl_load_skip_invalid(tmp_path: Path) -> None:
    target = tmp_path / "bad.jsonl"
    target.write_text("{\ninvalid\n", encoding="utf-8")
    rows = _common.jsonl_load(target, skip_invalid=True, max_errors=1)
    assert rows == []


def test_jsonl_save_validation_error(tmp_path: Path) -> None:
    target = tmp_path / "example.jsonl"

    def validate(row: dict) -> None:
        raise ValueError("bad row")

    with pytest.raises(ValueError, match="Validation failed"):
        _common.jsonl_save(target, [{"a": 1}], validate=validate)
    assert not target.exists()


def test_batcher() -> None:
    assert list(_common.Batcher([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    assert list(_common.Batcher([], 3)) == []


def test_manifest_append(tmp_path: Path) -> None:
    manifest = _common.data_manifests() / "docparse.chunks.manifest.jsonl"
    _common.manifest_append("chunks", "doc-1", "success", duration_s=1.23)
    content = manifest.read_text(encoding="utf-8").strip()
    record = json.loads(content)
    assert record["stage"] == "chunks"
    assert record["doc_id"] == "doc-1"
    assert record["status"] == "success"


# ---------------------------------------------------------------------------
# Schema validation tests


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
        "Qwen3_4B": {"model_id": "model", "vector": [0.1, 0.2], "dimension": 2},
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


# ---------------------------------------------------------------------------
# Golden fixture validation


FIXTURE_ROOT = Path("tests/data/docparsing/golden")


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@pytest.mark.parametrize("relative", ["sample.chunks.jsonl"])
def test_chunk_golden_rows_validate(relative: str) -> None:
    """Golden chunk fixtures must conform to the active schema."""

    rows = _load_jsonl(FIXTURE_ROOT / relative)
    assert rows, "expected at least one chunk row in fixture"

    for row in rows:
        validated = validate_chunk_row(row)
        assert validated.schema_version == CHUNK_SCHEMA_VERSION


@pytest.mark.parametrize("relative", ["sample.vectors.jsonl"])
def test_vector_golden_rows_validate(relative: str) -> None:
    """Golden vector fixtures must conform to the active schema."""

    rows = _load_jsonl(FIXTURE_ROOT / relative)
    assert rows, "expected at least one vector row in fixture"

    for row in rows:
        validated = validate_vector_row(row)
        assert validated.schema_version == VECTOR_SCHEMA_VERSION
