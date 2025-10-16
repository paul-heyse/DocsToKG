# === NAVMAP v1 ===
# {
#   "module": "tests.docparsing.test_docparsing_core",
#   "purpose": "Pytest coverage for docparsing docparsing core scenarios",
#   "sections": [
#     {
#       "id": "requestsresponse",
#       "name": "_RequestsResponse",
#       "anchor": "class-requestsresponse",
#       "kind": "class"
#     },
#     {
#       "id": "stub-module",
#       "name": "_stub_module",
#       "anchor": "function-stub-module",
#       "kind": "function"
#     },
#     {
#       "id": "dummytqdm",
#       "name": "_DummyTqdm",
#       "anchor": "class-dummytqdm",
#       "kind": "class"
#     },
#     {
#       "id": "reset-env",
#       "name": "_reset_env",
#       "anchor": "function-reset-env",
#       "kind": "function"
#     },
#     {
#       "id": "test-detect-data-root-env",
#       "name": "test_detect_data_root_env",
#       "anchor": "function-test-detect-data-root-env",
#       "kind": "function"
#     },
#     {
#       "id": "test-detect-data-root-scan",
#       "name": "test_detect_data_root_scan",
#       "anchor": "function-test-detect-data-root-scan",
#       "kind": "function"
#     },
#     {
#       "id": "test-detect-data-root-fallback",
#       "name": "test_detect_data_root_fallback",
#       "anchor": "function-test-detect-data-root-fallback",
#       "kind": "function"
#     },
#     {
#       "id": "test-data-directories-created",
#       "name": "test_data_directories_created",
#       "anchor": "function-test-data-directories-created",
#       "kind": "function"
#     },
#     {
#       "id": "test-get-logger-idempotent",
#       "name": "test_get_logger_idempotent",
#       "anchor": "function-test-get-logger-idempotent",
#       "kind": "function"
#     },
#     {
#       "id": "test-find-free-port-basic",
#       "name": "test_find_free_port_basic",
#       "anchor": "function-test-find-free-port-basic",
#       "kind": "function"
#     },
#     {
#       "id": "test-find-free-port-fallback",
#       "name": "test_find_free_port_fallback",
#       "anchor": "function-test-find-free-port-fallback",
#       "kind": "function"
#     },
#     {
#       "id": "test-atomic-write-success",
#       "name": "test_atomic_write_success",
#       "anchor": "function-test-atomic-write-success",
#       "kind": "function"
#     },
#     {
#       "id": "test-atomic-write-failure",
#       "name": "test_atomic_write_failure",
#       "anchor": "function-test-atomic-write-failure",
#       "kind": "function"
#     },
#     {
#       "id": "test-iter-doctags",
#       "name": "test_iter_doctags",
#       "anchor": "function-test-iter-doctags",
#       "kind": "function"
#     },
#     {
#       "id": "test-iter-chunks",
#       "name": "test_iter_chunks",
#       "anchor": "function-test-iter-chunks",
#       "kind": "function"
#     },
#     {
#       "id": "test-jsonl-load-and-save",
#       "name": "test_jsonl_load_and_save",
#       "anchor": "function-test-jsonl-load-and-save",
#       "kind": "function"
#     },
#     {
#       "id": "test-jsonl-load-skip-invalid",
#       "name": "test_jsonl_load_skip_invalid",
#       "anchor": "function-test-jsonl-load-skip-invalid",
#       "kind": "function"
#     },
#     {
#       "id": "test-jsonl-save-validation-error",
#       "name": "test_jsonl_save_validation_error",
#       "anchor": "function-test-jsonl-save-validation-error",
#       "kind": "function"
#     },
#     {
#       "id": "test-batcher",
#       "name": "test_batcher",
#       "anchor": "function-test-batcher",
#       "kind": "function"
#     },
#     {
#       "id": "test-manifest-append",
#       "name": "test_manifest_append",
#       "anchor": "function-test-manifest-append",
#       "kind": "function"
#     },
#     {
#       "id": "make-chunk-row",
#       "name": "make_chunk_row",
#       "anchor": "function-make-chunk-row",
#       "kind": "function"
#     },
#     {
#       "id": "make-vector-row",
#       "name": "make_vector_row",
#       "anchor": "function-make-vector-row",
#       "kind": "function"
#     },
#     {
#       "id": "test-chunk-row-valid",
#       "name": "test_chunk_row_valid",
#       "anchor": "function-test-chunk-row-valid",
#       "kind": "function"
#     },
#     {
#       "id": "test-chunk-row-missing-field",
#       "name": "test_chunk_row_missing_field",
#       "anchor": "function-test-chunk-row-missing-field",
#       "kind": "function"
#     },
#     {
#       "id": "test-chunk-row-invalid-num-tokens",
#       "name": "test_chunk_row_invalid_num_tokens",
#       "anchor": "function-test-chunk-row-invalid-num-tokens",
#       "kind": "function"
#     },
#     {
#       "id": "test-chunk-row-invalid-page-numbers",
#       "name": "test_chunk_row_invalid_page_numbers",
#       "anchor": "function-test-chunk-row-invalid-page-numbers",
#       "kind": "function"
#     },
#     {
#       "id": "test-provenance-invalid-engine",
#       "name": "test_provenance_invalid_engine",
#       "anchor": "function-test-provenance-invalid-engine",
#       "kind": "function"
#     },
#     {
#       "id": "test-vector-row-valid",
#       "name": "test_vector_row_valid",
#       "anchor": "function-test-vector-row-valid",
#       "kind": "function"
#     },
#     {
#       "id": "test-vector-row-mismatched-terms",
#       "name": "test_vector_row_mismatched_terms",
#       "anchor": "function-test-vector-row-mismatched-terms",
#       "kind": "function"
#     },
#     {
#       "id": "test-vector-row-negative-weights",
#       "name": "test_vector_row_negative_weights",
#       "anchor": "function-test-vector-row-negative-weights",
#       "kind": "function"
#     },
#     {
#       "id": "test-dense-vector-dimension-mismatch",
#       "name": "test_dense_vector_dimension_mismatch",
#       "anchor": "function-test-dense-vector-dimension-mismatch",
#       "kind": "function"
#     },
#     {
#       "id": "test-get-docling-version",
#       "name": "test_get_docling_version",
#       "anchor": "function-test-get-docling-version",
#       "kind": "function"
#     },
#     {
#       "id": "test-validate-schema-version",
#       "name": "test_validate_schema_version",
#       "anchor": "function-test-validate-schema-version",
#       "kind": "function"
#     },
#     {
#       "id": "test-chunk-row-invalid-schema-version",
#       "name": "test_chunk_row_invalid_schema_version",
#       "anchor": "function-test-chunk-row-invalid-schema-version",
#       "kind": "function"
#     },
#     {
#       "id": "test-vector-row-invalid-schema-version",
#       "name": "test_vector_row_invalid_schema_version",
#       "anchor": "function-test-vector-row-invalid-schema-version",
#       "kind": "function"
#     },
#     {
#       "id": "load-jsonl",
#       "name": "_load_jsonl",
#       "anchor": "function-load-jsonl",
#       "kind": "function"
#     },
#     {
#       "id": "test-chunk-golden-rows-validate",
#       "name": "test_chunk_golden_rows_validate",
#       "anchor": "function-test-chunk-golden-rows-validate",
#       "kind": "function"
#     },
#     {
#       "id": "test-vector-golden-rows-validate",
#       "name": "test_vector_golden_rows_validate",
#       "anchor": "function-test-vector-golden-rows-validate",
#       "kind": "function"
#     },
#     {
#       "id": "test-set-spawn-or-warn-warns-on-incompatible-method",
#       "name": "test_set_spawn_or_warn_warns_on_incompatible_method",
#       "anchor": "function-test-set-spawn-or-warn-warns-on-incompatible-method",
#       "kind": "function"
#     },
#     {
#       "id": "test-compute-relative-doc-id-handles-subdirectories",
#       "name": "test_compute_relative_doc_id_handles_subdirectories",
#       "anchor": "function-test-compute-relative-doc-id-handles-subdirectories",
#       "kind": "function"
#     },
#     {
#       "id": "test-derive-doc-id-and-output-path",
#       "name": "test_derive_doc_id_and_vectors_path",
#       "anchor": "function-test-derive-doc-id-and-vectors-path",
#       "kind": "function"
#     },
#     {
#       "id": "test-chunk-row-fields-unique",
#       "name": "test_chunk_row_fields_unique",
#       "anchor": "function-test-chunk-row-fields-unique",
#       "kind": "function"
#     },
#     {
#       "id": "test-qwen-embed-caches-llm",
#       "name": "test_qwen_embed_caches_llm",
#       "anchor": "function-test-qwen-embed-caches-llm",
#       "kind": "function"
#     },
#     {
#       "id": "test-pdf-model-path-resolution-precedence",
#       "name": "test_pdf_model_path_resolution_precedence",
#       "anchor": "function-test-pdf-model-path-resolution-precedence",
#       "kind": "function"
#     },
#     {
#       "id": "test-pdf-model-path-cli-normalization",
#       "name": "test_pdf_model_path_cli_normalization",
#       "anchor": "function-test-pdf-model-path-cli-normalization",
#       "kind": "function"
#     },
#     {
#       "id": "test-pdf-pipeline-mirrors-output-paths",
#       "name": "test_pdf_pipeline_mirrors_output_paths",
#       "anchor": "function-test-pdf-pipeline-mirrors-output-paths",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Core DocParsing utility and schema validation tests."""

from __future__ import annotations

import argparse
import importlib
import json
import socket
import sys
import uuid
import warnings
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


class _RequestsResponse:
    headers = {"content-type": "application/json"}
    text = "{}"
    status_code = 200

    def json(self):  # pragma: no cover - simple stub
        return {"data": []}


class _StubHuggingFaceTokenizer:
    def __init__(self, tokenizer, max_tokens):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens


def _stub_module(name: str, **attrs):
    module = ModuleType(name)
    module.__dict__.update(attrs)
    return module


sys.modules.setdefault(
    "requests", _stub_module("requests", get=lambda *args, **kwargs: _RequestsResponse())
)

sys.modules.setdefault("tqdm", _stub_module("tqdm", tqdm=lambda *a, **kw: _DummyTqdm(*a, **kw)))

sys.modules.setdefault(
    "docling_core.transforms.chunker.base",
    _stub_module("docling_core.transforms.chunker.base", BaseChunk=object),
)
sys.modules.setdefault(
    "docling_core.transforms.chunker.hybrid_chunker",
    _stub_module("docling_core.transforms.chunker.hybrid_chunker", HybridChunker=object),
)
sys.modules.setdefault(
    "docling_core.transforms.chunker.tokenizer.huggingface",
    _stub_module(
        "docling_core.transforms.chunker.tokenizer.huggingface",
        HuggingFaceTokenizer=_StubHuggingFaceTokenizer,
    ),
)
sys.modules.setdefault(
    "docling_core.transforms.chunker.hierarchical_chunker",
    _stub_module(
        "docling_core.transforms.chunker.hierarchical_chunker",
        ChunkingDocSerializer=object,
        ChunkingSerializerProvider=object,
    ),
)
sys.modules.setdefault(
    "docling_core.types.doc.document",
    _stub_module(
        "docling_core.types.doc.document",
        DoclingDocument=object,
        DocTagsDocument=object,
        PictureClassificationData=object,
        PictureDescriptionData=object,
        PictureItem=object,
        PictureMoleculeData=object,
    ),
)
sys.modules.setdefault(
    "transformers",
    _stub_module(
        "transformers",
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace()),
    ),
)
sys.modules.setdefault(
    "docling_core.transforms.serializer.base",
    _stub_module(
        "docling_core.transforms.serializer.base",
        BaseDocSerializer=object,
        SerializationResult=object,
    ),
)
sys.modules.setdefault(
    "docling_core.transforms.serializer.common",
    _stub_module(
        "docling_core.transforms.serializer.common",
        create_ser_result=lambda *a, **k: {},
    ),
)
sys.modules.setdefault(
    "docling_core.transforms.serializer.markdown",
    _stub_module(
        "docling_core.transforms.serializer.markdown",
        MarkdownParams=object,
        MarkdownPictureSerializer=object,
        MarkdownTableSerializer=object,
    ),
)


class _DummyTqdm:
    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - simple stub
        self.total = kwargs.get("total", 0)

    def __enter__(self):  # pragma: no cover - simple stub
        return self

    def __exit__(self, exc_type, exc, tb):  # pragma: no cover - simple stub
        return False

    def update(self, *_args, **_kwargs) -> None:  # pragma: no cover - simple stub
        pass


from DocsToKG.DocParsing import _common, schemas  # noqa: E402
from DocsToKG.DocParsing.schemas import (  # noqa: E402
    CHUNK_SCHEMA_VERSION,
    VECTOR_SCHEMA_VERSION,
    validate_chunk_row,
    validate_vector_row,
)

# --- _common utility tests ---


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
        _common.data_vectors(): "Embeddings",
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
    nested = chunks_dir / "teamA" / "doc.chunks.jsonl"
    other = chunks_dir / "doc.jsonl"
    good.write_text("{}\n", encoding="utf-8")
    nested.parent.mkdir(parents=True, exist_ok=True)
    nested.write_text("{}\n", encoding="utf-8")
    other.write_text("{}\n", encoding="utf-8")
    results = list(_common.iter_chunks(chunks_dir))
    assert results == sorted({good.resolve(), nested.resolve()})


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


# --- Schema validation tests ---


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


# --- Golden fixture validation ---

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


# --- New robustness and configuration tests ---


def test_set_spawn_or_warn_warns_on_incompatible_method(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import multiprocessing as mp

    def raise_runtime_error(*_args, **_kwargs):
        raise RuntimeError("already set")

    monkeypatch.setattr(mp, "set_start_method", raise_runtime_error)
    monkeypatch.setattr(mp, "get_start_method", lambda allow_none=True: "fork")

    class DummyLogger:
        def __init__(self) -> None:
            self.warnings: list[str] = []

        def warning(self, message: str) -> None:  # pragma: no cover - simple stub
            self.warnings.append(message)

        def debug(self, *_args, **_kwargs) -> None:  # pragma: no cover - simple stub
            pass

    dummy = DummyLogger()
    _common.set_spawn_or_warn(dummy)
    assert dummy.warnings
    assert "spawn" in dummy.warnings[0]


def test_compute_relative_doc_id_handles_subdirectories(tmp_path: Path) -> None:
    from DocsToKG.DocParsing import _common

    root = tmp_path / "docs"
    nested = root / "teamA" / "report.doctags"
    nested.parent.mkdir(parents=True, exist_ok=True)
    nested.write_text("{}", encoding="utf-8")

    assert _common.compute_relative_doc_id(nested, root) == "teamA/report.doctags"


def test_derive_doc_id_and_vectors_path(tmp_path: Path) -> None:
    from DocsToKG.DocParsing._common import derive_doc_id_and_vectors_path

    chunks_root = tmp_path / "chunks"
    chunk_file = chunks_root / "teamA" / "report.chunks.jsonl"
    chunk_file.parent.mkdir(parents=True, exist_ok=True)
    chunk_file.write_text("{}\n", encoding="utf-8")
    vectors_root = tmp_path / "vectors"

    doc_id, out_path = derive_doc_id_and_vectors_path(chunk_file, chunks_root, vectors_root)

    assert doc_id == "teamA/report.doctags"
    assert out_path == vectors_root / "teamA" / "report.vectors.jsonl"


def test_chunk_row_fields_unique() -> None:
    fields = schemas.ChunkRow.model_fields
    for name in ("has_image_captions", "has_image_classification", "num_images"):
        assert name in fields


def test_detect_mode_pdf_only(tmp_path: Path) -> None:
    from DocsToKG.DocParsing import cli as cli_module

    input_dir = tmp_path / "pdf-only"
    nested = input_dir / "nested"
    nested.mkdir(parents=True, exist_ok=True)
    (nested / "doc.PDF").write_text("content", encoding="utf-8")

    assert cli_module._detect_mode(input_dir) == "pdf"


def test_detect_mode_html_only(tmp_path: Path) -> None:
    from DocsToKG.DocParsing import cli as cli_module

    input_dir = tmp_path / "html-only"
    nested = input_dir / "nested"
    nested.mkdir(parents=True, exist_ok=True)
    (nested / "page.HTM").write_text("<html/>", encoding="utf-8")

    assert cli_module._detect_mode(input_dir) == "html"


def test_detect_mode_raises_when_both_present(tmp_path: Path) -> None:
    from DocsToKG.DocParsing import cli as cli_module

    input_dir = tmp_path / "mixed"
    nested = input_dir / "nested"
    nested.mkdir(parents=True, exist_ok=True)
    (nested / "doc.pdf").write_text("content", encoding="utf-8")
    (nested / "page.html").write_text("<html/>", encoding="utf-8")

    with pytest.raises(ValueError, match="Cannot auto-detect mode"):
        cli_module._detect_mode(input_dir)


def test_ensure_uuid_deterministic_generation() -> None:
    from DocsToKG.DocParsing import EmbeddingV2 as embedding

    rows = [
        {
            "doc_id": "teamA/report.doctags",
            "source_chunk_idxs": [0, 2],
            "text": "Chunk content for reproducible UUID.",
        }
    ]

    assert embedding.ensure_uuid(rows) is True
    generated = rows[0]["uuid"]
    assert uuid.UUID(generated).version == 5

    replica_rows = [
        {
            "doc_id": "teamA/report.doctags",
            "source_chunk_idxs": [0, 2],
            "text": "Chunk content for reproducible UUID.",
        }
    ]
    embedding.ensure_uuid(replica_rows)

    assert replica_rows[0]["uuid"] == generated
    assert embedding.ensure_uuid(rows) is False


def test_qwen_embed_caches_llm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from DocsToKG.DocParsing import EmbeddingV2 as embedding

    class DummyOutput:
        def __init__(self) -> None:
            self.outputs = SimpleNamespace(embedding=[0.1, 0.2])

    class DummyLLM:
        instances = 0

        def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - simple counter
            type(self).instances += 1

        def embed(self, batch, pooling_params):  # pragma: no cover - simple stub
            return [DummyOutput() for _ in batch]

    class DummyPoolingParams:
        def __init__(self, normalize: bool = True, **kwargs: Any) -> None:
            self.normalize = normalize

    monkeypatch.setattr(embedding, "LLM", DummyLLM)
    monkeypatch.setattr(embedding, "PoolingParams", DummyPoolingParams)
    monkeypatch.setattr(embedding, "_VLLM_IMPORT_ERROR", None)
    monkeypatch.setattr(embedding, "_QWEN_LLM_CACHE", {})

    cfg = embedding.QwenCfg(model_dir=tmp_path, batch_size=2)
    first = embedding.qwen_embed(cfg, ["a", "b"])
    second = embedding.qwen_embed(cfg, ["c"])

    assert DummyLLM.instances == 1
    assert len(first) == 2
    assert len(second) == 1


def test_pdf_model_path_resolution_precedence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class DummyResponse:
        headers = {"content-type": "application/json"}
        text = "{}"
        status_code = 200

        def json(self):  # pragma: no cover - simple stub
            return {"data": []}

    monkeypatch.setitem(
        sys.modules,
        "requests",
        SimpleNamespace(get=lambda *args, **kwargs: DummyResponse()),
    )

    from DocsToKG.DocParsing import pipelines

    monkeypatch.delenv("DOCLING_PDF_MODEL", raising=False)
    monkeypatch.delenv("DOCSTOKG_MODEL_ROOT", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)

    cli_override = "explicit-model"
    assert pipelines.resolve_pdf_model_path(cli_override) == cli_override

    env_model = tmp_path / "custom"
    monkeypatch.setenv("DOCLING_PDF_MODEL", str(env_model))
    assert pipelines.resolve_pdf_model_path(None) == str(env_model.resolve())

    monkeypatch.delenv("DOCLING_PDF_MODEL", raising=False)
    model_root = tmp_path / "root"
    monkeypatch.setenv("DOCSTOKG_MODEL_ROOT", str(model_root))
    expected = (model_root / "granite-docling-258M").resolve()
    assert pipelines.resolve_pdf_model_path(None) == str(expected)

    monkeypatch.delenv("DOCSTOKG_MODEL_ROOT", raising=False)
    hf_home = tmp_path / "hf"
    monkeypatch.setenv("HF_HOME", str(hf_home))
    expected = (hf_home / "granite-docling-258M").resolve()
    assert pipelines.resolve_pdf_model_path(None) == str(expected)


def test_pdf_model_path_cli_normalization(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from DocsToKG.DocParsing import pipelines

    monkeypatch.delenv("DOCLING_PDF_MODEL", raising=False)
    monkeypatch.delenv("DOCSTOKG_MODEL_ROOT", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)

    local_dir = tmp_path / "models" / "granite"
    local_dir.mkdir(parents=True)
    assert pipelines.resolve_pdf_model_path(str(local_dir)) == str(local_dir.resolve())

    tilde_path = str(Path("~") / "granite-model")
    assert pipelines.resolve_pdf_model_path(tilde_path) == str(
        Path(tilde_path).expanduser().resolve()
    )

    repo_id = "ibm-granite/granite-docling-258M"
    assert pipelines.resolve_pdf_model_path(repo_id) == repo_id


def test_pdf_pipeline_mirrors_output_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyResponse:
        headers = {"content-type": "application/json"}
        text = "{}"
        status_code = 200

        def json(self):  # pragma: no cover - simple stub
            return {"data": []}

    monkeypatch.setitem(
        sys.modules,
        "requests",
        SimpleNamespace(get=lambda *args, **kwargs: DummyResponse()),
    )

    from DocsToKG.DocParsing import pipelines

    input_dir = tmp_path / "inputs"
    (input_dir / "teamA").mkdir(parents=True)
    (input_dir / "teamB").mkdir(parents=True)
    for folder in ("teamA", "teamB"):
        target = input_dir / folder / "report.pdf"
        target.write_text("content", encoding="utf-8")

    output_dir = tmp_path / "outputs"
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    data_root = tmp_path / "data-root"
    data_root.mkdir()
    monkeypatch.setenv("DOCSTOKG_DATA_ROOT", str(data_root))

    monkeypatch.setattr(pipelines, "ensure_vllm", lambda *args, **kwargs: (8000, None, False))
    monkeypatch.setattr(pipelines, "probe_metrics", lambda port: (True, 200))
    monkeypatch.setattr(pipelines, "detect_vllm_version", lambda: "test")

    class ImmediateExecutor:
        def __init__(self, *args, **kwargs) -> None:
            self._submitted = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, task):
            from concurrent.futures import Future

            fut = Future()
            try:
                fut.set_result(fn(task))
            except Exception as exc:  # pragma: no cover - defensive
                fut.set_exception(exc)
            self._submitted.append(task)
            return fut

    monkeypatch.setattr(pipelines, "ProcessPoolExecutor", ImmediateExecutor)

    class DummyProgress:
        def __init__(self, *args, total=0, **_kwargs) -> None:
            self.total = total

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, *_args, **_kwargs) -> None:  # pragma: no cover - simple stub
            pass

    monkeypatch.setattr(pipelines, "tqdm", lambda *args, **kwargs: DummyProgress(*args, **kwargs))

    def immediate_as_completed(futures):
        return list(futures)

    monkeypatch.setattr(pipelines, "as_completed", immediate_as_completed)

    def fake_convert(task):
        task.output_path.parent.mkdir(parents=True, exist_ok=True)
        task.output_path.write_text("{}", encoding="utf-8")
        return pipelines.PdfConversionResult(
            doc_id=task.doc_id,
            status="success",
            duration_s=0.1,
            input_path=str(task.pdf_path),
            input_hash=task.input_hash,
            output_path=str(task.output_path),
        )

    monkeypatch.setattr(pipelines, "pdf_convert_one", fake_convert)

    args = argparse.Namespace(
        data_root=data_root,
        input=input_dir,
        output=output_dir,
        workers=1,
        resume=False,
        force=False,
        served_model_names=None,
        model=str(model_dir),
        gpu_memory_utilization=0.1,
        vlm_prompt="Convert this page to docling.",
        vlm_stop=["</doctag>", "<|end_of_text|>"],
    )

    exit_code = pipelines.pdf_main(args)
    assert exit_code == 0
    assert (output_dir / "teamA" / "report.doctags").exists()
    assert (output_dir / "teamB" / "report.doctags").exists()

    manifest_path = data_root / "Manifests" / "docparse.doctags-pdf.manifest.jsonl"
    assert manifest_path.exists()
    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    doc_ids = {rec["doc_id"] for rec in records if rec.get("doc_id") != "__service__"}
    assert doc_ids == {"teamA/report.pdf", "teamB/report.pdf"}

    args.resume = True
    call_count = {"value": 0}

    def unexpected_convert(task):  # pragma: no cover - resume guard
        call_count["value"] += 1
        return pipelines.PdfConversionResult(
            doc_id=task.doc_id,
            status="unexpected",
            duration_s=0.0,
            input_path=str(task.pdf_path),
            input_hash=task.input_hash,
            output_path=str(task.output_path),
        )

    monkeypatch.setattr(pipelines, "pdf_convert_one", unexpected_convert)
    exit_code = pipelines.pdf_main(args)
    assert exit_code == 0
    assert call_count["value"] == 0

    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    skip_entries = [rec for rec in records if rec.get("status") == "skip"]
    assert {rec["doc_id"] for rec in skip_entries if rec["doc_id"] != "__service__"} == {
        "teamA/report.pdf",
        "teamB/report.pdf",
    }


def test_pdf_pipeline_import_emits_deprecation_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing the legacy shim should emit DeprecationWarning."""

    monkeypatch.delenv("DOCSTOKG_DOC_PARSING_DISABLE_SHIMS", raising=False)
    sys.modules.pop("DocsToKG.DocParsing.pdf_pipeline", None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        module = importlib.import_module("DocsToKG.DocParsing.pdf_pipeline")

    assert any(
        issubclass(w.category, DeprecationWarning)
        and "DocsToKG.DocParsing.pdf_pipeline is deprecated" in str(w.message)
        for w in caught
    ), "Import should emit DeprecationWarning"

    from DocsToKG.DocParsing import pipelines

    shim_args = module.parse_args([])
    baseline_args = pipelines.pdf_parse_args([])
    assert isinstance(shim_args, argparse.Namespace)
    assert vars(shim_args) == vars(baseline_args)
    assert module.main.__module__ == "DocsToKG.DocParsing.pdf_pipeline"
