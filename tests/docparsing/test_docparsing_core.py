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
import builtins
import contextlib
import importlib
import json
import logging
import os
import socket
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, ClassVar, Dict
from unittest import mock

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


try:  # pragma: no cover - optional dependency
    import requests  # type: ignore # noqa: F401
except Exception:  # pragma: no cover - fallback stub
    sys.modules.setdefault(
        "requests", _stub_module("requests", get=lambda *args, **kwargs: _RequestsResponse())
    )

try:  # pragma: no cover - optional dependency
    import tqdm  # type: ignore # noqa: F401
except Exception:  # pragma: no cover - fallback stub
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
        self._iterable = args[0] if args else ()

    def __enter__(self):  # pragma: no cover - simple stub
        return self

    def __exit__(self, exc_type, exc, tb):  # pragma: no cover - simple stub
        return False

    def update(self, *_args, **_kwargs) -> None:  # pragma: no cover - simple stub
        pass

    def __iter__(self):  # pragma: no cover - simple stub
        return iter(self._iterable)


import DocsToKG.DocParsing.chunking.runtime as doc_chunking  # noqa: E402
import DocsToKG.DocParsing.config as doc_config  # noqa: E402
import DocsToKG.DocParsing.env as doc_env  # noqa: E402
import DocsToKG.DocParsing.io as doc_io  # noqa: E402
import DocsToKG.DocParsing.logging as doc_logging  # noqa: E402
import DocsToKG.DocParsing.telemetry as doc_telemetry  # noqa: E402
from DocsToKG.DocParsing import core, formats  # noqa: E402
from DocsToKG.DocParsing.formats import (  # noqa: E402
    CHUNK_SCHEMA_VERSION,
    VECTOR_SCHEMA_VERSION,
    validate_chunk_row,
    validate_vector_row,
)

# --- _common utility tests ---


@pytest.fixture(autouse=True)
def _reset_env(tmp_path: Path) -> None:
    """Force utilities to operate inside a temporary data root."""

    data_root = tmp_path / "Data"
    data_root.mkdir(parents=True, exist_ok=True)
    previous = os.environ.get("DOCSTOKG_DATA_ROOT")
    os.environ["DOCSTOKG_DATA_ROOT"] = str(data_root)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("DOCSTOKG_DATA_ROOT", None)
        else:
            os.environ["DOCSTOKG_DATA_ROOT"] = previous


def test_detect_data_root_env(tmp_path: Path) -> None:
    data = tmp_path / "Data"
    data.mkdir(exist_ok=True)
    root = doc_env.detect_data_root()
    assert root == data.resolve()


def test_detect_data_root_scan(tmp_path: Path) -> None:
    os.environ.pop("DOCSTOKG_DATA_ROOT", None)
    project = tmp_path / "workspace" / "DocsToKG" / "src"
    target = tmp_path / "workspace" / "DocsToKG" / "Data"
    project.mkdir(parents=True)
    (target / "DocTagsFiles").mkdir(parents=True)
    original_cwd = Path.cwd()
    os.chdir(project)
    try:
        resolved = doc_env.detect_data_root()
    finally:
        os.chdir(original_cwd)
    assert resolved == target.resolve()


def test_detect_data_root_fallback(tmp_path: Path) -> None:
    os.environ.pop("DOCSTOKG_DATA_ROOT", None)
    start = tmp_path / "no_data_here"
    start.mkdir()
    resolved = doc_env.detect_data_root(start)
    assert resolved == (start / "Data").resolve()


def test_data_directories_created(tmp_path: Path) -> None:
    expected = {
        doc_env.data_doctags(): "DocTagsFiles",
        doc_env.data_chunks(): "ChunkedDocTagFiles",
        doc_env.data_vectors(): "Embeddings",
        doc_env.data_manifests(): "Manifests",
        doc_env.data_pdfs(): "PDFs",
        doc_env.data_html(): "HTML",
    }
    for path, folder in expected.items():
        assert folder in str(path)
        assert path.exists()


def test_get_logger_idempotent() -> None:
    logger1 = doc_logging.get_logger("docparse-test")
    logger2 = doc_logging.get_logger("docparse-test")
    assert logger1 is logger2


def test_find_free_port_basic() -> None:
    port = core.find_free_port(start=9000, span=4)
    assert isinstance(port, int)


def test_find_free_port_fallback() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    busy_port = sock.getsockname()[1]
    try:
        result = core.find_free_port(start=busy_port, span=1)
        assert result != busy_port
    finally:
        sock.close()


def test_atomic_write_success(tmp_path: Path) -> None:
    target = tmp_path / "file.json"
    with doc_io.atomic_write(target) as handle:
        handle.write("data")
    assert target.read_text(encoding="utf-8") == "data"


def test_atomic_write_failure(tmp_path: Path) -> None:
    target = tmp_path / "file.json"
    with pytest.raises(RuntimeError):
        with doc_io.atomic_write(target) as handle:
            handle.write("data")
            raise RuntimeError("boom")
    assert not target.exists()
    assert not target.with_suffix(".json.tmp").exists()


def test_iter_doctags(tmp_path: Path) -> None:
    doctags_dir = tmp_path / "DocTagsFiles"
    doctags_dir.mkdir()

    files = [doctags_dir / "a.doctags", doctags_dir / "b.doctag"]
    for file in files:
        file.write_text("content", encoding="utf-8")

    alias = doctags_dir / "alias.doctags"
    alias.symlink_to(files[0])

    results = list(doc_io.iter_doctags(doctags_dir))
    assert results == files
    assert alias not in results


def test_iter_doctags_symlink_outside_root(tmp_path: Path) -> None:
    doctags_dir = tmp_path / "DocTagsFiles"
    doctags_dir.mkdir()

    external_dir = tmp_path / "external"
    external_dir.mkdir()
    external_file = external_dir / "external.doctags"
    external_file.write_text("content", encoding="utf-8")

    symlink_path = doctags_dir / "linked" / "external.doctags"
    symlink_path.parent.mkdir(parents=True, exist_ok=True)
    symlink_path.symlink_to(external_file)

    results = list(doc_io.iter_doctags(doctags_dir))
    assert results == [symlink_path]
    assert results[0].resolve() == external_file.resolve()


def test_iter_chunks(tmp_path: Path) -> None:
    chunks_dir = doc_env.data_chunks()
    good = chunks_dir / "doc.chunks.jsonl"
    nested = chunks_dir / "teamA" / "doc.chunks.jsonl"
    other = chunks_dir / "doc.jsonl"
    good.write_text("{}\n", encoding="utf-8")
    nested.parent.mkdir(parents=True, exist_ok=True)
    nested.write_text("{}\n", encoding="utf-8")
    other.write_text("{}\n", encoding="utf-8")
    results = list(core.iter_chunks(chunks_dir))
    logical_paths = sorted(item.logical_path for item in results)
    resolved_paths = sorted(item.resolved_path for item in results)
    assert logical_paths == [Path("doc.chunks.jsonl"), Path("teamA/doc.chunks.jsonl")]
    assert resolved_paths == sorted({good.resolve(), nested.resolve()})


def test_jsonl_load_and_save(tmp_path: Path) -> None:
    target = tmp_path / "example.jsonl"
    doc_io.jsonl_save(target, [{"a": 1}, {"b": 2}])
    rows = list(doc_io.iter_jsonl(target))
    assert rows == [{"a": 1}, {"b": 2}]
    with pytest.warns(DeprecationWarning):
        assert doc_io.jsonl_load(target) == rows


def test_jsonl_load_skip_invalid(tmp_path: Path) -> None:
    target = tmp_path / "bad.jsonl"
    target.write_text("{\ninvalid\n", encoding="utf-8")
    rows = list(doc_io.iter_jsonl(target, skip_invalid=True, max_errors=1))
    assert rows == []
    with pytest.warns(DeprecationWarning):
        assert doc_io.jsonl_load(target, skip_invalid=True, max_errors=1) == []


def test_jsonl_save_validation_error(tmp_path: Path) -> None:
    target = tmp_path / "example.jsonl"

    def validate(row: dict) -> None:
        raise ValueError("bad row")

    with pytest.raises(ValueError, match="Validation failed"):
        doc_io.jsonl_save(target, [{"a": 1}], validate=validate)


def test_jsonl_append_iter_appends_atomically(tmp_path: Path) -> None:
    target = tmp_path / "logs" / "events.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    doc_io.jsonl_save(target, [{"id": 1}])

    appended = [{"id": 2}, {"id": 3}]
    count = doc_io.jsonl_append_iter(target, appended, atomic=True)
    assert count == 2

    records = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in records] == [1, 2, 3]


def test_jsonl_append_iter_non_atomic(tmp_path: Path) -> None:
    target = tmp_path / "data.jsonl"
    doc_io.jsonl_save(target, [{"value": "initial"}])
    count = doc_io.jsonl_append_iter(target, [{"value": "later"}], atomic=False)
    assert count == 1
    records = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert [row["value"] for row in records] == ["initial", "later"]


def test_extract_refs_and_pages_handles_invalid_metadata() -> None:
    chunk = SimpleNamespace(
        meta=SimpleNamespace(
            document_id="doc-1",
            doc_items=[
                SimpleNamespace(
                    self_ref="ref-1",
                    prov=[
                        SimpleNamespace(page_no=2),
                        SimpleNamespace(page_no="page-x"),
                    ],
                )
            ],
        )
    )

    class _Capture(logging.Handler):
        def __init__(self) -> None:
            super().__init__()
            self.records: list[logging.LogRecord] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.records.append(record)

    handler = _Capture()
    logger = doc_chunking._LOGGER.logger
    logger.addHandler(handler)
    try:
        refs, pages = doc_chunking.extract_refs_and_pages(chunk)
    finally:
        logger.removeHandler(handler)

    assert refs == ["ref-1"]
    assert pages == [2]
    error_codes = {
        getattr(record, "extra_fields", {}).get("error_code") for record in handler.records
    }
    assert "CHUNK_PAGE_INVALID" in error_codes


def test_extract_refs_and_pages_surfaces_doc_items_failures() -> None:
    class BrokenMeta:
        document_id = "doc-err"

        @property
        def doc_items(self):
            raise RuntimeError("boom")

    chunk = SimpleNamespace(meta=BrokenMeta())

    with pytest.raises(RuntimeError, match="boom"):
        doc_chunking.extract_refs_and_pages(chunk)


def test_summarize_image_metadata_logs_non_iterable_sources() -> None:
    non_iterable_annotations = object()
    predicted_holder = SimpleNamespace(predicted_classes=object())
    doc_items = [
        SimpleNamespace(
            doc_item=SimpleNamespace(
                _docstokg_flags={
                    "has_image_captions": True,
                    "has_image_classification": False,
                    "image_confidence": 0.7,
                },
                _docstokg_meta=None,
                annotations=non_iterable_annotations,
            ),
            annotations=None,
        ),
        SimpleNamespace(
            doc_item=SimpleNamespace(
                _docstokg_flags=None,
                _docstokg_meta=None,
                annotations=[predicted_holder],
            ),
            annotations=None,
        ),
    ]
    chunk = SimpleNamespace(meta=SimpleNamespace(document_id="doc-annot", doc_items=doc_items))

    class _Capture(logging.Handler):
        def __init__(self) -> None:
            super().__init__()
            self.records: list[logging.LogRecord] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.records.append(record)

    handler = _Capture()
    logger = doc_chunking._LOGGER.logger
    logger.addHandler(handler)
    try:
        has_caption, has_classification, num_images, confidence, metadata = (
            doc_chunking.summarize_image_metadata(chunk, "Figure caption: sample text")
        )
    finally:
        logger.removeHandler(handler)

    assert has_caption is True
    assert has_classification is False
    assert num_images == 1
    assert confidence == 0.7
    assert metadata == []
    error_codes = {
        getattr(record, "extra_fields", {}).get("error_code") for record in handler.records
    }
    assert "CHUNK_ANNOTATIONS_NON_ITERABLE" in error_codes
    assert "CHUNK_PREDICTED_CLASSES_NON_ITERABLE" in error_codes


def test_telemetry_sink_uses_lock_and_jsonl_append(tmp_path: Path) -> None:
    attempts_path = tmp_path / "telemetry" / "attempts.jsonl"
    manifest_path = tmp_path / "telemetry" / "manifest.jsonl"

    lock_calls: list[Path] = []

    @contextlib.contextmanager
    def _fake_lock(path: Path):
        lock_calls.append(path)
        yield True

    with mock.patch.object(doc_telemetry, "_acquire_lock_for", _fake_lock):
        sink = doc_telemetry.TelemetrySink(attempts_path, manifest_path)

        sink.write_attempt(
            doc_telemetry.Attempt(
                run_id="run-1",
                file_id="file-1",
                stage="chunking",
                status="success",
                reason=None,
                started_at=0.0,
                finished_at=1.0,
                bytes=256,
                metadata={"extra": "attempt"},
            )
        )
        sink.write_manifest_entry(
            doc_telemetry.ManifestEntry(
                run_id="run-1",
                file_id="file-1",
                stage="chunking",
                output_path="output.jsonl",
                tokens=42,
                schema_version="1.0",
                duration_s=0.5,
                metadata={"extra": "manifest"},
            )
        )

    attempts_records = [
        json.loads(line) for line in attempts_path.read_text(encoding="utf-8").splitlines()
    ]
    manifest_records = [
        json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()
    ]

    assert attempts_records == [
        {
            "run_id": "run-1",
            "file_id": "file-1",
            "stage": "chunking",
            "status": "success",
            "reason": None,
            "started_at": 0.0,
            "finished_at": 1.0,
            "bytes": 256,
            "extra": "attempt",
        }
    ]
    assert manifest_records == [
        {
            "run_id": "run-1",
            "file_id": "file-1",
            "stage": "chunking",
            "output_path": "output.jsonl",
            "tokens": 42,
            "schema_version": "1.0",
            "duration_s": 0.5,
            "extra": "manifest",
            "doc_id": "file-1",
        }
    ]
    assert lock_calls == [attempts_path, manifest_path]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, False),
        ("true", True),
        ("FALSE", False),
        ("  yEs  ", True),
        ("off", False),
        ("n", False),
        ("", False),
        (1, True),
        (0, False),
        (1.0, True),
        (0.0, False),
        (None, False),
    ],
)
def test_coerce_bool_accepts_supported_literals(value: object, expected: bool) -> None:
    assert doc_config._coerce_bool(value) is expected


@pytest.mark.parametrize(
    "value",
    [
        "treu",
        "enable",
        "2",
        2,
        0.5,
        object(),
    ],
)
def test_coerce_bool_rejects_unknown_literals(value: object) -> None:
    with pytest.raises(ValueError):
        doc_config._coerce_bool(value)


@dataclass
class _DummyCfg(doc_config.StageConfigBase):
    value: int = 256

    FIELD_PARSERS: ClassVar[Dict[str, Any]] = {
        "value": doc_config.StageConfigBase._coerce_int,
    }


def test_apply_args_skips_parser_defaults() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--value", type=int, default=256)
    namespace = doc_config.parse_args_with_overrides(parser, [])

    cfg = _DummyCfg()
    cfg.value = 384
    cfg.overrides.add("value")

    cfg.apply_args(namespace)
    assert cfg.value == 384
    assert cfg.is_overridden("value")
    assert namespace._cli_explicit_overrides == set()


def test_apply_args_honours_explicit_cli_overrides() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--value", type=int, default=256)
    namespace = doc_config.parse_args_with_overrides(parser, ["--value", "256"])

    cfg = _DummyCfg()
    cfg.value = 384
    cfg.overrides.add("value")

    cfg.apply_args(namespace)
    assert cfg.value == 256
    assert cfg.is_overridden("value")
    assert "value" in namespace._cli_explicit_overrides


def test_apply_args_handles_manual_namespaces() -> None:
    cfg = _DummyCfg()
    cfg.apply_args(argparse.Namespace(value=512))
    assert cfg.value == 512
    assert cfg.is_overridden("value")


def test_iter_jsonl_batches(tmp_path: Path) -> None:
    file_one = tmp_path / "one.jsonl"
    file_two = tmp_path / "two.jsonl"
    doc_io.jsonl_save(file_one, [{"id": 1}, {"id": 2}])
    doc_io.jsonl_save(file_two, [{"id": 3}])

    batches = list(doc_io.iter_jsonl_batches([file_one, file_two], batch_size=2))
    assert [len(batch) for batch in batches] == [2, 1]
    assert [record["id"] for batch in batches for record in batch] == [1, 2, 3]

    with pytest.raises(ValueError):
        list(doc_io.iter_jsonl_batches([file_one], batch_size=0))


def test_make_hasher_invalid_env_fallback(caplog: pytest.LogCaptureFixture) -> None:
    os.environ["DOCSTOKG_HASH_ALG"] = "sha-1"
    with caplog.at_level("WARNING"):
        hasher = doc_io.make_hasher()
    assert hasher.name == "sha256"
    assert "Unknown hash algorithm 'sha-1'" in caplog.text
    assert "falling back to sha256" in caplog.text.lower()


def test_make_hasher_prefers_explicit_algorithm(caplog: pytest.LogCaptureFixture) -> None:
    os.environ["DOCSTOKG_HASH_ALG"] = "sha-1"
    with caplog.at_level("WARNING"):
        hasher = doc_io.make_hasher(name="sha1")
    assert hasher.name == "sha1"
    # Warning emitted for env override but no fallback message because explicit value is valid
    assert "Unknown hash algorithm 'sha-1'" in caplog.text


def test_resolve_hash_algorithm_defaults() -> None:
    os.environ.pop("DOCSTOKG_HASH_ALG", None)
    assert doc_io.resolve_hash_algorithm() == "sha1"
    os.environ["DOCSTOKG_HASH_ALG"] = "sha256"
    assert doc_io.resolve_hash_algorithm() == "sha256"


def test_batcher() -> None:
    assert list(core.Batcher([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    assert list(core.Batcher([], 3)) == []


def test_batcher_length_policy_handles_small_lengths() -> None:
    items = ["one", "zero", "two", "three"]
    lengths = [1, 0, 2, 3]

    batches = list(core.Batcher(items, 2, policy="length", lengths=lengths))

    assert batches == [["zero", "one"], ["two", "three"]]


def test_manifest_append(tmp_path: Path) -> None:
    manifest = tmp_path / "docparse.chunks.manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with mock.patch.object(
        doc_io,
        "resolve_manifest_path",
        lambda stage, root=None: manifest,
    ):
        doc_io.manifest_append("chunks", "doc-1", "success", duration_s=1.23)
    content = manifest.read_text(encoding="utf-8").strip()
    record = json.loads(content)
    assert record["stage"] == "chunks"
    assert record["doc_id"] == "doc-1"
    assert record["status"] == "success"


def test_manifest_append_respects_atomic_flag(tmp_path: Path) -> None:
    calls = []

    def fake_append(path, rows, *, atomic):
        calls.append({"path": Path(path), "rows": list(rows), "atomic": atomic})
        return len(calls[-1]["rows"])

    with (
        mock.patch.object(doc_io, "jsonl_append_iter", fake_append),
        mock.patch.object(
            doc_io,
            "resolve_manifest_path",
            lambda stage, root=None: tmp_path / f"{stage}.jsonl",
        ),
    ):
        doc_io.manifest_append("chunks", "doc-a", "success")
        doc_io.manifest_append("chunks", "doc-b", "success", atomic=True)

    assert len(calls) == 2
    assert calls[0]["atomic"] is False
    assert calls[0]["path"] == tmp_path / "chunks.jsonl"
    assert calls[0]["rows"][0]["doc_id"] == "doc-a"
    assert calls[1]["atomic"] is True


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
    parsed = formats.validate_chunk_row(make_chunk_row())
    assert parsed.doc_id == "doc"
    assert parsed.schema_version == formats.CHUNK_SCHEMA_VERSION


def test_chunk_row_missing_field() -> None:
    with pytest.raises(ValueError):
        formats.validate_chunk_row({"doc_id": "doc"})


def test_chunk_row_invalid_num_tokens() -> None:
    with pytest.raises(ValueError):
        formats.validate_chunk_row(make_chunk_row(num_tokens=0))
    with pytest.raises(ValueError):
        formats.validate_chunk_row(make_chunk_row(num_tokens=200_000))


def test_chunk_row_invalid_page_numbers() -> None:
    with pytest.raises(ValueError):
        formats.validate_chunk_row(make_chunk_row(page_nos=[0, 1]))


def test_provenance_invalid_engine() -> None:
    with pytest.raises(ValueError):
        formats.ProvenanceMetadata(parse_engine="bad", docling_version="1")


def test_vector_row_valid() -> None:
    parsed = formats.validate_vector_row(make_vector_row())
    assert parsed.UUID == "uuid"
    assert parsed.schema_version == formats.VECTOR_SCHEMA_VERSION


def test_vector_row_mismatched_terms() -> None:
    data = make_vector_row(BM25={"terms": ["a"], "weights": [1.0, 2.0], "avgdl": 1.0, "N": 1})
    with pytest.raises(ValueError):
        formats.validate_vector_row(data)


def test_vector_row_negative_weights() -> None:
    data = make_vector_row(SPLADEv3={"tokens": ["a"], "weights": [-0.1]})
    with pytest.raises(ValueError):
        formats.validate_vector_row(data)


def test_dense_vector_dimension_mismatch() -> None:
    data = make_vector_row(Qwen3_4B={"model_id": "model", "vector": [0.1], "dimension": 2})
    with pytest.raises(ValueError):
        formats.validate_vector_row(data)


def test_get_docling_version() -> None:
    module = SimpleNamespace(__version__="1.2.3")
    with mock.patch.dict(sys.modules, {"docling": module}, clear=False):
        assert formats.get_docling_version() == "1.2.3"

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "docling":
            raise ImportError("docling missing")
        return original_import(name, *args, **kwargs)

    with mock.patch("builtins.__import__", new=fake_import):
        assert formats.get_docling_version() == "unknown"


def test_validate_schema_version() -> None:
    assert (
        formats.validate_schema_version(
            "docparse/1.1.0",
            formats.SchemaKind.CHUNK,
            compatible_versions=formats.COMPATIBLE_CHUNK_VERSIONS,
        )
        == "docparse/1.1.0"
    )
    with pytest.raises(ValueError):
        formats.validate_schema_version(
            "other",
            formats.SchemaKind.CHUNK,
            compatible_versions=formats.COMPATIBLE_CHUNK_VERSIONS,
        )
    with pytest.raises(ValueError):
        formats.validate_schema_version(
            None,
            formats.SchemaKind.CHUNK,
            compatible_versions=formats.COMPATIBLE_CHUNK_VERSIONS,
        )


def test_chunk_row_invalid_schema_version() -> None:
    with pytest.raises(ValueError):
        formats.validate_chunk_row(make_chunk_row(schema_version="docparse/0.9.0"))


def test_vector_row_invalid_schema_version() -> None:
    with pytest.raises(ValueError):
        formats.validate_vector_row(make_vector_row(schema_version="embeddings/0.9.0"))


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


def test_set_spawn_or_warn_warns_on_incompatible_method() -> None:
    import multiprocessing as mp

    def raise_runtime_error(*_args, **_kwargs):
        raise RuntimeError("already set")

    class DummyLogger:
        def __init__(self) -> None:
            self.warnings: list[str] = []

        def warning(self, message: str, *args, **kwargs) -> None:  # pragma: no cover
            self.warnings.append(message)

        def debug(self, *_args, **_kwargs) -> None:  # pragma: no cover - simple stub
            pass

    dummy = DummyLogger()
    with (
        mock.patch.object(mp, "set_start_method", raise_runtime_error),
        mock.patch.object(mp, "get_start_method", lambda allow_none=True: "fork"),
    ):
        core.set_spawn_or_warn(dummy)
    assert dummy.warnings
    assert "spawn" in dummy.warnings[0]


def test_compute_relative_doc_id_handles_subdirectories(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    nested = root / "teamA" / "report.doctags"
    nested.parent.mkdir(parents=True, exist_ok=True)
    nested.write_text("{}", encoding="utf-8")

    assert core.compute_relative_doc_id(nested, root) == "teamA/report.doctags"


def test_derive_doc_id_and_vectors_path(tmp_path: Path) -> None:
    from DocsToKG.DocParsing.core import derive_doc_id_and_vectors_path

    chunks_root = tmp_path / "chunks"
    chunk_file = chunks_root / "teamA" / "report.chunks.jsonl"
    chunk_file.parent.mkdir(parents=True, exist_ok=True)
    chunk_file.write_text("{}\n", encoding="utf-8")
    vectors_root = tmp_path / "vectors"

    doc_id, out_path = derive_doc_id_and_vectors_path(chunk_file, chunks_root, vectors_root)

    assert doc_id == "teamA/report.doctags"
    assert out_path == vectors_root / "teamA" / "report.vectors.jsonl"


def test_chunk_row_fields_unique() -> None:
    fields = formats.ChunkRow.model_fields
    for name in ("has_image_captions", "has_image_classification", "num_images"):
        assert name in fields


def test_detect_mode_pdf_only(tmp_path: Path) -> None:
    from DocsToKG.DocParsing.core import detect_mode

    input_dir = tmp_path / "pdf-only"
    nested = input_dir / "nested"
    nested.mkdir(parents=True, exist_ok=True)
    (nested / "doc.PDF").write_text("content", encoding="utf-8")

    assert detect_mode(input_dir) == "pdf"


def test_detect_mode_html_only(tmp_path: Path) -> None:
    from DocsToKG.DocParsing.core import detect_mode

    input_dir = tmp_path / "html-only"
    nested = input_dir / "nested"
    nested.mkdir(parents=True, exist_ok=True)
    (nested / "page.HTM").write_text("<html/>", encoding="utf-8")

    assert detect_mode(input_dir) == "html"


def test_detect_mode_raises_when_both_present(tmp_path: Path) -> None:
    from DocsToKG.DocParsing.core import detect_mode

    input_dir = tmp_path / "mixed"
    nested = input_dir / "nested"
    nested.mkdir(parents=True, exist_ok=True)
    (nested / "doc.pdf").write_text("content", encoding="utf-8")
    (nested / "page.html").write_text("<html/>", encoding="utf-8")

    with pytest.raises(ValueError, match="Cannot auto-detect mode"):
        detect_mode(input_dir)


def test_ensure_uuid_deterministic_generation() -> None:
    from DocsToKG.DocParsing import embedding as embedding
    from DocsToKG.DocParsing.core import compute_chunk_uuid

    rows = [
        {
            "doc_id": "teamA/report.doctags",
            "source_chunk_idxs": [0, 2],
            "text": "Chunk content for reproducible UUID.",
            "start_offset": 128,
        }
    ]

    assert embedding.ensure_uuid(rows) is True
    generated = rows[0]["uuid"]
    assert uuid.UUID(generated).version == 5
    assert generated == compute_chunk_uuid(
        "teamA/report.doctags", 128, "Chunk content for reproducible UUID."
    )

    replica_rows = [
        {
            "doc_id": "teamA/report.doctags",
            "source_chunk_idxs": [0, 2],
            "text": "Chunk content for reproducible UUID.",
            "start_offset": 128,
        }
    ]
    embedding.ensure_uuid(replica_rows)

    assert replica_rows[0]["uuid"] == generated
    assert embedding.ensure_uuid(rows) is False


def test_qwen_embed_caches_llm(tmp_path: Path) -> None:
    import DocsToKG.DocParsing.embedding.runtime as embedding

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

    embedding._QWEN_LLM_CACHE.clear()
    with (
        mock.patch.object(
            embedding, "_get_vllm_components", lambda: (DummyLLM, DummyPoolingParams)
        ),
        mock.patch.object(embedding, "ensure_qwen_dependencies", lambda: None),
    ):

        cfg = embedding.QwenCfg(model_dir=tmp_path, batch_size=2)
        first = embedding.qwen_embed(cfg, ["a", "b"])
        second = embedding.qwen_embed(cfg, ["c"])

    assert DummyLLM.instances == 1
    assert len(first) == 2
    assert len(second) == 1


def _pipeline_module():
    import DocsToKG.DocParsing as docparsing

    return getattr(docparsing, "pipelines", docparsing.doctags)


def test_pdf_model_path_resolution_precedence(tmp_path: Path) -> None:
    pipelines = _pipeline_module()

    os.environ.pop("DOCLING_PDF_MODEL", None)
    os.environ.pop("DOCSTOKG_MODEL_ROOT", None)
    os.environ.pop("HF_HOME", None)

    env_model = tmp_path / "custom"
    os.environ["DOCLING_PDF_MODEL"] = str(env_model)
    assert pipelines.resolve_pdf_model_path(None) == str(env_model.resolve())

    os.environ.pop("DOCLING_PDF_MODEL", None)
    model_root = tmp_path / "root"
    os.environ["DOCSTOKG_MODEL_ROOT"] = str(model_root)
    expected = (model_root / "granite-docling-258M").resolve()
    assert pipelines.resolve_pdf_model_path(None) == str(expected)

    os.environ.pop("DOCSTOKG_MODEL_ROOT", None)
    hf_home = tmp_path / "hf"
    os.environ["HF_HOME"] = str(hf_home)
    expected = (hf_home / "granite-docling-258M").resolve()
    assert pipelines.resolve_pdf_model_path(None) == str(expected)


def test_pdf_model_path_cli_normalization(tmp_path: Path) -> None:
    pipelines = _pipeline_module()

    os.environ.pop("DOCLING_PDF_MODEL", None)
    os.environ.pop("DOCSTOKG_MODEL_ROOT", None)
    os.environ.pop("HF_HOME", None)

    local_dir = tmp_path / "models" / "granite"
    local_dir.mkdir(parents=True)
    assert pipelines.resolve_pdf_model_path(str(local_dir)) == str(local_dir.resolve())

    tilde_path = str(Path("~") / "granite-model")
    assert pipelines.resolve_pdf_model_path(tilde_path) == str(
        Path(tilde_path).expanduser().resolve()
    )

    repo_id = "ibm-granite/granite-docling-258M"
    assert pipelines.resolve_pdf_model_path(repo_id) == repo_id


def test_pdf_pipeline_mirrors_output_paths(tmp_path: Path) -> None:
    pipelines = _pipeline_module()

    input_dir = tmp_path / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "report.pdf").write_text("content", encoding="utf-8")

    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    data_root = tmp_path / "data-root"
    data_root.mkdir()
    os.environ["DOCSTOKG_DATA_ROOT"] = str(data_root)

    def fake_main(args):
        (output_dir / "report.doctags").write_text("{}", encoding="utf-8")
        manifest = data_root / "Manifests" / "docparse.doctags-pdf.manifest.jsonl"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            json.dumps({"doc_id": "report.pdf", "status": "success"}) + "\n",
            encoding="utf-8",
        )
        return 0

    args = argparse.Namespace(
        data_root=data_root,
        input=input_dir,
        output=output_dir,
        workers=1,
        resume=False,
        force=False,
        served_model_names=None,
        model="placeholder",
        gpu_memory_utilization=0.1,
        vlm_prompt="Convert this page to docling.",
        vlm_stop=["</doctag>", "<|end_of_text|>"],
    )

    with mock.patch("DocsToKG.DocParsing.doctags.pdf_main", fake_main):
        exit_code = pipelines.pdf_main(args)

    assert exit_code == 0
    assert (output_dir / "report.doctags").exists()


def test_pdf_pipeline_import_emits_deprecation_warning() -> None:
    os.environ.pop("DOCSTOKG_DOC_PARSING_DISABLE_SHIMS", None)
    sys.modules.pop("DocsToKG.DocParsing.pdf_pipeline", None)

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("DocsToKG.DocParsing.pdf_pipeline")
