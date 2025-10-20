"""Unit tests covering Parquet vector writer behaviour."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def _runtime():
    return importlib.import_module("DocsToKG.DocParsing.embedding.runtime")


def _sample_row() -> dict:
    return {
        "UUID": "11111111-2222-3333-4444-555555555555",
        "BM25": {
            "terms": ["hybrid", "search"],
            "weights": [0.5, 0.25],
            "avgdl": 32.0,
            "N": 2,
        },
        "SPLADEv3": {
            "tokens": ["hybrid", "retrieval"],
            "weights": [0.4, 0.3],
        },
        "Qwen3-4B": {
            "model_id": "Qwen/Qwen3-Embedding-4B",
            "vector": [0.1, 0.2, 0.3, 0.4],
            "dimension": 4,
        },
        "model_metadata": {"source": "unit-test"},
        "schema_version": "embeddings/1.0.0",
    }


def test_parquet_writer_roundtrip(tmp_path: Path) -> None:
    runtime = _runtime()
    vector_path = tmp_path / "sample.vectors.parquet"
    row = _sample_row()
    with runtime.create_vector_writer(vector_path, "parquet") as writer:
        writer.write_rows([row])

    assert vector_path.exists(), "Parquet writer should produce an output artifact"

    batches = list(runtime._iter_vector_rows(vector_path, "parquet"))
    assert batches and batches[0], "Parquet reader should yield at least one row"
    restored = batches[0][0]
    assert restored["UUID"] == row["UUID"]
    assert restored["BM25"]["terms"] == row["BM25"]["terms"]
    assert restored["SPLADEv3"]["tokens"] == row["SPLADEv3"]["tokens"]
    assert restored["Qwen3-4B"]["vector"] == pytest.approx(row["Qwen3-4B"]["vector"])
    assert restored["model_metadata"] == row["model_metadata"]


def test_parquet_writer_crash_rollback(tmp_path: Path, monkeypatch) -> None:
    runtime = _runtime()
    vector_path = tmp_path / "crash.vectors.parquet"
    monkeypatch.setattr(runtime, "_crash_after_write", 0, raising=False)
    with pytest.raises(RuntimeError):
        with runtime.create_vector_writer(vector_path, "parquet") as writer:
            writer.write_rows([_sample_row()])
    assert not vector_path.exists(), "Crash simulation should not leave partial outputs"
    monkeypatch.setattr(runtime, "_crash_after_write", None)


def test_parquet_writer_missing_dependency(monkeypatch) -> None:
    runtime = _runtime()
    import builtins
    import sys

    original_import = builtins.__import__

    def _fail_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"pyarrow", "pyarrow.parquet"}:
            raise ImportError("pyarrow unavailable for test")
        return original_import(name, globals, locals, fromlist, level)

    # Clear any cached modules so the failing import path is triggered.
    monkeypatch.setattr(runtime, "_PYARROW_MODULE", None)
    monkeypatch.setattr(runtime, "_PYARROW_PARQUET", None)
    for module in ["pyarrow", "pyarrow.parquet"]:
        if module in sys.modules:
            monkeypatch.delitem(sys.modules, module, raising=False)
    monkeypatch.setattr(builtins, "__import__", _fail_import)

    with pytest.raises(runtime.EmbeddingCLIValidationError):
        runtime._ensure_pyarrow_vectors()

    # Restore importlib state for subsequent tests.
    monkeypatch.setattr(builtins, "__import__", original_import)
    runtime._PYARROW_MODULE = None
    runtime._PYARROW_PARQUET = None
    runtime._ensure_pyarrow_vectors()
