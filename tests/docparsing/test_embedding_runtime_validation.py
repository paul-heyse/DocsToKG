"""Tests for embedding runtime validation helpers."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest

from DocsToKG.DocParsing.core import cli as core_cli
from DocsToKG.DocParsing.embedding import runtime


def test_validate_vectors_missing_logs_capped(tmp_path, monkeypatch):
    """Missing vectors should log capped metadata and raise a helpful error."""

    chunks_dir = tmp_path / "chunks"
    vectors_dir = tmp_path / "vectors"
    chunks_dir.mkdir()
    vectors_dir.mkdir()

    for idx in range(7):
        chunk_path = chunks_dir / f"doc-{idx}.chunks.jsonl"
        chunk_path.write_text("{}\n")

    log_calls: list[tuple[str, str, dict[str, Any]]] = []

    def fake_log_event(logger, level, message, **fields):
        log_calls.append((level, message, fields))

    monkeypatch.setattr(runtime, "log_event", fake_log_event)

    with pytest.raises(FileNotFoundError) as exc:
        runtime._validate_vectors_for_chunks(chunks_dir, vectors_dir, logger=object())

    message = str(exc.value)
    assert "doc-0.doctags" in message
    assert "doc-4.doctags" in message
    assert "..." in message

    missing_call_fields = None
    for _, msg, fields in log_calls:
        if msg == "Missing vector files for chunk documents":
            missing_call_fields = fields
            break

    assert missing_call_fields is not None
    assert missing_call_fields["missing_count"] == 7
    assert missing_call_fields["missing_sample_size"] == 5
    assert missing_call_fields["missing_sample_truncated"] is True
    assert missing_call_fields["missing_doc_ids_sample"] == [
        "doc-0.doctags",
        "doc-1.doctags",
        "doc-2.doctags",
        "doc-3.doctags",
        "doc-4.doctags",
    ]
    assert missing_call_fields["missing_paths_sample"] == [
        str(vectors_dir / "doc-0.vectors.jsonl"),
        str(vectors_dir / "doc-1.vectors.jsonl"),
        str(vectors_dir / "doc-2.vectors.jsonl"),
        str(vectors_dir / "doc-3.vectors.jsonl"),
        str(vectors_dir / "doc-4.vectors.jsonl"),
    ]
    assert "missing" not in missing_call_fields


def test_docparse_embed_missing_chunks_warns_and_exits(tmp_path, monkeypatch):
    """`docparse embed` exits cleanly with a warning when chunks are absent."""

    import DocsToKG.DocParsing.embedding as embedding_module

    importlib.reload(runtime)
    importlib.reload(embedding_module)

    data_root = tmp_path / "Data"
    model_root = tmp_path / "models"
    qwen_dir = model_root / "qwen"
    splade_dir = model_root / "splade"
    data_root.mkdir(parents=True)
    qwen_dir.mkdir(parents=True)
    splade_dir.mkdir(parents=True)

    missing_chunks = data_root / "ChunkedDocTagFiles" / "missing"
    vectors_dir = data_root / "Embeddings"

    monkeypatch.setattr(runtime, "ensure_model_environment", lambda: (model_root, model_root))

    def _expand_path(candidate):
        if candidate is None:
            return model_root
        return Path(candidate).resolve()

    monkeypatch.setattr(runtime, "expand_path", _expand_path)
    monkeypatch.setattr(runtime, "_resolve_qwen_dir", lambda _root: qwen_dir)
    monkeypatch.setattr(runtime, "_resolve_splade_dir", lambda _root: splade_dir)
    monkeypatch.setattr(runtime, "ensure_splade_environment", lambda **_: {"device": "cpu"})
    monkeypatch.setattr(runtime, "ensure_qwen_environment", lambda **_: {"device": "cpu", "dtype": "bfloat16"})
    monkeypatch.setattr(runtime, "_ensure_splade_dependencies", lambda: None)
    monkeypatch.setattr(runtime, "_ensure_qwen_dependencies", lambda: None)

    def _prepare_data_root(override, detected):
        if override is not None:
            return Path(override).resolve()
        return Path(detected).resolve()

    monkeypatch.setattr(runtime, "prepare_data_root", _prepare_data_root)
    monkeypatch.setattr(runtime, "detect_data_root", lambda: data_root)
    monkeypatch.setattr(runtime, "data_chunks", lambda root, ensure=False: root / "ChunkedDocTagFiles")
    monkeypatch.setattr(runtime, "data_vectors", lambda root, ensure=False: root / "Embeddings")
    monkeypatch.setattr(runtime, "process_pass_a", lambda files, logger: runtime.BM25Stats(N=0, avgdl=0.0, df={}))

    events: list[tuple[str, str, dict[str, Any]]] = []

    def fake_log_event(logger, level, message, **fields):
        events.append((level, message, fields))

    monkeypatch.setattr(runtime, "log_event", fake_log_event)

    exit_code = core_cli._run_stage(
        core_cli.embed,
        [
            "--data-root",
            str(data_root),
            "--chunks-dir",
            str(missing_chunks),
            "--out-dir",
            str(vectors_dir),
        ],
    )

    assert exit_code == 0
    warning_fields = [fields for _, message, fields in events if message == "No chunk files found"]
    assert warning_fields, "Expected 'No chunk files found' warning"
    assert warning_fields[-1]["chunks_dir"] == str(missing_chunks.resolve())
