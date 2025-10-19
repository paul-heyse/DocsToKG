"""Tests for embedding runtime validation helpers."""

from __future__ import annotations

from typing import Any

import pytest

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
