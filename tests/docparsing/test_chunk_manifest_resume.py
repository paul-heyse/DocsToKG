"""Regression tests for chunk manifest stage resume semantics."""

from __future__ import annotations

import json
from DocsToKG.DocParsing.io import load_manifest_index
from tests.docparsing.stubs import dependency_stubs


def test_chunk_resume_uses_chunks_manifest_stage(monkeypatch, tmp_path):
    """Second resume run skips once manifest entries use the ``chunks`` stage."""

    dependency_stubs()

    # Import after installing dependency stubs so optional modules resolve.
    from DocsToKG.DocParsing.chunking import runtime as chunk_runtime

    data_root = tmp_path / "Data"
    doctags_dir = data_root / "DocTagsFiles"
    chunks_dir = data_root / "ChunkedDocTagFiles"
    doctags_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    doc_path = doctags_dir / "example.doctags"
    doc_path.write_text("{}\n", encoding="utf-8")

    monkeypatch.setenv("DOCSTOKG_DATA_ROOT", str(data_root))

    processed: list[str] = []

    def fake_initializer(_cfg):  # pragma: no cover - simple stub
        return None

    def fake_process(task: chunk_runtime.ChunkTask) -> chunk_runtime.ChunkResult:
        processed.append(task.doc_id)
        output_path = task.output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"doc_id": task.doc_id, "text": "stub"}
        output_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        return chunk_runtime.ChunkResult(
            doc_id=task.doc_id,
            doc_stem=task.doc_stem,
            status="success",
            duration_s=0.01,
            input_path=task.doc_path,
            output_path=output_path,
            input_hash=task.input_hash,
            chunk_count=1,
            total_tokens=1,
            parse_engine=task.parse_engine,
            sanitizer_profile=None,
            anchors_injected=False,
            error=None,
        )

    monkeypatch.setattr(chunk_runtime, "_chunk_worker_initializer", fake_initializer)
    monkeypatch.setattr(chunk_runtime, "_process_chunk_task", fake_process)
    monkeypatch.setattr(chunk_runtime, "ensure_model_environment", lambda: None)

    cli_args = [
        "--in-dir",
        str(doctags_dir),
        "--out-dir",
        str(chunks_dir),
        "--data-root",
        str(data_root),
        "--resume",
    ]

    first_exit = chunk_runtime.main(cli_args)
    assert first_exit == 0
    assert processed == ["example.doctags"]

    manifest_index = load_manifest_index("chunks", data_root)
    assert manifest_index["example.doctags"]["stage"] == "chunks"
    assert manifest_index["example.doctags"]["status"] == "success"

    second_exit = chunk_runtime.main(cli_args)
    assert second_exit == 0
    assert processed == ["example.doctags"]

    manifest_after = load_manifest_index("chunks", data_root)
    assert manifest_after["example.doctags"]["input_hash"] == manifest_index["example.doctags"]["input_hash"]
    assert manifest_after["example.doctags"]["status"] == "skip"
