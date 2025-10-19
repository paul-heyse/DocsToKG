"""Guard against performance regressions in the chunking runtime.

This suite focuses on behaviours that materially affect throughput: avoiding
unnecessary hashing or manifest lookups, respecting lazy iteration over DocTags,
and ensuring resume semantics do not trigger redundant file system operations.
By keeping these invariants locked down we protect chunking jobs from silent
slowdowns introduced by seemingly innocuous refactors.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from tests.conftest import PatchManager
from tests.docparsing.stubs import dependency_stubs


def _load_manifest_entry(path: Path, doc_id: str) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("doc_id") == doc_id:
                return record
    raise AssertionError(f"Missing manifest entry for {doc_id}")


def test_chunk_runtime_avoids_eager_hash(patcher: PatchManager, tmp_path: Path) -> None:
    """Chunking should not hash inputs ahead of processing when resume is disabled."""

    dependency_stubs()
    module = importlib.import_module("DocsToKG.DocParsing.chunking.runtime")
    module = importlib.reload(module)

    data_root = tmp_path / "Data"
    in_dir = data_root / "DocTagsFiles"
    out_dir = data_root / "ChunkedDocTagFiles"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    doc_stem = "sample"
    manifest_doc_id = f"{doc_stem}.doctags"
    doctags_path = in_dir / manifest_doc_id
    doctags_path.write_text("Paragraph one\n\nParagraph two", encoding="utf-8")

    call_count = 0
    original_hash = module.compute_content_hash

    def _counting_hash(path: Path) -> str:
        nonlocal call_count
        call_count += 1
        return original_hash(path)

    patcher.setattr(module, "compute_content_hash", _counting_hash)

    exit_code = module._main_inner(
        [
            "--data-root",
            str(data_root),
            "--in-dir",
            str(in_dir),
            "--out-dir",
            str(out_dir),
            "--workers",
            "1",
        ]
    )

    assert exit_code == 0
    assert call_count == 0, "chunk stage should not hash inputs when resume is disabled"

    manifest_path = module.resolve_manifest_path("chunks", data_root)
    assert manifest_path.exists(), "chunk manifest should be written"
    entry = _load_manifest_entry(manifest_path, manifest_doc_id)

    expected_hash = original_hash(doctags_path)
    assert entry.get("input_hash") == expected_hash


def test_chunk_runtime_records_manifest_failure(patcher: PatchManager, tmp_path: Path) -> None:
    """Chunk failures should emit a manifest row marked with ``status=\"failure\"``."""

    dependency_stubs()
    module = importlib.import_module("DocsToKG.DocParsing.chunking.runtime")
    module = importlib.reload(module)

    data_root = tmp_path / "Data"
    in_dir = data_root / "DocTagsFiles"
    out_dir = data_root / "ChunkedDocTagFiles"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    doc_stem = "broken"
    manifest_doc_id = f"{doc_stem}.doctags"
    doctags_path = in_dir / manifest_doc_id
    doctags_path.write_text("Paragraph one", encoding="utf-8")

    def _failing_read(path: Path) -> str:
        raise RuntimeError("forced chunk failure")

    patcher.setattr(module, "read_utf8", _failing_read)

    with pytest.raises(RuntimeError, match="forced chunk failure"):
        module._main_inner(
            [
                "--data-root",
                str(data_root),
                "--in-dir",
                str(in_dir),
                "--out-dir",
                str(out_dir),
                "--workers",
                "1",
            ]
        )

    manifest_path = module.resolve_manifest_path("chunks", data_root)
    assert manifest_path.exists(), "chunk manifest should capture failure rows"

    with manifest_path.open("r", encoding="utf-8") as handle:
        entries = [json.loads(line) for line in handle if line.strip()]

    failure_entries = [entry for entry in entries if entry.get("status") == "failure"]
    assert len(failure_entries) == 1, "exactly one failure entry should be recorded"

    failure_entry = failure_entries[0]
    assert failure_entry.get("doc_id") == manifest_doc_id
    assert failure_entry.get("schema_version") == module.CHUNK_SCHEMA_VERSION
    assert failure_entry.get("parse_engine") == "docling-html"
    assert failure_entry.get("input_path", "").endswith(manifest_doc_id)
    assert failure_entry.get("output_path", "").endswith(".chunks.jsonl")
    assert failure_entry.get("hash_alg") == module.resolve_hash_algorithm()
