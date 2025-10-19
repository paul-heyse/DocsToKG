"""Regression coverage for manifest iteration order with sparse metadata.

`iter_manifest_entries` merges manifest files across stages and must remain
deterministic even when timestamps are absent. These tests craft JSONL fixtures
with mixed timestamp coverage to confirm the iterator respects file order and
maintains stable merges for resume tooling.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

from DocsToKG.DocParsing.io import iter_manifest_entries


def _write_manifest(stage_dir: Path, stage: str, entries: list[dict]) -> None:
    path = stage_dir / f"docparse.{stage}.manifest.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry))
            handle.write("\n")


def test_iter_manifest_entries_preserves_file_order_without_timestamps(tmp_path) -> None:
    """Ensure manifests with missing timestamps fall back to stable file order."""

    manifest_dir = tmp_path / "Manifests"
    manifest_dir.mkdir()

    chunk_entries = [
        {"timestamp": "2025-01-01T00:00:00", "doc_id": "chunk-0", "status": "success"},
        {"doc_id": "chunk-no-ts", "status": "success"},
        {"timestamp": "2025-01-01T00:02:00", "doc_id": "chunk-2", "status": "success"},
    ]
    embedding_entries = [
        {"timestamp": "2025-01-01T00:00:30", "doc_id": "embed-0", "status": "success"},
        {"doc_id": "embed-no-ts", "status": "failure"},
        {"timestamp": "2025-01-01T00:03:00", "doc_id": "embed-2", "status": "success"},
    ]

    _write_manifest(manifest_dir, "chunks", chunk_entries)
    _write_manifest(manifest_dir, "embeddings", embedding_entries)

    merged = list(iter_manifest_entries(["chunks", "embeddings"], tmp_path))
    assert [entry["doc_id"] for entry in merged] == [
        "chunk-0",
        "embed-0",
        "chunk-no-ts",
        "embed-no-ts",
        "chunk-2",
        "embed-2",
    ]

    tail = deque(iter_manifest_entries(["chunks", "embeddings"], tmp_path), maxlen=3)
    assert [entry["doc_id"] for entry in tail] == ["embed-no-ts", "chunk-2", "embed-2"]
