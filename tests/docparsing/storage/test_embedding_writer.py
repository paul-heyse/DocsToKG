"""Tests for the unified embedding vector writer helpers."""

from __future__ import annotations

import json
from pathlib import Path

from DocsToKG.DocParsing.storage.embedding_integration import create_unified_vector_writer


def test_unified_vector_writer_jsonl_batches(tmp_path: Path) -> None:
    """JSONL mode should accumulate all batches into the final artefact."""

    output_path = tmp_path / "Vectors" / "doc.vectors.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_batch_1 = [{"uuid": "a", "vector": [1, 2, 3]}]
    rows_batch_2 = [{"uuid": "b", "vector": [4, 5, 6]}]

    with create_unified_vector_writer(output_path, fmt="jsonl") as writer:
        writer.write_rows(rows_batch_1)
        writer.write_rows(rows_batch_2)

    with output_path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]

    assert records == rows_batch_1 + rows_batch_2
