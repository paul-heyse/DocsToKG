"""Regression tests for DocParsing I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path

from DocsToKG.DocParsing.io import jsonl_save


def test_jsonl_save_creates_parent_directories(tmp_path: Path) -> None:
    """``jsonl_save`` should create missing parents before writing output."""

    target = tmp_path / "nested" / "dir" / "rows.jsonl"
    rows = [{"id": 1, "value": "ok"}]

    jsonl_save(target, rows)

    assert target.exists()
    with target.open("r", encoding="utf-8") as handle:
        contents = [json.loads(line) for line in handle]

    assert contents == rows
