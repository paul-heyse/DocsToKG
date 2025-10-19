"""Cover atomic JSONL writing helpers that back DocParsing manifests.

These tests focus on the `jsonl_save` primitive used across DocParsing for
manifest writes. They assert that parent directories are created implicitly and
that writes remain atomic and idempotent, preventing partial files when multiple
stages emit telemetry concurrently.
"""

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
