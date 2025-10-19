"""Tests for DocParsing concurrency utilities."""

from __future__ import annotations

from pathlib import Path

from DocsToKG.DocParsing.core.concurrency import acquire_lock


def test_acquire_lock_creates_nested_directory(tmp_path: Path) -> None:
    """Ensure locks can be acquired for targets in nested directories."""

    target_path = tmp_path / "nested" / "deeper" / "artifact.jsonl"
    lock_path = target_path.with_suffix(target_path.suffix + ".lock")

    with acquire_lock(target_path):
        assert lock_path.exists()

    assert not lock_path.exists()
