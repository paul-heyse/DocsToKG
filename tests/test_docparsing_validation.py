"""Validate golden JSONL fixtures against the DocParsing schemas."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from DocsToKG.DocParsing.schemas import (
    CHUNK_SCHEMA_VERSION,
    VECTOR_SCHEMA_VERSION,
    validate_chunk_row,
    validate_vector_row,
)


FIXTURE_ROOT = Path("tests/data/docparsing/golden")


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@pytest.mark.parametrize(
    "relative",
    ["sample.chunks.jsonl"],
)
def test_chunk_golden_rows_validate(relative: str) -> None:
    """Golden chunk fixtures must conform to the active schema."""

    rows = _load_jsonl(FIXTURE_ROOT / relative)
    assert rows, "expected at least one chunk row in fixture"

    for row in rows:
        validated = validate_chunk_row(row)
        assert validated.schema_version == CHUNK_SCHEMA_VERSION


@pytest.mark.parametrize(
    "relative",
    ["sample.vectors.jsonl"],
)
def test_vector_golden_rows_validate(relative: str) -> None:
    """Golden vector fixtures must conform to the active schema."""

    rows = _load_jsonl(FIXTURE_ROOT / relative)
    assert rows, "expected at least one vector row in fixture"

    for row in rows:
        validated = validate_vector_row(row)
        assert validated.schema_version == VECTOR_SCHEMA_VERSION
