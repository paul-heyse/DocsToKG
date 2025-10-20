# === NAVMAP v1 ===
# {
#   "module": "tests.ontology_download.test_cli_utils",
#   "purpose": "Regression coverage for CLI table/summary formatting helpers.",
#   "sections": [
#     {"id": "tests", "name": "Test Cases", "anchor": "TST", "kind": "tests"}
#   ]
# }
# === /NAVMAP ===

"""Regression coverage for CLI table/summary formatting helpers.

Verifies that plan/result tables, validation summaries, and inline hyperlinks
render with correct alignment, padding, and detail aggregation. Protects the
text output consumed by operators and documentation snippets."""

from __future__ import annotations

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("pydantic_settings")

from DocsToKG.OntologyDownload.api import format_table, format_validation_summary

# --- Test Cases ---


def test_format_table_basic() -> None:
    """Test basic table formatting with headers and rows."""

    headers = ["ID", "Status"]
    rows = [["hp", "success"], ["efo", "cached"]]

    result = format_table(headers, rows)
    expected = "\n".join(
        [
            "ID  | Status ",
            "----+--------",
            "hp  | success",
            "efo | cached ",
        ]
    )

    assert result == expected


def test_format_table_column_alignment() -> None:
    """Columns should align to the widest entry in each column."""

    headers = ["Short", "Longer Header"]
    rows = [["a", "b"], ["xyz", "abcdefgh"]]

    result = format_table(headers, rows)
    lines = result.split("\n")

    assert len(lines) >= 4
    # Lengths of header and row lines should match for alignment
    assert len(lines[0]) == len(lines[2])


def test_format_table_empty_rows() -> None:
    """Formatting should handle empty row sequences."""

    headers = ["Name", "Value"]
    rows: list[list[str]] = []

    result = format_table(headers, rows)
    expected = "\n".join(
        [
            "Name | Value",
            "-----+------",
        ]
    )

    assert result == expected


def test_format_validation_summary_success() -> None:
    """Validation summary should include successful validator details."""

    results = {"rdflib": {"ok": True, "details": {"triples": 1234, "elapsed": 2.5}}}

    result = format_validation_summary(results)
    expected = "\n".join(
        [
            "validator | status | details                  ",
            "----------+--------+--------------------------",
            "rdflib    | ok     | triples=1234, elapsed=2.5",
        ]
    )

    assert result == expected


def test_format_validation_summary_error() -> None:
    """Validation summary should display error messages."""

    results = {"pronto": {"ok": False, "details": {"error": "timeout after 60s"}}}

    result = format_validation_summary(results)
    expected = "\n".join(
        [
            "validator | status | details          ",
            "----------+--------+------------------",
            "pronto    | error  | timeout after 60s",
        ]
    )

    assert result == expected


def test_format_validation_summary_multiple() -> None:
    """Summary should include each validator as a row."""

    results = {
        "rdflib": {"ok": True, "details": {}},
        "pronto": {"ok": False, "details": {"error": "parse error"}},
        "owlready2": {"ok": True, "details": {"entities": 50}},
    }

    result = format_validation_summary(results)
    lines = result.split("\n")

    assert "rdflib" in result
    assert "pronto" in result
    assert "owlready2" in result
    assert len(lines) == 5


def test_format_validation_summary_non_dict_details() -> None:
    """Gracefully handle details that are not dictionaries."""

    results = {"custom": {"ok": True, "details": "unexpected"}}

    result = format_validation_summary(results)

    assert "custom" in result
    assert "ok" in result
    # Without dict, details column should be empty string
    lines = result.split("\n")
    cells = [cell.strip() for cell in lines[-1].split("|")]
    assert cells[0] == "custom"
    assert cells[1] == "ok"
    assert cells[2] == ""
