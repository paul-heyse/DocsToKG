"""
CLI Formatting Helpers

This module contains lightweight formatting utilities used by the ontology
downloader CLI. The helpers focus on producing human-friendly tables and
summaries while remaining dependency-free so they can run in constrained
environments such as CI workflows or air-gapped deployments.

Key Features:
- ASCII table rendering for deterministic console output
- Validation summary formatting that mirrors structured JSON payloads
- Status-aware formatting that highlights validator outcomes consistently
- Utilities designed for reuse across multiple CLI subcommands

Usage:
    from DocsToKG.OntologyDownload.cli_utils import format_table

    print(format_table([\"Name\", \"Status\"], [[\"hp\", \"ok\"]]))
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

__all__ = [
    "format_table",
    "format_validation_summary",
]


def format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Format tabular data as an ASCII table.

    Args:
        headers: Ordered column header strings.
        rows: Iterable of rows where each row is a sequence of cell strings.

    Returns:
        A string containing the formatted table with aligned columns.

    Examples:
        >>> print(format_table(["Name", "Status"], [["hp", "success"], ["efo", "cached"]]))
        Name | Status
        -----+--------
        hp   | success
        efo  | cached
    """

    column_widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            column_widths[index] = max(column_widths[index], len(cell))

    def _format_row(values: Sequence[str]) -> str:
        """Render a single table row with padded column widths.

        Args:
            values: Ordered cell values corresponding to the table headers.

        Returns:
            String containing the formatted row.
        """
        return " | ".join(value.ljust(column_widths[index]) for index, value in enumerate(values))

    separator = "-+-".join("-" * width for width in column_widths)
    lines = [_format_row(headers), separator]
    lines.extend(_format_row(row) for row in rows)
    return "\n".join(lines)


def format_validation_summary(results: Dict[str, Dict[str, Any]]) -> str:
    """Format validator results as a status table.

    Args:
        results: Mapping of validator name to dictionaries containing ``ok`` and
            ``details`` keys describing the validator outcome.

    Returns:
        A formatted table with validator names, status, and detail summaries.

    Examples:
        >>> summary = {"rdflib": {"ok": True, "details": {"triples": 100}}}
        >>> print(format_validation_summary(summary))
        validator | status | details
        ----------+--------+---------
        rdflib    | ok     | triples=100
    """

    formatted_rows: list[Tuple[str, str, str]] = []
    for name, payload in results.items():
        status = "ok" if payload.get("ok") else "error"
        details = payload.get("details", {})
        message = ""
        if isinstance(details, dict):
            if "error" in details:
                message = str(details["error"])
            elif details:
                message = ", ".join(f"{key}={value}" for key, value in details.items())
        formatted_rows.append((name, status, message))

    return format_table(("validator", "status", "details"), formatted_rows)
