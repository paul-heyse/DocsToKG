# 1. Module: cli_utils

This reference documents the DocsToKG module ``DocsToKG.OntologyDownload.cli_utils``.

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

    print(format_table(["Name", "Status"], [["hp", "ok"]]))

## 1. Functions

### `format_table(headers, rows)`

Format tabular data as an ASCII table.

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

### `format_validation_summary(results)`

Format validator results as a status table.

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

### `_format_row(values)`

Render a single table row with padded column widths.

Args:
values: Ordered cell values corresponding to the table headers.

Returns:
String containing the formatted row.
