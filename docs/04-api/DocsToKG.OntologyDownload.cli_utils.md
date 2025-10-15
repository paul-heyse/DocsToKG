# 1. Module: cli_utils

This reference documents the DocsToKG module ``DocsToKG.OntologyDownload.cli_utils``.

Formatting helpers supporting the ontology downloader CLI.

The CLI surfaces tabular summaries for resolver planning, download batches,
and validator health. These helpers convert rich planner and downloader
objects into aligned ASCII tables so operators can quickly scan fallback
chains, concurrency overrides, and validation diagnostics highlighted in the
refactored ontology download specification.

## 1. Functions

### `format_table(headers, rows)`

Render an ASCII table with padded columns and header separator.

Args:
headers: Ordered column headers rendered on the first row.
rows: Row data that should be left-aligned within the computed widths.

Returns:
Multiline string containing the table body and separator.

### `format_plan_rows(plans)`

Convert planner output into table rows.

Args:
plans: Iterable of planned fetch results capturing resolver metadata.

Returns:
List of tuples ``(id, resolver, service, media_type, url)`` ready to
pass to :func:`format_table`.

### `format_results_table(results)`

Render download results as a table summarizing outcome and location.

Args:
results: Iterable of :class:`~DocsToKG.OntologyDownload.core.FetchResult`
objects produced by the ``pull`` command.

Returns:
ASCII table summarising ontology id, resolver choice, status, checksum,
and final file path.

### `format_validation_summary(results)`

Summarise validator outcomes in a compact status table.

Args:
results: Mapping of validator name to dictionaries containing ``ok`` and
``details`` fields returned by the validation pipeline.

Returns:
ASCII table listing validator name, status, and condensed detail string.

### `_format_row(values)`

*No documentation available.*
