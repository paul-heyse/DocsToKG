"""Formatting helpers supporting the ontology downloader CLI.

The CLI surfaces tabular summaries for resolver planning, download batches,
and validator health. These helpers convert rich planner and downloader
objects into aligned ASCII tables so operators can quickly scan fallback
chains, concurrency overrides, and validation diagnostics highlighted in the
refactored ontology download specification.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Sequence, Tuple

if TYPE_CHECKING:  # pragma: no cover - import only when type checking
    from .core import FetchResult, PlannedFetch

__all__ = [
    "format_table",
    "format_plan_rows",
    "format_results_table",
    "format_validation_summary",
]


def format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render an ASCII table with padded columns and header separator.

    Args:
        headers: Ordered column headers rendered on the first row.
        rows: Row data that should be left-aligned within the computed widths.

    Returns:
        Multiline string containing the table body and separator.
    """

    column_widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            column_widths[index] = max(column_widths[index], len(cell))

    def _format_row(values: Sequence[str]) -> str:
        return " | ".join(value.ljust(column_widths[index]) for index, value in enumerate(values))

    separator = "-+-".join("-" * width for width in column_widths)
    lines = [_format_row(headers), separator]
    lines.extend(_format_row(row) for row in rows)
    return "\n".join(lines)


def format_plan_rows(plans: Iterable["PlannedFetch"]) -> List[Tuple[str, str, str, str, str]]:
    """Convert planner output into table rows.

    Args:
        plans: Iterable of planned fetch results capturing resolver metadata.

    Returns:
        List of tuples ``(id, resolver, service, media_type, url)`` ready to
        pass to :func:`format_table`.
    """

    rows: List[Tuple[str, str, str, str, str]] = []
    for plan in plans:
        rows.append(
            (
                plan.spec.id,
                plan.resolver,
                plan.plan.service or "",
                plan.plan.media_type or "",
                plan.plan.url,
            )
        )
    return rows


def format_results_table(results: Iterable["FetchResult"]) -> str:
    """Render download results as a table summarizing outcome and location.

    Args:
        results: Iterable of :class:`~DocsToKG.OntologyDownload.core.FetchResult`
            objects produced by the ``pull`` command.

    Returns:
        ASCII table summarising ontology id, resolver choice, status, checksum,
        and final file path.
    """

    rows = [
        (
            result.spec.id,
            result.spec.resolver,
            result.status,
            result.sha256,
            str(result.local_path),
        )
        for result in results
    ]
    return format_table(("id", "resolver", "status", "sha256", "file"), rows)


def format_validation_summary(results: Dict[str, Dict[str, Any]]) -> str:
    """Summarise validator outcomes in a compact status table.

    Args:
        results: Mapping of validator name to dictionaries containing ``ok`` and
            ``details`` fields returned by the validation pipeline.

    Returns:
        ASCII table listing validator name, status, and condensed detail string.
    """

    formatted: List[Tuple[str, str, str]] = []
    for name, payload in results.items():
        status = "ok" if payload.get("ok") else "error"
        details = payload.get("details", {})
        message = ""
        if isinstance(details, dict):
            if "error" in details:
                message = str(details["error"])
            elif details:
                message = ", ".join(f"{key}={value}" for key, value in details.items())
        formatted.append((name, status, message))
    return format_table(("validator", "status", "details"), formatted)
