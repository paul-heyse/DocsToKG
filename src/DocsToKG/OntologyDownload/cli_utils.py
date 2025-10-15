"""Formatting helpers for the ontology downloader CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Sequence, Tuple

if TYPE_CHECKING:  # pragma: no cover - import only for typing
    from .core import FetchResult, PlannedFetch

__all__ = [
    "format_plan_rows",
    "format_results_table",
    "format_table",
    "format_plan_rows",
    "format_results_table",
    "format_validation_summary",
]


def format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render an ASCII table with aligned columns."""

    column_widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            column_widths[index] = max(column_widths[index], len(cell))

    def _format_row(values: Sequence[str]) -> str:
        return " | ".join(
            value.ljust(column_widths[index]) for index, value in enumerate(values)
        )

    separator = "-+-".join("-" * width for width in column_widths)
    lines = [_format_row(headers), separator]
    lines.extend(_format_row(row) for row in rows)
    return "\n".join(lines)


<<<<<<< HEAD
def format_plan_rows(plans: Iterable["PlannedFetch"]) -> List[Tuple[str, str, str, str, str]]:
    """Convert planned fetches into table rows for CLI presentation."""
=======
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


def format_plan_rows(plans: Sequence["PlannedFetch"]) -> Sequence[Sequence[str]]:
    """Return formatted table rows for planned fetches.

    Args:
        plans: Iterable of planned fetch objects produced by the planner.

    Returns:
        Sequence of tuples containing ``id``, ``resolver``, ``service``,
        ``media_type``, and ``url`` suitable for :func:`format_table`.
    """

    rows = []
def format_plan_rows(plans: Iterable[PlannedFetch]) -> List[Tuple[str, str, str, str, str]]:
    """Return plan metadata rows for human-readable tabular output.

    Args:
        plans: Sequence of planned fetch objects describing resolver outcomes.

    Returns:
        List of tuples containing ontology identifier, resolver, service,
        media type, and URL for table rendering.
    """
>>>>>>> 9b35f42188d4e1aa83a450f8ffa471e6683bfdc8

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


<<<<<<< HEAD
def format_results_table(results: Iterable["FetchResult"]) -> str:
    """Render fetch results (pull command) as an ASCII table."""
=======
def format_results_table(results: Sequence["FetchResult"]) -> str:
    """Render fetch results as a human-readable table.
>>>>>>> 9b35f42188d4e1aa83a450f8ffa471e6683bfdc8

    Args:
        results: Sequence of :class:`~DocsToKG.OntologyDownload.core.FetchResult`
            objects returned by the download workflow.

    Returns:
        ASCII table summarising resolver id, status, checksum, and file path.
    """

    rows = []
    for result in results:
        rows.append(
            (
                result.spec.id,
                result.spec.resolver,
                result.status,
                result.sha256,
                str(result.local_path),
            )
        )
<<<<<<< HEAD
        for result in results
    ]
    return format_table(("id", "resolver", "status", "file", "sha256"), rows)


def format_validation_summary(results: Dict[str, Dict[str, Any]]) -> str:
    """Summarise validator outcomes in table form."""

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
=======
    return format_table(("id", "resolver", "status", "sha256", "file"), rows)
>>>>>>> 9b35f42188d4e1aa83a450f8ffa471e6683bfdc8
