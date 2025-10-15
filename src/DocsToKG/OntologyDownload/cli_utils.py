"""Formatting helpers for the ontology downloader CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Sequence, Tuple

if TYPE_CHECKING:  # pragma: no cover - import only for typing
    from .core import FetchResult, PlannedFetch

__all__ = [
    "format_plan_rows",
    "format_results_table",
    "format_table",
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


def format_plan_rows(plans: Iterable["PlannedFetch"]) -> List[Tuple[str, str, str, str, str]]:
    """Convert planned fetches into table rows for CLI presentation."""

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
    """Render fetch results (pull command) as an ASCII table."""

    rows = [
        (
            result.spec.id,
            result.spec.resolver,
            result.status,
            str(result.local_path),
            result.sha256,
        )
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
