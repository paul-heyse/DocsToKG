"""Formatting helpers for turning ontology plans and results into rich tables.

The CLI exposes compact tabular renderings for plan, download, and validation
phases.  This module defines the header schemas used in those displays and
provides formatter functions that turn structured objects into terminal-ready
rows, while masking sensitive values and harmonising column order.  Keeping
the logic here allows both the CLI and documentation tooling to reuse the same
presentation semantics.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .io import format_bytes
from .planning import FetchResult, PlannedFetch

PlanRow = Tuple[str, str, str, str, str, str, str, str]

PLAN_TABLE_HEADERS: Tuple[str, ...] = (
    "id",
    "resolver",
    "service",
    "media_type",
    "version",
    "license",
    "expected_checksum",
    "url",
)

RESULT_TABLE_HEADERS: Tuple[str, ...] = (
    "id",
    "resolver",
    "status",
    "content_type",
    "bytes",
    "etag",
    "cache",
    "sha256",
    "expected_checksum",
    "file",
)

VALIDATION_TABLE_HEADERS: Tuple[str, ...] = ("validator", "status", "details")

__all__ = [
    "PlanRow",
    "PLAN_TABLE_HEADERS",
    "RESULT_TABLE_HEADERS",
    "VALIDATION_TABLE_HEADERS",
    "format_table",
    "format_plan_rows",
    "format_results_table",
    "format_validation_summary",
]


def format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render a padded ASCII table."""

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


def format_plan_rows(plans: Iterable[PlannedFetch]) -> List[PlanRow]:
    """Convert planner output into rows for table rendering."""

    rows: List[PlanRow] = []
    for plan in plans:
        expected_checksum = ""
        checksum = plan.metadata.get("expected_checksum")
        if isinstance(checksum, dict):
            algorithm = checksum.get("algorithm")
            value = checksum.get("value")
            if isinstance(algorithm, str) and isinstance(value, str):
                expected_checksum = (
                    f"{algorithm}:{value[:12]}…" if len(value) > 12 else f"{algorithm}:{value}"
                )

        rows.append(
            (
                plan.spec.id,
                plan.resolver,
                plan.plan.service or "",
                plan.plan.media_type or "",
                plan.plan.version or "",
                plan.plan.license or "",
                expected_checksum,
                plan.plan.url,
            )
        )
    return rows


def format_results_table(results: Iterable[FetchResult]) -> str:
    """Render download results as a table summarizing status and file paths."""

    rows: List[Tuple[str, str, str, str, str, str, str, str, str, str]] = []
    for result in results:
        checksum_text = ""
        checksum_obj = getattr(result, "expected_checksum", None)
        if checksum_obj is not None:
            checksum_text = checksum_obj.to_known_hash()
            if ":" in checksum_text:
                algo, value = checksum_text.split(":", 1)
                checksum_text = f"{algo}:{value[:12]}…" if len(value) > 12 else checksum_text
        else:
            extras = result.spec.extras if isinstance(result.spec.extras, dict) else {}
            checksum = extras.get("expected_checksum") if isinstance(extras, dict) else None
            if isinstance(checksum, dict):
                algo = checksum.get("algorithm")
                value = checksum.get("value")
                if isinstance(algo, str) and isinstance(value, str):
                    checksum_text = (
                        f"{algo}:{value[:12]}…" if len(value) > 12 else f"{algo}:{value}"
                    )
        content_type = getattr(result, "content_type", "") or ""
        content_length = getattr(result, "content_length", None)
        bytes_text = (
            format_bytes(int(content_length)) if isinstance(content_length, int) else ""
        )
        etag = getattr(result, "etag", "") or ""
        cache_info = getattr(result, "cache_status", None)
        cache_label = ""
        if isinstance(cache_info, Mapping):
            if cache_info.get("from_cache"):
                cache_label = "hit"
                if cache_info.get("revalidated"):
                    cache_label += " (revalidated)"
            elif cache_info:
                cache_label = "miss"
        rows.append(
            (
                result.spec.id,
                result.spec.resolver,
                result.status,
                content_type,
                bytes_text,
                etag,
                cache_label,
                result.sha256,
                checksum_text,
                str(result.local_path),
            )
        )
    return format_table(RESULT_TABLE_HEADERS, rows)


def format_validation_summary(results: Dict[str, Dict[str, Any]]) -> str:
    """Summarise validator outcomes in a compact status table."""

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
    return format_table(VALIDATION_TABLE_HEADERS, formatted)
# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.formatters",
#   "purpose": "Turn planning, download, and validation results into reusable table renderings",
#   "sections": [
#     {"id": "schema", "name": "Table Schemas", "anchor": "SCH", "kind": "constants"},
#     {"id": "plan", "name": "Plan Row Formatting", "anchor": "PLN", "kind": "helpers"},
#     {"id": "results", "name": "Download Result Formatting", "anchor": "RES", "kind": "helpers"},
#     {"id": "validation", "name": "Validation Summary Formatting", "anchor": "VAL", "kind": "helpers"}
#   ]
# }
# === /NAVMAP ===
