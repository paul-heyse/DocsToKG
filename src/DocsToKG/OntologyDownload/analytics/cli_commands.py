# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.analytics.cli_commands",
#   "purpose": "CLI commands for analytics and report generation",
#   "sections": [
#     {"id": "types", "name": "CLI Result Types", "anchor": "TYP", "kind": "models"},
#     {"id": "latest", "name": "Latest Report Command", "anchor": "LATEST", "kind": "api"},
#     {"id": "growth", "name": "Growth Report Command", "anchor": "GROWTH", "kind": "api"},
#     {"id": "validation", "name": "Validation Report Command", "anchor": "VAL", "kind": "api"},
#     {"id": "export", "name": "Export Formatters", "anchor": "EXPORT", "kind": "api"}
#   ]
# }
# === /NAVMAP ===

"""CLI commands for analytics and report generation."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Optional

try:  # pragma: no cover
    import polars as pl
except ImportError as exc:  # pragma: no cover
    raise ImportError("polars required for CLI commands") from exc

from DocsToKG.OntologyDownload.analytics.reports import (
    GrowthReport,
    LatestVersionReport,
    ValidationReport,
    generate_growth_report,
    generate_latest_report,
    generate_validation_report,
)

logger = logging.getLogger(__name__)


# ============================================================================
# CLI RESULT TYPES (TYP)
# ============================================================================


class CLIResult:
    """Base class for CLI command results."""

    def to_json(self) -> str:
        """Convert to JSON string."""
        raise NotImplementedError

    def to_table(self) -> str:
        """Convert to table string."""
        raise NotImplementedError


# ============================================================================
# LATEST REPORT COMMAND (LATEST)
# ============================================================================


def cmd_report_latest(
    files_df: pl.DataFrame | pl.LazyFrame,
    validations_df: Optional[pl.DataFrame | pl.LazyFrame] = None,
    output_format: str = "table",
) -> str:
    """Generate latest version report command.

    Args:
        files_df: Extracted files data
        validations_df: Optional validations data
        output_format: 'table', 'json', or 'csv'

    Returns:
        Formatted report string
    """
    report = generate_latest_report(files_df, validations_df)

    if output_format == "json":
        return _format_report_json(report)
    elif output_format == "csv":
        return _format_report_csv(report)
    else:  # table
        return _format_report_table(report)


# ============================================================================
# GROWTH REPORT COMMAND (GROWTH)
# ============================================================================


def cmd_report_growth(
    v1_files: pl.DataFrame | pl.LazyFrame,
    v2_files: pl.DataFrame | pl.LazyFrame,
    v1_id: str,
    v2_id: str,
    output_format: str = "table",
) -> str:
    """Generate growth report command.

    Args:
        v1_files: Version 1 files
        v2_files: Version 2 files
        v1_id: Version 1 identifier
        v2_id: Version 2 identifier
        output_format: 'table', 'json', or 'csv'

    Returns:
        Formatted report string
    """
    report = generate_growth_report(v1_files, v2_files, v1_id, v2_id)

    if output_format == "json":
        return _format_report_json(report)
    elif output_format == "csv":
        return _format_report_csv(report)
    else:  # table
        return _format_report_table(report)


# ============================================================================
# VALIDATION REPORT COMMAND (VAL)
# ============================================================================


def cmd_report_validation(
    version_id: str,
    validations_df: pl.DataFrame | pl.LazyFrame,
    output_format: str = "table",
) -> str:
    """Generate validation report command.

    Args:
        version_id: Version identifier
        validations_df: Validations data
        output_format: 'table', 'json', or 'csv'

    Returns:
        Formatted report string
    """
    report = generate_validation_report(version_id, validations_df)

    if output_format == "json":
        return _format_report_json(report)
    elif output_format == "csv":
        return _format_report_csv(report)
    else:  # table
        return _format_report_table(report)


# ============================================================================
# EXPORT FORMATTERS (EXPORT)
# ============================================================================


def _format_report_json(
    report: LatestVersionReport | GrowthReport | ValidationReport,
) -> str:
    """Format report as JSON.

    Args:
        report: Report object

    Returns:
        JSON string
    """
    d = asdict(report)
    return json.dumps(d, indent=2, default=str)


def _format_report_csv(
    report: LatestVersionReport | GrowthReport | ValidationReport,
) -> str:
    """Format report as CSV.

    Args:
        report: Report object

    Returns:
        CSV string
    """
    d = asdict(report)
    headers = ",".join(d.keys())
    values = ",".join(str(v) for v in d.values())
    return f"{headers}\n{values}"


def _format_report_table(
    report: LatestVersionReport | GrowthReport | ValidationReport,
) -> str:
    """Format report as ASCII table.

    Args:
        report: Report object

    Returns:
        Table string
    """
    d = asdict(report)
    lines = [f"{'Report':<20} {type(report).__name__}"]
    lines.append("─" * 60)

    for key, value in d.items():
        formatted_key = key.replace("_", " ").title()
        lines.append(f"{formatted_key:<30} {value}")

    return "\n".join(lines)


def format_latest_report(
    report: LatestVersionReport,
    output_format: str = "table",
) -> str:
    """Format latest version report.

    Args:
        report: LatestVersionReport
        output_format: 'table', 'json', or 'csv'

    Returns:
        Formatted string
    """
    if output_format == "json":
        return _format_report_json(report)
    elif output_format == "csv":
        return _format_report_csv(report)
    else:
        return _format_report_table(report)


def format_growth_report(
    report: GrowthReport,
    output_format: str = "table",
) -> str:
    """Format growth report.

    Args:
        report: GrowthReport
        output_format: 'table', 'json', or 'csv'

    Returns:
        Formatted string
    """
    if output_format == "json":
        return _format_report_json(report)
    elif output_format == "csv":
        return _format_report_csv(report)
    else:
        return _format_report_table(report)


def format_validation_report(
    report: ValidationReport,
    output_format: str = "table",
) -> str:
    """Format validation report.

    Args:
        report: ValidationReport
        output_format: 'table', 'json', or 'csv'

    Returns:
        Formatted string
    """
    if output_format == "json":
        return _format_report_json(report)
    elif output_format == "csv":
        return _format_report_csv(report)
    else:
        return _format_report_table(report)
