# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.analytics.__init__",
#   "purpose": "Analytics package for OntologyDownload.",
#   "sections": []
# }
# === /NAVMAP ===

"""Analytics package for OntologyDownload.

Core modules:
- pipelines: Lazy Polars pipelines (latest summary, version delta)
- reports: High-level report generation (latest, growth, validation)
- cli_commands: CLI command wrappers and output formatters
"""

from __future__ import annotations

from .cli_commands import (
    cmd_report_growth,
    cmd_report_latest,
    cmd_report_validation,
    format_growth_report,
    format_latest_report,
    format_validation_report,
)
from .pipelines import (
    LatestSummary,
    VersionDelta,
    arrow_to_lazy_frame,
    build_latest_summary_pipeline,
    build_version_delta_pipeline,
    compute_latest_summary,
    compute_version_delta,
    duckdb_to_lazy_frame,
    lazy_frame_to_arrow,
)
from .reports import (
    GrowthReport,
    LatestVersionReport,
    ValidationReport,
    generate_growth_report,
    generate_latest_report,
    generate_validation_report,
    report_to_dict,
    report_to_table,
)

__all__ = [
    "LatestSummary",
    "VersionDelta",
    "build_latest_summary_pipeline",
    "compute_latest_summary",
    "build_version_delta_pipeline",
    "compute_version_delta",
    "arrow_to_lazy_frame",
    "duckdb_to_lazy_frame",
    "lazy_frame_to_arrow",
    "LatestVersionReport",
    "GrowthReport",
    "ValidationReport",
    "generate_latest_report",
    "generate_growth_report",
    "generate_validation_report",
    "report_to_dict",
    "report_to_table",
    "cmd_report_latest",
    "cmd_report_growth",
    "cmd_report_validation",
    "format_latest_report",
    "format_growth_report",
    "format_validation_report",
]
