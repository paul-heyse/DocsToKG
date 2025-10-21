# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.analytics.reports",
#   "purpose": "High-level reports for OntologyDownload analytics",
#   "sections": [
#     {"id": "types", "name": "Report Types", "anchor": "TYP", "kind": "models"},
#     {"id": "latest", "name": "Latest Version Reports", "anchor": "LATEST", "kind": "api"},
#     {"id": "growth", "name": "Growth Reports", "anchor": "GROWTH", "kind": "api"},
#     {"id": "validation", "name": "Validation Reports", "anchor": "VAL", "kind": "api"},
#     {"id": "export", "name": "Export Helpers", "anchor": "EXPORT", "kind": "api"}
#   ]
# }
# === /NAVMAP ===

"""High-level reports for OntologyDownload analytics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:  # pragma: no cover
    import polars as pl
except ImportError as exc:  # pragma: no cover
    raise ImportError("polars required for reports") from exc


# ============================================================================
# REPORT TYPES (TYP)
# ============================================================================


@dataclass(frozen=True)
class LatestVersionReport:
    """Latest version report."""

    version_id: str
    total_files: int
    total_bytes: int
    formats: dict[str, int]
    largest_files: list[tuple[str, int]]
    validation_pass_rate: float


@dataclass(frozen=True)
class GrowthReport:
    """Version-to-version growth report."""

    from_version: str
    to_version: str
    files_added: int
    files_removed: int
    bytes_added: int
    bytes_removed: int
    net_growth: int


@dataclass(frozen=True)
class ValidationReport:
    """Validation health report."""

    version_id: str
    total_validations: int
    pass_count: int
    fail_count: int
    pass_rate: float


# ============================================================================
# LATEST VERSION REPORTS (LATEST)
# ============================================================================


def generate_latest_report(
    files_df: pl.DataFrame | pl.LazyFrame,
    validations_df: Optional[pl.DataFrame | pl.LazyFrame] = None,
) -> LatestVersionReport:
    """Generate comprehensive latest version report.

    Args:
        files_df: Extracted files
        validations_df: Optional validations

    Returns:
        LatestVersionReport
    """
    if isinstance(files_df, pl.LazyFrame):
        files = files_df.collect()
    else:
        files = files_df

    # Basic stats
    total_files = files.height
    total_bytes = files.select(pl.sum("size")).item() or 0

    # Format distribution
    formats = dict(
        zip(
            files.group_by("format").agg(pl.len()).select("format").to_series().to_list(),
            files.group_by("format").agg(pl.len()).select("len").to_series().to_list(),
        )
    )

    # Top files
    largest = files.sort("size", descending=True).limit(5)
    largest_files = list(
        zip(
            largest.select("relpath").to_series().to_list(),
            largest.select("size").to_series().to_list(),
        )
    )

    # Validation pass rate
    pass_rate = 1.0
    if validations_df is not None:
        if isinstance(validations_df, pl.LazyFrame):
            validations = validations_df.collect()
        else:
            validations = validations_df

        total_val = validations.height
        pass_count = validations.filter(pl.col("status") == "pass").height
        pass_rate = pass_count / total_val if total_val > 0 else 1.0

    return LatestVersionReport(
        version_id="latest",
        total_files=total_files,
        total_bytes=total_bytes,
        formats=formats,
        largest_files=largest_files,
        validation_pass_rate=pass_rate,
    )


# ============================================================================
# GROWTH REPORTS (GROWTH)
# ============================================================================


def generate_growth_report(
    v1_files: pl.DataFrame | pl.LazyFrame,
    v2_files: pl.DataFrame | pl.LazyFrame,
    v1_id: str,
    v2_id: str,
) -> GrowthReport:
    """Generate version-to-version growth report.

    Args:
        v1_files: Version 1 files
        v2_files: Version 2 files
        v1_id: Version 1 ID
        v2_id: Version 2 ID

    Returns:
        GrowthReport
    """
    if isinstance(v1_files, pl.LazyFrame):
        v1_df = v1_files.collect()
    else:
        v1_df = v1_files

    if isinstance(v2_files, pl.LazyFrame):
        v2_df = v2_files.collect()
    else:
        v2_df = v2_files

    # Get file IDs
    v1_ids = set(v1_df.select("file_id").to_series().to_list())
    v2_ids = set(v2_df.select("file_id").to_series().to_list())

    # Compute deltas
    added = v2_df.filter(~pl.col("file_id").is_in(list(v1_ids)))
    removed = v1_df.filter(~pl.col("file_id").is_in(list(v2_ids)))

    files_added = added.height
    files_removed = removed.height
    bytes_added = added.select(pl.sum("size")).item() or 0
    bytes_removed = removed.select(pl.sum("size")).item() or 0

    return GrowthReport(
        from_version=v1_id,
        to_version=v2_id,
        files_added=files_added,
        files_removed=files_removed,
        bytes_added=bytes_added,
        bytes_removed=bytes_removed,
        net_growth=bytes_added - bytes_removed,
    )


# ============================================================================
# VALIDATION REPORTS (VAL)
# ============================================================================


def generate_validation_report(
    version_id: str,
    validations_df: pl.DataFrame | pl.LazyFrame,
) -> ValidationReport:
    """Generate validation health report.

    Args:
        version_id: Version identifier
        validations_df: Validations

    Returns:
        ValidationReport
    """
    if isinstance(validations_df, pl.LazyFrame):
        validations = validations_df.collect()
    else:
        validations = validations_df

    total_val = validations.height
    pass_count = validations.filter(pl.col("status") == "pass").height
    fail_count = validations.filter(pl.col("status") == "fail").height
    pass_rate = pass_count / total_val if total_val > 0 else 0.0

    return ValidationReport(
        version_id=version_id,
        total_validations=total_val,
        pass_count=pass_count,
        fail_count=fail_count,
        pass_rate=pass_rate,
    )


# ============================================================================
# EXPORT HELPERS (EXPORT)
# ============================================================================


def report_to_dict(report: LatestVersionReport | GrowthReport | ValidationReport) -> dict:
    """Convert report to dictionary.

    Args:
        report: Report object

    Returns:
        Dictionary representation
    """
    return {k: v for k, v in report.__dict__.items()}


def report_to_table(report: LatestVersionReport | GrowthReport | ValidationReport) -> str:
    """Convert report to table string.

    Args:
        report: Report object

    Returns:
        Table string
    """
    d = report_to_dict(report)
    lines = [f"Report: {type(report).__name__}"]
    for k, v in d.items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)
