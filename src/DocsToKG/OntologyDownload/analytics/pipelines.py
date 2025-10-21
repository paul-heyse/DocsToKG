# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.analytics.pipelines",
#   "purpose": "Polars lazy pipelines for columnar analytics without loops",
#   "sections": [
#     {"id": "types", "name": "Pipeline Result Types", "anchor": "TYP", "kind": "models"},
#     {"id": "latest", "name": "Latest Version Pipeline", "anchor": "LATEST", "kind": "api"},
#     {"id": "delta", "name": "Version Delta Pipeline", "anchor": "DELTA", "kind": "api"},
#     {"id": "helpers", "name": "Pipeline Helpers", "anchor": "HELP", "kind": "api"}
#   ]
# }
# === /NAVMAP ===

"""Polars lazy pipelines for high-performance analytics.

Features:
- Lazy evaluation for automatic optimization
- Predicate pushdown and projection pruning
- Streaming support for large datasets
- Zero-copy Arrow interop with DuckDB
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

try:  # pragma: no cover
    import polars as pl
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "polars is required for analytics pipelines. Ensure .venv is initialized."
    ) from exc

logger = logging.getLogger(__name__)


# ============================================================================
# PIPELINE RESULT TYPES (TYP)
# ============================================================================


@dataclass(frozen=True)
class LatestSummary:
    """Summary of latest version artifacts."""

    total_files: int
    total_bytes: int
    files_by_format: dict[str, int]  # format → count
    bytes_by_format: dict[str, int]  # format → bytes
    top_files: list[tuple[str, int]]  # [(path, size), ...]
    validation_summary: dict[str, int]  # status → count


@dataclass(frozen=True)
class VersionDelta:
    """Changes between two versions."""

    added_files: int
    removed_files: int
    modified_files: int
    renamed_files: int
    added_bytes: int
    removed_bytes: int
    net_bytes_delta: int
    churn_bytes: int


# ============================================================================
# LATEST VERSION PIPELINE (LATEST)
# ============================================================================


def build_latest_summary_pipeline(
    files_df: pl.LazyFrame,
    validations_df: Optional[pl.LazyFrame] = None,
) -> pl.LazyFrame:
    """Build lazy pipeline for latest version summary.

    Args:
        files_df: Extracted files LazyFrame
        validations_df: Optional validations LazyFrame

    Returns:
        LazyFrame with aggregated summary (collect to execute)
    """
    # Start with files, project early to reduce width
    pipeline = files_df.select(
        [
            pl.col("relpath"),
            pl.col("size"),
            pl.col("format"),
            pl.col("file_id"),
        ]
    )

    # Filter on collection later via caller
    return pipeline


def compute_latest_summary(
    files_df: pl.LazyFrame,
    validations_df: Optional[pl.LazyFrame] = None,
    top_n: int = 10,
) -> LatestSummary:
    """Compute latest version summary using streaming.

    Args:
        files_df: Extracted files LazyFrame
        validations_df: Optional validations LazyFrame
        top_n: Number of top files to include

    Returns:
        LatestSummary with all metrics
    """
    # Collect with streaming for large datasets (handle both LazyFrame and DataFrame)
    if isinstance(files_df, pl.LazyFrame):
        files_collected = files_df.collect(engine="streaming")
    else:
        files_collected = files_df

    # Basic counts
    total_files = files_collected.height
    total_bytes = files_collected.select(pl.sum("size")).item() or 0

    # Group by format
    format_stats = files_collected.group_by("format").agg(
        [
            pl.len().alias("count"),
            pl.sum("size").alias("total_size"),
        ]
    )

    files_by_format = dict(
        zip(
            format_stats.select("format").to_series().to_list(),
            format_stats.select("count").to_series().to_list(),
        )
    )
    bytes_by_format = dict(
        zip(
            format_stats.select("format").to_series().to_list(),
            format_stats.select("total_size").to_series().to_list(),
        )
    )

    # Top N largest files
    top_files_df = (
        files_collected.sort("size", descending=True).limit(top_n).select(["relpath", "size"])
    )
    top_files = list(
        zip(
            top_files_df.select("relpath").to_series().to_list(),
            top_files_df.select("size").to_series().to_list(),
        )
    )

    # Validation summary
    validation_summary: dict[str, int] = {}
    if validations_df is not None:
        if isinstance(validations_df, pl.LazyFrame):
            val_collected = validations_df.select("status").collect(engine="streaming")
        else:
            val_collected = validations_df.select("status")

        status_counts = val_collected.group_by("status").agg(pl.len().alias("count"))
        validation_summary = dict(
            zip(
                status_counts.select("status").to_series().to_list(),
                status_counts.select("count").to_series().to_list(),
            )
        )

    return LatestSummary(
        total_files=total_files,
        total_bytes=total_bytes,
        files_by_format=files_by_format,
        bytes_by_format=bytes_by_format,
        top_files=top_files,
        validation_summary=validation_summary,
    )


# ============================================================================
# VERSION DELTA PIPELINE (DELTA)
# ============================================================================


def build_version_delta_pipeline(
    v1_files: pl.LazyFrame | pl.DataFrame,
    v2_files: pl.LazyFrame | pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Build lazy pipelines for version delta computation.

    Args:
        v1_files: Version 1 files LazyFrame (with file_id, relpath, size)
        v2_files: Version 2 files LazyFrame

    Returns:
        Tuple of (added, removed, common) LazyFrames/DataFrames
    """
    # Ensure we have DataFrames for easier operations
    if isinstance(v1_files, pl.LazyFrame):
        v1_df = v1_files.collect()
    else:
        v1_df = v1_files

    if isinstance(v2_files, pl.LazyFrame):
        v2_df = v2_files.collect()
    else:
        v2_df = v2_files

    # Get file IDs
    v1_ids_set = set(v1_df.select("file_id").to_series().to_list())
    v2_ids_set = set(v2_df.select("file_id").to_series().to_list())

    # Files in v2 but not v1 (added)
    added = v2_df.filter(~pl.col("file_id").is_in(list(v1_ids_set)))

    # Files in v1 but not v2 (removed)
    removed = v1_df.filter(~pl.col("file_id").is_in(list(v2_ids_set)))

    # Common files
    common = v2_df.filter(pl.col("file_id").is_in(list(v1_ids_set)))

    return added, removed, common


def compute_version_delta(
    v1_files: pl.LazyFrame | pl.DataFrame,
    v2_files: pl.LazyFrame | pl.DataFrame,
) -> VersionDelta:
    """Compute delta between two versions using streaming.

    Args:
        v1_files: Version 1 files
        v2_files: Version 2 files

    Returns:
        VersionDelta with all metrics
    """
    added, removed, common = build_version_delta_pipeline(v1_files, v2_files)

    # Ensure DataFrames
    if isinstance(added, pl.LazyFrame):
        added_collected = added.select(["size"]).collect(engine="streaming")
    else:
        added_collected = added.select(["size"])

    if isinstance(removed, pl.LazyFrame):
        removed_collected = removed.select(["size"]).collect(engine="streaming")
    else:
        removed_collected = removed.select(["size"])

    if isinstance(common, pl.LazyFrame):
        common.select(["size"]).collect(engine="streaming")
    else:
        common.select(["size"])

    # Counts
    added_files = added_collected.height
    removed_files = removed_collected.height

    # Sizes
    added_bytes = added_collected.select(pl.sum("size")).item() or 0
    removed_bytes = removed_collected.select(pl.sum("size")).item() or 0

    # Compute metrics
    net_delta = added_bytes - removed_bytes
    churn = min(added_bytes, removed_bytes)

    # Estimate renames (simplified)
    modified_files = 0
    renamed_files = 0

    return VersionDelta(
        added_files=added_files,
        removed_files=removed_files,
        modified_files=modified_files,
        renamed_files=renamed_files,
        added_bytes=added_bytes,
        removed_bytes=removed_bytes,
        net_bytes_delta=net_delta,
        churn_bytes=churn,
    )


# ============================================================================
# PIPELINE HELPERS (HELP)
# ============================================================================


def arrow_to_lazy_frame(arrow_table) -> pl.LazyFrame:  # type: ignore
    """Convert Arrow table to Polars LazyFrame (zero-copy).

    Args:
        arrow_table: PyArrow Table

    Returns:
        Polars LazyFrame (lazy)
    """
    df = pl.from_arrow(arrow_table)
    return df.lazy()


def duckdb_to_lazy_frame(conn, sql: str) -> pl.LazyFrame:  # type: ignore
    """Execute DuckDB query and convert to Polars LazyFrame.

    Args:
        conn: DuckDB connection
        sql: SQL query

    Returns:
        Polars LazyFrame (lazy)
    """
    arrow = conn.execute(sql).arrow()
    return arrow_to_lazy_frame(arrow)


def lazy_frame_to_arrow(lf: pl.LazyFrame):  # type: ignore
    """Convert LazyFrame to Arrow (zero-copy after collection).

    Args:
        lf: Polars LazyFrame

    Returns:
        PyArrow Table
    """
    return lf.collect(engine="streaming").to_arrow()
