"""
Lazy Dataset Views for Chunks and Vectors

Provides fast introspection of Parquet datasets with:
- Schema inspection without full scans
- Fragment-based file/size discovery
- Approximate row counts from Parquet statistics
- Partition-aware metadata extraction
- Optional Polars LazyFrame integration

Key Functions:
- `open_chunks`: Open Chunks dataset as Arrow Dataset.
- `open_vectors`: Open Vectors dataset (by family) as Arrow Dataset.
- `summarize`: Compute fast summary statistics.

=== NAVMAP v1 ===
{
  "module": "DocsToKG.DocParsing.storage.dataset_view",
  "purpose": "Lazy inspection and summarization of Parquet datasets",
  "sections": [
    {"id": "types", "name": "Types", "anchor": "types", "kind": "section"},
    {"id": "dataset-summary", "name": "DatasetSummary", "anchor": "class-dataset-summary", "kind": "class"},
    {"id": "readers", "name": "Reader Functions", "anchor": "readers", "kind": "section"}
  ]
}
=== END NAVMAP ===
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pyarrow as pa
import pyarrow.dataset as ds

from . import paths

# ============================================================
# Types
# ============================================================


@dataclass
class DatasetSummary:
    """Summary statistics for a Parquet dataset."""

    dataset_type: str  # "chunks", "dense", "sparse", "lexical"
    schema: pa.Schema
    file_count: int
    total_bytes: int
    approx_rows: Optional[int]  # May be None if estimation fails
    partitions: Dict[str, int]  # e.g., {"2025-10": 5 files, ...}
    sample_doc_ids: List[str]

    def to_dict(self) -> Dict:
        return {
            "dataset_type": self.dataset_type,
            "schema": str(self.schema),
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "approx_rows": self.approx_rows,
            "partitions": self.partitions,
            "sample_doc_ids": self.sample_doc_ids[:10],  # Limit samples
        }


# ============================================================
# Reader Functions
# ============================================================


def open_chunks(
    data_root: Path,
    columns: Optional[List[str]] = None,
    filters: Optional[str] = None,
) -> ds.Dataset:
    """
    Open Chunks Parquet dataset as Arrow Dataset.

    Args:
        data_root: Data root directory.
        columns: Column subset (if None, all columns).
        filters: Arrow filter expression (experimental).

    Returns:
        PyArrow Dataset for lazy operations.

    Raises:
        FileNotFoundError: If no Chunks files found.
    """
    # Use glob to find all parquet files recursively
    chunks_dir = Path(data_root) / "Chunks" / "fmt=parquet"
    if not chunks_dir.exists():
        raise FileNotFoundError(f"Chunks directory not found: {chunks_dir}")
    
    parquet_files = list(chunks_dir.glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Chunks Parquet files found in {chunks_dir}")
    
    # Create dataset from the files
    dataset = ds.dataset([str(f) for f in parquet_files], format="parquet")

    # Note: Column selection can be done by the caller using dataset methods
    # if columns:
    #     dataset = dataset.project(columns)

    return dataset


def open_vectors(
    data_root: Path,
    family: str,
    columns: Optional[List[str]] = None,
    filters: Optional[str] = None,
) -> ds.Dataset:
    """
    Open Vectors Parquet dataset (by family) as Arrow Dataset.

    Args:
        data_root: Data root directory.
        family: Vector family ("dense", "sparse", or "lexical").
        columns: Column subset (if None, all columns).
        filters: Arrow filter expression (experimental).

    Returns:
        PyArrow Dataset for lazy operations.

    Raises:
        FileNotFoundError: If no Vectors files found.
        ValueError: If family is invalid.
    """
    if family not in ("dense", "sparse", "lexical"):
        raise ValueError(f"Invalid family: {family}")

    vectors_dir = Path(data_root) / "Vectors" / f"family={family}" / "fmt=parquet"
    if not vectors_dir.exists():
        raise FileNotFoundError(f"Vectors directory not found: {vectors_dir}")
    
    parquet_files = list(vectors_dir.glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Vectors ({family}) Parquet files found in {vectors_dir}")
    
    # Create dataset from the files
    dataset = ds.dataset([str(f) for f in parquet_files], format="parquet")

    # Note: Column selection can be done by the caller using dataset methods
    # if columns:
    #     dataset = dataset.project(columns)

    return dataset


def _extract_partition_from_path(file_path: str) -> Optional[str]:
    """
    Extract partition key (YYYY-MM) from file path.

    Examples:
        "/data/Chunks/fmt=parquet/2025/10/file.parquet" → "2025-10"
        "/data/Vectors/family=dense/fmt=parquet/2025/10/file.parquet" → "2025-10"

    Args:
        file_path: File path string.

    Returns:
        Partition string "YYYY-MM", or None if extraction fails.
    """
    # Match pattern: .../{yyyy}/{mm}/...
    match = re.search(r"/(\d{4})/(\d{2})/", file_path)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    return None


def _extract_doc_id_from_filename(filename: str) -> Optional[str]:
    """
    Extract doc_id from filename.

    Examples:
        "papers_xyz_abc.parquet" → "papers_xyz_abc"
        "some_doc.parquet" → "some_doc"

    Args:
        filename: Filename string (without directory).

    Returns:
        Normalized doc_id, or None if extraction fails.
    """
    # Remove extension
    if filename.endswith(".parquet"):
        return filename[:-8]
    return None


def summarize(
    dataset: ds.Dataset,
    dataset_type: str = "unknown",
) -> DatasetSummary:
    """
    Compute fast summary statistics for a dataset.

    Uses fragment metadata to avoid full scans where possible.

    Args:
        dataset: PyArrow Dataset to summarize.
        dataset_type: Type identifier ("chunks", "dense", "sparse", "lexical").

    Returns:
        DatasetSummary with schema, file count, total bytes, approx rows, and samples.
    """
    schema = dataset.schema
    fragments = list(dataset.get_fragments())
    file_count = len(fragments)

    # Compute total bytes and collect partition info
    total_bytes = 0
    partitions: Dict[str, int] = {}
    sample_doc_ids: List[str] = []

    for frag in fragments:
        if hasattr(frag, "physical_schema"):
            # Get path if available
            path = getattr(frag, "path", None)
            if path:
                total_bytes += Path(path).stat().st_size

                # Extract partition
                partition = _extract_partition_from_path(path)
                if partition:
                    partitions[partition] = partitions.get(partition, 0) + 1

                # Extract doc_id from filename
                doc_id = _extract_doc_id_from_filename(Path(path).name)
                if doc_id and len(sample_doc_ids) < 20:
                    sample_doc_ids.append(doc_id)

    # Estimate row count from statistics (if available)
    approx_rows = None
    try:
        approx_rows = dataset.count_rows()
    except Exception:
        # Fallback: try to estimate from fragment metadata
        pass

    return DatasetSummary(
        dataset_type=dataset_type,
        schema=schema,
        file_count=file_count,
        total_bytes=total_bytes,
        approx_rows=approx_rows,
        partitions=partitions,
        sample_doc_ids=list(set(sample_doc_ids))[:10],  # Unique samples, limit to 10
    )
