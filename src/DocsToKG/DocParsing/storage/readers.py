# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.DocParsing.storage.readers",
#   "purpose": "Dataset Readers for DocParsing Artifacts.",
#   "sections": [
#     {
#       "id": "datasetview",
#       "name": "DatasetView",
#       "anchor": "class-datasetview",
#       "kind": "class"
#     },
#     {
#       "id": "scanresult",
#       "name": "ScanResult",
#       "anchor": "class-scanresult",
#       "kind": "class"
#     },
#     {
#       "id": "inspect-dataset",
#       "name": "inspect_dataset",
#       "anchor": "function-inspect-dataset",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""
Dataset Readers for DocParsing Artifacts

Provides lazy scan utilities for Chunks and Vectors (Dense, Sparse, Lexical) using
PyArrow Datasets with optional DuckDB integration for SQL queries and Polars export.

Key Classes:
- `DatasetView`: Lazy scanner for Chunks or Vectors with filtering and projection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.dataset as ds

from . import paths

logger = logging.getLogger(__name__)


class DatasetView:
    """
    Lazy dataset scanner for Chunks or Vectors (Parquet-backed).

    Supports predicate pushdown, column projection, and export to Polars/Pandas/DuckDB.
    """

    def __init__(
        self,
        data_root: str | Path,
        dataset_type: str,  # "chunks" or "dense"|"sparse"|"lexical" for vectors
        use_threads: bool = True,
    ):
        """
        Initialize a DatasetView.

        Args:
            data_root: Data root directory.
            dataset_type: Dataset type. For vectors, specify family ("dense", "sparse", "lexical").
            use_threads: Enable threaded scans.
        """
        self.data_root = Path(data_root)
        self.dataset_type = dataset_type
        self.use_threads = use_threads
        self._dataset: Optional[ds.Dataset] = None

    def _load_dataset(self) -> ds.Dataset:
        """Lazy-load the Arrow dataset."""
        if self._dataset is not None:
            return self._dataset

        if self.dataset_type == "chunks":
            pattern = paths.chunk_file_glob_pattern(self.data_root, family=None)
        elif self.dataset_type in ("dense", "sparse", "lexical"):
            pattern = paths.chunk_file_glob_pattern(self.data_root, family=self.dataset_type)
        else:
            raise ValueError(f"Invalid dataset_type: {self.dataset_type}")

        logger.debug(f"Loading dataset with pattern: {pattern}")
        self._dataset = ds.dataset(pattern, format="parquet")
        return self._dataset

    def schema(self) -> pa.Schema:
        """Return the Arrow schema of the dataset."""
        return self._load_dataset().schema

    def count(self) -> int:
        """Return the total row count (exact scan)."""
        return len(self._load_dataset().to_table())

    def count_approx(self) -> int:
        """Return approximate row count using Parquet statistics (fast)."""
        # Approximate via fragments
        ds_loaded = self._load_dataset()
        count = 0
        for fragment in ds_loaded.get_fragments():
            metadata = fragment.metadata
            if metadata is not None:
                count += metadata.num_rows
        return count

    def scan(
        self,
        columns: Optional[List[str]] = None,
        filters: Optional[Any] = None,
    ) -> "ScanResult":
        """
        Create a lazy scan with optional column projection and filters.

        Args:
            columns: List of column names to project (None = all).
            filters: PyArrow compute expression for filtering (e.g., ds.field("doc_id") == "X").

        Returns:
            ScanResult object for further operations.
        """
        ds_loaded = self._load_dataset()
        scanner = ds_loaded.scanner(columns=columns, filters=filters, use_threads=self.use_threads)
        return ScanResult(scanner)

    def head(self, n: int = 10) -> pa.Table:
        """Return first n rows as an Arrow Table."""
        ds_loaded = self._load_dataset()
        return ds_loaded.to_table().slice(0, n)

    def to_polars(self, columns: Optional[List[str]] = None) -> Any:
        """
        Export to Polars DataFrame (lazy).

        Requires: polars installed.
        """
        try:
            import polars as pl
        except ImportError as exc:
            raise ImportError("polars is required for to_polars()") from exc

        pattern = paths.chunk_file_glob_pattern(
            self.data_root, family=self.dataset_type if self.dataset_type != "chunks" else None
        )
        lf = pl.scan_parquet(pattern)
        if columns:
            lf = lf.select(columns)
        return lf

    def to_duckdb(self) -> Any:
        """
        Export to DuckDB connection for SQL queries.

        Requires: duckdb installed.
        Returns a DuckDB connection with a virtual table "ds" pointing to the dataset.
        """
        try:
            import duckdb
        except ImportError as exc:
            raise ImportError("duckdb is required for to_duckdb()") from exc

        con = duckdb.connect()
        pattern = paths.chunk_file_glob_pattern(
            self.data_root, family=self.dataset_type if self.dataset_type != "chunks" else None
        )
        con.sql(f"CREATE VIEW ds AS SELECT * FROM read_parquet('{pattern}')")
        return con


class ScanResult:
    """Wrapper around a PyArrow Scanner for deferred execution."""

    def __init__(self, scanner: ds.Scanner):
        """Initialize with a PyArrow Scanner."""
        self.scanner = scanner

    def to_table(self) -> pa.Table:
        """Materialize the scan result to an Arrow Table."""
        return self.scanner.to_table()

    def to_batches(self, max_chunksize: Optional[int] = None) -> List[pa.RecordBatch]:
        """Return results as a list of RecordBatches."""
        table = self.to_table()
        if max_chunksize:
            return table.to_batches(max_chunksize=max_chunksize)
        return table.to_batches()

    def to_pandas(self) -> Any:
        """Export to Pandas DataFrame."""
        return self.to_table().to_pandas()

    def to_polars(self) -> Any:
        """Export to Polars DataFrame."""
        try:
            import polars as pl
        except ImportError as exc:
            raise ImportError("polars is required for to_polars()") from exc
        return pl.from_arrow(self.to_table())

    def count(self) -> int:
        """Count rows in the scan result."""
        return len(self.to_table())


# ============================================================
# Inspection CLI Helpers (for docparse inspect command)
# ============================================================


def inspect_dataset(data_root: str | Path, dataset_type: str) -> Dict[str, Any]:
    """
    Inspect a dataset and return metadata for CLI output.

    Args:
        data_root: Data root directory.
        dataset_type: Dataset type ("chunks", "dense", "sparse", "lexical").

    Returns:
        Dict with schema, row count (approx/exact), file count, total bytes, etc.
    """
    view = DatasetView(data_root, dataset_type)

    try:
        schema = view.schema()
        count_approx = view.count_approx()
        ds_loaded = view._load_dataset()
        file_count = len(list(ds_loaded.get_fragments()))

        return {
            "dataset_type": dataset_type,
            "schema_fields": [f.name for f in schema],
            "row_count_approx": count_approx,
            "file_count": file_count,
            "schema_json": schema.to_json(),
        }
    except Exception as e:
        logger.error(f"Error inspecting dataset: {e}")
        return {
            "dataset_type": dataset_type,
            "error": str(e),
        }
