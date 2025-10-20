"""
Embedding Stage Integration with Storage Layer

Provides unified factories for vector writers that work with both legacy JSONL
format and new Parquet columnar format. This enables gradual migration of the
embedding stage to use the new storage layer without immediate refactoring of
all downstream code.

Key Interface:
- `create_unified_vector_writer()`: Factory returning either JSONL or Parquet
  writer based on format string and configuration.
- JSONL writers maintain backward compatibility; Parquet writers leverage
  the new schemas and footer metadata for provenance.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Iterator, List, Optional, Sequence

from . import parquet_schemas


class UnifiedVectorWriter:
    """
    Adapter that provides unified interface for both JSONL and Parquet writers.

    This allows the embedding stage to write vectors in either format while
    the implementation details are abstracted away.
    """

    def __init__(
        self,
        output_path: Path,
        fmt: str = "jsonl",
        vector_format_override: Optional[str] = None,
        **writer_kwargs: Any,
    ):
        """
        Initialize unified writer.

        Args:
            output_path: Path where vectors will be written.
            fmt: Format string ('jsonl' or 'parquet'); default 'jsonl'.
            vector_format_override: Optional override for format selection.
            **writer_kwargs: Additional kwargs for writer initialization.
        """
        self.output_path = Path(output_path)
        fmt = str(vector_format_override or fmt or "jsonl").lower()
        self.fmt = fmt
        self.writer_kwargs = writer_kwargs
        self._context = None
        self._writer = None

        if fmt not in ("jsonl", "parquet"):
            raise ValueError(f"Unsupported vector format: {fmt}")

    def __enter__(self) -> UnifiedVectorWriter:
        """Enter context manager and initialize underlying writer."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.fmt == "jsonl":
            # Legacy JSONL: use atomic_write pattern
            from DocsToKG.DocParsing.io import atomic_write

            self._context = atomic_write(self.output_path)
            self._writer = self._context.__enter__()
        else:
            # Parquet: note that Parquet writer will be created when write_rows is called
            # This matches the legacy interface where writers are context managers
            pass

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Exit context manager and finalize underlying writer."""
        if self._context is not None:
            return self._context.__exit__(exc_type, exc_val, exc_tb)
        return False

    def write_rows(self, rows: Sequence[dict]) -> None:
        """
        Write a batch of vector rows to the underlying storage.

        Args:
            rows: List of vector row dictionaries.
        """
        if not rows:
            return

        if self.fmt == "jsonl":
            if self._writer is None:
                raise RuntimeError("JSONL writer not initialized; call __enter__ first.")
            for row in rows:
                self._writer.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:  # parquet
            # Convert rows to Arrow table and write with footer metadata
            table = self._rows_to_arrow_table(rows)
            # Write directly (file will be created on first write)
            import pyarrow.parquet as pq

            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.output_path.with_name(f"{self.output_path.name}.tmp.{uuid.uuid4().hex}")
            try:
                pq.write_table(
                    table,
                    str(tmp_path),
                    compression="zstd",
                    compression_level=5,
                    write_statistics=True,
                )
                try:
                    with open(tmp_path, "rb") as f:
                        os.fsync(f.fileno())
                except OSError:
                    pass
                tmp_path.replace(self.output_path)
            except Exception:
                tmp_path.unlink(missing_ok=True)
                raise

    @staticmethod
    def _rows_to_arrow_table(rows: Sequence[dict]) -> Any:
        """
        Convert vector rows to Arrow table with legacy schema.

        This maintains compatibility with existing vector row format
        while preparing for transition to new schemas.
        """
        import pyarrow as pa

        if not rows:
            raise ValueError("rows must contain at least one vector payload")

        # Use the schema from the legacy vector format
        # The schema includes all three vector families in one row
        schema = parquet_schemas._legacy_vector_schema()
        return pa.Table.from_pylist(rows, schema=schema)


def create_unified_vector_writer(
    output_path: Path,
    fmt: str = "jsonl",
    vector_format_override: Optional[str] = None,
    **kwargs: Any,
) -> UnifiedVectorWriter:
    """
    Factory for creating unified vector writers.

    Args:
        output_path: File path for output vectors.
        fmt: Format string ('jsonl' or 'parquet').
        vector_format_override: Optional format override (checked first).
        **kwargs: Additional writer configuration.

    Returns:
        UnifiedVectorWriter instance ready for use in a context manager.
    """
    return UnifiedVectorWriter(
        output_path,
        fmt=fmt,
        vector_format_override=vector_format_override,
        **kwargs,
    )


def iter_vector_rows(
    path: Path,
    fmt: str,
    *,
    batch_size: int = 4096,
) -> Iterator[List[dict]]:
    """
    Iterate over vector rows from a file in the specified format.

    Supports both JSONL and Parquet formats, yielding batches of row dicts.

    Args:
        path: Path to vector file (JSONL or Parquet).
        fmt: Format string ('jsonl' or 'parquet').
        batch_size: Number of rows per batch.

    Yields:
        Lists of vector row dictionaries.
    """
    fmt_normalized = str(fmt or "jsonl").lower()

    if fmt_normalized == "jsonl":
        from DocsToKG.DocParsing.embedding.runtime import iter_rows_in_batches

        yield from iter_rows_in_batches(path, batch_size)
    elif fmt_normalized == "parquet":
        import json

        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(path)
        for record_batch in parquet_file.iter_batches(batch_size=batch_size):
            rows = record_batch.to_pylist()
            normalized: List[dict] = []
            for entry in rows:
                # Normalize model_metadata if it's a JSON string
                metadata = entry.get("model_metadata")
                if isinstance(metadata, str) and metadata:
                    try:
                        entry["model_metadata"] = json.loads(metadata)
                    except json.JSONDecodeError:
                        entry["model_metadata"] = {}
                elif metadata in (None, ""):
                    entry["model_metadata"] = {}
                normalized.append(entry)
            if normalized:
                yield normalized
    else:
        raise ValueError(f"Unsupported vector format: {fmt}")
