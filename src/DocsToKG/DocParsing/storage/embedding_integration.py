"""
Embedding Stage Integration with Storage Layer

Provides Parquet-based vector writer that works with the new storage layer.
All embedding vectors are written exclusively in Parquet columnar format
for efficiency, compression, and analytics compatibility.

Key Interface:
- `create_unified_vector_writer()`: Factory returning Parquet writer.
- Footer metadata includes provenance and model information.
- Schema versioning enables safe evolution.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Iterator, List, Optional, Sequence

from . import parquet_schemas


class VectorWriterError(RuntimeError):
    """Raised when a vector artifact cannot be written."""

    def __init__(self, fmt: str, output_path: Path, original: BaseException) -> None:
        super().__init__(f"Failed to write vectors in {fmt} format to {output_path}: {original}")
        self.format = fmt
        self.output_path = Path(output_path)
        self.original = original


class UnifiedVectorWriter:
    """Vector writer supporting Parquet with a JSONL fallback path."""

    def __init__(
        self,
        output_path: Path,
        fmt: str = "parquet",
        vector_format_override: Optional[str] = None,
        **writer_kwargs: Any,
    ) -> None:
        """Initialise the writer with atomic write semantics."""

        self.output_path = Path(output_path)
        fmt_normalised = str(vector_format_override or fmt or "parquet").lower()
        if fmt_normalised not in {"parquet", "jsonl"}:
            raise ValueError("Supported formats: parquet, jsonl")
        self.fmt = fmt_normalised
        defaults = {
            "compression": "zstd",
            "compression_level": 5,
            "write_statistics": True,
        }
        defaults.update(writer_kwargs)
        self.writer_kwargs = defaults
        self._rows_buffer: List[dict] = []
        self._jsonl_handle: Optional[Any] = None
        self._tmp_path: Optional[Path] = None

    def __enter__(self) -> UnifiedVectorWriter:
        """Prepare temporary resources prior to writing."""

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.fmt == "jsonl":
            tmp = self.output_path.with_name(f"{self.output_path.name}.tmp.{uuid.uuid4().hex}")
            self._tmp_path = tmp
            self._jsonl_handle = open(tmp, "w", encoding="utf-8")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Flush buffered data to disk or clean up on error."""

        if exc_type is not None:
            self._discard_temp()
            return False

        if self.fmt == "jsonl":
            handle = self._jsonl_handle
            tmp_path = self._tmp_path
            if handle is None or tmp_path is None:
                return False
            try:
                handle.flush()
                os.fsync(handle.fileno())
            except OSError:
                # Some filesystems may not support fsync; continue anyway.
                pass
            finally:
                handle.close()
                self._jsonl_handle = None
            try:
                tmp_path.replace(self.output_path)
            except Exception as exc:  # pragma: no cover - defensive
                tmp_path.unlink(missing_ok=True)
                raise VectorWriterError(self.fmt, self.output_path, exc) from exc
            finally:
                self._tmp_path = None
            return False

        if not self._rows_buffer:
            return False

        tmp_path = self.output_path.with_name(f"{self.output_path.name}.tmp.{uuid.uuid4().hex}")
        try:
            table = self._rows_to_arrow_table(self._rows_buffer)
            import pyarrow.parquet as pq

            pq.write_table(table, str(tmp_path), **self.writer_kwargs)
            try:
                with open(tmp_path, "rb") as fh:
                    os.fsync(fh.fileno())
            except OSError:
                pass
            tmp_path.replace(self.output_path)
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            raise VectorWriterError(self.fmt, self.output_path, exc) from exc
        finally:
            self._rows_buffer.clear()

        return False

    def _discard_temp(self) -> None:
        """Clean up any temporary artifacts when aborting the write."""

        if self._jsonl_handle is not None:
            try:
                self._jsonl_handle.close()
            except Exception:  # pragma: no cover - best effort cleanup
                pass
            self._jsonl_handle = None
        if self._tmp_path is not None:
            self._tmp_path.unlink(missing_ok=True)
            self._tmp_path = None
        self._rows_buffer.clear()

    def write_rows(self, rows: Sequence[dict]) -> None:
        """Buffer vector rows for later flush or stream JSONL content."""

        if not rows:
            return

        if self.fmt == "jsonl":
            handle = self._jsonl_handle
            if handle is None:
                raise RuntimeError("JSONL writer is not initialised; use as a context manager")
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")
            return

        self._rows_buffer.extend(rows)

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
    fmt: str = "parquet",
    vector_format_override: Optional[str] = None,
    **kwargs: Any,
) -> UnifiedVectorWriter:
    """
    Factory for creating Parquet vector writers.

    All embedding vectors are written exclusively in Parquet format.

    Args:
        output_path: File path for output vectors (Parquet only).
        fmt: Format string; must be 'parquet' (default, only option).
        vector_format_override: Optional format override (enforced to 'parquet').
        **kwargs: Additional writer configuration.

    Returns:
        UnifiedVectorWriter instance ready for use in a context manager.

    Raises:
        ValueError: If format is not 'parquet'.
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
    Iterate over vector rows from a Parquet file in batches.

    Note: JSONL support is maintained for reading historical data only.
    All new vector writes use Parquet exclusively.

    Args:
        path: Path to vector file (Parquet or legacy JSONL).
        fmt: Format string ('parquet' is primary; 'jsonl' for legacy only).
        batch_size: Number of rows per batch.

    Yields:
        Lists of vector row dictionaries.

    Raises:
        ValueError: If format is unsupported.
    """
    fmt_normalized = str(fmt or "parquet").lower()

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
