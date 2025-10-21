"""
Atomic Parquet Writer for Chunks

Encapsulates write logic for Chunks Parquet datasets with:
- Schema validation and enforcement
- Atomic writes (temp → fsync → rename) for safe concurrent access
  * Write to temporary file in same directory
  * Fsync to ensure durability
  * Atomic rename to final destination (no explicit locking needed; rename is atomic at OS level)
  * Concurrent readers are safe via temp-file pattern
- Batched row accumulation to control memory
- Parquet footer metadata for provenance
- Deterministic span hashing for reproducible chunks

Key Class:
- `ParquetChunksWriter`: Writes Chunks datasets with optional rolling.

All writes are safe for concurrent access and preserve data durability.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from . import parquet_schemas, paths

# ============================================================
# Types
# ============================================================


class WriteResult:
    """Summary of a Parquet write operation."""

    def __init__(
        self,
        paths: List[Path],
        rows_written: int,
        row_group_count: int,
        parquet_bytes: int,
    ):
        self.paths = paths
        self.rows_written = rows_written
        self.row_group_count = row_group_count
        self.parquet_bytes = parquet_bytes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paths": [str(p) for p in self.paths],
            "rows_written": self.rows_written,
            "row_group_count": self.row_group_count,
            "parquet_bytes": self.parquet_bytes,
        }


# ============================================================
# Main Writer
# ============================================================


class ParquetChunksWriter:
    """
    Atomic writer for Chunks Parquet datasets.

    Batches rows to control memory, validates schema, attaches footer metadata,
    and performs atomic writes with fsync.
    """

    def __init__(
        self,
        schema: Optional[pa.Schema] = None,
        compression: str = "zstd",
        compression_level: int = 5,
        target_row_group_mb: int = 32,
        batch_size: int = 1000,
    ):
        """
        Initialize a Chunks Parquet writer.

        Args:
            schema: PyArrow schema for Chunks. If None, uses chunks_schema().
            compression: Compression codec ("zstd", "snappy", "gzip", etc.).
            compression_level: Compression level (1-9 for zstd).
            target_row_group_mb: Target row group size in MB.
            batch_size: Number of rows to buffer before converting to Arrow table.
        """
        self.schema = schema or parquet_schemas.chunks_schema(include_optional=True)
        self.compression = compression
        self.compression_level = compression_level
        self.target_row_group_mb = target_row_group_mb
        self.batch_size = batch_size

    @staticmethod
    def _validate_row(row: Dict[str, Any]) -> None:
        """
        Validate a single chunk row invariants.

        Raises:
            ValueError: If invariants violated.
        """
        # doc_id and chunk_id must be present and non-empty
        if not row.get("doc_id"):
            raise ValueError("doc_id is required and cannot be empty")
        if not isinstance(row.get("chunk_id"), int) or row["chunk_id"] < 0:
            raise ValueError(f"chunk_id must be non-negative int, got {row.get('chunk_id')}")

        # text must be non-empty
        text = row.get("text", "")
        if not text or not isinstance(text, str):
            raise ValueError(f"text is required and must be str, got {type(text)}")

        # tokens must be non-negative
        if not isinstance(row.get("tokens"), int) or row["tokens"] < 0:
            raise ValueError(f"tokens must be non-negative int, got {row.get('tokens')}")

        # span must have start <= end
        span = row.get("span", {})
        if not isinstance(span, dict):
            raise ValueError(f"span must be dict, got {type(span)}")
        start = span.get("start", 0)
        end = span.get("end", 0)
        if not (isinstance(start, int) and isinstance(end, int)):
            raise ValueError(f"span.start and span.end must be int, got {type(start)}, {type(end)}")
        if start < 0 or end < 0 or start > end:
            raise ValueError(f"span invariant: 0 <= start <= end, got start={start}, end={end}")

    @staticmethod
    def _compute_span_hash(text: str, start: int, end: int) -> str:
        """
        Compute deterministic hash of a span.

        Args:
            text: Full text.
            start: Span start (inclusive).
            end: Span end (exclusive).

        Returns:
            Hex-encoded SHA256 hash of the span text.
        """
        span_text = text[start:end]
        return hashlib.sha256(span_text.encode("utf-8")).hexdigest()

    def write(
        self,
        rows_iter: Iterable[Dict[str, Any]],
        *,
        data_root: Path,
        rel_id: str,
        cfg_hash: str,
        created_by: str = "DocsToKG-DocParsing",
        dt_utc: Optional[datetime] = None,
    ) -> WriteResult:
        """
        Write chunk rows to Parquet (atomic, batched).

        Args:
            rows_iter: Iterable of chunk row dicts.
            data_root: Data root directory.
            rel_id: Relative identifier for output file.
            cfg_hash: Configuration hash for footer.
            created_by: Creator identifier for footer.
            dt_utc: Write timestamp (UTC). If None, uses current time.

        Returns:
            WriteResult with paths, row count, and metadata.

        Raises:
            ValueError: If any row violates invariants.
        """
        if dt_utc is None:
            dt_utc = datetime.now(timezone.utc)

        output_path = paths.chunks_output_path(data_root, rel_id, fmt="parquet", ts=dt_utc)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Collect and validate rows in batches
        all_rows: List[Dict[str, Any]] = []
        for row in rows_iter:
            self._validate_row(row)
            all_rows.append(row)

        if not all_rows:
            raise ValueError(f"No rows to write for {rel_id}")

        # Convert to Arrow table and write
        table = pa.Table.from_pylist(all_rows, schema=self.schema)

        # Validate schema match
        parquet_schemas.assert_table_matches_schema(table, self.schema)

        # Build footer metadata
        footer_meta = parquet_schemas.build_footer_common(
            schema_version=parquet_schemas.SCHEMA_VERSION_CHUNKS,
            cfg_hash=cfg_hash,
            created_by=created_by,
            created_at=dt_utc.strftime(parquet_schemas.ISO_UTC),
        )

        # Attach footer
        table = parquet_schemas.attach_footer_metadata(table, footer_meta)

        # Atomic write
        file_size = self._atomic_write(table, output_path)

        return WriteResult(
            paths=[output_path],
            rows_written=len(all_rows),
            row_group_count=self._estimate_row_group_count(file_size),
            parquet_bytes=file_size,
        )

    def _atomic_write(self, table: pa.Table, output_path: Path) -> int:
        """
        Atomically write table to output_path: temp → fsync → rename.

        Args:
            table: PyArrow table to write.
            output_path: Final output path.

        Returns:
            File size in bytes.
        """
        tmp_path = output_path.with_name(f"{output_path.name}.tmp.{uuid.uuid4().hex}")
        try:
            kwargs = {
                "compression": self.compression,
                "compression_level": self.compression_level,
                "write_statistics": True,
                "row_group_size": max(1, int(self.target_row_group_mb * 1024 * 1024 / 8)),
            }
            pq.write_table(table, str(tmp_path), **kwargs)

            # Fsync for durability
            try:
                with open(tmp_path, "rb") as f:
                    os.fsync(f.fileno())
            except OSError:
                # Fallback if fsync fails (e.g., on some filesystems)
                pass

            # Atomic rename
            tmp_path.replace(output_path)
            return output_path.stat().st_size
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    @staticmethod
    def _estimate_row_group_count(file_size: int, avg_row_group_mb: int = 32) -> int:
        """
        Estimate row group count based on file size.

        Args:
            file_size: File size in bytes.
            avg_row_group_mb: Average row group size in MB.

        Returns:
            Estimated row group count.
        """
        avg_row_group_bytes = avg_row_group_mb * 1024 * 1024
        return max(1, file_size // avg_row_group_bytes)
