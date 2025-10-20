"""
Parquet Writers for DocParsing Artifacts

Encapsulates write logic for Chunks and Vectors (Dense, Sparse, Lexical) with:
- Schema validation and enforcement
- Atomic writes (temp → fsync → rename)
- Parquet footer metadata for provenance
- Row-group sizing and compression tuning
- Manifest integration helpers

Key Classes:
- `ParquetWriter`: Abstract base with common patterns.
- `ChunksParquetWriter`: Writes Chunks datasets.
- `DenseVectorWriter`, `SparseVectorWriter`, `LexicalVectorWriter`: Vector writers.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from . import parquet_schemas


class ParquetWriter:
    """Abstract base for Parquet writers with atomic write semantics."""

    def __init__(
        self,
        output_path: Path,
        schema: pa.Schema,
        dataset_type: str,  # "chunks", "dense", "sparse", "lexical"
        compression: str = "zstd",
        compression_level: int = 5,
        target_row_group_mb: int = 32,
    ):
        """
        Initialize a Parquet writer.

        Args:
            output_path: File path for output Parquet file.
            schema: PyArrow schema for the table.
            dataset_type: Type identifier for dataset ("chunks", "dense", "sparse",
                "lexical").
            compression: Compression codec ("zstd", "snappy", "gzip", etc.).
            compression_level: Compression level (1-9 for zstd).
            target_row_group_mb: Target row group size in MB.
        """
        self.output_path = Path(output_path)
        self.schema = schema
        self.dataset_type = dataset_type
        self.compression = compression
        self.compression_level = compression_level
        self.target_row_group_mb = target_row_group_mb

    def _get_writer_kwargs(self) -> Dict[str, Any]:
        """Return pyarrow.parquet.write_table kwargs."""
        kwargs = {
            "compression": self.compression,
            "compression_level": self.compression_level,
            "write_statistics": True,
            "row_group_size": max(
                1, int(self.target_row_group_mb * 1024 * 1024 / 8)
            ),  # rough estimate
        }
        return kwargs

    def _atomic_write(self, table: pa.Table) -> int:
        """
        Atomically write table to output_path: temp → fsync → rename.

        Returns:
            File size in bytes.
        """
        # Ensure parent directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to temporary file
        tmp_path = self.output_path.with_name(f"{self.output_path.name}.tmp.{uuid.uuid4().hex}")
        try:
            kwargs = self._get_writer_kwargs()
            pq.write_table(table, str(tmp_path), **kwargs)

            # Fsync for durability
            try:
                with open(tmp_path, "rb") as f:
                    os.fsync(f.fileno())
            except OSError:
                # Fallback if fsync fails (e.g., on some filesystems)
                pass

            # Atomic rename
            tmp_path.replace(self.output_path)
            return self.output_path.stat().st_size
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    def write(
        self, records: List[Dict[str, Any]], footer_metadata: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Write records to Parquet with footer metadata.

        Args:
            records: List of dictionaries matching the schema.
            footer_metadata: Key-value metadata for Parquet footer.

        Returns:
            Summary dict with row_count, file_size, output_path, etc.

        Raises:
            ValueError: If records don't match schema.
        """
        # Convert to Arrow table
        table = pa.Table.from_pylist(records, schema=self.schema)

        # Validate schema match
        parquet_schemas.assert_table_matches_schema(table, self.schema)

        # Attach footer metadata
        table = parquet_schemas.attach_footer_metadata(table, footer_metadata)

        # Atomic write
        file_size = self._atomic_write(table)

        return {
            "output_path": str(self.output_path),
            "row_count": len(records),
            "file_size_bytes": file_size,
            "schema_version": footer_metadata.get("docparse.schema_version"),
            "compression": self.compression,
            "compression_level": self.compression_level,
        }


class ChunksParquetWriter(ParquetWriter):
    """Writer for Chunks Parquet datasets."""

    def __init__(
        self,
        output_path: Path,
        compression: str = "zstd",
        compression_level: int = 5,
        target_row_group_mb: int = 32,
        include_optional: bool = True,
    ):
        """
        Initialize a Chunks writer.

        Args:
            output_path: File path for output Parquet file.
            compression: Compression codec.
            compression_level: Compression level.
            target_row_group_mb: Target row group size.
            include_optional: Whether to include optional columns (section, meta).
        """
        schema = parquet_schemas.chunks_schema(include_optional=include_optional)
        super().__init__(
            output_path,
            schema,
            "chunks",
            compression=compression,
            compression_level=compression_level,
            target_row_group_mb=target_row_group_mb,
        )

    def write(
        self,
        records: List[Dict[str, Any]],
        cfg_hash: str,
        created_by: str = "DocsToKG-DocParsing",
    ) -> Dict[str, Any]:
        """
        Write chunk records with standard footer.

        Args:
            records: List of chunk dicts.
            cfg_hash: Config hash for reproducibility.
            created_by: Creator identifier.

        Returns:
            Summary dict.
        """
        footer_meta = parquet_schemas.build_footer_common(
            parquet_schemas.SCHEMA_VERSION_CHUNKS,
            cfg_hash,
            created_by,
        )
        return super().write(records, footer_meta)


class DenseVectorWriter(ParquetWriter):
    """Writer for Dense vector Parquet datasets."""

    def __init__(
        self,
        output_path: Path,
        dim: int,
        fixed_size: bool = True,
        compression: str = "zstd",
        compression_level: int = 5,
        target_row_group_mb: int = 32,
    ):
        """
        Initialize a Dense vector writer.

        Args:
            output_path: File path for output Parquet file.
            dim: Vector dimension.
            fixed_size: Whether to use fixed-size lists (recommended).
            compression: Compression codec.
            compression_level: Compression level.
            target_row_group_mb: Target row group size.
        """
        schema = parquet_schemas.dense_schema(dim, fixed_size=fixed_size)
        super().__init__(
            output_path,
            schema,
            "dense",
            compression=compression,
            compression_level=compression_level,
            target_row_group_mb=target_row_group_mb,
        )
        self.dim = dim
        self.fixed_size = fixed_size

    def write(
        self,
        records: List[Dict[str, Any]],
        provider: str,
        model_id: str,
        cfg_hash: str,
        device: Optional[str] = None,
        created_by: str = "DocsToKG-DocParsing",
    ) -> Dict[str, Any]:
        """
        Write dense vector records with standard footer.

        Args:
            records: List of dense vector dicts.
            provider: Provider identifier (e.g., "dense.qwen_vllm").
            model_id: Model identifier (with optional @revision).
            cfg_hash: Config hash.
            device: Device identifier (e.g., "cuda:0").
            created_by: Creator identifier.

        Returns:
            Summary dict.
        """
        footer_meta = parquet_schemas.build_footer_dense(
            provider=provider,
            model_id=model_id,
            dim=self.dim,
            cfg_hash=cfg_hash,
            device=device,
            created_by=created_by,
        )
        return super().write(records, footer_meta)


class SparseVectorWriter(ParquetWriter):
    """Writer for Sparse vector (SPLADE) Parquet datasets."""

    def __init__(
        self,
        output_path: Path,
        compression: str = "zstd",
        compression_level: int = 5,
        target_row_group_mb: int = 32,
    ):
        """
        Initialize a Sparse vector writer.

        Args:
            output_path: File path for output Parquet file.
            compression: Compression codec.
            compression_level: Compression level.
            target_row_group_mb: Target row group size.
        """
        schema = parquet_schemas.sparse_schema_idspace()
        super().__init__(
            output_path,
            schema,
            "sparse",
            compression=compression,
            compression_level=compression_level,
            target_row_group_mb=target_row_group_mb,
        )

    def write(
        self,
        records: List[Dict[str, Any]],
        provider: str,
        model_id: str,
        cfg_hash: str,
        vocab_id: Optional[str] = None,
        hash_scheme: Optional[str] = None,
        created_by: str = "DocsToKG-DocParsing",
    ) -> Dict[str, Any]:
        """
        Write sparse vector records with standard footer.

        Args:
            records: List of sparse vector dicts.
            provider: Provider identifier (e.g., "sparse.splade_st").
            model_id: Model identifier.
            cfg_hash: Config hash.
            vocab_id: Optional vocabulary identifier.
            hash_scheme: Optional hashing scheme description.
            created_by: Creator identifier.

        Returns:
            Summary dict.
        """
        footer_meta = parquet_schemas.build_footer_sparse(
            provider=provider,
            model_id=model_id,
            cfg_hash=cfg_hash,
            vocab_id=vocab_id,
            hash_scheme=hash_scheme,
            created_by=created_by,
        )
        return super().write(records, footer_meta)


class LexicalVectorWriter(ParquetWriter):
    """Writer for Lexical vector (BM25) Parquet datasets."""

    def __init__(
        self,
        output_path: Path,
        representation: str = "indices",  # "indices" or "terms"
        compression: str = "zstd",
        compression_level: int = 5,
        target_row_group_mb: int = 32,
    ):
        """
        Initialize a Lexical vector writer.

        Args:
            output_path: File path for output Parquet file.
            representation: "indices" (id-space, default) or "terms" (term-space).
            compression: Compression codec.
            compression_level: Compression level.
            target_row_group_mb: Target row group size.
        """
        if representation == "indices":
            schema = parquet_schemas.lexical_schema_idspace()
        elif representation == "terms":
            schema = parquet_schemas.lexical_schema_terms()
        else:
            raise ValueError(f"Invalid representation: {representation}")

        super().__init__(
            output_path,
            schema,
            "lexical",
            compression=compression,
            compression_level=compression_level,
            target_row_group_mb=target_row_group_mb,
        )
        self.representation = representation

    def write(
        self,
        records: List[Dict[str, Any]],
        tokenizer_id: str,
        k1: float,
        b: float,
        stopwords_policy: str,
        min_df: int,
        max_df_ratio: float,
        cfg_hash: str,
        created_by: str = "DocsToKG-DocParsing",
    ) -> Dict[str, Any]:
        """
        Write lexical vector records with standard footer.

        Args:
            records: List of lexical vector dicts.
            tokenizer_id: Tokenizer identifier.
            k1: BM25 k1 parameter.
            b: BM25 b parameter.
            stopwords_policy: Stopwords policy description.
            min_df: Minimum document frequency.
            max_df_ratio: Maximum document frequency ratio.
            cfg_hash: Config hash.
            created_by: Creator identifier.

        Returns:
            Summary dict.
        """
        footer_meta = parquet_schemas.build_footer_lexical(
            representation=self.representation,
            tokenizer_id=tokenizer_id,
            k1=k1,
            b=b,
            stopwords_policy=stopwords_policy,
            min_df=min_df,
            max_df_ratio=max_df_ratio,
            cfg_hash=cfg_hash,
            created_by=created_by,
        )
        return super().write(records, footer_meta)


# ============================================================
# Factory / Convenience Helpers
# ============================================================


def create_chunks_writer(
    output_path: str | Path,
    **kwargs: Any,
) -> ChunksParquetWriter:
    """Convenience factory for ChunksParquetWriter."""
    return ChunksParquetWriter(Path(output_path), **kwargs)


def create_dense_writer(
    output_path: str | Path,
    dim: int,
    **kwargs: Any,
) -> DenseVectorWriter:
    """Convenience factory for DenseVectorWriter."""
    return DenseVectorWriter(Path(output_path), dim, **kwargs)


def create_sparse_writer(
    output_path: str | Path,
    **kwargs: Any,
) -> SparseVectorWriter:
    """Convenience factory for SparseVectorWriter."""
    return SparseVectorWriter(Path(output_path), **kwargs)


def create_lexical_writer(
    output_path: str | Path,
    representation: str = "indices",
    **kwargs: Any,
) -> LexicalVectorWriter:
    """Convenience factory for LexicalVectorWriter."""
    return LexicalVectorWriter(Path(output_path), representation=representation, **kwargs)
