"""
DocParsing Storage Layer

This package encapsulates Parquet/Arrow schemas, partitioning layouts, and writers
for Chunks and Vectors (Dense, Sparse, Lexical). It replaces ad-hoc JSONL writers
with schema-enforced, provenance-rich columnar output.

Key modules:
- `parquet_schemas.py`: Arrow/Parquet schema declarations and footer contracts
- `paths.py`: Dataset layout builders (partitions, rel_id normalization)
- `writers.py`: Unified writer abstraction for Chunks and Vectors
- `readers.py`: Dataset view helpers (lazy scans via Polars/DuckDB/Arrow)
- `validation.py`: Schema and data quality checks

Usage:
    from DocsToKG.DocParsing.storage import schemas, paths, writers

    schema = schemas.chunks_schema()
    output_dir = paths.chunks_output_path("Data", doc_id="papers/physics/123")
    writer = writers.ChunksParquetWriter(output_dir)
    writer.write(chunk_rows)
"""

from __future__ import annotations

__all__ = [
    "schemas",
    "paths",
    "writers",
    "readers",
    "validation",
]
