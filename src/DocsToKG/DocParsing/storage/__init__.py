"""
DocParsing Storage Layer

Provides columnar (Parquet) and streaming (JSONL) writers and readers for
Chunks and Vectors (Dense, Sparse, Lexical) with:
- Explicit Arrow/Parquet schemas with semantic versioning
- Atomic writes with footer metadata for provenance
- Lazy dataset readers with DuckDB/Polars integration
- Partition-aware path construction

Modules:
- schemas: Arrow schema declarations and Parquet footer contracts
- paths: Dataset layout and path builders
- writers: Atomic Parquet writers for all artifact types
- readers: Lazy dataset scans and analytics integration
"""

from __future__ import annotations

__all__ = [
    "schemas",
    "paths",
    "writers",
    "readers",
]
