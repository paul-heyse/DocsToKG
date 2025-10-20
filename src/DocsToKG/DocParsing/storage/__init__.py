"""
DocParsing Storage Layer

Provides columnar (Parquet) writers and readers for Chunks and Vectors
(Dense, Sparse, Lexical) with:
- Explicit Arrow/Parquet schemas with semantic versioning
- Atomic writes with footer metadata for provenance
- Lazy dataset readers with DuckDB/Polars integration
- Partition-aware path construction

Embedding Vectors:
- All new vector writes use Parquet exclusively
- Legacy JSONL support maintained for reading historical data only
- Chunks can use either Parquet (default) or JSONL format

Modules:
- schemas: Arrow schema declarations and Parquet footer contracts
- paths: Dataset layout and path builders
- writers: Atomic Parquet writers for all artifact types
- readers: Lazy dataset scans and analytics integration
- embedding_integration: Parquet writer factory for embedding vectors
"""

from __future__ import annotations

__all__ = [
    "schemas",
    "paths",
    "writers",
    "readers",
]
