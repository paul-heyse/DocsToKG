# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.DocParsing.core.models",
#   "purpose": "Typed data models used to coordinate DocParsing chunking and embedding.",
#   "sections": [
#     {
#       "id": "bm25stats",
#       "name": "BM25Stats",
#       "anchor": "class-bm25stats",
#       "kind": "class"
#     },
#     {
#       "id": "spladecfg",
#       "name": "SpladeCfg",
#       "anchor": "class-spladecfg",
#       "kind": "class"
#     },
#     {
#       "id": "qwencfg",
#       "name": "QwenCfg",
#       "anchor": "class-qwencfg",
#       "kind": "class"
#     },
#     {
#       "id": "chunkworkerconfig",
#       "name": "ChunkWorkerConfig",
#       "anchor": "class-chunkworkerconfig",
#       "kind": "class"
#     },
#     {
#       "id": "chunktask",
#       "name": "ChunkTask",
#       "anchor": "class-chunktask",
#       "kind": "class"
#     },
#     {
#       "id": "chunkresult",
#       "name": "ChunkResult",
#       "anchor": "class-chunkresult",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""Typed data models used to coordinate DocParsing chunking and embedding.

This module defines the small dataclasses passed between worker processes:
chunking tasks, sparse/dense encoder configuration objects, and the corpus
statistics needed for BM25 scoring. Centralising these definitions keeps
serialization stable and allows tests to construct realistic payloads without
duplicating field order or defaults.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "BM25Stats",
    "ChunkResult",
    "ChunkTask",
    "ChunkWorkerConfig",
    "DEFAULT_SERIALIZER_PROVIDER",
    "DEFAULT_TOKENIZER",
    "QwenCfg",
    "SpladeCfg",
    "UUID_NAMESPACE",
]

UUID_NAMESPACE = uuid.UUID("00000000-0000-0000-0000-000000000000")
DEFAULT_SERIALIZER_PROVIDER = "DocsToKG.DocParsing.formats:RichSerializerProvider"
DEFAULT_TOKENIZER = "Qwen/Qwen3-Embedding-4B"


@dataclass(slots=True)
class BM25Stats:
    """Corpus-level statistics required for BM25 weighting."""

    N: int
    avgdl: float
    df: dict[str, int]


@dataclass(slots=True)
class SpladeCfg:
    """Runtime configuration for SPLADE sparse encoding."""

    model_dir: Path
    device: str = "cuda"
    batch_size: int = 32
    cache_folder: Path | None = None
    max_active_dims: int | None = None
    attn_impl: str | None = None
    local_files_only: bool = True


@dataclass(slots=True)
class QwenCfg:
    """Configuration for generating dense embeddings with Qwen via vLLM."""

    model_dir: Path
    dtype: str = "bfloat16"
    tp: int = 1
    gpu_mem_util: float = 0.60
    batch_size: int = 32
    quantization: str | None = None
    dim: int = 2560
    cache_enabled: bool = True


@dataclass(slots=True)
class ChunkWorkerConfig:
    """Lightweight configuration shared across chunker worker processes."""

    tokenizer_model: str
    min_tokens: int
    max_tokens: int
    soft_barrier_margin: int
    heading_markers: tuple[str, ...]
    caption_markers: tuple[str, ...]
    docling_version: str
    serializer_provider_spec: str = DEFAULT_SERIALIZER_PROVIDER
    inject_anchors: bool = False
    data_root: Path | None = None
    format: str = "parquet"


@dataclass(slots=True)
class ChunkTask:
    """Work unit describing a single DocTags file to chunk."""

    doc_path: Path
    output_path: Path
    doc_id: str
    doc_stem: str
    input_hash: str
    parse_engine: str
    sanitizer_profile: str | None = None


@dataclass(slots=True)
class ChunkResult:
    """Result envelope emitted by chunker workers."""

    doc_id: str
    doc_stem: str
    status: str
    duration_s: float
    input_path: Path
    output_path: Path
    input_hash: str
    chunk_count: int
    total_tokens: int
    parse_engine: str
    sanitizer_profile: str | None = None
    anchors_injected: bool = False
    error: str | None = None
    artifact_paths: tuple[Path, ...] = ()
    parquet_bytes: int | None = None
    row_group_count: int | None = None
    rows_written: int | None = None
