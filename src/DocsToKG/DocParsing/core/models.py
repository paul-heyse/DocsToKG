"""Lightweight data containers shared across DocParsing stages."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

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
    df: Dict[str, int]


@dataclass(slots=True)
class SpladeCfg:
    """Runtime configuration for SPLADE sparse encoding."""

    model_dir: Path
    device: str = "cuda"
    batch_size: int = 32
    cache_folder: Optional[Path] = None
    max_active_dims: Optional[int] = None
    attn_impl: Optional[str] = None
    local_files_only: bool = True


@dataclass(slots=True)
class QwenCfg:
    """Configuration for generating dense embeddings with Qwen via vLLM."""

    model_dir: Path
    dtype: str = "bfloat16"
    tp: int = 1
    gpu_mem_util: float = 0.60
    batch_size: int = 32
    quantization: Optional[str] = None
    dim: int = 2560
    cache_enabled: bool = True


@dataclass(slots=True)
class ChunkWorkerConfig:
    """Lightweight configuration shared across chunker worker processes."""

    tokenizer_model: str
    min_tokens: int
    max_tokens: int
    soft_barrier_margin: int
    heading_markers: Tuple[str, ...]
    caption_markers: Tuple[str, ...]
    docling_version: str
    serializer_provider_spec: str = DEFAULT_SERIALIZER_PROVIDER
    inject_anchors: bool = False
    data_root: Optional[Path] = None


@dataclass(slots=True)
class ChunkTask:
    """Work unit describing a single DocTags file to chunk."""

    doc_path: Path
    output_path: Path
    doc_id: str
    doc_stem: str
    input_hash: str
    parse_engine: str
    sanitizer_profile: Optional[str] = None


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
    parse_engine: str
    sanitizer_profile: Optional[str] = None
    anchors_injected: bool = False
    error: Optional[str] = None
