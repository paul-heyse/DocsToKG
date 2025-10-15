"""
Configuration models and manager for hybrid search.

This module provides comprehensive configuration management for DocsToKG's
hybrid search capabilities, including chunking, indexing, fusion, and
retrieval parameters with thread-safe configuration management.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Literal, Optional


@dataclass(frozen=True)
class ChunkingConfig:
    """Configuration for document chunking operations.

    Defines how documents are split into smaller chunks for processing
    and indexing, balancing between context preservation and search
    granularity.

    Attributes:
        max_tokens: Maximum number of tokens per chunk (800 default)
        overlap: Number of tokens to overlap between chunks (150 default)

    Examples:
        >>> config = ChunkingConfig(max_tokens=1000, overlap=200)
        >>> # Creates chunks with 1000 tokens max, 200 token overlap
    """

    max_tokens: int = 800
    overlap: int = 150


@dataclass(frozen=True)
class DenseIndexConfig:
    """Configuration for FAISS dense vector indexing.

    Controls the behavior of vector similarity search using FAISS,
    including index type selection and performance parameters.

    Attributes:
        index_type: Type of FAISS index ("flat", "ivf_flat", "ivf_pq")
        nlist: Number of Voronoi cells for IVF indexes (1024 default)
        nprobe: Number of cells to search for IVF indexes (8 default)
        pq_m: Number of sub-quantizers for PQ indexes (16 default)
        pq_bits: Bits per sub-quantizer for PQ indexes (8 default)
        oversample: Oversampling factor for IVF training samples (2 default)
        device: GPU device ordinal for FAISS operations (0 default)
        ivfpq_use_precomputed: Use precomputed IVFPQ lookup tables (True default)
        ivfpq_float16_lut: Use float16 IVFPQ lookup tables when available (True default)
        multi_gpu_mode: Replica strategy for multi-GPU hosts ("single" default)
        gpu_temp_memory_bytes: Optional temporary memory pool size for FAISS GPU ops
        gpu_indices_32_bit: When True, store FAISS indices in 32-bit format to save VRAM
        expected_ntotal: Hint for anticipated index size; used to pre-reserve GPU memory
        rebuild_delete_threshold: Pending delete count before forcing full rebuild
        force_64bit_ids: Force FAISS to use 64-bit IDs even when 32-bit would suffice
        interleaved_layout: Enable GPU interleaved layout optimisations when supported
        flat_use_fp16: Use float16 compute for flat indexes when available

    Examples:
        >>> config = DenseIndexConfig(
        ...     index_type="ivf_pq",
        ...     nlist=4096,
        ...     nprobe=32,
        ...     pq_m=32,
        ...     pq_bits=8
        ... )
    """

    index_type: Literal["flat", "ivf_flat", "ivf_pq"] = "flat"
    nlist: int = 1024
    nprobe: int = 8
    pq_m: int = 16
    pq_bits: int = 8
    oversample: int = 2
    device: int = 0
    ivfpq_use_precomputed: bool = True
    ivfpq_float16_lut: bool = True
    multi_gpu_mode: Literal["single", "replicate"] = "single"
    gpu_temp_memory_bytes: Optional[int] = None
    gpu_indices_32_bit: bool = True
    expected_ntotal: int = 0
    rebuild_delete_threshold: int = 10000
    force_64bit_ids: bool = False
    interleaved_layout: bool = True
    flat_use_fp16: bool = False


@dataclass(frozen=True)
class FusionConfig:
    """Configuration for result fusion and ranking.

    Controls how results from different retrieval methods (BM25, SPLADE,
    dense vectors) are combined and ranked for optimal relevance.

    Attributes:
        k0: RRF (Reciprocal Rank Fusion) parameter (60.0 default)
        mmr_lambda: MMR diversification parameter (0.6 default)
        enable_mmr: Whether to apply MMR diversification (True default)
        cosine_dedupe_threshold: Cosine similarity threshold for deduplication (0.98 default)
        max_chunks_per_doc: Maximum chunks to return per document (3 default)

    Examples:
        >>> config = FusionConfig(
        ...     k0=50.0,
        ...     mmr_lambda=0.7,
        ...     enable_mmr=True,
        ...     max_chunks_per_doc=5
        ... )
    """

    k0: float = 60.0
    mmr_lambda: float = 0.6
    enable_mmr: bool = True
    cosine_dedupe_threshold: float = 0.98
    max_chunks_per_doc: int = 3


@dataclass(frozen=True)
class RetrievalConfig:
    """Configuration for individual retrieval methods.

    Controls the behavior of each retrieval method (BM25, SPLADE, dense)
    including how many candidates each method should return.

    Attributes:
        bm25_top_k: Number of BM25 candidates to retrieve (50 default)
        splade_top_k: Number of SPLADE candidates to retrieve (50 default)
        dense_top_k: Number of dense vector candidates to retrieve (50 default)

    Examples:
        >>> config = RetrievalConfig(
        ...     bm25_top_k=100,
        ...     splade_top_k=75,
        ...     dense_top_k=25
        ... )
    """

    bm25_top_k: int = 50
    splade_top_k: int = 50
    dense_top_k: int = 50


@dataclass(frozen=True)
class HybridSearchConfig:
    """Complete configuration for hybrid search operations.

    This class aggregates all configuration for hybrid search functionality,
    providing a single source of truth for all search-related parameters.

    Attributes:
        chunking: Document chunking configuration
        dense: Dense vector indexing configuration
        fusion: Result fusion and ranking configuration
        retrieval: Individual retrieval method configuration

    Examples:
        >>> config = HybridSearchConfig(
        ...     chunking=ChunkingConfig(max_tokens=1000),
        ...     dense=DenseIndexConfig(index_type="ivf_pq"),
        ...     fusion=FusionConfig(enable_mmr=True),
        ...     retrieval=RetrievalConfig(bm25_top_k=100)
        ... )
    """

    chunking: ChunkingConfig = ChunkingConfig()
    dense: DenseIndexConfig = DenseIndexConfig()
    fusion: FusionConfig = FusionConfig()
    retrieval: RetrievalConfig = RetrievalConfig()

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "HybridSearchConfig":
        """Construct a config object from a dictionary payload.

        Args:
            payload: Nested mapping containing `chunking`, `dense`, `fusion`,
                and `retrieval` sections compatible with dataclass fields.

        Returns:
            Fully populated `HybridSearchConfig` instance.
        """
        chunking = ChunkingConfig(**payload.get("chunking", {}))
        dense = DenseIndexConfig(**payload.get("dense", {}))
        fusion = FusionConfig(**payload.get("fusion", {}))
        retrieval = RetrievalConfig(**payload.get("retrieval", {}))
        return HybridSearchConfig(
            chunking=chunking, dense=dense, fusion=fusion, retrieval=retrieval
        )


class HybridSearchConfigManager:
    """File-backed configuration manager with reload support.

    Attributes:
        _path: Path to the JSON/YAML configuration file.
        _lock: Threading lock guarding concurrent reloads.
        _config: Cached HybridSearchConfig instance.

    Examples:
        >>> manager = HybridSearchConfigManager(Path("config.json"))  # doctest: +SKIP
        >>> isinstance(manager.get(), HybridSearchConfig)  # doctest: +SKIP
        True
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = RLock()
        self._config = self._load()

    def get(self) -> HybridSearchConfig:
        """Return the currently cached hybrid search configuration.

        Args:
            None

        Returns:
            Latest `HybridSearchConfig` loaded from disk.
        """
        with self._lock:
            return self._config

    def reload(self) -> HybridSearchConfig:
        """Reload configuration from disk, replacing the cached instance.

        Args:
            None

        Returns:
            Freshly loaded `HybridSearchConfig`.

        Raises:
            FileNotFoundError: If the configuration path is missing.
            ValueError: If the config file is invalid JSON or YAML.
        """
        with self._lock:
            self._config = self._load()
            return self._config

    def _load(self) -> HybridSearchConfig:
        if not self._path.exists():
            raise FileNotFoundError(f"Configuration file {self._path} not found")
        raw = self._path.read_text(encoding="utf-8")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = self._load_yaml(raw)
        return HybridSearchConfig.from_dict(payload)

    def _load_yaml(self, raw: str) -> Dict[str, Any]:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - exercised in tests
            raise ValueError("YAML configuration requires PyYAML dependency") from exc
        data = yaml.safe_load(raw)
        if not isinstance(data, dict):
            raise ValueError("YAML configuration must define a mapping")
        return data
