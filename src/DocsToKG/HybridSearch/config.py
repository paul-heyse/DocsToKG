# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.HybridSearch.config",
#   "purpose": "Hybrid search configuration models and manager",
#   "sections": [
#     {
#       "id": "chunkingconfig",
#       "name": "ChunkingConfig",
#       "anchor": "class-chunkingconfig",
#       "kind": "class"
#     },
#     {
#       "id": "denseindexconfig",
#       "name": "DenseIndexConfig",
#       "anchor": "class-denseindexconfig",
#       "kind": "class"
#     },
#     {
#       "id": "fusionconfig",
#       "name": "FusionConfig",
#       "anchor": "class-fusionconfig",
#       "kind": "class"
#     },
#     {
#       "id": "retrievalconfig",
#       "name": "RetrievalConfig",
#       "anchor": "class-retrievalconfig",
#       "kind": "class"
#     },
#     {
#       "id": "hybridsearchconfig",
#       "name": "HybridSearchConfig",
#       "anchor": "class-hybridsearchconfig",
#       "kind": "class"
#     },
#     {
#       "id": "hybridsearchconfigmanager",
#       "name": "HybridSearchConfigManager",
#       "anchor": "class-hybridsearchconfigmanager",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""Configuration surface area for DocsToKG hybrid search.

The dataclasses defined here mirror the YAML/JSON snippets in the repository
README and describe every user-tunable aspect of the system:

- ``ChunkingConfig`` governs deterministic sliding-window chunking performed by
  :mod:`DocsToKG.HybridSearch.pipeline`. Adjusting ``max_tokens`` or ``overlap``
  directly alters the lexical payloads consumed by the FAISS adapters.
- ``DenseIndexConfig`` maps one-to-one to FAISS GPU primitives documented in
  ``faiss-gpu-wheel-reference.md``. Fields such as ``index_type``, ``nlist``,
  ``nprobe``, ``pq_m``/``pq_bits`` select between ``GpuIndexFlat*`` and
  ``GpuIndexIVF(PQ|ScalarQuantizer)`` families, while ``multi_gpu_mode``,
  ``replication_gpu_ids``, and the memory toggles feed into
  ``faiss.GpuMultipleClonerOptions`` and ``StandardGpuResources`` sizing. Flags
  like ``use_cuvs`` and ``flat_use_fp16`` control whether GPU distance kernels
  spin up ``GpuDistanceParams`` instances with FP16 or cuVS acceleration.
- ``FusionConfig`` and ``RetrievalConfig`` encode scoring budgets for the hybrid
  service layer (RRF/MMR weights, per-channel top-k cut-offs, cosine dedupe
  thresholds). ``HybridSearchService`` reads these values verbatim to plan
  concurrent lexical + dense queries and to guard pagination state.
- ``HybridSearchConfig`` groups the four components into a single snapshot,
  making it the hand-off shape for ingestion and query orchestration.

``HybridSearchConfigManager`` is a thread-safe facade for loading and persisting
configuration files. It supports JSON *and* YAML, normalises legacy field names
(``gpu_default_null_stream*``), and caches the current config while providing
atomic reloads so background workers can hot-swap FAISS settings without racing
ingestion or query threads.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Literal, Mapping, Optional, Tuple

# --- Globals ---

__all__ = (
    "ChunkingConfig",
    "DenseIndexConfig",
    "FusionConfig",
    "HybridSearchConfig",
    "HybridSearchConfigManager",
    "RetrievalConfig",
)


# --- Public Classes ---


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
        replication_gpu_ids: Explicit GPU ids to target when multi-GPU replication is enabled
        gpu_temp_memory_bytes: Optional temporary memory pool size for FAISS GPU ops
        gpu_pinned_memory_bytes: Optional pinned memory pool size for faster H2D/D2H transfers
        gpu_indices_32_bit: When True, store FAISS indices in 32-bit format to save VRAM
        expected_ntotal: Hint for anticipated index size; used to pre-reserve GPU memory
        rebuild_delete_threshold: Pending delete count before forcing full rebuild
        force_64bit_ids: Force FAISS to use 64-bit IDs even when 32-bit would suffice
        interleaved_layout: Enable GPU interleaved layout optimisations when supported
        flat_use_fp16: Use float16 compute for flat indexes when available
        enable_replication: Allow replication to additional GPUs when configured
        enable_reserve_memory: Enable GPU memory reservation based on expected_ntotal
        use_pinned_memory: Use pinned host buffers for large add/search batches
        gpu_use_default_null_stream: Use FAISS APIs to bind CUDA's default null stream
            for the configured device when available (False default)
        gpu_use_default_null_stream_all_devices: Use FAISS APIs to bind the default
            CUDA null stream across every visible GPU (False default)
        use_cuvs: Optional override to force-enable or disable cuVS acceleration
            for FAISS GPU distance kernels. ``None`` defers to FAISS heuristics
            (``faiss.should_use_cuvs``).
        ingest_dedupe_threshold: Skip ingest when cosine similarity exceeds this value (0 disables)
        persist_mode: Persistence policy ("cpu_bytes" default, disable for GPU-only runtime)
        snapshot_refresh_interval_seconds: Minimum seconds between CPU snapshot refreshes (0 disables interval throttle)
        snapshot_refresh_writes: Minimum write operations between CPU snapshot refreshes (0 disables write throttle)
        ivf_train_factor: Controls IVF training sample size per nlist shard

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
    multi_gpu_mode: Literal["single", "replicate", "shard"] = "single"
    replication_gpu_ids: Optional[Tuple[int, ...]] = None
    gpu_temp_memory_bytes: Optional[int] = None
    gpu_pinned_memory_bytes: Optional[int] = None
    gpu_indices_32_bit: bool = True
    expected_ntotal: int = 0
    rebuild_delete_threshold: int = 10000
    force_64bit_ids: bool = False
    interleaved_layout: bool = True
    flat_use_fp16: bool = False
    use_cuvs: Optional[bool] = None
    enable_replication: bool = True
    enable_reserve_memory: bool = True
    use_pinned_memory: bool = True
    gpu_use_default_null_stream: bool = False
    gpu_use_default_null_stream_all_devices: bool = False
    ingest_dedupe_threshold: float = 0.0
    force_remove_ids_fallback: bool = False
    # Persistence: "cpu_bytes" (default, serialize via CPU), or "disabled"
    persist_mode: Literal["cpu_bytes", "disabled"] = "cpu_bytes"
    # Snapshot throttling: refresh if either interval or write thresholds are satisfied (>0 to enable)
    snapshot_refresh_interval_seconds: float = 0.0
    snapshot_refresh_writes: int = 0
    # Controls training sample size for IVF: min(total, max(1024, nlist * ivf_train_factor))
    ivf_train_factor: int = 8


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
    # Mandatory shaping budgets
    token_budget: int = 2_000  # total tokens allowed across shaped results
    byte_budget: int = 32_000  # total UTF-8 bytes allowed across shaped results
    strict_highlights: bool = True  # Only accept lexical index highlight spans
    channel_weights: Mapping[str, float] = field(
        default_factory=lambda: {"bm25": 1.0, "splade": 1.0, "dense": 1.0}
    )
    mmr_pool_size: int = 200


@dataclass(frozen=True)
class RetrievalConfig:
    """Configuration for individual retrieval methods.

    Controls the behavior of each retrieval method (BM25, SPLADE, dense)
    including how many candidates each method should return.

    Attributes:
        bm25_top_k: Number of BM25 candidates to retrieve (50 default)
        splade_top_k: Number of SPLADE candidates to retrieve (50 default)
        dense_top_k: Number of dense vector candidates to retrieve (50 default)
        dense_overfetch_factor: Multiplier applied to oversampled dense requests (1.5 default)
        dense_oversample: Query-time oversample multiplier for dense retrieval (2.0 default)
        dense_score_floor: Minimum similarity score required for dense results before fusion (0.0 default)
        dense_calibration_batch_size: Optional batch size for validator dense calibration sweeps
        bm25_scoring: \"compat\" for legacy dot-product, \"true\" for Okapi BM25
        bm25_k1: Okapi BM25 k1 parameter (used when bm25_scoring == \"true\")
        bm25_b: Okapi BM25 b parameter (used when bm25_scoring == \"true\")
        executor_max_workers: Optional override for the service thread pool size

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
    # Over-fetch multiplier applied at query time (page_size * dense_oversample).
    dense_overfetch_factor: float = 1.5
    # Query-time oversample (separate from DenseIndexConfig.oversample used for IVF training)
    dense_oversample: float = 2.0
    # Minimum dense similarity threshold used prior to fusion (0 disables thresholding)
    dense_score_floor: float = 0.0
    # Optional batch size for dense calibration sweeps (None disables batching)
    dense_calibration_batch_size: Optional[int] = None
    # BM25 scoring mode (compat: legacy dot product, true: Okapi BM25)
    bm25_scoring: Literal["compat", "true"] = "compat"
    # Okapi BM25 hyperparameters (only used when bm25_scoring == "true")
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    # Optional override for HybridSearchService executor parallelism (None uses default)
    executor_max_workers: Optional[int] = None

    def __post_init__(self) -> None:
        max_workers = self.executor_max_workers
        if max_workers is None:
            return
        if not isinstance(max_workers, int):
            raise TypeError(
                "RetrievalConfig.executor_max_workers must be an int, "
                f"received {type(max_workers).__name__}"
            )
        if max_workers <= 0:
            raise ValueError("RetrievalConfig.executor_max_workers must be positive")


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
    def from_dict(payload: Mapping[str, Any]) -> "HybridSearchConfig":
        """Construct a config object from a dictionary payload.

        Args:
            payload: Nested mapping containing `chunking`, `dense`, `fusion`,
                and `retrieval` sections compatible with dataclass fields.

        Returns:
            Fully populated `HybridSearchConfig` instance.
        """
        if not isinstance(payload, Mapping):
            raise ValueError(
                "HybridSearchConfig.from_dict expected a mapping payload, "
                f"received {type(payload).__name__}"
            )

        def coerce_section(name: str) -> Dict[str, Any]:
            section = payload.get(name)
            if section is None:
                return {}
            if not isinstance(section, Mapping):
                raise ValueError(
                    f"HybridSearchConfig.{name} must be a mapping or null, "
                    f"received {type(section).__name__}"
                )
            return dict(section)

        chunking = ChunkingConfig(**coerce_section("chunking"))
        dense_payload = coerce_section("dense")
        alias_pairs = (
            ("gpu_default_null_stream", "gpu_use_default_null_stream"),
            (
                "gpu_default_null_stream_all_devices",
                "gpu_use_default_null_stream_all_devices",
            ),
        )
        for legacy_key, canonical_key in alias_pairs:
            if canonical_key in dense_payload:
                dense_payload.pop(legacy_key, None)
                continue
            if legacy_key in dense_payload:
                dense_payload[canonical_key] = dense_payload.pop(legacy_key)
        dense = DenseIndexConfig(**dense_payload)
        fusion = FusionConfig(**coerce_section("fusion"))
        retrieval = RetrievalConfig(**coerce_section("retrieval"))
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
        """Create a configuration manager bound to the supplied file path.

        Args:
            path: Filesystem path pointing to a JSON or YAML configuration file.

        Returns:
            None
        """

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
        """Load configuration data from the configured path (JSON or YAML).

        Args:
            None

        Returns:
            :class:`HybridSearchConfig` constructed from the on-disk payload.

        Raises:
            FileNotFoundError: If the configuration file is missing.
            ValueError: If JSON or YAML decoding fails or yields invalid structure.
        """

        if not self._path.exists():
            raise FileNotFoundError(f"Configuration file {self._path} not found")
        raw = self._path.read_text(encoding="utf-8")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = self._load_yaml(raw)
        return HybridSearchConfig.from_dict(payload)

    def _load_yaml(self, raw: str) -> Dict[str, Any]:
        """Parse YAML configuration content into a dictionary.

        Args:
            raw: Raw YAML string read from disk.

        Returns:
            Dictionary representation suitable for :class:`HybridSearchConfig.from_dict`.

        Raises:
            ValueError: If PyYAML is unavailable or the content does not define a mapping.
        """

        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - exercised in tests
            raise ValueError("YAML configuration requires PyYAML dependency") from exc

        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise ValueError(f"Failed to parse YAML configuration at {self._path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("YAML configuration must define a mapping")
        return data
