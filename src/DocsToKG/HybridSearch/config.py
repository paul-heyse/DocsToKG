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
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Literal

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

    Key fields:
    - ``max_tokens``: Maximum number of tokens per chunk (800 default).
    - ``overlap``: Number of tokens to overlap between chunks (150 default).

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

    Key tunables include:
    - Index topology: ``index_type``, ``nlist``, ``nprobe``, ``pq_m``, ``pq_bits``, and ``oversample``.
    - GPU deployment knobs such as ``device``, replication controls, memory pool sizing, and 32-bit index toggles.
    - Precision/layout options (``ivfpq_use_precomputed``, ``ivfpq_float16_lut``, ``interleaved_layout``, ``flat_use_fp16``).
    - Persistence and snapshot settings (``persist_mode``, snapshot refresh intervals, ``ivf_train_factor``).
    - Optional accelerators and safety valves (cuVS toggles, dedupe thresholds, forced 64-bit IDs).

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
    replication_gpu_ids: tuple[int, ...] | None = None
    gpu_temp_memory_bytes: int | None = None
    gpu_pinned_memory_bytes: int | None = None
    gpu_indices_32_bit: bool = True
    expected_ntotal: int = 0
    rebuild_delete_threshold: int = 10000
    force_64bit_ids: bool = False
    interleaved_layout: bool = True
    flat_use_fp16: bool = False
    use_cuvs: bool | None = None
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

    Key fields:
    - ``k0``: RRF (Reciprocal Rank Fusion) parameter (60.0 default).
    - ``mmr_lambda`` / ``enable_mmr``: MMR diversification controls.
    - ``cosine_dedupe_threshold``: Cosine similarity threshold for deduplication (0.98 default).
    - ``max_chunks_per_doc``: Maximum chunks to return per document (3 default).

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

    Key fields:
    - ``bm25_top_k`` / ``splade_top_k`` / ``dense_top_k``: Candidate counts per retrieval modality.
    - ``dense_overfetch_factor`` / ``dense_oversample``: Dense oversampling controls.
    - ``dense_score_floor`` / ``dense_calibration_batch_size``: Dense filtering and calibration knobs.
    - ``bm25_scoring`` / ``bm25_k1`` / ``bm25_b``: BM25 scoring configuration.
    - ``executor_max_workers``: Optional override for service thread pool size.

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
    dense_calibration_batch_size: int | None = None
    # BM25 scoring mode (compat: legacy dot product, true: Okapi BM25)
    bm25_scoring: Literal["compat", "true"] = "compat"
    # Okapi BM25 hyperparameters (only used when bm25_scoring == "true")
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    # Optional override for HybridSearchService executor parallelism (None uses default)
    executor_max_workers: int | None = None

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

    Components:
    - ``chunking``: Document chunking configuration.
    - ``dense``: Dense vector indexing configuration.
    - ``fusion``: Result fusion and ranking configuration.
    - ``retrieval``: Individual retrieval method configuration.

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
    def from_dict(payload: Mapping[str, Any]) -> HybridSearchConfig:
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

        def coerce_section(name: str) -> dict[str, Any]:
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

    Internals:
    - ``_path``: Path to the JSON/YAML configuration file.
    - ``_lock``: Threading lock guarding concurrent reloads.
    - ``_config``: Cached :class:`HybridSearchConfig` instance.

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

    def _load_yaml(self, raw: str) -> dict[str, Any]:
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
