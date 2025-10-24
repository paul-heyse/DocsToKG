# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.DocParsing.settings",
#   "purpose": "Unified Pydantic v2 Settings for DocParsing configuration management.",
#   "sections": [
#     {
#       "id": "loglevel",
#       "name": "LogLevel",
#       "anchor": "class-loglevel",
#       "kind": "class"
#     },
#     {
#       "id": "logformat",
#       "name": "LogFormat",
#       "anchor": "class-logformat",
#       "kind": "class"
#     },
#     {
#       "id": "runnerpolicy",
#       "name": "RunnerPolicy",
#       "anchor": "class-runnerpolicy",
#       "kind": "class"
#     },
#     {
#       "id": "runnerschedule",
#       "name": "RunnerSchedule",
#       "anchor": "class-runnerschedule",
#       "kind": "class"
#     },
#     {
#       "id": "runneradaptive",
#       "name": "RunnerAdaptive",
#       "anchor": "class-runneradaptive",
#       "kind": "class"
#     },
#     {
#       "id": "doctagsmode",
#       "name": "DoctagsMode",
#       "anchor": "class-doctagsmode",
#       "kind": "class"
#     },
#     {
#       "id": "format",
#       "name": "Format",
#       "anchor": "class-format",
#       "kind": "class"
#     },
#     {
#       "id": "densebackend",
#       "name": "DenseBackend",
#       "anchor": "class-densebackend",
#       "kind": "class"
#     },
#     {
#       "id": "teicompression",
#       "name": "TeiCompression",
#       "anchor": "class-teicompression",
#       "kind": "class"
#     },
#     {
#       "id": "attnbackend",
#       "name": "AttnBackend",
#       "anchor": "class-attnbackend",
#       "kind": "class"
#     },
#     {
#       "id": "spladenorm",
#       "name": "SpladeNorm",
#       "anchor": "class-spladenorm",
#       "kind": "class"
#     },
#     {
#       "id": "appcfg",
#       "name": "AppCfg",
#       "anchor": "class-appcfg",
#       "kind": "class"
#     },
#     {
#       "id": "runnercfg",
#       "name": "RunnerCfg",
#       "anchor": "class-runnercfg",
#       "kind": "class"
#     },
#     {
#       "id": "doctagscfg",
#       "name": "DocTagsCfg",
#       "anchor": "class-doctagscfg",
#       "kind": "class"
#     },
#     {
#       "id": "chunkcfg",
#       "name": "ChunkCfg",
#       "anchor": "class-chunkcfg",
#       "kind": "class"
#     },
#     {
#       "id": "qwenvllmcfg",
#       "name": "QwenVLLMCfg",
#       "anchor": "class-qwenvllmcfg",
#       "kind": "class"
#     },
#     {
#       "id": "teicfg",
#       "name": "TEICfg",
#       "anchor": "class-teicfg",
#       "kind": "class"
#     },
#     {
#       "id": "sentencetransformerscfg",
#       "name": "SentenceTransformersCfg",
#       "anchor": "class-sentencetransformerscfg",
#       "kind": "class"
#     },
#     {
#       "id": "densecfg",
#       "name": "DenseCfg",
#       "anchor": "class-densecfg",
#       "kind": "class"
#     },
#     {
#       "id": "spladestcfg",
#       "name": "SpladeSTCfg",
#       "anchor": "class-spladestcfg",
#       "kind": "class"
#     },
#     {
#       "id": "sparsecfg",
#       "name": "SparseCfg",
#       "anchor": "class-sparsecfg",
#       "kind": "class"
#     },
#     {
#       "id": "localbm25cfg",
#       "name": "LocalBM25Cfg",
#       "anchor": "class-localbm25cfg",
#       "kind": "class"
#     },
#     {
#       "id": "lexicalcfg",
#       "name": "LexicalCfg",
#       "anchor": "class-lexicalcfg",
#       "kind": "class"
#     },
#     {
#       "id": "vectorscfg",
#       "name": "VectorsCfg",
#       "anchor": "class-vectorscfg",
#       "kind": "class"
#     },
#     {
#       "id": "embedcfg",
#       "name": "EmbedCfg",
#       "anchor": "class-embedcfg",
#       "kind": "class"
#     },
#     {
#       "id": "settings",
#       "name": "Settings",
#       "anchor": "class-settings",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""
Unified Pydantic v2 Settings for DocParsing configuration management.

This module provides typed, reproducible settings for all DocParsing stages
using Pydantic v2 BaseSettings with a consistent ENV prefix (``DOCSTOKG_``) and
field validators. The models support layering (CLI > ENV > profile > defaults)
and provide clear separation of concerns across stages.

NAVMAP:
- AppCfg: Global application-level configuration
- RunnerCfg: Shared execution runner settings
- DocTagsCfg: PDF/HTML → DocTags conversion stage
- ChunkCfg: DocTags → Chunks chunking stage
- EmbedCfg: Chunks → Vectors embedding stage (with provider subtrees)
- Settings: Root aggregation of all configs
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_core import PydanticUndefined

from DocsToKG.DocParsing.core.discovery import (
    DEFAULT_DISCOVERY_IGNORE,
    parse_discovery_ignore,
)

# ============================================================================
# Enums for validated choices
# ============================================================================


class LogLevel(str, Enum):
    """Supported logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class LogFormat(str, Enum):
    """Supported log output formats."""

    CONSOLE = "console"
    JSON = "json"


class RunnerPolicy(str, Enum):
    """Execution policy for the unified runner."""

    IO = "io"
    CPU = "cpu"
    GPU = "gpu"


class RunnerSchedule(str, Enum):
    """Task scheduling strategy."""

    FIFO = "fifo"
    SJF = "sjf"  # shortest-job-first


class RunnerAdaptive(str, Enum):
    """Adaptive worker tuning modes."""

    OFF = "off"
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"


class DoctagsMode(str, Enum):
    """DocTags conversion mode."""

    AUTO = "auto"
    PDF = "pdf"
    HTML = "html"


class Format(str, Enum):
    """Data format choices."""

    PARQUET = "parquet"
    JSONL = "jsonl"


class DenseBackend(str, Enum):
    """Dense embedding provider backends."""

    QWEN_VLLM = "qwen_vllm"
    TEI = "tei"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    NONE = "none"


class TeiCompression(str, Enum):
    """TEI HTTP compression options."""

    AUTO = "auto"
    GZIP = "gzip"
    NONE = "none"


class AttnBackend(str, Enum):
    """Attention backend for transformers."""

    AUTO = "auto"
    SDPA = "sdpa"
    EAGER = "eager"
    FLASH_ATTENTION_2 = "flash_attention_2"


class SpladeNorm(str, Enum):
    """SPLADE normalization modes."""

    NONE = "none"
    L2 = "l2"


# ============================================================================
# Global configuration (AppCfg)
# ============================================================================


class AppCfg(BaseSettings):
    """Global application-level configuration."""

    model_config = SettingsConfigDict(
        env_prefix="DOCSTOKG_",
        case_sensitive=False,
        extra="ignore",
    )

    profile: str | None = Field(None, description="Profile name to load from docstokg.toml/.yaml")
    data_root: Path = Field(
        Path("Data"), description="Root directory for all data (DocTags/Chunks/Vectors/Manifests)"
    )
    manifests_root: Path | None = Field(
        None, description="Override Manifests directory (default: data_root/Manifests)"
    )
    models_root: Path = Field(
        Path("~/.cache/docstokg/models").expanduser(),
        description="Local cache for models/tokenizers",
    )
    log_level: LogLevel = Field(LogLevel.INFO, description="Root logging level")
    log_format: LogFormat = Field(
        LogFormat.CONSOLE, description="Pretty console or structured JSON"
    )
    log_dir: Path = Field(Path("Data/Logs"), description="Directory for structured logs (JSONL)")
    metrics_enabled: bool = Field(False, description="Expose Prometheus metrics HTTP endpoint")
    metrics_port: int = Field(
        9108,
        description="Prometheus metrics port",
        ge=1,
        le=65535,
    )
    tracing_enabled: bool = Field(False, description="Enable OpenTelemetry export")
    strict_config: bool = Field(
        True, description="Treat unknown/deprecated keys as errors (false = warn)"
    )
    random_seed: int | None = Field(None, description="Random seed for reproducibility")
    discovery_ignore: tuple[str, ...] = Field(
        (),
        description=(
            "Glob patterns ignored during discovery (use !pattern to drop defaults)."
        ),
    atomic_writes: bool = Field(
        False,
        description="Write outputs via temporary files and atomic os.replace",
    )
    retain_lock_files: bool = Field(
        False,
        description="Retain .lock sentinels after releasing FileLock handles",
    )

    @field_validator("data_root", "manifests_root", "models_root", "log_dir", mode="before")
    @classmethod
    def expand_paths(cls, v: Any) -> Any:
        """Expand user home and make absolute."""
        if v is None:
            return None
        if isinstance(v, str):
            return Path(v).expanduser().resolve()
        if isinstance(v, Path):
            return v.expanduser().resolve()
        return v

    @field_validator("metrics_port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate port is in valid range."""
        if not (1 <= v <= 65535):
            raise ValueError(f"Port must be 1-65535, got {v}")
        return v

    @field_validator("discovery_ignore", mode="before")
    @classmethod
    def normalize_discovery_ignore(cls, value: Any) -> tuple[str, ...]:
        """Normalize discovery ignore patterns with defaults and modifiers."""

        if value is None or value is PydanticUndefined or value == ():
            return parse_discovery_ignore(None, base=DEFAULT_DISCOVERY_IGNORE)
        return parse_discovery_ignore(value, base=DEFAULT_DISCOVERY_IGNORE)


# ============================================================================
# Runner configuration (RunnerCfg)
# ============================================================================


class RunnerCfg(BaseSettings):
    """Shared execution runner configuration."""

    model_config = SettingsConfigDict(
        env_prefix="DOCSTOKG_RUNNER_",
        case_sensitive=False,
        extra="ignore",
    )

    policy: RunnerPolicy = Field(RunnerPolicy.CPU, description="Execution policy (io/cpu/gpu)")
    workers: int = Field(
        8,
        description="Max parallel files",
        ge=1,
    )
    schedule: RunnerSchedule = Field(RunnerSchedule.FIFO, description="Task scheduling (fifo/sjf)")
    retries: int = Field(
        0,
        description="Retry attempts for retryable errors",
        ge=0,
    )
    retry_backoff_s: float = Field(
        0.5,
        description="Exponential backoff base (seconds)",
        ge=0,
    )
    per_item_timeout_s: float = Field(
        0,
        description="Per-item timeout (0 = disabled)",
        ge=0,
    )
    error_budget: int = Field(
        0,
        description="Stop after N failures (0 = stop on first)",
        ge=0,
    )
    max_queue: int = Field(
        16,
        description="Submission backpressure depth",
        ge=1,
    )
    adaptive: RunnerAdaptive = Field(
        RunnerAdaptive.OFF, description="Auto-tune workers (off/conservative/aggressive)"
    )
    fingerprinting: bool = Field(True, description="Use *.fp.json for exact resume")


# ============================================================================
# DocTags stage configuration (DocTagsCfg)
# ============================================================================


class DocTagsCfg(BaseSettings):
    """PDF/HTML → DocTags conversion stage configuration."""

    model_config = SettingsConfigDict(
        env_prefix="DOCSTOKG_DOCTAGS_",
        case_sensitive=False,
        extra="ignore",
    )

    input_dir: Path = Field(Path("Data/Raw"), description="Input directory for PDF/HTML files")
    output_dir: Path = Field(Path("Data/DocTags"), description="Output directory for DocTags JSONL")
    mode: DoctagsMode = Field(DoctagsMode.AUTO, description="Conversion mode (auto/pdf/html)")
    model_id: str = Field("granite-docling-258M", description="DocTags model ID (PDF path)")
    vllm_wait_timeout_s: float = Field(
        60.0,
        description="Wait timeout for auxiliary VLM (seconds)",
        ge=0,
    )
    resume: bool = Field(True, description="Skip when outputs+fingerprint match")
    force: bool = Field(False, description="Recompute and atomically replace")

    @field_validator("input_dir", "output_dir", mode="before")
    @classmethod
    def expand_paths(cls, v: Any) -> Any:
        """Expand paths."""
        if isinstance(v, str):
            return Path(v).expanduser().resolve()
        if isinstance(v, Path):
            return v.expanduser().resolve()
        return v

    @model_validator(mode="after")
    def validate_resume_force(self) -> DocTagsCfg:
        """Ensure resume and force are not both True."""
        if self.resume and self.force:
            raise ValueError("Cannot set both resume=True and force=True; force takes precedence")
        return self


# ============================================================================
# Chunk stage configuration (ChunkCfg)
# ============================================================================


class ChunkCfg(BaseSettings):
    """DocTags → Chunks chunking stage configuration."""

    model_config = SettingsConfigDict(
        env_prefix="DOCSTOKG_CHUNK_",
        case_sensitive=False,
        extra="ignore",
    )

    input_dir: Path = Field(Path("Data/DocTags"), description="Source DocTags directory")
    output_dir: Path = Field(Path("Data/Chunks"), description="Target chunks dataset root")
    format: Format = Field(Format.PARQUET, description="Output format (parquet/jsonl)")
    min_tokens: int = Field(
        120,
        description="Minimum tokens per chunk",
        ge=1,
    )
    max_tokens: int = Field(
        800,
        description="Maximum tokens per chunk",
        ge=1,
    )
    tokenizer_model: str = Field(
        "cl100k_base", description="Tokenizer model ID to align with dense embedder"
    )
    shard_count: int = Field(
        1,
        description="Deterministic sharding count",
        ge=1,
    )
    shard_index: int = Field(
        0,
        description="Shard index (0-based)",
        ge=0,
    )
    resume: bool = Field(True, description="Skip when outputs+fingerprint match")
    force: bool = Field(False, description="Recompute and atomically replace")

    @field_validator("input_dir", "output_dir", mode="before")
    @classmethod
    def expand_paths(cls, v: Any) -> Any:
        """Expand paths."""
        if isinstance(v, str):
            return Path(v).expanduser().resolve()
        if isinstance(v, Path):
            return v.expanduser().resolve()
        return v

    @model_validator(mode="after")
    def validate_tokens(self) -> ChunkCfg:
        """Ensure min_tokens <= max_tokens."""
        if self.min_tokens > self.max_tokens:
            raise ValueError(
                f"min_tokens ({self.min_tokens}) must be <= max_tokens ({self.max_tokens})"
            )
        return self


# ============================================================================
# Dense embedding provider configs (nested under EmbedCfg)
# ============================================================================


class QwenVLLMCfg(BaseModel):
    """Qwen embedding via vLLM."""

    model_id: str = "Qwen2-7B-Embedding"
    batch_size: int = 64
    dtype: str = "auto"  # auto|float16|bfloat16
    device: str = "cuda:0"
    tensor_parallelism: int = 1
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.9
    download_dir: Path | None = None
    warmup: bool = True

    @field_validator("gpu_memory_utilization")
    @classmethod
    def validate_gpu_mem(cls, v: float) -> float:
        """Validate GPU memory utilization."""
        if not (0.1 <= v <= 0.99):
            raise ValueError(f"GPU memory utilization must be 0.1-0.99, got {v}")
        return v


class TEICfg(BaseModel):
    """Text Embeddings Inference HTTP backend."""

    url: str | None = None  # Required when backend=tei
    api_key: str | None = None
    timeout_s: float = 30.0
    verify_tls: bool = True
    compression: TeiCompression = TeiCompression.AUTO
    max_inflight_requests: int = 32
    batch_size: int = 64

    @model_validator(mode="after")
    def validate_tei_url(self) -> TEICfg:
        """TEI URL required when TEI backend is active."""
        # Note: full validation happens in parent when backend=tei
        return self


class SentenceTransformersCfg(BaseModel):
    """Sentence-Transformers dense embedding."""

    model_id: str | None = None
    revision: str | None = None
    device: str = "auto"
    dtype: str = "auto"  # auto|float32|float16|bfloat16
    batch_size: int = 64
    max_seq_length: int | None = None
    trust_remote_code: bool = False
    use_memory_map: bool = True
    intra_op_threads: int | None = None


class DenseCfg(BaseModel):
    """Dense embedding provider configuration."""

    backend: DenseBackend = DenseBackend.QWEN_VLLM
    qwen_vllm: QwenVLLMCfg = Field(default_factory=QwenVLLMCfg)
    tei: TEICfg = Field(default_factory=TEICfg)
    sentence_transformers: SentenceTransformersCfg = Field(default_factory=SentenceTransformersCfg)


# ============================================================================
# Sparse embedding provider configs (nested under EmbedCfg)
# ============================================================================


class SpladeSTCfg(BaseModel):
    """SPLADE sparse embedding via Sentence-Transformers."""

    model_id: str = "naver/splade-cocondenser-ensembledistil"
    revision: str | None = None
    device: str = "auto"
    dtype: str = "auto"  # auto|float32|float16|bfloat16
    batch_size: int = 64
    topk_per_doc: int = 0  # 0 = all
    prune_below: float = 0.0
    normalize_doclen: SpladeNorm = SpladeNorm.L2
    tokenizer_id: str | None = None
    attn_backend: AttnBackend = AttnBackend.AUTO


class SparseCfg(BaseModel):
    """Sparse embedding provider configuration."""

    backend: str = "splade_st"
    splade_st: SpladeSTCfg = Field(default_factory=SpladeSTCfg)


# ============================================================================
# Lexical embedding provider configs (nested under EmbedCfg)
# ============================================================================


class LocalBM25Cfg(BaseModel):
    """Local BM25 lexical search."""

    k1: float = 1.5
    b: float = 0.75
    stopwords: str = "none"  # none|english|PATH
    tokenizer: str = "simple"  # simple|spacy_en|regexp:...
    min_df: int = 1
    max_df_ratio: float = 1.0

    @field_validator("k1")
    @classmethod
    def validate_k1(cls, v: float) -> float:
        """k1 must be > 0."""
        if v <= 0:
            raise ValueError(f"BM25 k1 must be > 0, got {v}")
        return v

    @field_validator("b")
    @classmethod
    def validate_b(cls, v: float) -> float:
        """b must be in [0, 1]."""
        if not (0 <= v <= 1):
            raise ValueError(f"BM25 b must be in [0, 1], got {v}")
        return v


class LexicalCfg(BaseModel):
    """Lexical embedding provider configuration."""

    backend: str = "local_bm25"
    local_bm25: LocalBM25Cfg = Field(default_factory=LocalBM25Cfg)


# ============================================================================
# Vectors format configuration (nested under EmbedCfg)
# ============================================================================


class VectorsCfg(BaseModel):
    """Vector output format configuration."""

    format: Format = Format.PARQUET
    expected_dim: int | None = None  # For validation


# ============================================================================
# Embedding stage configuration (EmbedCfg)
# ============================================================================


class EmbedCfg(BaseSettings):
    """Chunks → Vectors embedding stage configuration."""

    model_config = SettingsConfigDict(
        env_prefix="DOCSTOKG_EMBED_",
        case_sensitive=False,
        extra="ignore",
    )

    input_chunks_dir: Path = Field(Path("Data/Chunks"), description="Source chunks dataset")
    output_vectors_dir: Path = Field(
        Path("Data/Vectors"), description="Target vectors dataset root"
    )
    vectors: VectorsCfg = Field(default_factory=VectorsCfg, description="Vector output format")

    # Family toggles
    enable_dense: bool = Field(True, description="Generate dense vectors")
    enable_sparse: bool = Field(True, description="Generate SPLADE sparse vectors")
    enable_lexical: bool = Field(True, description="Generate BM25 lexical vectors")

    # Provider configurations
    dense: DenseCfg = Field(default_factory=DenseCfg, description="Dense embedding provider config")
    sparse: SparseCfg = Field(
        default_factory=SparseCfg, description="Sparse embedding provider config"
    )
    lexical: LexicalCfg = Field(
        default_factory=LexicalCfg, description="Lexical embedding provider config"
    )

    # Workflow
    plan_only: bool = Field(False, description="Plan without writing vectors")
    resume: bool = Field(True, description="Skip when outputs+fingerprint match")
    force: bool = Field(False, description="Recompute and atomically replace")

    @field_validator("input_chunks_dir", "output_vectors_dir", mode="before")
    @classmethod
    def expand_paths(cls, v: Any) -> Any:
        """Expand paths."""
        if isinstance(v, str):
            return Path(v).expanduser().resolve()
        if isinstance(v, Path):
            return v.expanduser().resolve()
        return v

    @model_validator(mode="after")
    def validate_tei_url_if_backend(self) -> EmbedCfg:
        """TEI URL required when backend=tei."""
        if self.dense.backend == DenseBackend.TEI and not self.dense.tei.url:
            raise ValueError(
                "TEI URL required when dense.backend=tei; set via --tei-url or DOCSTOKG_TEI_URL"
            )
        return self


# ============================================================================
# Root Settings aggregation
# ============================================================================


class Settings(BaseModel):
    """Aggregated configuration for all DocParsing stages."""

    app: AppCfg = Field(default_factory=AppCfg)
    runner: RunnerCfg = Field(default_factory=RunnerCfg)
    doctags: DocTagsCfg = Field(default_factory=DocTagsCfg)
    chunk: ChunkCfg = Field(default_factory=ChunkCfg)
    embed: EmbedCfg = Field(default_factory=EmbedCfg)

    def model_dump_redacted(self, **kwargs: Any) -> dict[str, Any]:
        """Dump config with sensitive fields redacted."""
        import re

        result = self.model_dump(**kwargs)
        pattern = re.compile(r"(?i)(token|secret|password|api[_-]?key)")

        def redact_dict(d: dict[str, Any]) -> dict[str, Any]:
            for key, val in d.items():
                if pattern.search(key):
                    d[key] = "***REDACTED***"
                elif isinstance(val, dict):
                    redact_dict(val)
            return d

        return redact_dict(result)

    def compute_stage_hashes(self) -> dict[str, str]:
        """Compute stable content hashes per stage."""
        hashes = {}
        for stage_name in ["app", "runner", "doctags", "chunk", "embed"]:
            stage_cfg = getattr(self, stage_name)
            # Use exclude_none and sort keys for determinism
            data = stage_cfg.model_dump(exclude_none=True)
            json_str = json.dumps(data, sort_keys=True, default=str)
            hashes[stage_name] = hashlib.sha256(json_str.encode()).hexdigest()[:8]
        return hashes


__all__ = [
    "Settings",
    "AppCfg",
    "RunnerCfg",
    "DocTagsCfg",
    "ChunkCfg",
    "EmbedCfg",
    "DenseCfg",
    "SparseCfg",
    "LexicalCfg",
    "VectorsCfg",
    "QwenVLLMCfg",
    "TEICfg",
    "SentenceTransformersCfg",
    "SpladeSTCfg",
    "LocalBM25Cfg",
    # Enums
    "LogLevel",
    "LogFormat",
    "RunnerPolicy",
    "RunnerSchedule",
    "RunnerAdaptive",
    "DoctagsMode",
    "Format",
    "DenseBackend",
    "TeiCompression",
    "AttnBackend",
    "SpladeNorm",
]
