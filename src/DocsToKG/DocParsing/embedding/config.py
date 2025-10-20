"""
Embedding configuration dataclasses, presets, and validation helpers.

Embedding workloads combine dense LLM inference and sparse SPLADE encoding, so
this module captures all tunable knobs—batch sizes, tensor parallelism, cache
directories—in a single ``EmbedCfg`` dataclass. It also publishes curated
profiles (CPU, GPU) and validation routines that the CLI uses to translate
YAML/JSON inputs into runtime-ready settings.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, Mapping, Optional

from DocsToKG.DocParsing.cli_errors import EmbeddingCLIValidationError
from DocsToKG.DocParsing.config import ConfigLoadError, StageConfigBase
from DocsToKG.DocParsing.env import data_chunks, data_vectors, detect_data_root

SPLADE_SPARSITY_WARN_THRESHOLD_PCT: float = 1.0

EMBED_PROFILE_PRESETS: Dict[str, Dict[str, Any]] = {
    "cpu-small": {
        "batch_size_splade": 16,
        "batch_size_qwen": 16,
        "files_parallel": 1,
        "offline": True,
        "splade_attn": "auto",
        "splade_max_active_dims": 10000,
        "qwen_dtype": "float32",
        "qwen_dim": 2560,
        "qwen_quant": None,
        "tp": 1,
    },
    "gpu-default": {
        "batch_size_splade": 32,
        "batch_size_qwen": 64,
        "files_parallel": 1,
        "offline": False,
        "splade_attn": "flash",
        "splade_max_active_dims": 20000,
        "qwen_dtype": "bfloat16",
        "qwen_dim": 2560,
        "qwen_quant": None,
        "tp": 1,
    },
    "gpu-max": {
        "batch_size_splade": 64,
        "batch_size_qwen": 128,
        "files_parallel": 4,
        "tp": 2,
        "offline": False,
        "splade_attn": "flash",
        "splade_max_active_dims": 32000,
        "qwen_dtype": "bfloat16",
        "qwen_dim": 2560,
        "qwen_quant": "nf4",
    },
}


@dataclass
class EmbedCfg(StageConfigBase):
    """Stage configuration container for the embedding pipeline."""

    log_level: str = "INFO"
    data_root: Optional[Path] = None
    chunks_dir: Optional[Path] = None
    out_dir: Optional[Path] = None
    vector_format: str = "jsonl"
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    batch_size_splade: int = 32
    batch_size_qwen: int = 64
    splade_max_active_dims: Optional[int] = None
    splade_model_dir: Optional[Path] = None
    splade_attn: str = "auto"
    qwen_dtype: str = "bfloat16"
    qwen_quant: Optional[str] = None
    qwen_model_dir: Optional[Path] = None
    qwen_dim: int = 2560
    tp: int = 1
    sparsity_warn_threshold_pct: float = SPLADE_SPARSITY_WARN_THRESHOLD_PCT
    sparsity_report_top_n: int = 10
    files_parallel: int = 1
    validate_only: bool = False
    offline: bool = False
    resume: bool = False
    force: bool = False
    no_cache: bool = False
    shard_count: int = 1
    shard_index: int = 0
    # Provider-centric configuration
    embedding_device: str = "auto"
    embedding_dtype: str = "auto"
    embedding_batch_size: Optional[int] = None
    embedding_max_concurrency: Optional[int] = None
    embedding_normalize_l2: bool = True
    embedding_cache_dir: Optional[Path] = None
    embedding_telemetry_tags: Dict[str, str] = field(default_factory=dict)
    dense_backend: str = "qwen_vllm"
    dense_qwen_vllm_model_id: Optional[str] = None
    dense_qwen_vllm_download_dir: Optional[Path] = None
    dense_qwen_vllm_batch_size: Optional[int] = None
    dense_qwen_vllm_queue_depth: Optional[int] = None
    dense_qwen_vllm_quantization: Optional[str] = None
    dense_qwen_vllm_dimension: Optional[int] = None
    dense_tei_url: Optional[str] = None
    dense_tei_timeout_seconds: float = 30.0
    dense_tei_max_inflight: Optional[int] = None
    dense_sentence_transformers_model_id: Optional[str] = None
    dense_sentence_transformers_batch_size: Optional[int] = None
    dense_sentence_transformers_normalize_l2: Optional[bool] = None
    dense_fallback_backend: Optional[str] = None
    sparse_backend: str = "splade_st"
    sparse_splade_st_model_dir: Optional[Path] = None
    sparse_splade_st_batch_size: Optional[int] = None
    sparse_splade_st_attn_backend: Optional[str] = None
    sparse_splade_st_max_active_dims: Optional[int] = None
    lexical_backend: str = "local_bm25"
    lexical_local_bm25_k1: float = 1.5
    lexical_local_bm25_b: float = 0.75

    ENV_VARS: ClassVar[Dict[str, str]] = {
        "log_level": "DOCSTOKG_EMBED_LOG_LEVEL",
        "data_root": "DOCSTOKG_EMBED_DATA_ROOT",
        "chunks_dir": "DOCSTOKG_EMBED_CHUNKS_DIR",
        "out_dir": "DOCSTOKG_EMBED_OUT_DIR",
        "vector_format": "DOCSTOKG_EMBED_VECTOR_FORMAT",
        "bm25_k1": "DOCSTOKG_EMBED_BM25_K1",
        "bm25_b": "DOCSTOKG_EMBED_BM25_B",
        "batch_size_splade": "DOCSTOKG_EMBED_BATCH_SIZE_SPLADE",
        "batch_size_qwen": "DOCSTOKG_EMBED_BATCH_SIZE_QWEN",
        "splade_max_active_dims": "DOCSTOKG_EMBED_SPLADE_MAX_ACTIVE_DIMS",
        "splade_model_dir": "DOCSTOKG_SPLADE_DIR",
        "splade_attn": "DOCSTOKG_EMBED_SPLADE_ATTN",
        "qwen_dtype": "DOCSTOKG_EMBED_QWEN_DTYPE",
        "qwen_quant": "DOCSTOKG_EMBED_QWEN_QUANT",
        "qwen_model_dir": "DOCSTOKG_QWEN_DIR",
        "qwen_dim": "DOCSTOKG_EMBED_QWEN_DIM",
        "tp": "DOCSTOKG_EMBED_TP",
        "sparsity_warn_threshold_pct": "DOCSTOKG_EMBED_SPARSITY_WARN_PCT",
        "sparsity_report_top_n": "DOCSTOKG_EMBED_SPARSITY_REPORT_TOP_N",
        "files_parallel": "DOCSTOKG_EMBED_FILES_PARALLEL",
        "validate_only": "DOCSTOKG_EMBED_VALIDATE_ONLY",
        "offline": "DOCSTOKG_EMBED_OFFLINE",
        "resume": "DOCSTOKG_EMBED_RESUME",
        "force": "DOCSTOKG_EMBED_FORCE",
        "no_cache": "DOCSTOKG_EMBED_NO_CACHE",
        "shard_count": "DOCSTOKG_EMBED_SHARD_COUNT",
        "shard_index": "DOCSTOKG_EMBED_SHARD_INDEX",
        "config": "DOCSTOKG_EMBED_CONFIG",
        "embedding_device": "DOCSTOKG_EMBED_DEVICE",
        "embedding_dtype": "DOCSTOKG_EMBED_DTYPE",
        "embedding_batch_size": "DOCSTOKG_EMBED_BATCH_SIZE",
        "embedding_max_concurrency": "DOCSTOKG_EMBED_MAX_CONCURRENCY",
        "embedding_normalize_l2": "DOCSTOKG_EMBED_NORMALIZE_L2",
        "embedding_cache_dir": "DOCSTOKG_EMBED_CACHE_DIR",
        "dense_backend": "DOCSTOKG_DENSE_BACKEND",
        "dense_qwen_vllm_model_id": "DOCSTOKG_QWEN_MODEL_ID",
        "dense_qwen_vllm_download_dir": "DOCSTOKG_QWEN_DIR",
        "dense_qwen_vllm_batch_size": "DOCSTOKG_EMBED_BATCH_SIZE_QWEN",
        "dense_qwen_vllm_queue_depth": "DOCSTOKG_QWEN_QUEUE_DEPTH",
        "dense_qwen_vllm_quantization": "DOCSTOKG_QWEN_QUANT",
        "dense_qwen_vllm_dimension": "DOCSTOKG_EMBED_QWEN_DIM",
        "dense_tei_url": "DOCSTOKG_TEI_URL",
        "dense_tei_timeout_seconds": "DOCSTOKG_TEI_TIMEOUT",
        "dense_tei_max_inflight": "DOCSTOKG_TEI_MAX_INFLIGHT",
        "dense_sentence_transformers_model_id": "DOCSTOKG_SENTENCE_TRANSFORMERS_MODEL",
        "dense_sentence_transformers_batch_size": "DOCSTOKG_SENTENCE_TRANSFORMERS_BATCH",
        "dense_sentence_transformers_normalize_l2": "DOCSTOKG_SENTENCE_TRANSFORMERS_NORMALIZE",
        "dense_fallback_backend": "DOCSTOKG_DENSE_FALLBACK",
        "sparse_backend": "DOCSTOKG_SPARSE_BACKEND",
        "sparse_splade_st_model_dir": "DOCSTOKG_SPLADE_DIR",
        "sparse_splade_st_batch_size": "DOCSTOKG_EMBED_BATCH_SIZE_SPLADE",
        "sparse_splade_st_attn_backend": "DOCSTOKG_EMBED_SPLADE_ATTN",
        "sparse_splade_st_max_active_dims": "DOCSTOKG_EMBED_SPLADE_MAX_ACTIVE_DIMS",
        "lexical_backend": "DOCSTOKG_LEXICAL_BACKEND",
        "lexical_local_bm25_k1": "DOCSTOKG_EMBED_BM25_K1",
        "lexical_local_bm25_b": "DOCSTOKG_EMBED_BM25_B",
    }

    FIELD_PARSERS: ClassVar[Dict[str, Callable[[Any, Optional[Path]], Any]]] = {
        "config": StageConfigBase._coerce_optional_path,
        "log_level": StageConfigBase._coerce_str,
        "data_root": StageConfigBase._coerce_optional_path,
        "chunks_dir": StageConfigBase._coerce_path,
        "out_dir": StageConfigBase._coerce_path,
        "vector_format": StageConfigBase._coerce_str,
        "bm25_k1": StageConfigBase._coerce_float,
        "bm25_b": StageConfigBase._coerce_float,
        "batch_size_splade": StageConfigBase._coerce_int,
        "batch_size_qwen": StageConfigBase._coerce_int,
        "splade_max_active_dims": lambda value, base_dir: (
            None if value in (None, "", []) else StageConfigBase._coerce_int(value, base_dir)
        ),
        "splade_model_dir": StageConfigBase._coerce_optional_path,
        "splade_attn": StageConfigBase._coerce_str,
        "qwen_dtype": StageConfigBase._coerce_str,
        "qwen_quant": StageConfigBase._coerce_str,
        "qwen_model_dir": StageConfigBase._coerce_optional_path,
        "qwen_dim": StageConfigBase._coerce_int,
        "tp": StageConfigBase._coerce_int,
        "sparsity_warn_threshold_pct": StageConfigBase._coerce_float,
        "sparsity_report_top_n": StageConfigBase._coerce_int,
        "files_parallel": StageConfigBase._coerce_int,
        "validate_only": StageConfigBase._coerce_bool,
        "offline": StageConfigBase._coerce_bool,
        "resume": StageConfigBase._coerce_bool,
        "force": StageConfigBase._coerce_bool,
        "no_cache": StageConfigBase._coerce_bool,
        "shard_count": StageConfigBase._coerce_int,
        "shard_index": StageConfigBase._coerce_int,
        "embedding_device": StageConfigBase._coerce_str,
        "embedding_dtype": StageConfigBase._coerce_str,
        "embedding_batch_size": StageConfigBase._coerce_int,
        "embedding_max_concurrency": StageConfigBase._coerce_int,
        "embedding_normalize_l2": StageConfigBase._coerce_bool,
        "embedding_cache_dir": StageConfigBase._coerce_optional_path,
        "embedding_telemetry_tags": lambda value, _base: EmbedCfg._coerce_str_dict(value),
        "dense_backend": StageConfigBase._coerce_str,
        "dense_qwen_vllm_model_id": StageConfigBase._coerce_str,
        "dense_qwen_vllm_download_dir": StageConfigBase._coerce_optional_path,
        "dense_qwen_vllm_batch_size": StageConfigBase._coerce_int,
        "dense_qwen_vllm_queue_depth": StageConfigBase._coerce_int,
        "dense_qwen_vllm_quantization": StageConfigBase._coerce_str,
        "dense_qwen_vllm_dimension": StageConfigBase._coerce_int,
        "dense_tei_url": StageConfigBase._coerce_str,
        "dense_tei_timeout_seconds": StageConfigBase._coerce_float,
        "dense_tei_max_inflight": StageConfigBase._coerce_int,
        "dense_sentence_transformers_model_id": StageConfigBase._coerce_str,
        "dense_sentence_transformers_batch_size": StageConfigBase._coerce_int,
        "dense_sentence_transformers_normalize_l2": StageConfigBase._coerce_bool,
        "dense_fallback_backend": StageConfigBase._coerce_str,
        "sparse_backend": StageConfigBase._coerce_str,
        "sparse_splade_st_model_dir": StageConfigBase._coerce_optional_path,
        "sparse_splade_st_batch_size": StageConfigBase._coerce_int,
        "sparse_splade_st_attn_backend": StageConfigBase._coerce_str,
        "sparse_splade_st_max_active_dims": StageConfigBase._coerce_int,
        "lexical_backend": StageConfigBase._coerce_str,
        "lexical_local_bm25_k1": StageConfigBase._coerce_float,
        "lexical_local_bm25_b": StageConfigBase._coerce_float,
    }

    @classmethod
    def from_env(cls, defaults: Optional[Dict[str, Any]] = None) -> "EmbedCfg":
        """Construct configuration from environment variables."""

        cfg = cls(**(defaults or {}))
        cfg.apply_env()
        if cfg.data_root is None:
            fallback_root = os.getenv("DOCSTOKG_DATA_ROOT")
            if fallback_root:
                cfg.data_root = StageConfigBase._coerce_optional_path(fallback_root, None)
        cfg.finalize()
        return cfg

    @classmethod
    def from_args(
        cls,
        args: object,
        defaults: Optional[Dict[str, Any]] = None,
    ) -> "EmbedCfg":
        """Merge CLI arguments, configuration files, and environment variables."""

        cfg = cls.from_env(defaults=defaults)
        config_path = getattr(args, "config", None)
        if config_path:
            try:
                cfg.update_from_file(Path(config_path))
            except ConfigLoadError as exc:
                raise EmbeddingCLIValidationError(
                    option="--config",
                    message=str(exc),
                ) from exc
        cfg.apply_args(args)
        cfg.finalize()
        return cfg

    def finalize(self) -> None:
        """Normalise paths and casing after all sources have been applied."""

        if self.data_root is not None:
            resolved_root = StageConfigBase._coerce_optional_path(self.data_root, None)
        else:
            env_root = os.getenv("DOCSTOKG_DATA_ROOT")
            if env_root:
                resolved_root = StageConfigBase._coerce_optional_path(env_root, None)
            else:
                resolved_root = detect_data_root()
        self.data_root = resolved_root

        if self.chunks_dir is None:
            self.chunks_dir = data_chunks(resolved_root, ensure=False)
        else:
            self.chunks_dir = StageConfigBase._coerce_path(self.chunks_dir, None)

        if self.out_dir is None:
            self.out_dir = data_vectors(resolved_root, ensure=False)
        else:
            self.out_dir = StageConfigBase._coerce_path(self.out_dir, None)

        if self.splade_model_dir is not None:
            self.splade_model_dir = StageConfigBase._coerce_optional_path(
                self.splade_model_dir, None
            )
        if self.qwen_model_dir is not None:
            self.qwen_model_dir = StageConfigBase._coerce_optional_path(self.qwen_model_dir, None)
        if self.config is not None:
            self.config = StageConfigBase._coerce_optional_path(self.config, None)
        self.log_level = str(self.log_level or "INFO").upper()
        self.vector_format = str(self.vector_format or "jsonl").lower()
        if self.vector_format not in {"jsonl", "parquet"}:
            raise EmbeddingCLIValidationError(
                option="--format",
                message="must be one of: jsonl, parquet",
            )
        self.splade_attn = str(self.splade_attn or "auto").lower()
        if self.splade_max_active_dims in (None, "", []):
            self.splade_max_active_dims = None
        self.validate_only = bool(self.validate_only)
        self.offline = bool(self.offline)
        self.resume = bool(self.resume)
        self.force = bool(self.force)
        self.no_cache = bool(self.no_cache)

        if self.batch_size_splade < 1:
            raise EmbeddingCLIValidationError(
                option="--batch-size-splade",
                message="must be >= 1",
            )
        if self.batch_size_qwen < 1:
            raise EmbeddingCLIValidationError(
                option="--batch-size-qwen",
                message="must be >= 1",
            )
        if self.files_parallel < 1:
            raise EmbeddingCLIValidationError(
                option="--files-parallel",
                message="must be >= 1",
            )
        if self.tp < 1:
            raise EmbeddingCLIValidationError(option="--tp", message="must be >= 1")
        if self.shard_count < 1:
            raise EmbeddingCLIValidationError(option="--shard-count", message="must be >= 1")
        if not (0 <= self.shard_index < self.shard_count):
            raise EmbeddingCLIValidationError(
                option="--shard-index",
                message="must be between 0 and shard-count-1",
            )
        if self.splade_max_active_dims is not None and self.splade_max_active_dims < 1:
            raise EmbeddingCLIValidationError(
                option="--splade-max-active-dims",
                message="must be >= 1 when provided",
            )

        # Provider schema defaults & validation
        self.embedding_device = (self.embedding_device or "auto").strip() or "auto"
        self.embedding_dtype = (self.embedding_dtype or "auto").strip() or "auto"
        if self.embedding_batch_size is not None and self.embedding_batch_size < 1:
            raise EmbeddingCLIValidationError(
                option="embedding.batch_size",
                message="must be >= 1 when provided",
            )
        if self.embedding_max_concurrency is not None and self.embedding_max_concurrency < 1:
            raise EmbeddingCLIValidationError(
                option="embedding.max_concurrency",
                message="must be >= 1 when provided",
            )
        if self.embedding_max_concurrency is None:
            self.embedding_max_concurrency = self.files_parallel
        if self.embedding_cache_dir is not None:
            self.embedding_cache_dir = StageConfigBase._coerce_optional_path(
                self.embedding_cache_dir, None
            )
        self.embedding_telemetry_tags = self._coerce_str_dict(self.embedding_telemetry_tags)

        self.dense_backend = (self.dense_backend or "qwen_vllm").strip().lower() or "qwen_vllm"
        if self.dense_fallback_backend:
            self.dense_fallback_backend = self.dense_fallback_backend.strip().lower()
        if not self.dense_qwen_vllm_model_id:
            self.dense_qwen_vllm_model_id = "Qwen/Qwen3-Embedding-4B"
        if self.dense_qwen_vllm_download_dir is None and self.qwen_model_dir is not None:
            self.dense_qwen_vllm_download_dir = self.qwen_model_dir
        if self.dense_qwen_vllm_download_dir is not None:
            self.dense_qwen_vllm_download_dir = StageConfigBase._coerce_optional_path(
                self.dense_qwen_vllm_download_dir, None
            )
        if self.dense_qwen_vllm_batch_size is None:
            self.dense_qwen_vllm_batch_size = self.batch_size_qwen
        if self.dense_qwen_vllm_batch_size is not None and self.dense_qwen_vllm_batch_size < 1:
            raise EmbeddingCLIValidationError(
                option="dense.qwen_vllm.batch_size",
                message="must be >= 1 when provided",
            )
        if self.dense_qwen_vllm_queue_depth is not None and self.dense_qwen_vllm_queue_depth < 1:
            raise EmbeddingCLIValidationError(
                option="dense.qwen_vllm.queue_depth",
                message="must be >= 1 when provided",
            )
        if self.dense_qwen_vllm_dimension is None:
            self.dense_qwen_vllm_dimension = self.qwen_dim
        if self.dense_qwen_vllm_quantization is None:
            self.dense_qwen_vllm_quantization = self.qwen_quant

        self.sparse_backend = (self.sparse_backend or "splade_st").strip().lower() or "splade_st"
        if self.sparse_splade_st_model_dir is None and self.splade_model_dir is not None:
            self.sparse_splade_st_model_dir = self.splade_model_dir
        if self.sparse_splade_st_model_dir is not None:
            self.sparse_splade_st_model_dir = StageConfigBase._coerce_optional_path(
                self.sparse_splade_st_model_dir, None
            )
        if self.sparse_splade_st_batch_size is None:
            self.sparse_splade_st_batch_size = self.batch_size_splade
        if (
            self.sparse_splade_st_batch_size is not None
            and self.sparse_splade_st_batch_size < 1
        ):
            raise EmbeddingCLIValidationError(
                option="sparse.splade_st.batch_size",
                message="must be >= 1 when provided",
            )
        if self.sparse_splade_st_attn_backend is None:
            self.sparse_splade_st_attn_backend = self.splade_attn
        if self.sparse_splade_st_max_active_dims is None:
            self.sparse_splade_st_max_active_dims = self.splade_max_active_dims

        self.lexical_backend = (self.lexical_backend or "local_bm25").strip().lower()
        if self.lexical_backend != "local_bm25":
            raise EmbeddingCLIValidationError(
                option="lexical.backend",
                message="currently only 'local_bm25' is supported",
            )
        if self.lexical_local_bm25_k1 <= 0:
            raise EmbeddingCLIValidationError(
                option="lexical.local_bm25.k1",
                message="must be > 0",
            )
        if not (0 <= self.lexical_local_bm25_b <= 1):
            raise EmbeddingCLIValidationError(
                option="lexical.local_bm25.b",
                message="must be between 0 and 1",
            )

        # Backwards compatibility with legacy fields
        self.bm25_k1 = self.lexical_local_bm25_k1
        self.bm25_b = self.lexical_local_bm25_b
        self.batch_size_qwen = self.dense_qwen_vllm_batch_size or self.batch_size_qwen
        self.batch_size_splade = self.sparse_splade_st_batch_size or self.batch_size_splade
        self.splade_attn = self.sparse_splade_st_attn_backend or self.splade_attn
        self.splade_max_active_dims = (
            self.sparse_splade_st_max_active_dims or self.splade_max_active_dims
        )
        if self.dense_qwen_vllm_download_dir is not None:
            self.qwen_model_dir = self.dense_qwen_vllm_download_dir
        if self.sparse_splade_st_model_dir is not None:
            self.splade_model_dir = self.sparse_splade_st_model_dir

        deprecation_warnings = []
        if self.is_overridden("bm25_k1") and not self.is_overridden("lexical_local_bm25_k1"):
            deprecation_warnings.append(
                "--bm25-k1 is deprecated; use --lexical-local-bm25-k1"
            )
        if self.is_overridden("bm25_b") and not self.is_overridden("lexical_local_bm25_b"):
            deprecation_warnings.append("--bm25-b is deprecated; use --lexical-local-bm25-b")
        if self.is_overridden("batch_size_qwen") and not self.is_overridden(
            "dense_qwen_vllm_batch_size"
        ):
            deprecation_warnings.append(
                "--batch-size-qwen is deprecated; use --dense-qwen-batch-size"
            )
        if self.is_overridden("batch_size_splade") and not self.is_overridden(
            "sparse_splade_st_batch_size"
        ):
            deprecation_warnings.append(
                "--batch-size-splade is deprecated; use --sparse-splade-batch-size"
            )
        if self.is_overridden("splade_attn") and not self.is_overridden(
            "sparse_splade_st_attn_backend"
        ):
            deprecation_warnings.append(
                "--splade-attn is deprecated; use --sparse-splade-attn-backend"
            )
        if self.is_overridden("qwen_quant") and not self.is_overridden(
            "dense_qwen_vllm_quantization"
        ):
            deprecation_warnings.append(
                "--qwen-quant is deprecated; use --dense-qwen-quantization"
            )
        if self.is_overridden("qwen_dim") and not self.is_overridden(
            "dense_qwen_vllm_dimension"
        ):
            deprecation_warnings.append(
                "--qwen-dim is deprecated; use --dense-qwen-dimension"
            )

        for message in deprecation_warnings:
            print(f"[docparse embed] {message}", file=sys.stderr)

    @staticmethod
    def _coerce_str_dict(value: Any) -> Dict[str, str]:
        if value in (None, "", {}):
            return {}
        if isinstance(value, Mapping):
            return {str(k): str(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            merged: Dict[str, str] = {}
            for item in value:
                if not item:
                    continue
                text = str(item).strip()
                if not text:
                    continue
                if "=" in text:
                    key, val = text.split("=", 1)
                    key = key.strip()
                    val = val.strip()
                    if key:
                        merged[key] = val
                else:
                    raise ValueError(
                        "telemetry tags supplied via CLI must use KEY=VALUE format"
                    )
            return merged
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return {}
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                pairs: Dict[str, str] = {}
                for segment in text.split(","):
                    if "=" in segment:
                        key, val = segment.split("=", 1)
                        key = key.strip()
                        val = val.strip()
                        if key:
                            pairs[key] = val
                if pairs:
                    return pairs
                raise ValueError("telemetry tags must be JSON object or key=value pairs")
            else:
                if isinstance(parsed, Mapping):
                    return {str(k): str(v) for k, v in parsed.items()}
                raise ValueError("telemetry tags must be provided as a mapping")
        raise ValueError(f"Unsupported telemetry tag payload: {value!r}")

    def provider_settings(self) -> Dict[str, Any]:
        """Return provider-focused configuration for the embedding factory."""

        return {
            "embedding": {
                "device": self.embedding_device,
                "dtype": self.embedding_dtype,
                "batch_size": self.embedding_batch_size,
                "max_concurrency": self.embedding_max_concurrency,
                "normalize_l2": bool(self.embedding_normalize_l2),
                "offline": bool(self.offline),
                "cache_dir": self.embedding_cache_dir,
                "telemetry_tags": dict(self.embedding_telemetry_tags),
            },
            "dense": {
                "backend": self.dense_backend,
                "fallback": self.dense_fallback_backend,
                "qwen_vllm": {
                    "model_id": self.dense_qwen_vllm_model_id,
                    "download_dir": self.dense_qwen_vllm_download_dir,
                    "batch_size": self.dense_qwen_vllm_batch_size,
                    "queue_depth": self.dense_qwen_vllm_queue_depth,
                    "quantization": self.dense_qwen_vllm_quantization,
                    "dimension": self.dense_qwen_vllm_dimension,
                },
                "tei": {
                    "url": self.dense_tei_url,
                    "timeout_seconds": self.dense_tei_timeout_seconds,
                    "max_inflight": self.dense_tei_max_inflight,
                },
                "sentence_transformers": {
                    "model_id": self.dense_sentence_transformers_model_id,
                    "batch_size": self.dense_sentence_transformers_batch_size,
                    "normalize_l2": self.dense_sentence_transformers_normalize_l2,
                },
            },
            "sparse": {
                "backend": self.sparse_backend,
                "splade_st": {
                    "model_dir": self.sparse_splade_st_model_dir,
                    "batch_size": self.sparse_splade_st_batch_size,
                    "attn_backend": self.sparse_splade_st_attn_backend,
                    "max_active_dims": self.sparse_splade_st_max_active_dims,
                },
            },
            "lexical": {
                "backend": self.lexical_backend,
                "local_bm25": {
                    "k1": self.lexical_local_bm25_k1,
                    "b": self.lexical_local_bm25_b,
                },
            },
        }


__all__ = ["EmbedCfg", "EMBED_PROFILE_PRESETS", "SPLADE_SPARSITY_WARN_THRESHOLD_PCT"]
