"""Configuration objects and presets for the embedding stage."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, Optional

from DocsToKG.DocParsing.config import StageConfigBase
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
            cfg.update_from_file(Path(config_path))
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
        self.splade_attn = str(self.splade_attn or "auto").lower()
        if self.splade_max_active_dims in (None, "", []):
            self.splade_max_active_dims = None
        self.validate_only = bool(self.validate_only)
        self.offline = bool(self.offline)
        self.resume = bool(self.resume)
        self.force = bool(self.force)
        self.no_cache = bool(self.no_cache)

        if self.batch_size_splade < 1:
            raise ValueError("batch_size_splade must be >= 1")
        if self.batch_size_qwen < 1:
            raise ValueError("batch_size_qwen must be >= 1")
        if self.files_parallel < 1:
            raise ValueError("files_parallel must be >= 1")
        if self.tp < 1:
            raise ValueError("tp must be >= 1")
        if self.shard_count < 1:
            raise ValueError("shard_count must be >= 1")
        if not (0 <= self.shard_index < self.shard_count):
            raise ValueError("shard_index must be in [0, shard_count)")
        if self.splade_max_active_dims is not None and self.splade_max_active_dims < 1:
            raise ValueError("splade_max_active_dims must be >= 1 when provided")


__all__ = ["EmbedCfg", "EMBED_PROFILE_PRESETS", "SPLADE_SPARSITY_WARN_THRESHOLD_PCT"]
