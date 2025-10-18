"""Configuration objects for the chunking stage."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, Optional

from DocsToKG.DocParsing.config import StageConfigBase
from DocsToKG.DocParsing.core import DEFAULT_SERIALIZER_PROVIDER, DEFAULT_TOKENIZER
from DocsToKG.DocParsing.env import data_chunks, data_doctags, detect_data_root

SOFT_BARRIER_MARGIN: int = 64


@dataclass
class ChunkerCfg(StageConfigBase):
    """Configuration values for the chunking stage."""

    log_level: str = "INFO"
    data_root: Optional[Path] = None
    in_dir: Optional[Path] = None
    out_dir: Optional[Path] = None
    min_tokens: int = 256
    max_tokens: int = 512
    shard_count: int = 1
    shard_index: int = 0
    tokenizer_model: str = DEFAULT_TOKENIZER
    soft_barrier_margin: int = SOFT_BARRIER_MARGIN
    structural_markers: Optional[Path] = None
    serializer_provider: str = DEFAULT_SERIALIZER_PROVIDER
    workers: int = 1
    validate_only: bool = False
    resume: bool = False
    force: bool = False
    inject_anchors: bool = False

    ENV_VARS: ClassVar[Dict[str, str]] = {
        "log_level": "DOCSTOKG_CHUNK_LOG_LEVEL",
        "data_root": "DOCSTOKG_CHUNK_DATA_ROOT",
        "in_dir": "DOCSTOKG_CHUNK_IN_DIR",
        "out_dir": "DOCSTOKG_CHUNK_OUT_DIR",
        "min_tokens": "DOCSTOKG_CHUNK_MIN_TOKENS",
        "max_tokens": "DOCSTOKG_CHUNK_MAX_TOKENS",
        "shard_count": "DOCSTOKG_CHUNK_SHARD_COUNT",
        "shard_index": "DOCSTOKG_CHUNK_SHARD_INDEX",
        "tokenizer_model": "DOCSTOKG_CHUNK_TOKENIZER",
        "soft_barrier_margin": "DOCSTOKG_CHUNK_SOFT_BARRIER_MARGIN",
        "structural_markers": "DOCSTOKG_CHUNK_STRUCTURAL_MARKERS",
        "serializer_provider": "DOCSTOKG_CHUNK_SERIALIZER_PROVIDER",
        "workers": "DOCSTOKG_CHUNK_WORKERS",
        "validate_only": "DOCSTOKG_CHUNK_VALIDATE_ONLY",
        "resume": "DOCSTOKG_CHUNK_RESUME",
        "force": "DOCSTOKG_CHUNK_FORCE",
        "inject_anchors": "DOCSTOKG_CHUNK_INJECT_ANCHORS",
        "config": "DOCSTOKG_CHUNK_CONFIG",
    }

    FIELD_PARSERS: ClassVar[Dict[str, Callable[[Any, Optional[Path]], Any]]] = {
        "config": StageConfigBase._coerce_optional_path,
        "log_level": StageConfigBase._coerce_str,
        "data_root": StageConfigBase._coerce_optional_path,
        "in_dir": StageConfigBase._coerce_path,
        "out_dir": StageConfigBase._coerce_path,
        "min_tokens": StageConfigBase._coerce_int,
        "max_tokens": StageConfigBase._coerce_int,
        "shard_count": StageConfigBase._coerce_int,
        "shard_index": StageConfigBase._coerce_int,
        "tokenizer_model": StageConfigBase._coerce_str,
        "soft_barrier_margin": StageConfigBase._coerce_int,
        "structural_markers": StageConfigBase._coerce_optional_path,
        "serializer_provider": StageConfigBase._coerce_str,
        "workers": StageConfigBase._coerce_int,
        "validate_only": StageConfigBase._coerce_bool,
        "resume": StageConfigBase._coerce_bool,
        "force": StageConfigBase._coerce_bool,
        "inject_anchors": StageConfigBase._coerce_bool,
    }

    @classmethod
    def from_env(cls, defaults: Optional[Dict[str, Any]] = None) -> "ChunkerCfg":
        """Instantiate configuration derived solely from environment variables."""

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
    ) -> "ChunkerCfg":
        """Create a configuration by layering env vars, config files, and CLI args."""

        cfg = cls.from_env(defaults=defaults)
        config_path = getattr(args, "config", None)
        if config_path:
            cfg.update_from_file(Path(config_path))
        cfg.apply_args(args)
        cfg.finalize()
        return cfg

    def finalize(self) -> None:
        """Normalise paths and derived values after merging all sources."""

        if self.data_root is not None:
            resolved_root = StageConfigBase._coerce_optional_path(self.data_root, None)
        else:
            env_root = os.getenv("DOCSTOKG_DATA_ROOT")
            resolved_root = (
                StageConfigBase._coerce_optional_path(env_root, None) if env_root else detect_data_root()
            )
        self.data_root = resolved_root

        if self.in_dir is None:
            self.in_dir = data_doctags(resolved_root, ensure=False)
        else:
            self.in_dir = StageConfigBase._coerce_path(self.in_dir, None)

        if self.out_dir is None:
            self.out_dir = data_chunks(resolved_root, ensure=False)
        else:
            self.out_dir = StageConfigBase._coerce_path(self.out_dir, None)

        if self.structural_markers is not None:
            self.structural_markers = StageConfigBase._coerce_optional_path(self.structural_markers, None)

        if self.config is not None:
            self.config = StageConfigBase._coerce_optional_path(self.config, None)
        self.log_level = str(self.log_level or "INFO").upper()
        self.tokenizer_model = (
            str(self.tokenizer_model or DEFAULT_TOKENIZER).strip() or DEFAULT_TOKENIZER
        )
        self.serializer_provider = (
            str(self.serializer_provider or DEFAULT_SERIALIZER_PROVIDER).strip()
            or DEFAULT_SERIALIZER_PROVIDER
        )
        self.validate_only = bool(self.validate_only)
        self.resume = bool(self.resume)
        self.force = bool(self.force)
        self.inject_anchors = bool(self.inject_anchors)

        if self.min_tokens < 0 or self.max_tokens < 0:
            raise ValueError("min_tokens and max_tokens must be non-negative")
        if self.min_tokens > self.max_tokens:
            raise ValueError("min_tokens must be <= max_tokens")
        if self.shard_count < 1:
            raise ValueError("shard_count must be >= 1")
        if not (0 <= self.shard_index < self.shard_count):
            raise ValueError("shard_index must be in [0, shard_count)")
        if self.workers < 1:
            raise ValueError("workers must be >= 1")
        if self.soft_barrier_margin < 0:
            raise ValueError("soft_barrier_margin must be >= 0")


CHUNK_PROFILE_PRESETS: Dict[str, Dict[str, Any]] = {
    "cpu-small": {
        "workers": 1,
        "min_tokens": 128,
        "max_tokens": 256,
        "soft_barrier_margin": 24,
        "tokenizer_model": DEFAULT_TOKENIZER,
    },
    "gpu-default": {
        "workers": 1,
        "min_tokens": 256,
        "max_tokens": 512,
        "soft_barrier_margin": SOFT_BARRIER_MARGIN,
        "tokenizer_model": DEFAULT_TOKENIZER,
    },
    "gpu-max": {
        "workers": max(1, (os.cpu_count() or 16) - 2),
        "min_tokens": 256,
        "max_tokens": 768,
        "soft_barrier_margin": max(32, SOFT_BARRIER_MARGIN * 2),
        "tokenizer_model": DEFAULT_TOKENIZER,
    },
    "bert-compat": {
        "workers": 1,
        "min_tokens": 192,
        "max_tokens": 320,
        "soft_barrier_margin": 32,
        "tokenizer_model": "bert-base-uncased",
    },
}


__all__ = ["ChunkerCfg", "SOFT_BARRIER_MARGIN", "CHUNK_PROFILE_PRESETS"]
