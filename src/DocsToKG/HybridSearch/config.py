"""Configuration models and manager for hybrid search."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Literal, Optional


@dataclass(frozen=True)
class ChunkingConfig:
    max_tokens: int = 800
    overlap: int = 150


@dataclass(frozen=True)
class DenseIndexConfig:
    index_type: Literal["flat", "ivf_flat", "ivf_pq"] = "flat"
    nlist: int = 1024
    nprobe: int = 8
    pq_m: int = 16
    pq_bits: int = 8
    oversample: int = 2


@dataclass(frozen=True)
class FusionConfig:
    k0: float = 60.0
    mmr_lambda: float = 0.6
    enable_mmr: bool = True
    cosine_dedupe_threshold: float = 0.98
    max_chunks_per_doc: int = 3


@dataclass(frozen=True)
class RetrievalConfig:
    bm25_top_k: int = 50
    splade_top_k: int = 50
    dense_top_k: int = 50


@dataclass(frozen=True)
class HybridSearchConfig:
    chunking: ChunkingConfig = ChunkingConfig()
    dense: DenseIndexConfig = DenseIndexConfig()
    fusion: FusionConfig = FusionConfig()
    retrieval: RetrievalConfig = RetrievalConfig()

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "HybridSearchConfig":
        chunking = ChunkingConfig(**payload.get("chunking", {}))
        dense = DenseIndexConfig(**payload.get("dense", {}))
        fusion = FusionConfig(**payload.get("fusion", {}))
        retrieval = RetrievalConfig(**payload.get("retrieval", {}))
        return HybridSearchConfig(chunking=chunking, dense=dense, fusion=fusion, retrieval=retrieval)


class HybridSearchConfigManager:
    """File-backed configuration manager with reload support."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = RLock()
        self._config = self._load()

    def get(self) -> HybridSearchConfig:
        with self._lock:
            return self._config

    def reload(self) -> HybridSearchConfig:
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

