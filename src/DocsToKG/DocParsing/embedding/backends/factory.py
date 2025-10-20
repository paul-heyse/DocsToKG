"""Provider factory wiring dense, sparse, and lexical backends."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from DocsToKG.DocParsing.embedding.config import EmbedCfg

from .base import (
    DenseEmbeddingBackend,
    LexicalEmbeddingBackend,
    ProviderContext,
    ProviderError,
    ProviderIdentity,
    ProviderTelemetryEmitter,
    SparseEmbeddingBackend,
)
from .dense.qwen_vllm import QwenVLLMConfig, QwenVLLMProvider
from .dense.sentence_transformers import SentenceTransformersConfig, SentenceTransformersProvider
from .dense.tei import TEIConfig, TEIProvider
from .lexical.local_bm25 import LocalBM25Config, LocalBM25Provider
from .sparse.splade_st import SpladeSTConfig, SpladeSTProvider


@dataclass
class ProviderBundle:
    """Context-managed bundle of embedding providers."""

    dense: Optional[DenseEmbeddingBackend]
    sparse: Optional[SparseEmbeddingBackend]
    lexical: Optional[LexicalEmbeddingBackend]
    context: ProviderContext

    def __enter__(self) -> "ProviderBundle":
        for provider in (self.dense, self.sparse, self.lexical):
            if provider:
                provider.open(self.context)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        for provider in (self.dense, self.sparse, self.lexical):
            if not provider:
                continue
            try:
                provider.close()
            except Exception:  # pragma: no cover - defensive shutdown
                pass

    def identities(self) -> Dict[str, ProviderIdentity]:
        identities: Dict[str, ProviderIdentity] = {}
        if self.dense:
            identities["dense"] = self.dense.identity
        if self.sparse:
            identities["sparse"] = self.sparse.identity
        if self.lexical:
            identities["lexical"] = self.lexical.identity
        return identities


class ProviderFactory:
    """Construct embedding providers based on ``EmbedCfg`` provider settings."""

    @staticmethod
    def create(
        cfg: EmbedCfg,
        *,
        telemetry_emitter: Optional[ProviderTelemetryEmitter] = None,
    ) -> ProviderBundle:
        settings = cfg.provider_settings()
        embedding_cfg = settings["embedding"]
        context = ProviderContext(
            device=embedding_cfg["device"] or "auto",
            dtype=embedding_cfg["dtype"] or "auto",
            batch_hint=embedding_cfg["batch_size"],
            max_concurrency=embedding_cfg["max_concurrency"],
            normalize_l2=bool(embedding_cfg["normalize_l2"]),
            offline=bool(embedding_cfg["offline"]),
            cache_dir=embedding_cfg["cache_dir"],
            telemetry_tags=embedding_cfg["telemetry_tags"],
            telemetry_emitter=telemetry_emitter,
        )

        dense_provider = ProviderFactory._build_dense(cfg, settings["dense"])
        sparse_provider = ProviderFactory._build_sparse(cfg, settings["sparse"])
        lexical_provider = ProviderFactory._build_lexical(cfg, settings["lexical"])

        return ProviderBundle(dense=dense_provider, sparse=sparse_provider, lexical=lexical_provider, context=context)

    @staticmethod
    def _build_dense(cfg: EmbedCfg, dense_settings: Dict[str, Any]) -> Optional[DenseEmbeddingBackend]:
        backend = dense_settings.get("backend") or "qwen_vllm"
        backend = backend.lower()
        if backend in {"none", "null"}:
            raise ProviderError(
                provider="dense",
                category="validation",
                detail="Dense backend cannot be 'none' under the current vector schema.",
                retryable=False,
            )
        if backend == "qwen_vllm":
            qwen_cfg = dense_settings.get("qwen_vllm", {})
            download_dir = qwen_cfg.get("download_dir") or cfg.qwen_model_dir
            if download_dir is None:
                raise ProviderError(
                    provider="dense.qwen_vllm",
                    category="init",
                    detail="Qwen download directory is not configured.",
                    retryable=False,
                )
            model_dir = Path(download_dir).expanduser().resolve()
            config = QwenVLLMConfig(
                model_dir=model_dir,
                model_id=qwen_cfg.get("model_id"),
                dtype=cfg.qwen_dtype,
                tensor_parallelism=cfg.tp,
                gpu_memory_utilization=0.60,
                batch_size=qwen_cfg.get("batch_size") or cfg.batch_size_qwen,
                quantization=qwen_cfg.get("quantization") or cfg.qwen_quant,
                dimension=qwen_cfg.get("dimension") or cfg.qwen_dim,
                cache_enabled=not cfg.no_cache,
                queue_depth=qwen_cfg.get("queue_depth") or cfg.files_parallel,
            )
            return QwenVLLMProvider(config)
        if backend == "tei":
            tei_cfg = dense_settings.get("tei", {})
            url = tei_cfg.get("url")
            if not url:
                raise ProviderError(
                    provider="dense.tei",
                    category="validation",
                    detail="dense.tei.url must be provided when using the TEI backend.",
                    retryable=False,
                )
            config = TEIConfig(
                url=url,
                timeout_seconds=tei_cfg.get("timeout_seconds", 30.0),
                max_inflight=tei_cfg.get("max_inflight") or cfg.files_parallel,
            )
            return TEIProvider(config)
        if backend == "sentence_transformers":
            st_cfg = dense_settings.get("sentence_transformers", {})
            model_id = st_cfg.get("model_id")
            if not model_id:
                raise ProviderError(
                    provider="dense.sentence_transformers",
                    category="validation",
                    detail="dense.sentence_transformers.model_id must be specified.",
                    retryable=False,
                )
            config = SentenceTransformersConfig(
                model_id=model_id,
                batch_size=st_cfg.get("batch_size") or cfg.batch_size_qwen,
                normalize_l2=(
                    st_cfg.get("normalize_l2")
                    if st_cfg.get("normalize_l2") is not None
                    else cfg.embedding_normalize_l2
                ),
            )
            return SentenceTransformersProvider(config)
        raise ProviderError(
            provider="dense",
            category="validation",
            detail=f"Unsupported dense backend: {backend}",
            retryable=False,
        )

    @staticmethod
    def _build_sparse(cfg: EmbedCfg, sparse_settings: Dict[str, Any]) -> Optional[SparseEmbeddingBackend]:
        backend = sparse_settings.get("backend") or "splade_st"
        backend = backend.lower()
        if backend in {"none", "null"}:
            raise ProviderError(
                provider="sparse",
                category="validation",
                detail="Sparse backend cannot be 'none' under the current vector schema.",
                retryable=False,
            )
        if backend != "splade_st":
            raise ProviderError(
                provider="sparse",
                category="validation",
                detail=f"Unsupported sparse backend: {backend}",
                retryable=False,
            )
        splade_cfg = sparse_settings.get("splade_st", {})
        model_dir = splade_cfg.get("model_dir") or cfg.splade_model_dir
        if model_dir is None:
            raise ProviderError(
                provider="sparse.splade_st",
                category="init",
                detail="sparse.splade_st.model_dir is required.",
                retryable=False,
            )
        config = SpladeSTConfig(
            model_dir=Path(model_dir),
            device=cfg.embedding_device,
            batch_size=splade_cfg.get("batch_size") or cfg.batch_size_splade,
            cache_folder=cfg.embedding_cache_dir or cfg.splade_model_dir,
            max_active_dims=splade_cfg.get("max_active_dims") or cfg.splade_max_active_dims,
            attn_impl=(splade_cfg.get("attn_backend") or cfg.splade_attn),
            local_files_only=bool(cfg.offline),
        )
        return SpladeSTProvider(config)

    @staticmethod
    def _build_lexical(cfg: EmbedCfg, lexical_settings: Dict[str, Any]) -> Optional[LexicalEmbeddingBackend]:
        backend = lexical_settings.get("backend") or "local_bm25"
        backend = backend.lower()
        if backend in {"none", "null"}:
            raise ProviderError(
                provider="lexical",
                category="validation",
                detail="Lexical backend cannot be 'none' under the current vector schema.",
                retryable=False,
            )
        if backend != "local_bm25":
            raise ProviderError(
                provider="lexical",
                category="validation",
                detail=f"Unsupported lexical backend: {backend}",
                retryable=False,
            )
        local_cfg = lexical_settings.get("local_bm25", {})
        config = LocalBM25Config(
            k1=float(local_cfg.get("k1") or cfg.lexical_local_bm25_k1),
            b=float(local_cfg.get("b") or cfg.lexical_local_bm25_b),
        )
        return LocalBM25Provider(config)


__all__ = ["ProviderFactory", "ProviderBundle"]
