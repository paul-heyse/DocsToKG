# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.DocParsing.embedding.backends.sparse.splade_st",
#   "purpose": "SPLADE sentence-transformers sparse embedding provider.",
#   "sections": [
#     {
#       "id": "spladestconfig",
#       "name": "SpladeSTConfig",
#       "anchor": "class-spladestconfig",
#       "kind": "class"
#     },
#     {
#       "id": "get-sparse-encoder-cls",
#       "name": "_get_sparse_encoder_cls",
#       "anchor": "function-get-sparse-encoder-cls",
#       "kind": "function"
#     },
#     {
#       "id": "detect-splade-backend",
#       "name": "_detect_splade_backend",
#       "anchor": "function-detect-splade-backend",
#       "kind": "function"
#     },
#     {
#       "id": "get-encoder",
#       "name": "_get_encoder",
#       "anchor": "function-get-encoder",
#       "kind": "function"
#     },
#     {
#       "id": "spladestprovider",
#       "name": "SpladeSTProvider",
#       "anchor": "class-spladestprovider",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""SPLADE sentence-transformers sparse embedding provider."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from DocsToKG.DocParsing.core.models import SpladeCfg
from DocsToKG.DocParsing.env import ensure_splade_dependencies, ensure_splade_environment

from ..base import ProviderContext, ProviderError, ProviderIdentity, SparseEmbeddingBackend


@dataclass(slots=True)
class SpladeSTConfig:
    model_dir: Path
    device: str = "auto"
    batch_size: int = 32
    cache_folder: Optional[Path] = None
    max_active_dims: Optional[int] = None
    attn_impl: Optional[str] = None
    local_files_only: bool = True


_SPLADE_ENCODER_CACHE: Dict[Tuple[str, str, Optional[str], Optional[int]], object] = {}
_SPLADE_ENCODER_BACKENDS: Dict[Tuple[str, str, Optional[str], Optional[int]], str] = {}
_SPLADE_LOCK = threading.Lock()


def _get_sparse_encoder_cls():
    ensure_splade_dependencies()
    try:
        from sentence_transformers import SparseEncoder as cls  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency missing
        raise ProviderError(
            provider="sparse.splade_st",
            category="init",
            detail="sentence-transformers must be installed for SPLADE embeddings.",
            retryable=False,
            wrapped=exc,
        ) from exc
    return cls


def _detect_splade_backend(encoder, requested: Optional[str]) -> str:
    candidates = (
        ("model", "model", "config", "attn_implementation"),
        ("model", "config", "attn_implementation"),
        ("config", "attn_implementation"),
        ("model", "model", "attn_implementation"),
    )
    for path in candidates:
        value = encoder
        for attr in path:
            value = getattr(value, attr, None)
            if value is None:
                break
        else:
            if isinstance(value, str) and value:
                return value

    if requested in {"sdpa", "eager", "flash_attention_2"}:
        return requested
    return "auto" if requested is None else requested


def _get_encoder(cfg: SpladeCfg):
    key = (str(cfg.model_dir), cfg.device, cfg.attn_impl, cfg.max_active_dims)
    with _SPLADE_LOCK:
        if key in _SPLADE_ENCODER_CACHE:
            if key not in _SPLADE_ENCODER_BACKENDS:
                _SPLADE_ENCODER_BACKENDS[key] = cfg.attn_impl or "auto"
            return _SPLADE_ENCODER_CACHE[key]

        encoder_cls = _get_sparse_encoder_cls()
        model_kwargs: Dict[str, object] = {}
        if cfg.attn_impl:
            model_kwargs["attn_implementation"] = cfg.attn_impl
        if cfg.max_active_dims is not None:
            model_kwargs["max_active_dims"] = cfg.max_active_dims

        backend_used: Optional[str] = cfg.attn_impl
        try:
            encoder = encoder_cls(
                str(cfg.model_dir),
                device=cfg.device,
                cache_folder=str(cfg.cache_folder) if cfg.cache_folder else None,
                model_kwargs=model_kwargs,
                local_files_only=cfg.local_files_only,
            )
            backend_used = _detect_splade_backend(encoder, backend_used)
        except (ValueError, ImportError) as exc:
            if cfg.attn_impl == "flash_attention_2" and "Flash Attention 2" in str(exc):
                fallback_kwargs = dict(model_kwargs)
                fallback_kwargs["attn_implementation"] = "sdpa"
                encoder = encoder_cls(
                    str(cfg.model_dir),
                    device=cfg.device,
                    cache_folder=str(cfg.cache_folder) if cfg.cache_folder else None,
                    model_kwargs=fallback_kwargs,
                    local_files_only=cfg.local_files_only,
                )
                backend_used = _detect_splade_backend(encoder, "sdpa")
            else:
                raise

        _SPLADE_ENCODER_CACHE[key] = encoder
        _SPLADE_ENCODER_BACKENDS[key] = backend_used or "auto"
        return encoder


class SpladeSTProvider(SparseEmbeddingBackend):
    identity = ProviderIdentity(name="sparse.splade_st", version="1.0.0")

    def __init__(self, config: SpladeSTConfig) -> None:
        self._cfg_spec = config
        self._ctx: ProviderContext | None = None
        self._splade_cfg: SpladeCfg | None = None
        self._lock = threading.Lock()

    def open(self, context: ProviderContext) -> None:
        self._ctx = context

        env = ensure_splade_environment(
            device=context.device if context.device != "auto" else None,
            cache_dir=self._cfg_spec.cache_folder,
        )
        device = env.get("device", "cpu")
        cache_dir = env.get("model_dir")

        model_dir = self._cfg_spec.model_dir.expanduser().resolve()
        device_override = self._cfg_spec.device if self._cfg_spec.device != "auto" else device

        self._splade_cfg = SpladeCfg(
            model_dir=model_dir,
            device=device_override,
            batch_size=int(self._cfg_spec.batch_size),
            cache_folder=Path(cache_dir) if cache_dir else self._cfg_spec.cache_folder,
            max_active_dims=self._cfg_spec.max_active_dims,
            attn_impl=self._cfg_spec.attn_impl,
            local_files_only=bool(self._cfg_spec.local_files_only),
        )
        # Warm encoder for telemetry
        _get_encoder(self._splade_cfg)

    def close(self) -> None:
        if self._ctx:
            self._ctx.emit(
                self.identity,
                phase="close",
                data={"status": "closed"},
            )
        self._ctx = None
        self._splade_cfg = None

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[Tuple[str, float]]]:
        if not texts:
            return []
        if self._splade_cfg is None:
            raise ProviderError(
                provider=self.identity.name,
                category="runtime",
                detail="Provider has not been opened before use.",
                retryable=False,
            )
        encoder = _get_encoder(self._splade_cfg)
        token_lists: List[List[Tuple[str, float]]] = []
        batch_size = self._splade_cfg.batch_size
        with self._lock:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                s = encoder.encode(batch)
                for r in range(s.shape[0]):
                    nnz = s[r].coalesce().values().numel()
                    decoded = encoder.decode(s[r], top_k=int(nnz))
                    token_lists.append([(str(tok), float(weight)) for tok, weight in decoded])
        if self._ctx:
            self._ctx.emit(
                self.identity,
                phase="encode",
                data={"batch_size_effective": batch_size, "vector_count": len(token_lists)},
            )
        return token_lists


__all__ = ["SpladeSTConfig", "SpladeSTProvider"]
