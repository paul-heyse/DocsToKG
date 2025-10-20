"""Dense embedding provider using SentenceTransformer models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from ..base import DenseEmbeddingBackend, ProviderContext, ProviderError, ProviderIdentity
from ..utils import bounded_batch_size, normalise_vectors, resolve_device


@dataclass(slots=True)
class SentenceTransformersConfig:
    model_id: str
    batch_size: int = 32
    normalize_l2: bool = True
    trust_remote_code: bool = False


class SentenceTransformersProvider(DenseEmbeddingBackend):
    """Dense provider backed by sentence-transformers."""

    identity = ProviderIdentity(name="dense.sentence_transformers", version="1.0.0")

    def __init__(self, config: SentenceTransformersConfig) -> None:
        self._cfg = config
        self._ctx: ProviderContext | None = None
        self._model = None

    def open(self, context: ProviderContext) -> None:
        self._ctx = context
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency missing
            raise ProviderError(
                provider=self.identity.name,
                category="init",
                detail="sentence-transformers must be installed for the dense backend.",
                retryable=False,
                wrapped=exc,
            ) from exc

        device_hint = resolve_device(context.device)
        kwargs = {"device": device_hint} if device_hint else {}
        kwargs["trust_remote_code"] = bool(self._cfg.trust_remote_code)
        kwargs["local_files_only"] = context.offline
        self._model = SentenceTransformer(self._cfg.model_id, **kwargs)

    def close(self) -> None:
        self._model = None
        self._ctx = None

    def embed(
        self,
        texts: Sequence[str],
        *,
        batch_hint: Optional[int] = None,
    ) -> Sequence[Sequence[float]]:
        if not texts:
            return []
        if self._model is None:
            raise ProviderError(
                provider=self.identity.name,
                category="runtime",
                detail="Provider has not been opened before use.",
                retryable=False,
            )
        batch_size = bounded_batch_size(
            preferred=batch_hint or self._cfg.batch_size,
            fallback=32,
        )
        vectors = self._model.encode(  # type: ignore[call-arg]
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self._cfg.normalize_l2 if self._cfg.normalize_l2 else False,
        )
        results: List[List[float]] = [[float(x) for x in embedding] for embedding in vectors]
        if self._ctx and self._ctx.normalize_l2 and not self._cfg.normalize_l2:
            results = normalise_vectors(results, normalise=True)
        if self._ctx:
            self._ctx.emit(
                self.identity,
                phase="embed",
                data={"batch_size_effective": batch_size, "vector_count": len(results)},
            )
        return results


__all__ = ["SentenceTransformersConfig", "SentenceTransformersProvider"]
