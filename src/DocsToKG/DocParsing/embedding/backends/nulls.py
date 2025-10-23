"""Null provider implementations used for disabled backends.

These providers intentionally raise :class:`ProviderError` with a
``validation`` category so callers can surface descriptive messages or fall
back to alternate providers.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import (
    DenseEmbeddingBackend,
    LexicalEmbeddingBackend,
    ProviderContext,
    ProviderError,
    ProviderIdentity,
    SparseEmbeddingBackend,
)


@dataclass(slots=True)
class _NullMixin:
    backend_label: str

    def _error(self) -> ProviderError:
        return ProviderError(
            provider=self.identity.name,
            category="validation",
            detail=(
                f"The {self.backend_label} backend is disabled (value 'none'). "
                "Configure a concrete provider or supply a fallback backend."
            ),
            retryable=False,
        )

    def open(self, context: ProviderContext) -> None:  # pragma: no cover - trivial
        raise self._error()

    def close(self) -> None:  # pragma: no cover - trivial
        return None


class NullDenseProvider(_NullMixin, DenseEmbeddingBackend):
    identity = ProviderIdentity(name="dense.none", version="0.0.0")

    def __init__(self) -> None:
        super().__init__(backend_label="dense")

    def embed(self, texts, *, batch_hint=None):  # pragma: no cover - trivial
        raise self._error()


class NullSparseProvider(_NullMixin, SparseEmbeddingBackend):
    identity = ProviderIdentity(name="sparse.none", version="0.0.0")

    def __init__(self) -> None:
        super().__init__(backend_label="sparse")

    def encode(self, texts):  # pragma: no cover - trivial
        raise self._error()


class NullLexicalProvider(_NullMixin, LexicalEmbeddingBackend):
    identity = ProviderIdentity(name="lexical.none", version="0.0.0")

    def __init__(self) -> None:
        super().__init__(backend_label="lexical")

    def accumulate_stats(self, chunks):  # pragma: no cover - trivial
        raise self._error()

    def vector(self, text, stats):  # pragma: no cover - trivial
        raise self._error()


__all__ = [
    "NullDenseProvider",
    "NullSparseProvider",
    "NullLexicalProvider",
]
