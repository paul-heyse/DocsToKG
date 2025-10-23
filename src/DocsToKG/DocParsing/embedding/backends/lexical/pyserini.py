"""Placeholder lexical provider for the future Pyserini integration."""

from __future__ import annotations

from dataclasses import dataclass

from ..base import LexicalEmbeddingBackend, ProviderContext, ProviderError, ProviderIdentity


@dataclass(slots=True)
class PyseriniBM25Provider(LexicalEmbeddingBackend):
    identity = ProviderIdentity(name="lexical.pyserini", version="0.0.0")

    def open(self, context: ProviderContext) -> None:  # pragma: no cover - trivial
        raise ProviderError(
            provider=self.identity.name,
            category="init",
            detail="lexical.pyserini is not implemented yet.",
            retryable=False,
        )

    def close(self) -> None:  # pragma: no cover - trivial
        return None

    def accumulate_stats(self, chunks):  # pragma: no cover - trivial
        raise ProviderError(
            provider=self.identity.name,
            category="runtime",
            detail="lexical.pyserini cannot accumulate stats yet.",
            retryable=False,
        )

    def vector(self, text, stats):  # pragma: no cover - trivial
        raise ProviderError(
            provider=self.identity.name,
            category="runtime",
            detail="lexical.pyserini cannot produce vectors yet.",
            retryable=False,
        )


__all__ = ["PyseriniBM25Provider"]
