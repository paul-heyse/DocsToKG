# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.DocParsing.embedding.backends.dense.fallback",
#   "purpose": "Placeholder dense fallback provider.",
#   "sections": [
#     {
#       "id": "densefallbackprovider",
#       "name": "DenseFallbackProvider",
#       "anchor": "class-densefallbackprovider",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""Placeholder dense fallback provider.

The implementation simply raises a :class:`ProviderError` explaining that the
concrete backend has not been implemented yet.  It acts as a marker so the
factory can report helpful error messages when users request
``dense.backend=fallback``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..base import DenseEmbeddingBackend, ProviderContext, ProviderError, ProviderIdentity


@dataclass(slots=True)
class DenseFallbackProvider(DenseEmbeddingBackend):
    identity = ProviderIdentity(name="dense.fallback", version="0.0.0")

    def open(self, context: ProviderContext) -> None:  # pragma: no cover - trivial
        raise ProviderError(
            provider=self.identity.name,
            category="init",
            detail="dense.fallback is a placeholder. Configure a concrete backend instead.",
            retryable=False,
        )

    def close(self) -> None:  # pragma: no cover - trivial
        return None

    def embed(self, texts, *, batch_hint=None):  # pragma: no cover - trivial
        raise ProviderError(
            provider=self.identity.name,
            category="runtime",
            detail="dense.fallback cannot produce embeddings; select a real backend.",
            retryable=False,
        )


__all__ = ["DenseFallbackProvider"]
