# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.DocParsing.embedding.backends.__init__",
#   "purpose": "Embedding backend abstractions for DocsToKG DocParsing.",
#   "sections": []
# }
# === /NAVMAP ===

"""Embedding backend abstractions for DocsToKG DocParsing."""

from .base import (
    DenseEmbeddingBackend,
    LexicalEmbeddingBackend,
    ProviderContext,
    ProviderError,
    ProviderIdentity,
    ProviderTelemetryEmitter,
    ProviderTelemetryEvent,
    SparseEmbeddingBackend,
)
from .dense.fallback import DenseFallbackProvider
from .factory import ProviderBundle, ProviderFactory
from .lexical.pyserini import PyseriniBM25Provider
from .nulls import NullDenseProvider, NullLexicalProvider, NullSparseProvider

__all__ = [
    "DenseEmbeddingBackend",
    "LexicalEmbeddingBackend",
    "ProviderContext",
    "ProviderError",
    "ProviderIdentity",
    "ProviderTelemetryEmitter",
    "ProviderTelemetryEvent",
    "SparseEmbeddingBackend",
    "ProviderBundle",
    "ProviderFactory",
    "NullDenseProvider",
    "NullSparseProvider",
    "NullLexicalProvider",
    "DenseFallbackProvider",
    "PyseriniBM25Provider",
]
