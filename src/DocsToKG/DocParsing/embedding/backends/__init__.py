"""Embedding backend abstractions for DocsToKG DocParsing."""

from .base import (
    DenseEmbeddingBackend,
    LexicalEmbeddingBackend,
    ProviderContext,
    ProviderError,
    ProviderIdentity,
    ProviderTelemetryEvent,
    SparseEmbeddingBackend,
)

__all__ = [
    "DenseEmbeddingBackend",
    "LexicalEmbeddingBackend",
    "ProviderContext",
    "ProviderError",
    "ProviderIdentity",
    "ProviderTelemetryEvent",
    "SparseEmbeddingBackend",
]
