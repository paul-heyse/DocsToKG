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
from .factory import ProviderBundle, ProviderFactory

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
]
