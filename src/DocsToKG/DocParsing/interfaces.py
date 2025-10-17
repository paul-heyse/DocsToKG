"""Shared protocol interfaces for DocParsing plugin points."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from docling_core.transforms.chunker.hierarchical_chunker import ChunkingDocSerializer
from docling_core.types.doc.document import DoclingDocument


@runtime_checkable
class ChunkingSerializerProvider(Protocol):
    """Protocol describing serializer providers accepted by the chunking stage."""

    def get_serializer(self, doc: DoclingDocument) -> ChunkingDocSerializer:
        """Return a serializer instance capable of handling ``doc``."""
