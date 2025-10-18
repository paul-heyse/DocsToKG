from __future__ import annotations

from typing import Any

from ..serializer.base import BaseDocSerializer

__all__ = ["ChunkingDocSerializer", "ChunkingSerializerProvider"]


class ChunkingDocSerializer(BaseDocSerializer):
    def __init__(
        self,
        doc: object | None = None,
        table_serializer: object | None = None,
        picture_serializer: object | None = None,
        params: object | None = None,
    ) -> None:
        self.doc = doc
        self.table_serializer = table_serializer
        self.picture_serializer = picture_serializer
        self.params = params


class ChunkingSerializerProvider:
    def get_serializer(self, doc: Any) -> ChunkingDocSerializer:
        return ChunkingDocSerializer(doc=doc)
