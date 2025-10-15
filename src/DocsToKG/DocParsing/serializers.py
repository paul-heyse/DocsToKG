"""Picture and table serializers for DocParsing pipeline.

These serializers extract rich metadata from Docling documents including image
captions, classifications, and SMILES molecular structures.

Example Usage:
    >>> from DocsToKG.DocParsing.serializers import RichSerializerProvider
    >>> provider = RichSerializerProvider()
    >>> isinstance(provider.get_serializer(None), ChunkingDocSerializer)  # doctest: +SKIP
    True
"""

from __future__ import annotations

from typing import Any, List

from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.serializer.base import BaseDocSerializer, SerializationResult
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.markdown import (
    MarkdownParams,
    MarkdownPictureSerializer,
    MarkdownTableSerializer,
)
from docling_core.types.doc.document import (
    DoclingDocument,
    PictureClassificationData,
    PictureDescriptionData,
    PictureItem,
    PictureMoleculeData,
)
from docling_core.types.doc.document import DoclingDocument as _Doc
from typing_extensions import override

__all__ = [
    "CaptionPlusAnnotationPictureSerializer",
    "RichSerializerProvider",
]


class CaptionPlusAnnotationPictureSerializer(MarkdownPictureSerializer):
    """Serialize picture items with captions and rich annotation metadata."""

    @override
    def serialize(
        self,
        *,
        item: PictureItem,
        doc_serializer: BaseDocSerializer,
        doc: _Doc,
        **_: Any,
    ) -> SerializationResult:
        """Render picture metadata into Markdown-friendly text."""

        parts: List[str] = []
        try:
            caption = (item.caption_text(doc) or "").strip()
            if caption:
                parts.append(f"Figure caption: {caption}")
        except Exception:  # pragma: no cover - defensive catch
            pass
        try:
            for annotation in item.annotations or []:
                if isinstance(annotation, PictureDescriptionData) and annotation.text:
                    parts.append(f"Picture description: {annotation.text}")
                elif isinstance(annotation, PictureClassificationData) and annotation.predicted_classes:
                    parts.append(
                        f"Picture type: {annotation.predicted_classes[0].class_name}"
                    )
                elif isinstance(annotation, PictureMoleculeData) and annotation.smi:
                    parts.append(f"SMILES: {annotation.smi}")
        except Exception:  # pragma: no cover - defensive catch
            pass
        if not parts:
            parts.append("<!-- image -->")
        text = doc_serializer.post_process(text="\n".join(parts))
        return create_ser_result(text=text, span_source=item)


class RichSerializerProvider(ChunkingSerializerProvider):
    """Provide a serializer that augments tables and pictures with Markdown."""

    def get_serializer(self, doc: DoclingDocument) -> ChunkingDocSerializer:
        """Construct a ChunkingDocSerializer tailored for DocTags documents."""

        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
            picture_serializer=CaptionPlusAnnotationPictureSerializer(),
            params=MarkdownParams(image_placeholder="<!-- image -->"),
        )
