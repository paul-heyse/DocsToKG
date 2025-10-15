"""
DocParsing Rich Serializers

This module extends Docling serializers to capture captions, classifications,
and molecular annotations when generating DocTags. It provides picture and
table serializers tailored to DocsToKG along with a convenience provider that
bundles them for hybrid chunking pipelines.

Key Features:
- Extract human-readable captions and classification labels for images
- Preserve SMILES strings discovered during document parsing
- Wrap the enhanced serializers in a provider compatible with Docling chunkers

Usage:
    from DocsToKG.DocParsing.serializers import RichSerializerProvider

    provider = RichSerializerProvider()
    serializer = provider.get_serializer(doc)  # doctest: +SKIP

Dependencies:
- docling_core: Supplies Docling data structures and base serializer classes.
- typing_extensions.override: Ensures subclass overrides remain explicit.
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
    """Serialize picture items with captions and rich annotation metadata.

    Attributes:
        image_placeholder: Fallback marker inserted when an image lacks metadata.

    Examples:
        >>> serializer = CaptionPlusAnnotationPictureSerializer()
        >>> isinstance(serializer, CaptionPlusAnnotationPictureSerializer)
        True
    """

    @override
    def serialize(
        self,
        *,
        item: PictureItem,
        doc_serializer: BaseDocSerializer,
        doc: _Doc,
        **_: Any,
    ) -> SerializationResult:
        """Render picture metadata into Markdown-friendly text.

        Args:
            item: Picture element extracted from the Docling document.
            doc_serializer: Serializer orchestrating the chunking pipeline.
            doc: Source document providing contextual accessors.
            **_: Additional keyword arguments ignored by this implementation.

        Returns:
            Serialization result containing Markdown-ready text and spans.
        """

        parts: List[str] = []
        annotations: List[object] = []
        try:
            annotations = list(item.annotations or [])
        except Exception:  # pragma: no cover - defensive catch
            annotations = []

        try:
            caption = (item.caption_text(doc) or "").strip()
            if caption:
                parts.append(f"Figure caption: {caption}")
        except Exception:  # pragma: no cover - defensive catch
            pass

        for annotation in annotations:
            try:
                if isinstance(annotation, PictureDescriptionData) and annotation.text:
                    parts.append(f"Picture description: {annotation.text}")
                elif (
                    isinstance(annotation, PictureClassificationData)
                    and annotation.predicted_classes
                ):
                    parts.append(f"Picture type: {annotation.predicted_classes[0].class_name}")
                elif isinstance(annotation, PictureMoleculeData) and annotation.smi:
                    parts.append(f"SMILES: {annotation.smi}")
            except Exception:  # pragma: no cover - defensive catch
                continue
        if not parts:
            parts.append("<!-- image -->")

        has_caption = len(parts) > 1 or (parts and parts[0] != "<!-- image -->")
        has_classification = any(
            isinstance(annotation, PictureClassificationData)
            and getattr(annotation, "predicted_classes", None)
            for annotation in annotations
        )
        try:  # pragma: no cover - metadata enrichment best effort
            flags = getattr(item, "_docstokg_flags", {})
            if not isinstance(flags, dict):
                flags = {}
            flags.setdefault("has_image_captions", False)
            flags.setdefault("has_image_classification", False)
            flags["has_image_captions"] = flags["has_image_captions"] or has_caption
            flags["has_image_classification"] = (
                flags["has_image_classification"] or has_classification
            )
            setattr(item, "_docstokg_flags", flags)
        except Exception:
            pass

        text = doc_serializer.post_process(text="\n".join(parts))
        return create_ser_result(text=text, span_source=item)


class RichSerializerProvider(ChunkingSerializerProvider):
    """Provide a serializer that augments tables and pictures with Markdown.

    Attributes:
        markdown_params: Default Markdown parameters passed to serializers.

    Examples:
        >>> provider = RichSerializerProvider()
        >>> isinstance(provider, RichSerializerProvider)
        True
    """

    def get_serializer(self, doc: DoclingDocument) -> ChunkingDocSerializer:
        """Construct a ChunkingDocSerializer tailored for DocTags documents.

        Args:
            doc: Docling document to serialize into DocTags-compatible chunks.

        Returns:
            Chunking serializer configured with Markdown enrichments.
        """

        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
            picture_serializer=CaptionPlusAnnotationPictureSerializer(),
            params=MarkdownParams(image_placeholder="<!-- image -->"),
        )
