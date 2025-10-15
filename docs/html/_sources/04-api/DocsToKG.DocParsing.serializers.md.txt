# 1. Module: serializers

This reference documents the DocsToKG module ``DocsToKG.DocParsing.serializers``.

Picture and table serializers for DocParsing pipeline.

These serializers extract rich metadata from Docling documents including image
captions, classifications, and SMILES molecular structures.

Example Usage:
    >>> from DocsToKG.DocParsing.serializers import RichSerializerProvider
    >>> provider = RichSerializerProvider()
    >>> isinstance(provider.get_serializer(None), ChunkingDocSerializer)  # doctest: +SKIP
    True

## 1. Functions

### `serialize(self)`

Render picture metadata into Markdown-friendly text.

Args:
item: Picture element extracted from the Docling document.
doc_serializer: Serializer orchestrating the chunking pipeline.
doc: Source document providing contextual accessors.
**_: Additional keyword arguments ignored by this implementation.

Returns:
Serialization result containing Markdown-ready text and spans.

### `get_serializer(self, doc)`

Construct a ChunkingDocSerializer tailored for DocTags documents.

Args:
doc: Docling document to serialize into DocTags-compatible chunks.

Returns:
Chunking serializer configured with Markdown enrichments.

## 2. Classes

### `CaptionPlusAnnotationPictureSerializer`

Serialize picture items with captions and rich annotation metadata.

Attributes:
image_placeholder: Fallback marker inserted when an image lacks metadata.

Examples:
>>> serializer = CaptionPlusAnnotationPictureSerializer()
>>> isinstance(serializer, CaptionPlusAnnotationPictureSerializer)
True

### `RichSerializerProvider`

Provide a serializer that augments tables and pictures with Markdown.

Attributes:
markdown_params: Default Markdown parameters passed to serializers.

Examples:
>>> provider = RichSerializerProvider()
>>> isinstance(provider, RichSerializerProvider)
True
