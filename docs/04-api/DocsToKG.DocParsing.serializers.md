# 1. Module: serializers

This reference documents the DocsToKG module ``DocsToKG.DocParsing.serializers``.

## 1. Overview

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

## 2. Functions

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

### `_maybe_add_conf(value)`

Collect numeric confidence scores when they can be coerced to float.

## 3. Classes

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
