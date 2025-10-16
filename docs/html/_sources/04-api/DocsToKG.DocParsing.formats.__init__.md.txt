# 1. Module: __init__

This reference documents the DocsToKG module ``DocsToKG.DocParsing.formats.__init__``.

## 1. Overview

DocParsing Formats

This module defines Pydantic models, validation helpers, and Docling serializer
providers used throughout the DocParsing pipeline. By gathering schema
definitions alongside the Markdown-aware serializers, downstream stages can
import one module for all data contract needs.

Key Features:
- Strict schemas for chunk JSONL rows and embedding vector rows
- Optional dependency stubs for environments without Pydantic installed
- Convenience validators for schema versions and provenance metadata

Usage:
    from DocsToKG.DocParsing import formats

    validated = formats.validate_chunk_row(raw_row)
    provider = formats.RichSerializerProvider()

Dependencies:
- pydantic (optional): Offers model validation when available; graceful fallbacks
  raise informative errors otherwise.
- docling_core: Supplies serializer base classes consumed by DocsToKG.

## 2. Functions

### `_missing_pydantic_message()`

Return a consistent optional dependency warning message.

### `validate_chunk_row(row)`

Validate and parse a chunk JSONL row.

Args:
row: Raw dictionary from JSONL.

Returns:
Validated :class:`ChunkRow` instance.

Raises:
ValueError: If the row fails schema validation.

Examples:
>>> chunk = validate_chunk_row({
...     "doc_id": "doc",
...     "source_path": "path",
...     "chunk_id": 0,
...     "source_chunk_idxs": [0],
...     "num_tokens": 10,
...     "text": "hello",
... })
>>> chunk.schema_version
'docparse/1.1.0'

### `validate_vector_row(row)`

Validate and parse a vector JSONL row.

Args:
row: Raw dictionary from JSONL.

Returns:
Validated :class:`VectorRow` instance.

Raises:
ValueError: If the row fails schema validation.

Examples:
>>> vector = validate_vector_row({
...     "UUID": "uuid",
...     "BM25": {"terms": [], "weights": [], "avgdl": 1.0, "N": 1},
...     "SPLADEv3": {"tokens": [], "weights": []},
...     "Qwen3-4B": {"model_id": "model", "vector": [0.1], "dimension": 1},
... })
>>> vector.UUID
'uuid'

### `get_docling_version()`

Detect installed Docling package version.

Args:
None: This helper does not accept arguments.

Returns:
Version string or ``"unknown"`` when docling is not installed.

Examples:
>>> isinstance(get_docling_version(), str)
True

Raises:
None: This helper does not raise exceptions.

### `validate_schema_version(version, compatible_versions)`

Ensure a schema version string is recognised.

Args:
version: Schema version string from a JSONL row.
compatible_versions: List of accepted version identifiers.
kind: Human-readable label describing the schema type.
source: Optional context describing where the version originated.

Returns:
The validated schema version.

Examples:
>>> validate_schema_version("docparse/1.1.0", COMPATIBLE_CHUNK_VERSIONS)
'docparse/1.1.0'

Raises:
ValueError: If the schema version is missing or unsupported.

### `validate_parse_engine(cls, value)`

Ensure the parse engine identifier references a supported parser.

Args:
value: Candidate parse engine label provided in the payload.

Returns:
Validated parse engine string.

Raises:
ValueError: If the parser name is not recognised.

### `_validate_schema_version(cls, value)`

Ensure chunk rows declare a supported schema identifier.

Args:
value: Schema version string provided by the chunk payload.

Returns:
Validated schema version string.

Raises:
ValueError: If the supplied schema version is not compatible.

### `validate_num_tokens(cls, value)`

Ensure token counts fall within the supported range.

Args:
value: Token count supplied in the payload.

Returns:
Validated token count.

Raises:
ValueError: If the token count is non-positive or exceeds bounds.

### `validate_page_nos(cls, value)`

Normalise and validate page numbering information.

Args:
value: Sequence of 1-based page identifiers.

Returns:
Sorted, de-duplicated list of page numbers.

Raises:
ValueError: If any page number is non-positive.

### `validate_parallel_lists(self)`

Verify that token and weight collections are aligned.

Args:
self: Instance whose term and weight lists require validation.

Returns:
The validated BM25 vector instance.

Raises:
ValueError: If the vector and weight lengths differ.

### `validate_parallel_lists(self)`

Ensure token and weight arrays stay in lock-step.

Args:
self: Instance whose tokens and weights are being validated.

Returns:
The validated SPLADE vector instance.

Raises:
ValueError: If list lengths differ or weights are negative.

### `validate_vector(cls, value)`

Ensure the dense vector contains values to embed.

Args:
value: Collection of embedding coefficients.

Returns:
The validated embedding vector.

Raises:
ValueError: If the supplied vector is empty.

### `validate_dimension(self)`

Confirm the vector length matches the declared dimension.

Args:
self: Instance whose vector dimensionality is being verified.

Returns:
The validated dense vector instance.

Raises:
ValueError: If the actual length differs from ``dimension``.

### `_validate_schema_version(cls, value)`

Ensure vector rows declare a supported schema identifier.

Args:
value: Schema version string provided by the vector payload.

Returns:
Validated schema version string.

Raises:
ValueError: If the supplied schema version is not compatible.

### `serialize(self)`

Render picture metadata into Markdown-friendly text.

Args:
item: Picture element extracted from the Docling document.
doc_serializer: Serializer orchestrating the chunking pipeline.
doc: Source document providing contextual accessors.
**_: Additional keyword arguments ignored by this implementation.

Returns:
Serialization result containing Markdown-ready text and spans.

### `_build_picture_serializer(self)`

Construct the picture serializer enriched with annotations.

### `_build_table_serializer(self)`

Construct the Markdown table serializer.

### `get_serializer(self, doc)`

Return a chunking serializer configured for ``doc``.

Args:
doc: Docling document whose metadata informs serializer selection.

Returns:
ChunkingDocSerializer: Serializer capable of producing DocTags with
caption and annotation metadata.

### `Field()`

Return default values in place of real Pydantic field descriptors.

Args:
*args: Positional arguments supplied to mimic :func:`pydantic.Field`.
**kwargs: Keyword arguments mirroring :func:`pydantic.Field` parameters.

Returns:
The provided default value or ``None`` when unspecified.

### `field_validator()`

Provide a decorator shim compatible with Pydantic field validators.

Args:
*_args: Positional decorator arguments (ignored in stub mode).
**_kwargs: Keyword decorator arguments (ignored in stub mode).

Returns:
Callable[[Callable[..., Any]], Callable[..., Any]]: Decorator passthrough.

### `model_validator()`

Provide a decorator shim compatible with model-level validators.

Args:
*_args: Positional decorator arguments (ignored in stub mode).
**_kwargs: Keyword decorator arguments (ignored in stub mode).

Returns:
Callable[[Callable[..., Any]], Callable[..., Any]]: Decorator passthrough.

### `ConfigDict()`

Return a dictionary mimicking Pydantic's ``ConfigDict`` helper.

Args:
**kwargs: Configuration keyword arguments.

Returns:
Dictionary containing the supplied configuration data.

### `_maybe_add_conf(value)`

Collect numeric confidence scores when they can be coerced to float.

### `_maybe_float(value)`

*No documentation available.*

### `model_dump(self)`

Mirror :meth:`pydantic.BaseModel.model_dump` error semantics.

Args:
*args: Positional arguments passed through from callers.
**kwargs: Keyword arguments passed through from callers.

Returns:
Never returns; the method always raises to signal missing dependency.

Raises:
RuntimeError: Always raised to indicate Pydantic is unavailable.

### `decorator(func)`

Return the wrapped function unchanged when validation is stubbed.

Args:
func: Function being decorated.

Returns:
The original function without modification.

### `decorator(func)`

Return the wrapped function unchanged when validation is stubbed.

Args:
func: Function being decorated.

Returns:
The original function without modification.

## 3. Classes

### `ProvenanceMetadata`

Stores provenance metadata extracted during chunk parsing.

Attributes:
parse_engine: Parser identifier such as ``"docling-html"`` or ``"docling-vlm"``.
docling_version: Installed Docling package version string.
has_image_captions: Flag indicating whether caption text accompanies the chunk.
has_image_classification: Flag indicating whether image classification labels exist.
num_images: Count of images referenced by the chunk.
image_confidence: Optional confidence score associated with image annotations.

Examples:
>>> ProvenanceMetadata(parse_engine="docling-html", docling_version="1.2.3")
ProvenanceMetadata(parse_engine='docling-html', docling_version='1.2.3', has_image_captions=False, has_image_classification=False, num_images=0)

### `ChunkRow`

Schema for chunk JSONL rows describing processed document segments.

Attributes:
doc_id: Document identifier shared across chunk rows.
source_path: Path to the originating DocTags file.
chunk_id: Sequential chunk index within the document.
source_chunk_idxs: Original chunk indices prior to coalescence.
num_tokens: Token count for the chunk's text body.
text: Extracted text for the chunk.
doc_items_refs: References to downstream document item metadata.
page_nos: List of 1-based page numbers touched by the chunk.
schema_version: Version identifier for the chunk schema.
provenance: Optional provenance metadata describing parsing context.
uuid: Optional stable identifier for the chunk.
has_image_captions: Optional duplicate of provenance flag for convenience.
has_image_classification: Optional duplicate of provenance flag for convenience.
num_images: Optional duplicate of provenance image count for convenience.
image_confidence: Optional duplicate of provenance image confidence for convenience.

Examples:
>>> chunk = ChunkRow(
...     doc_id="doc-123",
...     source_path="/tmp/doc-123.doctags.json",
...     chunk_id=0,
...     source_chunk_idxs=[0],
...     num_tokens=42,
...     text="Sample chunk text.",
... )
>>> chunk.doc_id
'doc-123'

### `BM25Vector`

Encapsulates BM25 sparse vector statistics for a chunk.

Attributes:
terms: Token vocabulary used in the sparse representation.
weights: BM25 weight assigned to each token.
k1: Tunable BM25 parameter controlling term frequency saturation.
b: Tunable BM25 parameter controlling length normalisation.
avgdl: Average document length across the source corpus.
N: Total document count in the source corpus.

Examples:
>>> BM25Vector(terms=["doc"], weights=[1.2], avgdl=100.0, N=10)
BM25Vector(terms=['doc'], weights=[1.2], k1=1.5, b=0.75, avgdl=100.0, N=10)

### `SPLADEVector`

Represents a SPLADE-v3 sparse activation vector.

Attributes:
model_id: Identifier of the SPLADE model that produced the vector.
tokens: Token vocabulary included in the activation map.
weights: Normalised activation weight for each token.

Examples:
>>> SPLADEVector(tokens=["term"], weights=[0.5])
SPLADEVector(model_id='naver/splade-v3', tokens=['term'], weights=[0.5])

### `DenseVector`

Stores dense embedding output from a neural encoder.

Attributes:
model_id: Identifier for the originating embedding model.
vector: Vector of numeric embedding values.
dimension: Expected dimensionality for the vector, if known.

Examples:
>>> DenseVector(model_id="encoder", vector=[0.1, 0.2], dimension=2)
DenseVector(model_id='encoder', vector=[0.1, 0.2], dimension=2)

### `VectorRow`

Schema for vector JSONL rows storing embedding artefacts.

Attributes:
UUID: Stable chunk identifier referenced by vector data.
BM25: Sparse BM25 representation for lexical retrieval.
SPLADEv3: SPLADE sparse activations supporting neural lexical search.
Qwen3_4B: Dense embedding produced by the Qwen3-4B encoder.
model_metadata: Additional metadata describing embedding provenance.
schema_version: Version identifier for the vector schema.

Examples:
>>> vector = VectorRow(
...     UUID="chunk-1",
...     BM25={"terms": ["doc"], "weights": [1.0], "avgdl": 10.0, "N": 2},
...     SPLADEv3={"tokens": ["doc"], "weights": [0.1]},
...     Qwen3_4B={"model_id": "encoder", "vector": [0.1], "dimension": 1},
... )
>>> vector.BM25.avgdl
10.0

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
>>> isinstance(provider.get_serializer(DoclingDocument()), ChunkingDocSerializer)  # doctest: +SKIP
True  # doctest: +SKIP

### `_PydanticStubBase`

Minimal stub that raises an actionable error on instantiation.

Attributes:
None: This stub purposely exposes no attributes to mimic the BaseModel API.

Examples:
>>> _PydanticStubBase()  # doctest: +IGNORE_EXCEPTION_DETAIL
Traceback (most recent call last):
...
RuntimeError: Optional dependency 'pydantic' is required ...

### `BaseModel`

Fallback BaseModel that raises informative errors when used.

Attributes:
model_config: Dictionary mirroring the ``model_config`` contract from Pydantic.

Examples:
>>> BaseModel()  # doctest: +IGNORE_EXCEPTION_DETAIL
Traceback (most recent call last):
...
RuntimeError: Optional dependency 'pydantic' is required ...
