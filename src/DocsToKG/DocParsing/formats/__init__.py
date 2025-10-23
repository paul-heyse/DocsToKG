# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.DocParsing.formats.__init__",
#   "purpose": "Data models and validators for DocParsing artifact formats",
#   "sections": [
#     {
#       "id": "schemakind",
#       "name": "SchemaKind",
#       "anchor": "class-schemakind",
#       "kind": "class"
#     },
#     {
#       "id": "coerce-kind",
#       "name": "_coerce_kind",
#       "anchor": "function-coerce-kind",
#       "kind": "function"
#     },
#     {
#       "id": "get-default-schema-version",
#       "name": "get_default_schema_version",
#       "anchor": "function-get-default-schema-version",
#       "kind": "function"
#     },
#     {
#       "id": "get-compatible-versions",
#       "name": "get_compatible_versions",
#       "anchor": "function-get-compatible-versions",
#       "kind": "function"
#     },
#     {
#       "id": "validate-schema-version",
#       "name": "validate_schema_version",
#       "anchor": "function-validate-schema-version",
#       "kind": "function"
#     },
#     {
#       "id": "ensure-chunk-schema",
#       "name": "ensure_chunk_schema",
#       "anchor": "function-ensure-chunk-schema",
#       "kind": "function"
#     },
#     {
#       "id": "provenancemetadata",
#       "name": "ProvenanceMetadata",
#       "anchor": "class-provenancemetadata",
#       "kind": "class"
#     },
#     {
#       "id": "chunkrow",
#       "name": "ChunkRow",
#       "anchor": "class-chunkrow",
#       "kind": "class"
#     },
#     {
#       "id": "bm25vector",
#       "name": "BM25Vector",
#       "anchor": "class-bm25vector",
#       "kind": "class"
#     },
#     {
#       "id": "spladevector",
#       "name": "SPLADEVector",
#       "anchor": "class-spladevector",
#       "kind": "class"
#     },
#     {
#       "id": "densevector",
#       "name": "DenseVector",
#       "anchor": "class-densevector",
#       "kind": "class"
#     },
#     {
#       "id": "vectorrow",
#       "name": "VectorRow",
#       "anchor": "class-vectorrow",
#       "kind": "class"
#     },
#     {
#       "id": "validate-chunk-row",
#       "name": "validate_chunk_row",
#       "anchor": "function-validate-chunk-row",
#       "kind": "function"
#     },
#     {
#       "id": "pydantic-validate-vector-row",
#       "name": "_pydantic_validate_vector_row",
#       "anchor": "function-pydantic-validate-vector-row",
#       "kind": "function"
#     },
#     {
#       "id": "validate-vector-row",
#       "name": "validate_vector_row",
#       "anchor": "function-validate-vector-row",
#       "kind": "function"
#     },
#     {
#       "id": "get-docling-version",
#       "name": "get_docling_version",
#       "anchor": "function-get-docling-version",
#       "kind": "function"
#     },
#     {
#       "id": "captionplusannotationpictureserializer",
#       "name": "CaptionPlusAnnotationPictureSerializer",
#       "anchor": "class-captionplusannotationpictureserializer",
#       "kind": "class"
#     },
#     {
#       "id": "richserializerprovider",
#       "name": "RichSerializerProvider",
#       "anchor": "class-richserializerprovider",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""
DocParsing Formats

This module defines Pydantic models, validation helpers, and Docling serializer
providers used throughout the DocParsing pipeline. By gathering schema
definitions alongside the Markdown-aware serializers, downstream stages can
import one module for all data contract needs without relying on auxiliary
schema packages.

Key Features:
- Strict schemas for chunk JSONL rows and embedding vector rows
- Canonical schema-version helpers co-located with the models
- Convenience validators for provenance metadata and serializer utilities

Usage:
    from DocsToKG.DocParsing import formats

    validated = formats.validate_chunk_row(raw_row)
    provider = formats.RichSerializerProvider()

Dependencies:
- pydantic>=2,<3: Required for schema validation and model parsing.
- docling_core: Supplies serializer base classes consumed by DocsToKG.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, override

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

try:
    from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
except ImportError as exc:  # pragma: no cover - hard dependency guard
    raise RuntimeError(
        "DocsToKG.DocParsing requires pydantic>=2,<3; install it via "
        '`pip install "pydantic>=2,<3"`.'
    ) from exc


# --- Schema Metadata ---

SchemaVersion = str


class SchemaKind(str, Enum):
    """Enumerate schema families handled by DocParsing pipelines."""

    CHUNK = "chunk"
    VECTOR = "vector"


_DEFAULT_VERSIONS: dict[SchemaKind, SchemaVersion] = {
    SchemaKind.CHUNK: "docparse/1.1.0",
    SchemaKind.VECTOR: "embeddings/1.0.0",
}

_COMPATIBLE_VERSIONS: dict[SchemaKind, tuple[SchemaVersion, ...]] = {
    SchemaKind.CHUNK: ("docparse/1.0.0", "docparse/1.1.0"),
    SchemaKind.VECTOR: ("embeddings/1.0.0",),
}

CHUNK_SCHEMA_VERSION: SchemaVersion = _DEFAULT_VERSIONS[SchemaKind.CHUNK]
VECTOR_SCHEMA_VERSION: SchemaVersion = _DEFAULT_VERSIONS[SchemaKind.VECTOR]
COMPATIBLE_CHUNK_VERSIONS: tuple[SchemaVersion, ...] = _COMPATIBLE_VERSIONS[SchemaKind.CHUNK]
COMPATIBLE_VECTOR_VERSIONS: tuple[SchemaVersion, ...] = _COMPATIBLE_VERSIONS[SchemaKind.VECTOR]


def _coerce_kind(kind: SchemaKind | str) -> SchemaKind:
    """Normalise raw schema identifiers to ``SchemaKind`` enums."""

    if isinstance(kind, SchemaKind):
        return kind
    try:
        return SchemaKind(kind)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Unknown schema kind: {kind!r}") from exc


def get_default_schema_version(kind: SchemaKind | str) -> SchemaVersion:
    """Return the canonical schema version for ``kind``."""

    return _DEFAULT_VERSIONS[_coerce_kind(kind)]


def get_compatible_versions(kind: SchemaKind | str) -> tuple[SchemaVersion, ...]:
    """Return the tuple of compatible versions for ``kind``."""

    return _COMPATIBLE_VERSIONS[_coerce_kind(kind)]


def validate_schema_version(
    version: SchemaVersion,
    kind: SchemaKind | str,
    *,
    compatible_versions: Iterable[SchemaVersion] | None = None,
    context: str | None = None,
) -> SchemaVersion:
    """Validate that ``version`` is compatible for ``kind``."""

    schema_kind = _coerce_kind(kind)
    compatible = tuple(compatible_versions or get_compatible_versions(schema_kind))
    if version not in compatible:
        suffix = f" ({context})" if context else ""
        raise ValueError(
            f"Unsupported {schema_kind.value} schema version {version!r}{suffix}; "
            f"supported versions: {', '.join(compatible)}"
        )
    return version


def ensure_chunk_schema(
    rec: dict,
    *,
    default_version: SchemaVersion | None = None,
    context: str | None = None,
) -> dict:
    """Ensure ``rec`` declares a compatible chunk schema version."""

    version = rec.get("schema_version")
    if not version:
        rec["schema_version"] = default_version or CHUNK_SCHEMA_VERSION
        return rec

    coerced = str(version)
    validate_schema_version(
        coerced,
        SchemaKind.CHUNK,
        context=context,
    )
    rec["schema_version"] = coerced
    return rec


# --- Globals ---

__all__ = [
    "SchemaKind",
    "SchemaVersion",
    "CHUNK_SCHEMA_VERSION",
    "VECTOR_SCHEMA_VERSION",
    "COMPATIBLE_CHUNK_VERSIONS",
    "COMPATIBLE_VECTOR_VERSIONS",
    "ProvenanceMetadata",
    "ChunkRow",
    "BM25Vector",
    "SPLADEVector",
    "DenseVector",
    "VectorRow",
    "validate_chunk_row",
    "validate_vector_row",
    "get_docling_version",
    "validate_schema_version",
    "get_default_schema_version",
    "get_compatible_versions",
    "ensure_chunk_schema",
    "CaptionPlusAnnotationPictureSerializer",
    "RichSerializerProvider",
]
# --- Public Classes ---


class ProvenanceMetadata(BaseModel):
    """Stores provenance metadata extracted during chunk parsing.

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
    """

    parse_engine: str = Field(..., description="Parser used: 'docling-html' or 'docling-vlm'")
    docling_version: str = Field(..., description="Docling package version")
    has_image_captions: bool = Field(
        default=False, description="Whether chunk includes image captions"
    )
    has_image_classification: bool = Field(
        default=False, description="Whether chunk includes image classifications"
    )
    num_images: int = Field(default=0, ge=0, description="Number of images in chunk")
    image_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score associated with image annotations when available",
    )
    picture_meta: dict[str, Any] | None = Field(
        default=None,
        description="Machine-readable picture metadata emitted by serializers (captions, classifications, SMILES).",
    )

    @field_validator("parse_engine")
    @classmethod
    def validate_parse_engine(cls, value: str) -> str:
        """Ensure the parse engine identifier references a supported parser.

        Args:
            value: Candidate parse engine label provided in the payload.

        Returns:
            Validated parse engine string.

        Raises:
            ValueError: If the parser name is not recognised.
        """
        allowed = {"docling-html", "docling-vlm"}
        if value not in allowed:
            raise ValueError(f"Invalid parse_engine: {value}")
        return value


class ChunkRow(BaseModel):
    """Schema for chunk JSONL rows describing processed document segments.

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
        start_offset: Character offset of the chunk text within the source document.
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
    """

    doc_id: str = Field(..., min_length=1, description="Document identifier")
    source_path: str = Field(..., description="Path to source DocTags file")
    chunk_id: int = Field(..., ge=0, description="Sequential chunk index within document")
    source_chunk_idxs: list[int] = Field(
        ..., description="Original chunk indices before coalescence"
    )
    num_tokens: int = Field(..., gt=0, description="Token count (must be positive)")
    text: str = Field(..., min_length=1, description="Chunk text content")
    doc_items_refs: list[str] = Field(default_factory=list, description="Document item references")
    page_nos: list[int] = Field(default_factory=list, description="Page numbers")
    schema_version: str = Field(
        default=CHUNK_SCHEMA_VERSION, description="Schema version identifier"
    )
    start_offset: int | None = Field(
        default=None,
        ge=0,
        description="Character offset of the chunk within the source document",
    )
    has_image_captions: bool | None = Field(
        default=None,
        description=(
            "Convenience flag mirroring provenance.has_image_captions for quick filtering"
        ),
    )
    has_image_classification: bool | None = Field(
        default=None,
        description=(
            "Convenience flag mirroring provenance.has_image_classification for quick filtering"
        ),
    )
    num_images: int | None = Field(
        default=None,
        ge=0,
        description="Convenience count mirroring provenance.num_images",
    )
    image_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional confidence mirroring provenance.image_confidence",
    )
    provenance: ProvenanceMetadata | None = Field(None, description="Optional provenance metadata")
    uuid: str | None = Field(None, description="Optional UUID for chunk")

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: str) -> str:
        """Ensure chunk rows declare a supported schema identifier.

        Args:
            value: Schema version string provided by the chunk payload.

        Returns:
            Validated schema version string.

        Raises:
            ValueError: If the supplied schema version is not compatible.
        """

        return validate_schema_version(value, SchemaKind.CHUNK)

    @field_validator("num_tokens")
    @classmethod
    def validate_num_tokens(cls, value: int) -> int:
        """Ensure token counts fall within the supported range.

        Args:
            value: Token count supplied in the payload.

        Returns:
            Validated token count.

        Raises:
            ValueError: If the token count is non-positive or exceeds bounds.
        """
        if value <= 0:
            raise ValueError("num_tokens must be positive")
        if value > 100_000:
            raise ValueError("num_tokens exceeds reasonable limit (100k)")
        return value

    @field_validator("page_nos")
    @classmethod
    def validate_page_nos(cls, value: list[int]) -> list[int]:
        """Normalise and validate page numbering information.

        Args:
            value: Sequence of 1-based page identifiers.

        Returns:
            Sorted, de-duplicated list of page numbers.

        Raises:
            ValueError: If any page number is non-positive.
        """
        if value and not all(page > 0 for page in value):
            raise ValueError("All page numbers must be positive")
        return sorted(set(value))

    model_config = ConfigDict(extra="forbid")


class BM25Vector(BaseModel):
    """Encapsulates BM25 sparse vector statistics for a chunk.

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
    """

    terms: list[str] = Field(..., description="Token terms")
    weights: list[float] = Field(..., description="BM25 weights for each term")
    k1: float = Field(default=1.5, ge=0, description="BM25 k1 parameter")
    b: float = Field(default=0.75, ge=0, le=1, description="BM25 b parameter")
    avgdl: float = Field(..., gt=0, description="Average document length in corpus")
    N: int = Field(..., gt=0, description="Total documents in corpus")

    @model_validator(mode="after")
    def validate_parallel_lists(self) -> BM25Vector:
        """Verify that token and weight collections are aligned.

        Args:
            self: Instance whose term and weight lists require validation.

        Returns:
            The validated BM25 vector instance.

        Raises:
            ValueError: If the vector and weight lengths differ.
        """
        if len(self.terms) != len(self.weights):
            raise ValueError("terms and weights must have same length")
        return self


class SPLADEVector(BaseModel):
    """Represents a SPLADE-v3 sparse activation vector.

    Attributes:
        model_id: Identifier of the SPLADE model that produced the vector.
        tokens: Token vocabulary included in the activation map.
        weights: Normalised activation weight for each token.

    Examples:
        >>> SPLADEVector(tokens=["term"], weights=[0.5])
        SPLADEVector(model_id='naver/splade-v3', tokens=['term'], weights=[0.5])
    """

    model_id: str = Field(default="naver/splade-v3", description="SPLADE model identifier")
    tokens: list[str] = Field(..., description="SPLADE token vocabulary")
    weights: list[float] = Field(..., description="SPLADE activation weights")

    @model_validator(mode="after")
    def validate_parallel_lists(self) -> SPLADEVector:
        """Ensure token and weight arrays stay in lock-step.

        Args:
            self: Instance whose tokens and weights are being validated.

        Returns:
            The validated SPLADE vector instance.

        Raises:
            ValueError: If list lengths differ or weights are negative.
        """
        if len(self.tokens) != len(self.weights):
            raise ValueError("tokens and weights must have same length")
        if any(weight < 0 for weight in self.weights):
            raise ValueError("SPLADE weights must be non-negative")
        return self


class DenseVector(BaseModel):
    """Stores dense embedding output from a neural encoder.

    Attributes:
        model_id: Identifier for the originating embedding model.
        vector: Vector of numeric embedding values.
        dimension: Expected dimensionality for the vector, if known.

    Examples:
        >>> DenseVector(model_id="encoder", vector=[0.1, 0.2], dimension=2)
        DenseVector(model_id='encoder', vector=[0.1, 0.2], dimension=2)
    """

    model_id: str = Field(..., description="Embedding model identifier")
    vector: list[float] = Field(..., description="Dense embedding vector")
    dimension: int | None = Field(None, description="Expected vector dimension")

    @field_validator("vector")
    @classmethod
    def validate_vector(cls, value: list[float]) -> list[float]:
        """Ensure the dense vector contains values to embed.

        Args:
            value: Collection of embedding coefficients.

        Returns:
            The validated embedding vector.

        Raises:
            ValueError: If the supplied vector is empty.
        """
        if not value:
            raise ValueError("vector cannot be empty")
        return value

    @model_validator(mode="after")
    def validate_dimension(self) -> DenseVector:
        """Confirm the vector length matches the declared dimension.

        Args:
            self: Instance whose vector dimensionality is being verified.

        Returns:
            The validated dense vector instance.

        Raises:
            ValueError: If the actual length differs from ``dimension``.
        """
        if self.dimension and len(self.vector) != self.dimension:
            raise ValueError(f"vector dimension {len(self.vector)} != expected {self.dimension}")
        return self


class VectorRow(BaseModel):
    """Schema for vector JSONL rows storing embedding artefacts.

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
    """

    UUID: str = Field(..., description="Chunk UUID (must match chunk file)")
    BM25: BM25Vector = Field(..., description="BM25 sparse vector")
    SPLADEv3: SPLADEVector = Field(..., description="SPLADE-v3 sparse vector")
    Qwen3_4B: DenseVector = Field(..., alias="Qwen3-4B", description="Qwen3-4B dense vector")
    model_metadata: dict[str, Any] | None = Field(
        default_factory=dict, description="Additional model metadata"
    )
    schema_version: str = Field(
        default=VECTOR_SCHEMA_VERSION, description="Schema version identifier"
    )

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: str) -> str:
        """Ensure vector rows declare a supported schema identifier.

        Args:
            value: Schema version string provided by the vector payload.

        Returns:
            Validated schema version string.

        Raises:
            ValueError: If the supplied schema version is not compatible.
        """

        return validate_schema_version(value, SchemaKind.VECTOR)


# --- Public Functions ---


def validate_chunk_row(row: dict) -> ChunkRow:
    """Validate and parse a chunk JSONL row.

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
    """

    try:
        return ChunkRow(**row)
    except Exception as exc:  # pragma: no cover - exercised by tests raising ValueError
        doc_id = row.get("doc_id", "unknown")
        raise ValueError(f"Chunk row validation failed for doc_id={doc_id}: {exc}") from exc


def _pydantic_validate_vector_row(row: dict) -> VectorRow:
    """Validate and parse a vector JSONL row using Pydantic models.

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
    """

    try:
        return VectorRow(**row)
    except Exception as exc:  # pragma: no cover - exercised by tests raising ValueError
        uuid = row.get("UUID") or row.get("uuid") or "unknown"
        raise ValueError(f"Vector row validation failed for UUID={uuid}: {exc}") from exc


def validate_vector_row(row: dict, *, expected_dimension: int | None = None) -> VectorRow:
    """Validate a vector JSONL row and optionally enforce the dense dimension."""

    vector_row = _pydantic_validate_vector_row(row)
    qwen_vector = vector_row.Qwen3_4B
    actual_dim = qwen_vector.dimension or len(qwen_vector.vector)
    if expected_dimension is not None and actual_dim != expected_dimension:
        raise ValueError(
            f"Qwen vector dimension {actual_dim} does not match expected {expected_dimension}"
        )
    return vector_row


def get_docling_version() -> str:
    """Detect installed Docling package version.

    Args:
        None: This helper does not accept arguments.

    Returns:
        Version string or ``"unknown"`` when docling is not installed.

    Examples:
        >>> isinstance(get_docling_version(), str)
        True

    Raises:
        None: This helper does not raise exceptions.
    """

    try:
        import docling  # type: ignore import-not-found

        return getattr(docling, "__version__", "unknown")
    except (ImportError, AttributeError):
        return "unknown"


# --- Serializer Helpers ---


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

        parts: list[str] = []
        parts_append = parts.append
        annotations: list[object] = []
        description_texts: list[str] = []
        descriptions_append = description_texts.append
        classification_entries: list[dict[str, Any]] = []
        classification_append = classification_entries.append
        molecule_entries: list[dict[str, Any]] = []
        molecule_append = molecule_entries.append
        try:
            annotations = list(item.annotations or [])
        except Exception:  # pragma: no cover - defensive catch
            annotations = []

        has_caption_flag = False
        has_classification_flag = False
        try:
            caption = (item.caption_text(doc) or "").strip()
            if caption:
                parts_append(f"Figure caption: {caption}")
                has_caption_flag = True
        except Exception:  # pragma: no cover - defensive catch
            pass

        confidence_candidates: list[float] = []
        add_confidence = confidence_candidates.append

        def _maybe_add_conf(value: object) -> None:
            """Collect numeric confidence scores when they can be coerced to float."""
            try:
                if value is None:
                    return
                add_confidence(float(value))
            except (TypeError, ValueError):
                pass

        def _maybe_float(value: object) -> float | None:
            """Convert ``value`` to ``float`` when possible, otherwise return ``None``."""
            try:
                if value is None:
                    return None
                return float(value)
            except (TypeError, ValueError):
                return None

        for annotation in annotations:
            try:
                _maybe_add_conf(getattr(annotation, "confidence", None))
                _maybe_add_conf(getattr(annotation, "score", None))
                predicted = getattr(annotation, "predicted_classes", None) or []
                if predicted:
                    has_classification_flag = True
                    for cls in predicted:
                        _maybe_add_conf(getattr(cls, "confidence", None))
                        _maybe_add_conf(getattr(cls, "probability", None))
                if isinstance(annotation, PictureDescriptionData) and annotation.text:
                    desc_text = annotation.text.strip()
                    if desc_text:
                        parts_append(f"Picture description: {desc_text}")
                        descriptions_append(desc_text)
                        has_caption_flag = True
                elif isinstance(annotation, PictureClassificationData) and predicted:
                    primary = predicted[0]
                    parts_append(f"Picture type: {primary.class_name}")
                    for cls in predicted:
                        classification_append(
                            {
                                "label": getattr(cls, "class_name", str(cls)),
                                "confidence": _maybe_float(
                                    getattr(cls, "confidence", getattr(cls, "probability", None))
                                ),
                            }
                        )
                elif isinstance(annotation, PictureMoleculeData) and annotation.smi:
                    parts_append(f"SMILES: {annotation.smi}")
                    molecule_append(
                        {
                            "smiles": annotation.smi,
                            "confidence": _maybe_float(getattr(annotation, "confidence", None)),
                        }
                    )
            except Exception:  # pragma: no cover - defensive catch
                continue
        if not parts:
            parts_append("<!-- image -->")

        has_caption = has_caption_flag or (parts and parts[0] != "<!-- image -->")
        has_classification = has_classification_flag
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
            if confidence_candidates:
                candidate = max(confidence_candidates)
                existing = flags.get("image_confidence")
                try:
                    existing_val = float(existing) if existing is not None else None
                except (TypeError, ValueError):
                    existing_val = None
                if existing_val is None or candidate > existing_val:
                    flags["image_confidence"] = candidate
            item._docstokg_flags = flags
        except Exception:
            pass

        structured_meta: dict[str, Any] = {}
        if caption:
            structured_meta["caption"] = caption
        if description_texts:
            structured_meta["descriptions"] = description_texts
        if classification_entries:
            structured_meta["classifications"] = classification_entries
        if molecule_entries:
            structured_meta["molecules"] = molecule_entries
        if confidence_candidates:
            structured_meta["confidence"] = max(confidence_candidates)
        if structured_meta:
            try:
                item._docstokg_meta = structured_meta
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
        >>> isinstance(provider.get_serializer(DoclingDocument()), ChunkingDocSerializer)  # doctest: +SKIP
        True  # doctest: +SKIP
    """

    markdown_params: MarkdownParams

    def __init__(self, markdown_params: MarkdownParams | None = None) -> None:
        """Initialise the serializer provider with Markdown defaults.

        Args:
            markdown_params: Optional override for Markdown rendering parameters.
        """

        self.markdown_params = markdown_params or MarkdownParams()

    def _build_picture_serializer(self) -> MarkdownPictureSerializer:
        """Construct the picture serializer enriched with annotations."""

        params = getattr(self.markdown_params, "picture", None)
        if params is None:
            return CaptionPlusAnnotationPictureSerializer()
        return CaptionPlusAnnotationPictureSerializer(params=params)

    def _build_table_serializer(self) -> MarkdownTableSerializer:
        """Construct the Markdown table serializer."""

        params = getattr(self.markdown_params, "table", None)
        if params is None:
            return MarkdownTableSerializer()
        return MarkdownTableSerializer(params=params)

    @override
    def get_serializer(self, doc: DoclingDocument) -> ChunkingDocSerializer:
        """Return a chunking serializer configured for ``doc``.

        Args:
            doc: Docling document whose metadata informs serializer selection.

        Returns:
            ChunkingDocSerializer: Serializer capable of producing DocTags with
            caption and annotation metadata.
        """

        serializer_kwargs: dict[str, object] = {
            "doc": doc,
            "picture_serializer": self._build_picture_serializer(),
            "table_serializer": self._build_table_serializer(),
        }
        core_params = getattr(self.markdown_params, "core", None)
        if core_params is not None:
            serializer_kwargs["params"] = core_params
        serializer = ChunkingDocSerializer(**serializer_kwargs)
        return serializer


ChunkRow.model_rebuild()
VectorRow.model_rebuild()
