"""
DocParsing Schema Definitions

This module defines Pydantic models and helpers that validate chunk and vector
payloads across the DocsToKG pipeline. By enforcing consistent schemas prior to
persistence or downstream processing, the utilities safeguard search, indexing,
and analytics stages from malformed data.

Key Features:
- Strict schemas for chunk JSONL rows and embedding vector rows
- Optional dependency stubs for environments without Pydantic installed
- Convenience validators for schema versions and provenance metadata

Usage:
    from DocsToKG.DocParsing import schemas

    validated = schemas.validate_chunk_row(raw_row)

Dependencies:
- pydantic (optional): Offers model validation when available; graceful fallbacks
  raise informative errors otherwise.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

PYDANTIC_AVAILABLE = True

try:  # Optional dependency
    from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

    _PYDANTIC_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised via tests with stubs
    PYDANTIC_AVAILABLE = False
    _PYDANTIC_IMPORT_ERROR = exc

    class _PydanticStubBase:
        """Minimal stub that raises an actionable error on instantiation.

        Attributes:
            None: This stub purposely exposes no attributes to mimic the BaseModel API.

        Examples:
            >>> _PydanticStubBase()  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
            ...
            RuntimeError: Optional dependency 'pydantic' is required ...
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Prevent instantiation when Pydantic is unavailable.

            Args:
                *args: Positional arguments forwarded by callers expecting Pydantic.
                **kwargs: Keyword arguments forwarded by callers expecting Pydantic.

            Returns:
                None

            Raises:
                RuntimeError: Always raised instructing users to install Pydantic.
            """

            raise RuntimeError(_missing_pydantic_message()) from _PYDANTIC_IMPORT_ERROR

        def model_dump(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - stub
            """Mirror :meth:`pydantic.BaseModel.model_dump` error semantics.

            Args:
                *args: Positional arguments passed through from callers.
                **kwargs: Keyword arguments passed through from callers.

            Returns:
                Never returns; the method always raises to signal missing dependency.

            Raises:
                RuntimeError: Always raised to indicate Pydantic is unavailable.
            """

            raise RuntimeError(_missing_pydantic_message()) from _PYDANTIC_IMPORT_ERROR

    class BaseModel(_PydanticStubBase):  # type: ignore[no-redef]
        """Fallback BaseModel that raises informative errors when used.

        Attributes:
            model_config: Dictionary mirroring the ``model_config`` contract from Pydantic.

        Examples:
            >>> BaseModel()  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
            ...
            RuntimeError: Optional dependency 'pydantic' is required ...
        """

        model_config: Dict[str, Any] = {}

    def Field(*args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        """Return default values in place of real Pydantic field descriptors.

        Args:
            *args: Positional arguments supplied to mimic :func:`pydantic.Field`.
            **kwargs: Keyword arguments mirroring :func:`pydantic.Field` parameters.

        Returns:
            The provided default value or ``None`` when unspecified.
        """

        return kwargs.get("default", args[0] if args else None)

    def field_validator(*_args: Any, **_kwargs: Any):  # type: ignore[override]
        """Provide a decorator shim compatible with Pydantic field validators.

        Args:
            *_args: Positional decorator arguments (ignored in stub mode).
            **_kwargs: Keyword decorator arguments (ignored in stub mode).

        Returns:
            Callable[[Callable[..., Any]], Callable[..., Any]]: Decorator passthrough.
        """

        def decorator(func):
            """Return the wrapped function unchanged when validation is stubbed.

            Args:
                func: Function being decorated.

            Returns:
                The original function without modification.
            """

            return func

        return decorator

    def model_validator(*_args: Any, **_kwargs: Any):  # type: ignore[override]
        """Provide a decorator shim compatible with model-level validators.

        Args:
            *_args: Positional decorator arguments (ignored in stub mode).
            **_kwargs: Keyword decorator arguments (ignored in stub mode).

        Returns:
            Callable[[Callable[..., Any]], Callable[..., Any]]: Decorator passthrough.
        """

        def decorator(func):
            """Return the wrapped function unchanged when validation is stubbed.

            Args:
                func: Function being decorated.

            Returns:
                The original function without modification.
            """

            return func

        return decorator

    def ConfigDict(**kwargs: Any) -> Dict[str, Any]:  # type: ignore[override]
        """Return a dictionary mimicking Pydantic's ``ConfigDict`` helper.

        Args:
            **kwargs: Configuration keyword arguments.

        Returns:
            Dictionary containing the supplied configuration data.
        """

        return dict(kwargs)


def _missing_pydantic_message() -> str:
    """Return a consistent optional dependency warning message."""

    return (
        "Optional dependency 'pydantic' is required for DocParsing schema "
        "validation. Install it with `pip install 'pydantic>=2,<3'` to enable "
        "the validation helpers."
    )


CHUNK_SCHEMA_VERSION = "docparse/1.1.0"
VECTOR_SCHEMA_VERSION = "embeddings/1.0.0"
COMPATIBLE_CHUNK_VERSIONS = ["docparse/1.0.0", "docparse/1.1.0"]
COMPATIBLE_VECTOR_VERSIONS = ["embeddings/1.0.0"]

__all__ = [
    "PYDANTIC_AVAILABLE",
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
]


class ProvenanceMetadata(BaseModel):
    """Stores provenance metadata extracted during chunk parsing.

    Attributes:
        parse_engine: Parser identifier such as ``"docling-html"`` or ``"docling-vlm"``.
        docling_version: Installed Docling package version string.
        has_image_captions: Flag indicating whether caption text accompanies the chunk.
        has_image_classification: Flag indicating whether image classification labels exist.
        num_images: Count of images referenced by the chunk.

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
        provenance: Optional provenance metadata describing parsing context.
        uuid: Optional stable identifier for the chunk.
        has_image_captions: Optional duplicate of provenance flag for convenience.
        has_image_classification: Optional duplicate of provenance flag for convenience.
        num_images: Optional duplicate of provenance image count for convenience.

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
    source_chunk_idxs: List[int] = Field(
        ..., description="Original chunk indices before coalescence"
    )
    num_tokens: int = Field(..., gt=0, description="Token count (must be positive)")
    text: str = Field(..., min_length=1, description="Chunk text content")
    doc_items_refs: List[str] = Field(default_factory=list, description="Document item references")
    page_nos: List[int] = Field(default_factory=list, description="Page numbers")
    schema_version: str = Field(
        default=CHUNK_SCHEMA_VERSION, description="Schema version identifier"
    )
    provenance: Optional["ProvenanceMetadata"] = Field(
        None, description="Optional provenance metadata"
    )
    uuid: Optional[str] = Field(None, description="Optional UUID for chunk")
    has_image_captions: Optional[bool] = Field(
        default=None,
        description=(
            "Convenience flag mirroring provenance.has_image_captions for quick filtering"
        ),
    )
    has_image_classification: Optional[bool] = Field(
        default=None,
        description=(
            "Convenience flag mirroring provenance.has_image_classification"
        ),
    )
    num_images: Optional[int] = Field(
        default=None,
        ge=0,
        description=("Convenience image count copied from provenance metadata"),
    )

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: str) -> str:
        """Ensure chunk rows declare a supported schema identifier."""

        return validate_schema_version(
            value,
            COMPATIBLE_CHUNK_VERSIONS,
            kind="chunk",
        )

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
    def validate_page_nos(cls, value: List[int]) -> List[int]:
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

    terms: List[str] = Field(..., description="Token terms")
    weights: List[float] = Field(..., description="BM25 weights for each term")
    k1: float = Field(default=1.5, ge=0, description="BM25 k1 parameter")
    b: float = Field(default=0.75, ge=0, le=1, description="BM25 b parameter")
    avgdl: float = Field(..., gt=0, description="Average document length in corpus")
    N: int = Field(..., gt=0, description="Total documents in corpus")

    @model_validator(mode="after")
    def validate_parallel_lists(self) -> "BM25Vector":
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
    tokens: List[str] = Field(..., description="SPLADE token vocabulary")
    weights: List[float] = Field(..., description="SPLADE activation weights")

    @model_validator(mode="after")
    def validate_parallel_lists(self) -> "SPLADEVector":
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
    vector: List[float] = Field(..., description="Dense embedding vector")
    dimension: Optional[int] = Field(None, description="Expected vector dimension")

    @field_validator("vector")
    @classmethod
    def validate_vector(cls, value: List[float]) -> List[float]:
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
    def validate_dimension(self) -> "DenseVector":
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
    model_metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional model metadata"
    )
    schema_version: str = Field(
        default=VECTOR_SCHEMA_VERSION, description="Schema version identifier"
    )

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: str) -> str:
        """Ensure vector rows declare a supported schema identifier."""

        return validate_schema_version(
            value,
            COMPATIBLE_VECTOR_VERSIONS,
            kind="vector",
        )


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

    if not PYDANTIC_AVAILABLE:
        raise RuntimeError(_missing_pydantic_message()) from _PYDANTIC_IMPORT_ERROR

    try:
        return ChunkRow(**row)
    except Exception as exc:  # pragma: no cover - exercised by tests raising ValueError
        doc_id = row.get("doc_id", "unknown")
        raise ValueError(f"Chunk row validation failed for doc_id={doc_id}: {exc}") from exc


def validate_vector_row(row: dict) -> VectorRow:
    """Validate and parse a vector JSONL row.

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

    if not PYDANTIC_AVAILABLE:
        raise RuntimeError(_missing_pydantic_message()) from _PYDANTIC_IMPORT_ERROR

    try:
        return VectorRow(**row)
    except Exception as exc:  # pragma: no cover - exercised by tests raising ValueError
        uuid = row.get("UUID") or row.get("uuid") or "unknown"
        raise ValueError(f"Vector row validation failed for UUID={uuid}: {exc}") from exc


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


def validate_schema_version(
    version: Optional[str],
    compatible_versions: List[str],
    *,
    kind: str = "schema",
    source: Optional[str] = None,
) -> str:
    """Ensure a schema version string is recognised.

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
    """

    if not version:
        location = f" from {source}" if source else ""
        expected = ", ".join(sorted(compatible_versions))
        raise ValueError(
            f"Missing {kind} schema_version{location}. Expected one of: {expected}"
        )
    if version not in compatible_versions:
        location = f" in {source}" if source else ""
        expected = ", ".join(sorted(compatible_versions))
        raise ValueError(
            f"Unsupported {kind} schema_version '{version}'{location}. "
            f"Supported versions: {expected}"
        )
    return version


if PYDANTIC_AVAILABLE:
    ChunkRow.model_rebuild()
    VectorRow.model_rebuild()
