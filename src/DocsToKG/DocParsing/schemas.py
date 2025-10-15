"""Pydantic schemas for DocParsing JSONL outputs with validation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CHUNK_SCHEMA_VERSION = "docparse/1.1.0"
VECTOR_SCHEMA_VERSION = "embeddings/1.0.0"
COMPATIBLE_CHUNK_VERSIONS = ["docparse/1.0.0", "docparse/1.1.0"]
COMPATIBLE_VECTOR_VERSIONS = ["embeddings/1.0.0"]

__all__ = [
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
    """Provenance metadata for chunks."""

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
        allowed = {"docling-html", "docling-vlm"}
        if value not in allowed:
            raise ValueError(f"Invalid parse_engine: {value}")
        return value


class ChunkRow(BaseModel):
    """Schema for chunk JSONL rows."""

    doc_id: str = Field(..., min_length=1, description="Document identifier")
    source_path: str = Field(..., description="Path to source DocTags file")
    chunk_id: int = Field(..., ge=0, description="Sequential chunk index within document")
    source_chunk_idxs: List[int] = Field(
        ..., description="Original chunk indices before coalescence"
    )
    num_tokens: int = Field(..., gt=0, description="Token count (must be positive)")
    text: str = Field(..., min_length=1, description="Chunk text content")
    doc_items_refs: List[str] = Field(
        default_factory=list, description="Document item references"
    )
    page_nos: List[int] = Field(default_factory=list, description="Page numbers")
    schema_version: str = Field(
        default=CHUNK_SCHEMA_VERSION, description="Schema version identifier"
    )
    provenance: Optional["ProvenanceMetadata"] = Field(
        None, description="Optional provenance metadata"
    )
    uuid: Optional[str] = Field(None, description="Optional UUID for chunk")

    @field_validator("num_tokens")
    @classmethod
    def validate_num_tokens(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("num_tokens must be positive")
        if value > 100_000:
            raise ValueError("num_tokens exceeds reasonable limit (100k)")
        return value

    @field_validator("page_nos")
    @classmethod
    def validate_page_nos(cls, value: List[int]) -> List[int]:
        if value and not all(page > 0 for page in value):
            raise ValueError("All page numbers must be positive")
        return sorted(set(value))

    model_config = ConfigDict(extra="forbid")


class BM25Vector(BaseModel):
    """BM25 sparse vector representation."""

    terms: List[str] = Field(..., description="Token terms")
    weights: List[float] = Field(..., description="BM25 weights for each term")
    k1: float = Field(default=1.5, ge=0, description="BM25 k1 parameter")
    b: float = Field(default=0.75, ge=0, le=1, description="BM25 b parameter")
    avgdl: float = Field(..., gt=0, description="Average document length in corpus")
    N: int = Field(..., gt=0, description="Total documents in corpus")

    @model_validator(mode="after")
    def validate_parallel_lists(self) -> "BM25Vector":
        if len(self.terms) != len(self.weights):
            raise ValueError("terms and weights must have same length")
        return self


class SPLADEVector(BaseModel):
    """SPLADE-v3 sparse vector representation."""

    model_id: str = Field(default="naver/splade-v3", description="SPLADE model identifier")
    tokens: List[str] = Field(..., description="SPLADE token vocabulary")
    weights: List[float] = Field(..., description="SPLADE activation weights")

    @model_validator(mode="after")
    def validate_parallel_lists(self) -> "SPLADEVector":
        if len(self.tokens) != len(self.weights):
            raise ValueError("tokens and weights must have same length")
        if any(weight < 0 for weight in self.weights):
            raise ValueError("SPLADE weights must be non-negative")
        return self


class DenseVector(BaseModel):
    """Dense embedding vector representation."""

    model_id: str = Field(..., description="Embedding model identifier")
    vector: List[float] = Field(..., description="Dense embedding vector")
    dimension: Optional[int] = Field(None, description="Expected vector dimension")

    @field_validator("vector")
    @classmethod
    def validate_vector(cls, value: List[float]) -> List[float]:
        if not value:
            raise ValueError("vector cannot be empty")
        return value

    @model_validator(mode="after")
    def validate_dimension(self) -> "DenseVector":
        if self.dimension and len(self.vector) != self.dimension:
            raise ValueError(
                f"vector dimension {len(self.vector)} != expected {self.dimension}"
            )
        return self


class VectorRow(BaseModel):
    """Schema for vector JSONL rows."""

    UUID: str = Field(..., description="Chunk UUID (must match chunk file)")
    BM25: BM25Vector = Field(..., description="BM25 sparse vector")
    SPLADEv3: SPLADEVector = Field(..., description="SPLADE-v3 sparse vector")
    Qwen3_4B: DenseVector = Field(
        ..., alias="Qwen3-4B", description="Qwen3-4B dense vector"
    )
    model_metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional model metadata"
    )
    schema_version: str = Field(
        default=VECTOR_SCHEMA_VERSION, description="Schema version identifier"
    )

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


def validate_chunk_row(row: dict) -> ChunkRow:
    """Validate and parse a chunk JSONL row.

    Args:
        row: Raw dictionary from JSONL.

    Returns:
        Validated :class:`ChunkRow` instance.

    Raises:
        ValueError: If the row fails schema validation.

    Examples:
        >>> validate_chunk_row({
        ...     "doc_id": "doc",
        ...     "source_path": "path",
        ...     "chunk_id": 0,
        ...     "source_chunk_idxs": [0],
        ...     "num_tokens": 10,
        ...     "text": "hello",
        ... })
        ChunkRow(doc_id='doc', source_path='path', chunk_id=0, source_chunk_idxs=[0], num_tokens=10, text='hello', doc_items_refs=[], page_nos=[], schema_version='docparse/1.1.0', provenance=None, uuid=None)
    """

    try:
        return ChunkRow(**row)
    except Exception as exc:  # pragma: no cover - exercised by tests raising ValueError
        doc_id = row.get("doc_id", "unknown")
        raise ValueError(
            f"Chunk row validation failed for doc_id={doc_id}: {exc}"
        ) from exc


def validate_vector_row(row: dict) -> VectorRow:
    """Validate and parse a vector JSONL row.

    Args:
        row: Raw dictionary from JSONL.

    Returns:
        Validated :class:`VectorRow` instance.

    Raises:
        ValueError: If the row fails schema validation.

    Examples:
        >>> validate_vector_row({
        ...     "UUID": "uuid",
        ...     "BM25": {"terms": [], "weights": [], "avgdl": 1.0, "N": 1},
        ...     "SPLADEv3": {"tokens": [], "weights": []},
        ...     "Qwen3-4B": {"model_id": "model", "vector": [0.1], "dimension": 1},
        ... })
        VectorRow(UUID='uuid', BM25=BM25Vector(terms=[], weights=[], k1=1.5, b=0.75, avgdl=1.0, N=1), SPLADEv3=SPLADEVector(model_id='naver/splade-v3', tokens=[], weights=[]), Qwen3_4B=DenseVector(model_id='model', vector=[0.1], dimension=1), model_metadata={}, schema_version='embeddings/1.0.0')
    """

    try:
        return VectorRow(**row)
    except Exception as exc:  # pragma: no cover - exercised by tests raising ValueError
        uuid = row.get("UUID") or row.get("uuid") or "unknown"
        raise ValueError(f"Vector row validation failed for UUID={uuid}: {exc}") from exc


def get_docling_version() -> str:
    """Detect installed docling package version.

    Returns:
        Version string or ``"unknown"`` when docling is not installed.

    Examples:
        >>> isinstance(get_docling_version(), str)
        True
    """

    try:
        import docling  # type: ignore import-not-found

        return getattr(docling, "__version__", "unknown")
    except (ImportError, AttributeError):
        return "unknown"


def validate_schema_version(version: str, compatible_versions: List[str]) -> bool:
    """Check if schema version is compatible.

    Args:
        version: Schema version string from JSONL row.
        compatible_versions: List of accepted version identifiers.

    Returns:
        ``True`` when the version is recognised, ``False`` otherwise.

    Examples:
        >>> validate_schema_version("docparse/1.1.0", COMPATIBLE_CHUNK_VERSIONS)
        True
    """

    return version in compatible_versions


ChunkRow.model_rebuild()
VectorRow.model_rebuild()
