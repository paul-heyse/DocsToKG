"""
Core typed structures for hybrid search components.

This module defines the fundamental data structures used throughout the DocsToKG
hybrid search system, including document inputs, chunk features, search requests,
and result structures. These types ensure type safety and provide clear contracts
for data exchange between system components.

The types support both traditional text search (BM25) and modern dense retrieval
methods, enabling seamless hybrid search capabilities across the platform.

Key Features:
- Type-safe data structures for all search operations
- Immutable data classes for thread safety
- Comprehensive validation and error handling
- Support for both lexical and semantic search modalities
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class DocumentInput:
    """Pre-computed chunk and vector artifacts for a document.

    This class represents the processed state of a document after chunking
    and vector embedding generation. It serves as the input for indexing
    operations and provides metadata about the document's processing state.

    Attributes:
        doc_id: Unique identifier for the document
        namespace: Logical grouping for the document (e.g., "research", "news")
        chunk_path: File system path to chunk data
        vector_path: File system path to vector embeddings
        metadata: Additional document metadata and processing information
        created_at: Timestamp when document was first processed
        updated_at: Timestamp of last modification

    Examples:
        >>> doc_input = DocumentInput(
        ...     doc_id="research_paper_123",
        ...     namespace="academic",
        ...     chunk_path=Path("chunks/doc_123.jsonl"),
        ...     vector_path=Path("vectors/doc_123.jsonl"),
        ...     metadata={"title": "AI Research Paper", "author": "Dr. Smith"}
        ... )
    """

    doc_id: str
    namespace: str
    chunk_path: Path
    vector_path: Path
    metadata: Mapping[str, Any]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass(slots=True)
class ChunkFeatures:
    """Sparse and dense features computed for a document chunk.

    This class encapsulates the multiple feature representations computed
    for a document chunk, supporting both traditional lexical search (BM25)
    and modern semantic search (SPLADE, dense embeddings).

    Attributes:
        bm25_terms: BM25 term frequency mapping (term -> weight)
        splade_weights: SPLADE sparse lexical and dense weights
        embedding: Dense vector embedding for semantic similarity

    Examples:
        >>> features = ChunkFeatures(
        ...     bm25_terms={"machine": 0.8, "learning": 0.6},
        ...     splade_weights={"neural": 0.9, "network": 0.7},
        ...     embedding=np.random.rand(768).astype(np.float32)
        ... )
    """

    bm25_terms: Mapping[str, float]
    splade_weights: Mapping[str, float]
    embedding: NDArray[np.float32]

    def copy(self) -> "ChunkFeatures":
        """Create a deep copy of the chunk features.

        This method creates independent copies of all feature data to prevent
        unintended mutations when features are shared across operations.

        Args:
            None

        Returns:
            New ChunkFeatures instance with copied data

        Examples:
            >>> original = ChunkFeatures(...)
            >>> copy = original.copy()
            >>> # Modifications to copy won't affect original
        """
        return ChunkFeatures(
            dict(self.bm25_terms),
            dict(self.splade_weights),
            self.embedding.copy(),
        )


@dataclass(slots=True)
class ChunkPayload:
    """Fully materialized chunk stored in OpenSearch and FAISS.

    This class represents a document chunk that has been processed and indexed
    for both lexical and semantic search. It contains all the information
    needed for hybrid retrieval operations.

    Attributes:
        doc_id: Source document identifier
        chunk_id: Unique chunk identifier within the document
        vector_id: Corresponding vector identifier in FAISS
        namespace: Logical grouping for search scoping
        text: Original chunk text content
        metadata: Mutable mapping of persisted chunk metadata
        features: Hybrid feature representations associated with the chunk
        token_count: Number of tokens contained in the chunk text
        source_chunk_idxs: Original chunk indices contributing to this payload
        doc_items_refs: References to DocTags items that produced the chunk
        char_offset: Optional character span of the chunk in the source document
    """

    doc_id: str
    chunk_id: str
    vector_id: str
    namespace: str
    text: str
    metadata: MutableMapping[str, Any]
    features: ChunkFeatures
    token_count: int
    source_chunk_idxs: Sequence[int]
    doc_items_refs: Sequence[str]
    char_offset: Optional[Tuple[int, int]] = None


@dataclass(slots=True)
class HybridSearchDiagnostics:
    """Per-channel diagnostics for a hybrid search result.

    This class provides detailed scoring information for each retrieval
    method (BM25, SPLADE, dense vectors) used in hybrid search, enabling
    analysis of individual method performance and result quality.

    Attributes:
        bm25_score: BM25 lexical similarity score (None if not used)
        splade_score: SPLADE sparse embedding score (None if not used)
        dense_score: Dense vector similarity score (None if not used)

    Examples:
        >>> diagnostics = HybridSearchDiagnostics(
        ...     bm25_score=0.85,
        ...     splade_score=0.92,
        ...     dense_score=0.78
        ... )
    """

    bm25_score: Optional[float] = None
    splade_score: Optional[float] = None
    dense_score: Optional[float] = None


@dataclass(slots=True)
class HybridSearchResult:
    """Output item returned to callers of the hybrid search service.

    This class represents a single result from hybrid search operations,
    containing the document content, metadata, and scoring information
    from all retrieval methods used in the search.

    Attributes:
        doc_id: Source document identifier
        chunk_id: Chunk identifier within the document
        namespace: Logical grouping for search scoping
        score: Final fused similarity score
        fused_rank: Position in the final result ranking
        text: Retrieved text content from the chunk
        highlights: Highlighted terms or phrases in the result
        provenance_offsets: Character offsets for result provenance
        diagnostics: Per-method scoring information
        metadata: Additional result metadata and context

    Examples:
        >>> result = HybridSearchResult(
        ...     doc_id="doc_123",
        ...     chunk_id="chunk_456",
        ...     namespace="research",
        ...     score=0.89,
        ...     text="Machine learning algorithms...",
        ...     highlights=["machine learning"],
        ...     diagnostics=HybridSearchDiagnostics(bm25_score=0.85)
        ... )
    """

    doc_id: str
    chunk_id: str
    namespace: str
    score: float
    fused_rank: int
    text: str
    highlights: Sequence[str]
    provenance_offsets: Sequence[Tuple[int, int]]
    diagnostics: HybridSearchDiagnostics
    metadata: Mapping[str, Any]


@dataclass(slots=True)
class HybridSearchRequest:
    """Validated request payload for `/v1/hybrid-search`.

    This class represents a validated hybrid search request containing
    all parameters needed for multi-modal document retrieval operations.

    Attributes:
        query: Natural language search query string
        namespace: Optional namespace for scoped search
        filters: Optional search filters as key-value pairs
        page_size: Maximum number of results to return
        cursor: Optional pagination cursor for continuation
        diversification: Whether to apply MMR diversification
        diagnostics: Whether to include per-method scoring information

    Examples:
        >>> request = HybridSearchRequest(
        ...     query="machine learning algorithms",
        ...     namespace="research",
        ...     page_size=10,
        ...     diversification=True
        ... )
    """

    query: str
    namespace: Optional[str]
    filters: Mapping[str, Any]
    page_size: int
    cursor: Optional[str] = None
    diversification: bool = False
    diagnostics: bool = True


@dataclass(slots=True)
class HybridSearchResponse:
    """Response envelope returned by the hybrid search API.

    This class encapsulates the complete response from hybrid search operations,
    including results, pagination information, and performance metrics.

    Attributes:
        results: Sequence of hybrid search results
        next_cursor: Optional cursor for pagination continuation
        total_candidates: Total number of candidates before filtering
        timings_ms: Performance timing information by operation

    Examples:
        >>> response = HybridSearchResponse(
        ...     results=[result1, result2],
        ...     next_cursor="cursor_123",
        ...     total_candidates=150,
        ...     timings_ms={"bm25": 45, "fusion": 12}
        ... )
    """

    results: Sequence[HybridSearchResult]
    next_cursor: Optional[str]
    total_candidates: int
    timings_ms: Mapping[str, float]


@dataclass(slots=True)
class FusionCandidate:
    """Intermediate structure used by fusion pipeline.

    This class represents a candidate result during the fusion process,
    tracking which retrieval method produced it, its original score,
    and its position in the original ranking.

    Attributes:
        source: Name of the retrieval method that produced this candidate
        score: Original similarity score from the source method
        chunk: Full chunk payload with metadata and features
        rank: Original rank position from the source method

    Examples:
        >>> candidate = FusionCandidate(
        ...     source="bm25",
        ...     score=0.85,
        ...     chunk=chunk_payload,
        ...     rank=3
        ... )
    """

    source: str
    score: float
    chunk: ChunkPayload
    rank: int


@dataclass(slots=True)
class ValidationReport:
    """Structured output for the validation harness.

    This class represents the result of a single validation check,
    containing the check name, pass/fail status, and detailed results.

    Attributes:
        name: Name or identifier of the validation check
        passed: Boolean indicating if validation passed
        details: Additional information about the validation results

    Examples:
        >>> report = ValidationReport(
        ...     name="api_health_check",
        ...     passed=True,
        ...     details={"response_time": 45, "status_code": 200}
        ... )
    """

    name: str
    passed: bool
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ValidationSummary:
    """Aggregate of validation reports.

    This class provides a summary of multiple validation reports,
    including timing information and overall pass/fail status.

    Attributes:
        reports: Sequence of individual validation reports
        started_at: Timestamp when validation began
        completed_at: Timestamp when validation finished

    Properties:
        passed: Boolean indicating if all validations passed

    Examples:
        >>> summary = ValidationSummary(
        ...     reports=[report1, report2],
        ...     started_at=datetime.now(),
        ...     completed_at=datetime.now()
        ... )
        >>> print(f"All validations passed: {summary.passed}")
    """

    reports: Sequence[ValidationReport]
    started_at: datetime
    completed_at: datetime

    @property
    def passed(self) -> bool:
        """Check if all validation reports passed.

        Args:
            None

        Returns:
            True if all reports passed, False otherwise
        """
        return all(report.passed for report in self.reports)
