"""Core typed structures for hybrid search components."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class DocumentInput:
    """Pre-computed chunk and vector artifacts for a document."""

    doc_id: str
    namespace: str
    chunk_path: Path
    vector_path: Path
    metadata: Mapping[str, Any]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass(slots=True)
class ChunkFeatures:
    """Sparse and dense features computed for a chunk."""

    bm25_terms: Mapping[str, float]
    splade_weights: Mapping[str, float]
    embedding: NDArray[np.float32]

    def copy(self) -> "ChunkFeatures":
        return ChunkFeatures(
            dict(self.bm25_terms),
            dict(self.splade_weights),
            self.embedding.copy(),
        )


@dataclass(slots=True)
class ChunkPayload:
    """Fully materialized chunk stored in OpenSearch and FAISS."""

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
    """Per-channel diagnostics for a result."""

    bm25_score: Optional[float] = None
    splade_score: Optional[float] = None
    dense_score: Optional[float] = None


@dataclass(slots=True)
class HybridSearchResult:
    """Output item returned to callers of the hybrid search service."""

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
    """Validated request payload for `/v1/hybrid-search`."""

    query: str
    namespace: Optional[str]
    filters: Mapping[str, Any]
    page_size: int
    cursor: Optional[str] = None
    diversification: bool = False
    diagnostics: bool = True


@dataclass(slots=True)
class HybridSearchResponse:
    """Response envelope returned by the hybrid search API."""

    results: Sequence[HybridSearchResult]
    next_cursor: Optional[str]
    total_candidates: int
    timings_ms: Mapping[str, float]


@dataclass(slots=True)
class FusionCandidate:
    """Intermediate structure used by fusion pipeline."""

    source: str
    score: float
    chunk: ChunkPayload
    rank: int


@dataclass(slots=True)
class ValidationReport:
    """Structured output for the validation harness."""

    name: str
    passed: bool
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ValidationSummary:
    """Aggregate of validation reports."""

    reports: Sequence[ValidationReport]
    started_at: datetime
    completed_at: datetime

    @property
    def passed(self) -> bool:
        return all(report.passed for report in self.reports)

