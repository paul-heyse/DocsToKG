# 1. Module: ranking

This reference documents the DocsToKG module ``DocsToKG.HybridSearch.ranking``.

Ranking, fusion, and result shaping utilities for DocsToKG hybrid search.

## 1. Functions

### `apply_mmr_diversification(fused_candidates, fused_scores, lambda_param, top_k)`

Apply maximal marginal relevance to promote diversity.

Args:
fused_candidates: Ordered fusion candidates before diversification.
fused_scores: Precomputed fused scores for each candidate vector ID.
lambda_param: Balancing factor between relevance and diversity (0-1).
top_k: Maximum number of diversified candidates to retain.
device: GPU device identifier used for similarity computations.
resources: FAISS GPU resources used for pairwise similarity.

Returns:
List[FusionCandidate]: Diversified candidate list ordered by MMR score.

Raises:
ValueError: If ``lambda_param`` falls outside ``[0, 1]``.
RuntimeError: When GPU resources are not available for diversification.

### `fuse(self, candidates)`

Fuse ranked candidates using Reciprocal Rank Fusion.

Args:
candidates: Ranked candidates from individual retrieval channels.

Returns:
Dict[str, float]: Mapping of vector IDs to fused reciprocal-rank scores.

### `shape(self, ordered_chunks, fused_scores, request, channel_scores)`

Shape ranked candidates into final results with highlights and diagnostics.

Args:
ordered_chunks: Ranked chunks produced by hybrid fusion.
fused_scores: Mapping of vector IDs to fused scores.
request: Original hybrid search request, used for query context.
channel_scores: Channel-specific score lookups keyed by resolver name.

Returns:
List[HybridSearchResult]: Finalised results respecting per-document quotas
and duplicate suppression.

### `_within_doc_limit(self, doc_id, doc_buckets)`

*No documentation available.*

### `_is_near_duplicate(self, embeddings, current_idx, emitted_indices, pairwise)`

*No documentation available.*

### `_build_highlights(self, chunk, query_tokens)`

*No documentation available.*

## 2. Classes

### `ReciprocalRankFusion`

Combine ranked lists using Reciprocal Rank Fusion.

Attributes:
_k0: Smoothing constant used in the reciprocal rank formula.

Examples:
>>> fusion = ReciprocalRankFusion(k0=60.0)
>>> scores = fusion.fuse([])
>>> scores
{}

### `ResultShaper`

Collapse duplicates, enforce quotas, and generate highlights.

Attributes:
_opensearch: Simulator providing metadata and highlights.
_fusion_config: Fusion configuration controlling result shaping.
_gpu_device: CUDA device used for optional similarity checks.
_gpu_resources: Optional FAISS resources for GPU pairwise similarity.

Examples:
>>> shaper = ResultShaper(OpenSearchSimulator(), FusionConfig())  # doctest: +SKIP
>>> shaper.shape([], {}, HybridSearchRequest(query="", namespace=None, filters={}, page_size=1), {})  # doctest: +SKIP
[]
