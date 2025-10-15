# 1. Module: results

This reference documents the DocsToKG module ``DocsToKG.HybridSearch.ranking``.

Result shaping utilities for hybrid search responses.

## 1. Functions

### `shape(self, ordered_chunks, fused_scores, request, channel_scores)`

Transform ordered chunks into shaped search results.

Args:
ordered_chunks: Ranked chunk payloads emitted by fusion.
fused_scores: Final fused score per vector ID.
request: Original hybrid search request used for context.
channel_scores: Per-channel scoring maps keyed by vector ID.

Returns:
List of `HybridSearchResult` objects ready for response serialization.

### `_within_doc_limit(self, doc_id, doc_buckets)`

Check and update per-document emission counts.

Args:
doc_id: Document identifier being considered for emission.
doc_buckets: Mutable counter of chunks emitted per document.

Returns:
True if the document is still below the configured limit.

### `_is_near_duplicate(self, embeddings, current_idx, emitted_indices, pairwise)`

Determine whether the current chunk is too similar to emitted ones.

Args:
embeddings: Matrix of chunk embeddings ordered to match `ordered_chunks`.
current_idx: Index of the chunk currently under consideration.
emitted_indices: Indices that have already been emitted in the final results.

Returns:
True when the current chunk's embedding exceeds the cosine similarity threshold.

### `_build_highlights(self, chunk, query_tokens)`

Generate highlight snippets for a chunk.

Args:
chunk: Chunk payload being rendered.
query_tokens: Tokens derived from the user's query.

Returns:
List of highlight strings; falls back to a snippet when necessary.

## 2. Classes

### `ResultShaper`

Collapse duplicates, enforce quotas, and generate highlights.

Attributes:
_opensearch: OpenSearch simulator providing highlighting hooks.
_fusion_config: Fusion configuration controlling dedupe thresholds and quotas.

Examples:
>>> shaper = ResultShaper(OpenSearchSimulator(), FusionConfig())
>>> shaper._fusion_config.max_chunks_per_doc
3
