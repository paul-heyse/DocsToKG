# 1. Module: features

This reference documents the DocsToKG module ``DocsToKG.HybridSearch.features``.

## 1. Overview

Feature generation utilities for SPLADE weights and dense embeddings.

## 2. Functions

### `tokenize(text)`

Tokenize text into lowercase alphanumeric tokens.

Args:
text: Raw text to segment into tokens.

Returns:
List[str]: Lowercased alphanumeric tokens extracted from ``text``.

### `tokenize_with_spans(text)`

Return tokens alongside their character spans.

Args:
text: Raw text to tokenize while preserving character offsets.

Returns:
Tuple[List[str], List[Tuple[int, int]]]: Token list and the corresponding
``(start, end)`` character spans for each token.

### `sliding_window(tokens, window, overlap)`

Yield token windows with configurable overlap.

Args:
tokens: Token sequence to window over.
window: Maximum number of tokens per window.
overlap: Number of tokens shared between consecutive windows.

Yields:
List[str]: Next token window respecting ``window`` and ``overlap`` constraints.

Returns:
Iterator[List[str]]: Iterator producing sliding windows across ``tokens``.

Raises:
ValueError: If ``window`` or ``overlap`` violate expected ranges.

### `embedding_dim(self)`

Return the dimensionality used for synthetic dense embeddings.

Args:
None

Returns:
Integer dimensionality for generated embeddings.

### `compute_features(self, text)`

Generate BM25, SPLADE, and dense features for the supplied text.

Args:
text: Chunk text that requires feature extraction.

Returns:
ChunkFeatures containing sparse and dense representations.

### `_compute_bm25(self, tokens)`

Calculate BM25-style term weights for the provided tokens.

Args:
tokens: Sequence of string tokens extracted from chunk text.

Returns:
Mapping of token → BM25-inspired weight using log-scaled term frequency.

### `_compute_splade(self, tokens)`

Generate SPLADE-style sparse lexical weights for the tokens.

Args:
tokens: Token sequence used to derive sparse lexical weights.

Returns:
Mapping of token → SPLADE-inspired weight capturing term salience.

### `_compute_dense_embedding(self, tokens)`

Aggregate per-token hashes into a normalized dense embedding.

Args:
tokens: Tokens present in the chunk text.

Returns:
L2-normalised dense embedding representing the chunk semantics.

### `_hash_to_vector(self, token)`

Project a token deterministically into dense vector space via hashing.

Args:
token: Token string that requires a dense vector representation.

Returns:
Dense vector derived from the SHA-1 digest of the token.

### `_normalize(self, vector)`

Return a unit-length copy of the provided dense vector.

Args:
vector: Vector to normalise to unit length.

Returns:
Normalised vector, or the original vector when zero-norm.

## 3. Classes

### `FeatureGenerator`

Derive sparse and dense features for chunk text.

Attributes:
_embedding_dim: Dimensionality used for synthetic dense embeddings.

Examples:
>>> generator = FeatureGenerator(embedding_dim=8)
>>> features = generator.compute_features("hybrid search")
>>> sorted(features.bm25_terms)
['hybrid', 'search']
