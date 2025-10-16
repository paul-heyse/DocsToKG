# 1. Module: features

This reference documents the DocsToKG module ``DocsToKG.HybridSearch.devtools.features``.

## 1. Overview

Synthetic feature generation utilities for development and testing.

Args:
    None

Returns:
    None

Raises:
    None

## 2. Functions

### `tokenize(text)`

Tokenize ``text`` into lower-cased alphanumeric tokens.

Args:
text: Source string that should be segmented into tokens.

Returns:
List of tokens extracted from ``text`` with all characters lower-cased.

Examples:
>>> tokenize("Hybrid Search FTW!")
['hybrid', 'search', 'ftw']

### `tokenize_with_spans(text)`

Tokenize ``text`` and return token spans for highlight generation.

Args:
text: Source string that should be segmented while preserving offsets.

Returns:
Tuple containing the token list and a list of ``(start, end)`` offsets.

### `sliding_window(tokens, window, overlap)`

Yield sliding windows across ``tokens`` with ``overlap`` between chunks.

Args:
tokens: Token sequence that should be windowed.
window: Maximum number of tokens per emitted window (must be > 0).
overlap: Number of tokens shared between consecutive windows. Must be
less than ``window`` and greater than or equal to zero.

Yields:
Token windows with the configured overlap.

Raises:
ValueError: If ``window`` or ``overlap`` violate expected constraints.

Returns:
Iterator over token windows respecting the configured parameters.

### `embedding_dim(self)`

Return the configured dense embedding dimensionality.

Args:
None

Returns:
Positive integer describing the dense embedding dimensionality.

### `compute_features(self, text)`

Compute synthetic BM25, SPLADE, and dense vector features.

Args:
text: Chunk text that should be transformed into features.

Returns:
ChunkFeatures containing sparse weights and a dense embedding.

### `_compute_bm25(self, tokens)`

Compute BM25-style weights for ``tokens``.

Args:
tokens: Token sequence produced from the source text.

Returns:
Mapping of token to BM25-inspired weight.

### `_compute_splade(self, tokens)`

Compute SPLADE-inspired sparse weights for ``tokens``.

Args:
tokens: Token sequence produced from the source text.

Returns:
Mapping of token to SPLADE-style activation weight.

### `_compute_dense_embedding(self, tokens)`

Aggregate token hashes into a deterministic dense embedding.

Args:
tokens: Token sequence produced from the source text.

Returns:
L2-normalised dense vector with dimensionality ``embedding_dim``.

### `_hash_to_vector(self, token)`

Map a token to a deterministic dense vector using a SHA1 hash.

Args:
token: Token string to transform.

Returns:
Dense vector representing the hashed token.

### `_normalize(self, vector)`

Normalise ``vector`` to unit length if its norm is non-zero.

Args:
vector: Dense vector to normalise.

Returns:
Normalised vector with L2 norm equal to one unless ``vector`` is zero.

## 3. Classes

### `FeatureGenerator`

Deterministic feature generator used by tests and validation harnesses.

Attributes:
_embedding_dim: Dimensionality of the dense embedding vectors produced.

Examples:
>>> generator = FeatureGenerator(embedding_dim=8)
>>> features = generator.compute_features("hello hybrid search")
>>> sorted(features.bm25_terms)
['hello', 'hybrid', 'search']
