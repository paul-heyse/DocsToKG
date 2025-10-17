# 1. Module: features

This reference documents the DocsToKG module ``DocsToKG.HybridSearch.features``.

## 1. Overview

Feature generation helpers for DocsToKG hybrid search.

## 2. Functions

### `tokenize(text)`

Tokenize ``text`` into lower-cased alphanumeric tokens.

### `tokenize_with_spans(text)`

Tokenize ``text`` and return token spans for highlight generation.

### `sliding_window(tokens, window, overlap)`

Yield sliding windows across ``tokens`` with ``overlap`` between chunks.

### `embedding_dim(self)`

Return the configured dense embedding dimensionality.

### `compute_features(self, text)`

Compute synthetic BM25, SPLADE, and dense vector features.

### `_compute_bm25(self, tokens)`

*No documentation available.*

### `_compute_splade(self, tokens)`

*No documentation available.*

### `_compute_dense_embedding(self, tokens)`

*No documentation available.*

### `_hash_to_vector(self, token)`

*No documentation available.*

### `_normalize(self, vector)`

*No documentation available.*

## 3. Classes

### `FeatureGenerator`

Deterministic feature generator used by tests and validation harnesses.
