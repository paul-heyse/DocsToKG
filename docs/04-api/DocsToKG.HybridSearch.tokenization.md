# 1. Module: tokenization

This reference documents the DocsToKG module ``DocsToKG.HybridSearch.features``.

Tokenization utilities shared across sparse/dense features.

## 1. Functions

### `tokenize(text)`

Tokenize text into lowercase alphanumeric tokens.

Args:
text: Input string to tokenize.

Returns:
List of lowercase tokens extracted from the text.

### `tokenize_with_spans(text)`

Return tokens alongside their character spans.

Args:
text: Input string to tokenize.

Returns:
Tuple containing the token list and matching (start, end) spans.

### `sliding_window(tokens, window, overlap)`

Yield token windows with configurable overlap.

Args:
tokens: Sequence of tokens to segment.
window: Number of tokens in each window.
overlap: Number of tokens overlapping between consecutive windows.

Yields:
Lists of tokens representing each window.

Returns:
Iterator producing token windows.

Raises:
ValueError: If parameters do not describe a valid sliding window.
