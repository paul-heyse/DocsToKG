# Module: tokenization

Tokenization utilities shared across sparse/dense features.

## Functions

### `tokenize(text)`

Tokenize text into lowercase alphanumeric tokens.

### `tokenize_with_spans(text)`

Return tokens alongside their character spans.

### `sliding_window(tokens, window, overlap)`

Yield token windows with configurable overlap.
