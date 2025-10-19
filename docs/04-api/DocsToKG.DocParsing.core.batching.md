# 1. Module: batching

This reference documents the DocsToKG module ``DocsToKG.DocParsing.core.batching``.

## 1. Overview

Batching utilities shared across DocParsing stages.

## 2. Functions

### `_length_bucket(length)`

Return the power-of-two bucket for ``length``.

### `_ordered_indices(self)`

Return indices ordered by bucketed length and original position.

### `__iter__(self)`

Yield successive batches respecting any active policy.

## 3. Classes

### `Batcher`

Yield fixed-size batches from an iterable with optional policies.

``Batcher`` operates in two distinct modes depending on ``policy``. When no
policy is supplied the iterable is consumed lazily: items are fetched on
demand from the underlying iterator via :mod:`itertools` without storing the
entire sequence in memory. Policy-aware batching (for example ``"length"``)
requires random access to items and therefore materialises the iterable to a
list so that indices can be reordered deterministically.
