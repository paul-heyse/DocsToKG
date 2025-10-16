"""Deprecated GPU similarity shim maintained for backward compatibility."""

from __future__ import annotations

import warnings

from .vectorstore import (
    cosine_against_corpus_gpu,
    max_inner_product,
    normalize_rows,
    pairwise_inner_products,
)

__all__ = [
    "cosine_against_corpus_gpu",
    "max_inner_product",
    "normalize_rows",
    "pairwise_inner_products",
]

warnings.warn(
    "DocsToKG.HybridSearch.similarity is deprecated and will be removed in v0.6.0. "
    "Import GPU similarity helpers from DocsToKG.HybridSearch.vectorstore instead.",
    DeprecationWarning,
    stacklevel=2,
)
