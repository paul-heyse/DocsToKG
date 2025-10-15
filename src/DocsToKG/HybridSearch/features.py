"""Feature generation utilities for SPLADE weights and dense embeddings."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Dict, Sequence

import numpy as np
from numpy.typing import NDArray

from .tokenization import tokenize
from .types import ChunkFeatures


class FeatureGenerator:
    """Derive sparse and dense features for chunk text."""

    def __init__(self, *, embedding_dim: int = 2560) -> None:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self._embedding_dim = embedding_dim

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality used for synthetic dense embeddings.

        Args:
            None

        Returns:
            Integer dimensionality for generated embeddings.
        """
        return self._embedding_dim

    def compute_features(self, text: str) -> ChunkFeatures:
        """Generate BM25, SPLADE, and dense features for the supplied text.

        Args:
            text: Chunk text that requires feature extraction.

        Returns:
            ChunkFeatures containing sparse and dense representations.
        """
        tokens = tokenize(text)
        bm25_terms = self._compute_bm25(tokens)
        splade_weights = self._compute_splade(tokens)
        embedding = self._compute_dense_embedding(tokens)
        return ChunkFeatures(
            bm25_terms=bm25_terms, splade_weights=splade_weights, embedding=embedding
        )

    def _compute_bm25(self, tokens: Sequence[str]) -> Dict[str, float]:
        counter = Counter(tokens)
        return {token: float(1.0 + np.log1p(freq)) for token, freq in counter.items()}

    def _compute_splade(self, tokens: Sequence[str]) -> Dict[str, float]:
        counter = Counter(tokens)
        if not counter:
            return {}
        max_tf = max(counter.values())
        weights: Dict[str, float] = {}
        for token, tf in counter.items():
            weight = float(np.log1p(tf) * (0.5 + tf / max_tf))
            weights[token] = weight
        return weights

    def _compute_dense_embedding(self, tokens: Sequence[str]) -> NDArray[np.float32]:
        if not tokens:
            return np.zeros(self._embedding_dim, dtype=np.float32)
        aggregate = np.zeros(self._embedding_dim, dtype=np.float32)
        for token in tokens:
            aggregate += self._hash_to_vector(token)
        return self._normalize(aggregate)

    def _hash_to_vector(self, token: str) -> NDArray[np.float32]:
        digest = hashlib.sha1(token.encode("utf-8")).digest()
        required = self._embedding_dim
        repeats = (required + len(digest) - 1) // len(digest)
        buf = (digest * repeats)[:required]
        arr = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        arr = arr / 255.0 - 0.5
        return arr

    def _normalize(self, vector: NDArray[np.float32]) -> NDArray[np.float32]:
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return vector
        return vector / norm
