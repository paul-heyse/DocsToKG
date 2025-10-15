"""Feature generation utilities for SPLADE weights and dense embeddings."""

from __future__ import annotations

import hashlib
from collections import Counter
import re
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .types import ChunkFeatures

_TOKEN_PATTERN = re.compile(r"[\w']+")


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric tokens."""

    return [token.lower() for token in _TOKEN_PATTERN.findall(text)]


def tokenize_with_spans(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Return tokens alongside their character spans."""

    tokens: List[str] = []
    spans: List[Tuple[int, int]] = []
    for match in _TOKEN_PATTERN.finditer(text):
        tokens.append(match.group(0).lower())
        spans.append((match.start(), match.end()))
    return tokens, spans


def sliding_window(tokens: Sequence[str], window: int, overlap: int) -> Iterator[List[str]]:
    """Yield token windows with configurable overlap."""

    if window <= 0:
        raise ValueError("window must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= window:
        raise ValueError("overlap must be smaller than window")

    start = 0
    step = window - overlap
    while start < len(tokens):
        end = min(len(tokens), start + window)
        yield list(tokens[start:end])
        start += step


class FeatureGenerator:
    """Derive sparse and dense features for chunk text.

    Attributes:
        _embedding_dim: Dimensionality used for synthetic dense embeddings.

    Examples:
        >>> generator = FeatureGenerator(embedding_dim=8)
        >>> features = generator.compute_features("hybrid search")
        >>> sorted(features.bm25_terms)
        ['hybrid', 'search']
    """

    def __init__(self, *, embedding_dim: int = 2560) -> None:
        """Create a feature generator with a deterministic dense embedding dimensionality.

        Args:
            embedding_dim: Dimensionality used for generated dense embeddings.

        Returns:
            None

        Raises:
            ValueError: If ``embedding_dim`` is not positive.
        """

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
        """Calculate BM25-style term weights for the provided tokens.

        Args:
            tokens: Sequence of string tokens extracted from chunk text.

        Returns:
            Mapping of token → BM25-inspired weight using log-scaled term frequency.
        """
        counter = Counter(tokens)
        return {token: float(1.0 + np.log1p(freq)) for token, freq in counter.items()}

    def _compute_splade(self, tokens: Sequence[str]) -> Dict[str, float]:
        """Generate SPLADE-style sparse lexical weights for the tokens.

        Args:
            tokens: Token sequence used to derive sparse lexical weights.

        Returns:
            Mapping of token → SPLADE-inspired weight capturing term salience.
        """
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
        """Aggregate per-token hashes into a normalized dense embedding.

        Args:
            tokens: Tokens present in the chunk text.

        Returns:
            L2-normalised dense embedding representing the chunk semantics.
        """
        if not tokens:
            return np.zeros(self._embedding_dim, dtype=np.float32)
        aggregate = np.zeros(self._embedding_dim, dtype=np.float32)
        for token in tokens:
            aggregate += self._hash_to_vector(token)
        return self._normalize(aggregate)

    def _hash_to_vector(self, token: str) -> NDArray[np.float32]:
        """Project a token deterministically into dense vector space via hashing.

        Args:
            token: Token string that requires a dense vector representation.

        Returns:
            Dense vector derived from the SHA-1 digest of the token.
        """
        digest = hashlib.sha1(token.encode("utf-8")).digest()
        required = self._embedding_dim
        repeats = (required + len(digest) - 1) // len(digest)
        buf = (digest * repeats)[:required]
        arr = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        arr = arr / 255.0 - 0.5
        return arr

    def _normalize(self, vector: NDArray[np.float32]) -> NDArray[np.float32]:
        """Return a unit-length copy of the provided dense vector.

        Args:
            vector: Vector to normalise to unit length.

        Returns:
            Normalised vector, or the original vector when zero-norm.
        """
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return vector
        return vector / norm
