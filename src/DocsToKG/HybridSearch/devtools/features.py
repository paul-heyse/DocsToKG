"""Synthetic feature generation utilities for development and testing.

Args:
    None

Returns:
    None

Raises:
    None
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from ..types import ChunkFeatures

__all__ = ("FeatureGenerator", "sliding_window", "tokenize", "tokenize_with_spans")

_TOKEN_PATTERN = re.compile(r"[\w']+")


def tokenize(text: str) -> List[str]:
    """Tokenize ``text`` into lower-cased alphanumeric tokens.

    Args:
        text: Source string that should be segmented into tokens.

    Returns:
        List of tokens extracted from ``text`` with all characters lower-cased.

    Examples:
        >>> tokenize("Hybrid Search FTW!")
        ['hybrid', 'search', 'ftw']
    """
    return [match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text)]


def tokenize_with_spans(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Tokenize ``text`` and return token spans for highlight generation.

    Args:
        text: Source string that should be segmented while preserving offsets.

    Returns:
        Tuple containing the token list and a list of ``(start, end)`` offsets.
    """
    tokens: List[str] = []
    spans: List[Tuple[int, int]] = []
    for match in _TOKEN_PATTERN.finditer(text):
        tokens.append(match.group(0).lower())
        spans.append((match.start(), match.end()))
    return tokens, spans


def sliding_window(
    tokens: Sequence[str],
    window: int,
    overlap: int,
) -> Iterator[List[str]]:
    """Yield sliding windows across ``tokens`` with ``overlap`` between chunks.

    Args:
        tokens: Token sequence that should be windowed.
        window: Maximum number of tokens per emitted window (must be > 0).
        overlap: Number of tokens shared between consecutive windows. Must be
            less than ``window`` and greater than or equal to zero.

    Yields:
        Token windows with the configured overlap.

    Raises:
        ValueError: If ``window`` or ``overlap`` violate expected constraints.
    """
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
    """Deterministic feature generator used by tests and validation harnesses.

    Attributes:
        _embedding_dim: Dimensionality of the dense embedding vectors produced.

    Examples:
        >>> generator = FeatureGenerator(embedding_dim=8)
        >>> features = generator.compute_features("hello hybrid search")
        >>> sorted(features.bm25_terms)
        ['hello', 'hybrid', 'search']
    """

    def __init__(self, *, embedding_dim: int = 2560) -> None:
        """Initialize a feature generator with a deterministic embedding dimension.

        Args:
            embedding_dim: Size of the dense embedding space to generate.

        Raises:
            ValueError: If ``embedding_dim`` is not positive.
        """
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self._embedding_dim = embedding_dim

    @property
    def embedding_dim(self) -> int:
        """Return the configured dense embedding dimensionality.

        Returns:
            Positive integer describing the dense embedding dimensionality.
        """
        return self._embedding_dim

    def compute_features(self, text: str) -> ChunkFeatures:
        """Compute synthetic BM25, SPLADE, and dense vector features.

        Args:
            text: Chunk text that should be transformed into features.

        Returns:
            ChunkFeatures containing sparse weights and a dense embedding.
        """
        tokens = tokenize(text)
        return ChunkFeatures(
            bm25_terms=self._compute_bm25(tokens),
            splade_weights=self._compute_splade(tokens),
            embedding=self._compute_dense_embedding(tokens),
        )

    def _compute_bm25(self, tokens: Sequence[str]) -> Dict[str, float]:
        """Compute BM25-style weights for ``tokens``.

        Args:
            tokens: Token sequence produced from the source text.

        Returns:
            Mapping of token to BM25-inspired weight.
        """
        counter = Counter(tokens)
        return {token: float(1.0 + np.log1p(freq)) for token, freq in counter.items()}

    def _compute_splade(self, tokens: Sequence[str]) -> Dict[str, float]:
        """Compute SPLADE-inspired sparse weights for ``tokens``.

        Args:
            tokens: Token sequence produced from the source text.

        Returns:
            Mapping of token to SPLADE-style activation weight.
        """
        counter = Counter(tokens)
        if not counter:
            return {}
        max_tf = max(counter.values())
        output: Dict[str, float] = {}
        for token, tf in counter.items():
            output[token] = float(np.log1p(tf) * (0.5 + tf / max_tf))
        return output

    def _compute_dense_embedding(self, tokens: Sequence[str]) -> NDArray[np.float32]:
        """Aggregate token hashes into a deterministic dense embedding.

        Args:
            tokens: Token sequence produced from the source text.

        Returns:
            L2-normalised dense vector with dimensionality ``embedding_dim``.
        """
        if not tokens:
            return np.zeros(self._embedding_dim, dtype=np.float32)
        aggregate = np.zeros(self._embedding_dim, dtype=np.float32)
        for token in tokens:
            aggregate += self._hash_to_vector(token)
        return self._normalize(aggregate)

    def _hash_to_vector(self, token: str) -> NDArray[np.float32]:
        """Map a token to a deterministic dense vector using a SHA1 hash.

        Args:
            token: Token string to transform.

        Returns:
            Dense vector representing the hashed token.
        """
        digest = hashlib.sha1(token.encode("utf-8")).digest()
        required = self._embedding_dim
        repetitions = (required + len(digest) - 1) // len(digest)
        buffer = (digest * repetitions)[:required]
        arr = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
        arr = arr / 255.0 - 0.5
        return arr

    def _normalize(self, vector: NDArray[np.float32]) -> NDArray[np.float32]:
        """Normalise ``vector`` to unit length if its norm is non-zero.

        Args:
            vector: Dense vector to normalise.

        Returns:
            Normalised vector with L2 norm equal to one unless ``vector`` is zero.
        """
        norm = float(np.linalg.norm(vector))
        return vector if norm == 0.0 else vector / norm
