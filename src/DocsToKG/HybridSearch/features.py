"""Feature generation helpers used during ingestion and synthetic tests.

The ingestion pipeline expects chunk payloads to carry BM25 stats, SPLADE-style
weights, and dense vectors. Production runs obtain these from DocParsing; test
fixtures use the deterministic helpers in this module (`FeatureGenerator`,
`tokenize`, `sliding_window`) to emulate the same shapes. The resulting
structures feed directly into `ChunkIngestionPipeline` and the FAISS-backed
store described in the README.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .types import ChunkFeatures

__all__ = ("FeatureGenerator", "sliding_window", "tokenize", "tokenize_with_spans")

_TOKEN_PATTERN = re.compile(r"[\w']+")


def tokenize(text: str) -> List[str]:
    """Tokenize ``text`` into lower-cased alphanumeric tokens."""

    return [match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text)]


def tokenize_with_spans(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Tokenize ``text`` and return token spans for highlight generation."""

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
    """Yield sliding windows across ``tokens`` with ``overlap`` between chunks."""

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
    """Deterministic feature generator used by tests and validation harnesses."""

    def __init__(self, *, embedding_dim: int = 2560) -> None:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self._embedding_dim = embedding_dim

    @property
    def embedding_dim(self) -> int:
        """Return the configured dense embedding dimensionality."""

        return self._embedding_dim

    def compute_features(self, text: str) -> ChunkFeatures:
        """Compute synthetic BM25, SPLADE, and dense vector features."""

        tokens = tokenize(text)
        return ChunkFeatures(
            bm25_terms=self._compute_bm25(tokens),
            splade_weights=self._compute_splade(tokens),
            embedding=self._compute_dense_embedding(tokens),
        )

    def _compute_bm25(self, tokens: Sequence[str]) -> Dict[str, float]:
        counter = Counter(tokens)
        return {token: float(1.0 + np.log1p(freq)) for token, freq in counter.items()}

    def _compute_splade(self, tokens: Sequence[str]) -> Dict[str, float]:
        counter = Counter(tokens)
        if not counter:
            return {}
        max_tf = max(counter.values())
        output: Dict[str, float] = {}
        for token, tf in counter.items():
            output[token] = float(np.log1p(tf) * (0.5 + tf / max_tf))
        return output

    def _compute_dense_embedding(self, tokens: Sequence[str]) -> NDArray[np.float32]:
        if not tokens:
            return np.zeros(self._embedding_dim, dtype=np.float32)
        aggregate = np.zeros(self._embedding_dim, dtype=np.float32)
        for token in tokens:
            aggregate += self._hash_to_vector(token)
        return self._normalize(aggregate)

    def _hash_to_vector(self, token: str) -> NDArray[np.float32]:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        chunk = np.frombuffer(digest, dtype=np.uint8)
        repeat = int(np.ceil(self._embedding_dim / chunk.size))
        tiled = np.tile(chunk, repeat)[: self._embedding_dim]
        return tiled.astype(np.float32) / 255.0

    def _normalize(self, vector: NDArray[np.float32]) -> NDArray[np.float32]:
        norm = np.linalg.norm(vector)
        if norm == 0.0:
            return vector
        return vector / norm
