"""Tokenization utilities shared across sparse/dense features."""
from __future__ import annotations

import re
from typing import Iterable, Iterator, List, Sequence, Tuple

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

