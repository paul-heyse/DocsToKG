from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

__all__ = [
    "SparseEncoder",
]


class _SparseValues:
    def __init__(self, count: int) -> None:
        self._count = count

    def numel(self) -> int:
        return self._count


class _SparseRow:
    def __init__(self, tokens: List[str], weights: List[float]) -> None:
        self.tokens = tokens
        self.weights = weights

    def coalesce(self) -> "_SparseRow":
        return self

    def values(self) -> _SparseValues:
        return _SparseValues(len(self.tokens))


class _SparseBatch:
    def __init__(self, rows: Iterable[Tuple[List[str], List[float]]]) -> None:
        self._rows = [_SparseRow(tokens, weights) for tokens, weights in rows]

    @property
    def shape(self) -> tuple[int, int]:
        if not self._rows:
            return (0, 0)
        width = max((len(row.tokens) for row in self._rows), default=0)
        return (len(self._rows), width)

    def __getitem__(self, index: int) -> _SparseRow:
        return self._rows[index]


class SparseEncoder:
    """
    Minimal stub for sentence-transformers `SparseEncoder`.

    The behaviour mirrors the dynamic version previously defined in
    `tests.docparsing.stubs`.
    """

    def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - signature compatibility
        pass

    def encode(self, texts: Sequence[str]) -> _SparseBatch:
        rows: List[Tuple[List[str], List[float]]] = []
        for text in texts:
            tokens = text.lower().split()
            if not tokens:
                tokens = ["synthetic"]
            weights = [float(len(tok)) / max(len(tokens), 1) for tok in tokens]
            rows.append((tokens, weights))
        return _SparseBatch(rows)

    def decode(self, row: _SparseRow, top_k: int) -> List[tuple[str, float]]:
        pairs = list(zip(row.tokens, row.weights))
        return pairs[:top_k]
