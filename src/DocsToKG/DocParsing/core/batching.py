"""Batching utilities shared across DocParsing stages."""

from __future__ import annotations

from typing import Iterable, Iterator, List, Optional, Sequence, TypeVar

T = TypeVar("T")

__all__ = ["Batcher"]


class Batcher(Iterable[List[T]]):
    """Yield fixed-size batches from an iterable with optional policies."""

    def __init__(
        self,
        iterable: Iterable[T],
        batch_size: int,
        *,
        policy: Optional[str] = None,
        lengths: Optional[Sequence[int]] = None,
    ) -> None:
        """Materialise items and batching metadata for subsequent iteration."""

        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self._items: List[T] = list(iterable)
        self._batch_size = batch_size
        self._policy = (policy or "").lower() or None
        if self._policy:
            if self._policy not in {"length"}:
                raise ValueError(f"Unsupported batching policy: {policy}")
            if lengths is None:
                raise ValueError("lengths must be provided when using a policy")
            if len(lengths) != len(self._items):
                raise ValueError("lengths must align with iterable length")
            self._lengths = [int(max(0, length)) for length in lengths]
        else:
            self._lengths = None

    @staticmethod
    def _length_bucket(length: int) -> int:
        """Return the power-of-two bucket for ``length``."""

        if length <= 0:
            return 0
        return 1 << ((length - 1).bit_length())

    def _ordered_indices(self) -> List[int]:
        """Return indices ordered by bucketed length and original position."""

        if not self._lengths:
            return list(range(len(self._items)))
        pairs = [(idx, self._length_bucket(self._lengths[idx])) for idx in range(len(self._items))]
        pairs.sort(key=lambda pair: (pair[1], pair[0]))
        return [idx for idx, _ in pairs]

    def __iter__(self) -> Iterator[List[T]]:
        """Yield successive batches respecting any active policy."""

        if not self._policy:
            for i in range(0, len(self._items), self._batch_size):
                yield self._items[i : i + self._batch_size]
            return

        ordered_indices = self._ordered_indices()
        for i in range(0, len(ordered_indices), self._batch_size):
            batch_indices = ordered_indices[i : i + self._batch_size]
            yield [self._items[idx] for idx in batch_indices]
