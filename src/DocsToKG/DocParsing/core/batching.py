# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.DocParsing.core.batching",
#   "purpose": "Flexible batching helpers used by DocParsing chunking and embedding flows.",
#   "sections": [
#     {
#       "id": "batcher",
#       "name": "Batcher",
#       "anchor": "class-batcher",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""Flexible batching helpers used by DocParsing chunking and embedding flows.

Chunking and embedding stages process large collections of documents, so they
rely on deterministic batching to manage throughput without exhausting memory.
This module provides a policy-aware ``Batcher`` iterable that can stream data in
fixed chunks or rebalance work based on precomputed lengths (for example token
counts). Keeping the batching logic isolated allows each stage to reuse the same
implementation while customising parameters suited to their workloads.
"""

from __future__ import annotations

from itertools import islice
from typing import Iterable, Iterator, List, Optional, Sequence, TypeVar

T = TypeVar("T")

__all__ = ["Batcher"]


class Batcher(Iterable[List[T]]):
    """Yield fixed-size batches from an iterable with optional policies.

    ``Batcher`` operates in two distinct modes depending on ``policy``. When no
    policy is supplied the iterable is consumed lazily: items are fetched on
    demand from the underlying iterator via :mod:`itertools` without storing the
    entire sequence in memory. Policy-aware batching (for example ``"length"``)
    requires random access to items and therefore materialises the iterable to a
    list so that indices can be reordered deterministically.
    """

    def __init__(
        self,
        iterable: Iterable[T],
        batch_size: int,
        *,
        policy: Optional[str] = None,
        lengths: Optional[Sequence[int]] = None,
    ) -> None:
        """Initialise batching metadata for streaming or materialised modes."""

        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self._batch_size = batch_size
        self._policy = (policy or "").lower() or None
        self._iterable: Optional[Iterable[T]] = None
        self._items: Optional[List[T]] = None
        self._lengths: Optional[List[int]] = None

        if not self._policy:
            self._iterable = iterable
            return

        if self._policy not in {"length"}:
            raise ValueError(f"Unsupported batching policy: {policy}")
        if lengths is None:
            raise ValueError("lengths must be provided when using a policy")

        self._items = list(iterable)
        if len(lengths) != len(self._items):
            raise ValueError("lengths must align with iterable length")
        self._lengths = [int(max(0, length)) for length in lengths]

    @staticmethod
    def _length_bucket(length: int) -> int:
        """Return the power-of-two bucket for ``length``."""

        if length <= 0:
            return 0
        return 1 << ((length - 1).bit_length())

    def _ordered_indices(self) -> List[int]:
        """Return indices ordered by bucketed length and original position."""

        assert self._items is not None
        if not self._lengths:
            return list(range(len(self._items)))
        pairs = [(idx, self._length_bucket(self._lengths[idx])) for idx in range(len(self._items))]
        pairs.sort(key=lambda pair: (pair[1], pair[0]))
        return [idx for idx, _ in pairs]

    def __iter__(self) -> Iterator[List[T]]:
        """Yield successive batches respecting any active policy."""

        if not self._policy:
            assert self._iterable is not None
            iterator = iter(self._iterable)
            while True:
                batch = list(islice(iterator, self._batch_size))
                if not batch:
                    break
                yield batch
            return

        assert self._items is not None
        ordered_indices = self._ordered_indices()
        for i in range(0, len(ordered_indices), self._batch_size):
            batch_indices = ordered_indices[i : i + self._batch_size]
            yield [self._items[idx] for idx in batch_indices]
