from __future__ import annotations

from collections.abc import Iterable
from typing import Any

__all__ = ["tqdm"]


def tqdm(iterable: Iterable[Any] | None = None, **_kwargs: Any) -> Iterable[Any] | list[Any]:
    """Return the iterable unchanged (or an empty list when None)."""

    return iterable if iterable is not None else []
