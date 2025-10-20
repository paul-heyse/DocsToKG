"""Shared helper utilities for embedding providers.

These helpers are intentionally lightweight so concrete providers can import
them without triggering heavy optional dependencies (for example, vLLM or
sentence-transformers).  Keeping normalisation and coercion logic in one place
reduces the amount of bespoke glue each provider has to maintain.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


def resolve_device(requested: Optional[str], *, default: str = "cpu") -> str:
    """Normalise device hints emitted by configuration layers.

    Providers treat ``"auto"`` as a cue to pick their preferred backend, while
    truthy values are returned verbatim.  Empty strings fall back to
    ``default`` to guard against CLI/env overrides that accidentally supply
    whitespace.
    """

    if not requested:
        return default
    candidate = str(requested).strip().lower()
    if candidate == "auto" or not candidate:
        return default
    return candidate


def resolve_cache_dir(path: Optional[Path]) -> Optional[Path]:
    """Return an absolute path suitable for model cache directories."""

    if path is None:
        return None
    return Path(path).expanduser().resolve()


def normalise_vectors(vectors: Sequence[Sequence[float]], *, normalise: bool) -> List[List[float]]:
    """Optionally L2-normalise each vector.

    Args:
        vectors: Iterable of numerical vectors.
        normalise: When ``True`` the vectors are L2 normalised.  When ``False``
            the original values are returned.

    Returns:
        A list containing the normalised (or untouched) vectors.
    """

    normalised: List[List[float]] = []
    for vector in vectors:
        values = [float(value) for value in vector]
        if not normalise:
            normalised.append(values)
            continue
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0:
            normalised.append(values)
            continue
        normalised.append([value / norm for value in values])
    return normalised


def bounded_batch_size(*, preferred: Optional[int], fallback: int) -> int:
    """Return a sane, positive batch size given configuration hints."""

    candidate = preferred or fallback
    return max(1, int(candidate))


def coerce_telemetry_tags(tags: Optional[Iterable[tuple[str, str]]]) -> dict[str, str]:
    """Convert tag key/value pairs into a dictionary with consistent str types."""

    payload: dict[str, str] = {}
    if not tags:
        return payload
    for key, value in tags:
        payload[str(key)] = str(value)
    return payload

