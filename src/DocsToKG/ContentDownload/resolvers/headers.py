"""Resolver header utilities shared across provider implementations.

This module exposes helpers that normalise HTTP header dictionaries into
hashable cache keys. Centralising the logic avoids subtle drift between
provider modules and eliminates cross-imports of private helpers.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple


def headers_cache_key(headers: Dict[str, str]) -> Tuple[Tuple[str, str], ...]:
    """Return a deterministic cache key for HTTP header dictionaries.

    Args:
        headers: Mapping of header names to values.

    Returns:
        Tuple of lowercase header names paired with their original values,
        sorted alphabetically for stable hashing.
    """

    items: Iterable[Tuple[str, str]] = ((key.lower(), value) for key, value in (headers or {}).items())
    return tuple(sorted(items))


__all__ = ["headers_cache_key"]
