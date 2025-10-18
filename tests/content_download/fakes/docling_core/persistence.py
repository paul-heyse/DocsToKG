"""Persistence helpers for the content download fake dependency tree."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

__all__ = ["manifest_append", "manifest_load", "reset_manifest"]

_MANIFESTS: Dict[str, List[Dict[str, Any]]] = {}


def _normalise_path(path: str | Path) -> str:
    return str(Path(path))


def manifest_append(path: str | Path, entry: Dict[str, Any]) -> None:
    """Record a manifest entry for the supplied path."""

    key = _normalise_path(path)
    entries = _MANIFESTS.setdefault(key, [])
    entries.append(dict(entry))


def manifest_load(path: str | Path) -> List[Dict[str, Any]]:
    """Return the recorded manifest entries for ``path``."""

    key = _normalise_path(path)
    return [dict(entry) for entry in _MANIFESTS.get(key, [])]


def reset_manifest(path: str | Path | None = None) -> None:
    """Clear the cached manifest entries (used by tests)."""

    if path is None:
        _MANIFESTS.clear()
    else:
        _MANIFESTS.pop(_normalise_path(path), None)
