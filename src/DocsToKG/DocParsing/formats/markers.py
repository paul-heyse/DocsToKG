"""Utilities for structural markers used by DocParsing chunking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

__all__ = [
    "DEFAULT_HEADING_MARKERS",
    "DEFAULT_CAPTION_MARKERS",
    "dedupe_preserve_order",
    "load_structural_marker_config",
]

# --- Defaults ---------------------------------------------------------------

DEFAULT_HEADING_MARKERS: Tuple[str, ...] = ("#",)
DEFAULT_CAPTION_MARKERS: Tuple[str, ...] = (
    "Figure caption:",
    "Table:",
    "Picture description:",
    "<!-- image -->",
)


# --- Helpers ----------------------------------------------------------------


def dedupe_preserve_order(markers: Sequence[str]) -> Tuple[str, ...]:
    """Return markers without duplicates while preserving input order."""

    seen: set[str] = set()
    ordered: List[str] = []
    for marker in markers:
        if not marker:
            continue
        if marker in seen:
            continue
        seen.add(marker)
        ordered.append(marker)
    return tuple(ordered)


def _ensure_str_list(value: object, label: str) -> List[str]:
    """Normalise configuration entries into string lists."""

    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Expected a list of strings for '{label}'")
    return [item for item in value if item]


def load_structural_marker_config(path: Path) -> Tuple[List[str], List[str]]:
    """Load user-provided heading and caption markers from JSON or YAML."""

    raw = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    data: object
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Loading YAML markers requires the 'PyYAML' package") from exc
        data = yaml.safe_load(raw) or {}
    else:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ValueError(
                    f"Unable to parse structural marker file {path}: invalid JSON and PyYAML missing."
                ) from exc
            data = yaml.safe_load(raw) or {}

    if isinstance(data, list):
        headings = _ensure_str_list(data, "headings")
        captions: List[str] = []
    elif isinstance(data, dict):
        headings = _ensure_str_list(data.get("headings"), "headings")
        captions = _ensure_str_list(data.get("captions"), "captions")
    else:
        raise ValueError(f"Unsupported structural marker format in {path!s}")

    return headings, captions


