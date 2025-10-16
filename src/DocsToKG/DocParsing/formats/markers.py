"""Deprecated structural marker helpers retained for backwards compatibility."""

from __future__ import annotations

from DocsToKG.DocParsing.core import (
    DEFAULT_CAPTION_MARKERS,
    DEFAULT_HEADING_MARKERS,
    dedupe_preserve_order,
    load_structural_marker_config,
    load_structural_marker_profile,
)

__all__ = [
    "DEFAULT_HEADING_MARKERS",
    "DEFAULT_CAPTION_MARKERS",
    "dedupe_preserve_order",
    "load_structural_marker_config",
    "load_structural_marker_profile",
]
