# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.DocParsing.formats.markers",
#   "purpose": "Shim layer preserving legacy structural marker helper imports.",
#   "sections": []
# }
# === /NAVMAP ===

"""Shim layer preserving legacy structural marker helper imports.

Earlier releases exposed structural marker utilities from ``formats.markers``.
Those helpers now live in ``DocParsing.core`` but we retain this module to avoid
breaking external automation. It re-exports the public constants and helper
functions so callers receive deprecation coverage while migrating.
"""

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
