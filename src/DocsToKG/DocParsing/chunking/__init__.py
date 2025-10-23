# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.DocParsing.chunking.__init__",
#   "purpose": "Chunking stage package exporting CLI, configuration, and runtime helpers.",
#   "sections": []
# }
# === /NAVMAP ===

"""Chunking stage package exporting CLI, configuration, and runtime helpers.

The chunking package stitches together the streaming runtime, CLI surface, and
configuration presets that slice DocTags into topic-aware chunks. Importing this
package provides convenient access to the canonical parser builders, reusable
logging utilities, atomic manifest writers, and backwards-compatible runtime
symbols so existing pipelines can adopt newer chunking features without
rewriting their integration code.
"""

from DocsToKG.DocParsing.io import atomic_write
from DocsToKG.DocParsing.logging import (
    manifest_log_failure,
    manifest_log_skip,
    manifest_log_success,
)

from . import runtime as _runtime
from .cli import CHUNK_CLI_OPTIONS, build_parser, parse_args
from .config import CHUNK_PROFILE_PRESETS, SOFT_BARRIER_MARGIN, ChunkerCfg
from .runtime import *  # noqa: F401,F403 - re-export legacy runtime symbols
from .runtime import main

__all__ = [
    "ChunkerCfg",
    "CHUNK_PROFILE_PRESETS",
    "SOFT_BARRIER_MARGIN",
    "CHUNK_CLI_OPTIONS",
    "build_parser",
    "parse_args",
    "main",
    "atomic_write",
    "manifest_log_failure",
    "manifest_log_skip",
    "manifest_log_success",
]

for _name in list(globals()):
    if _name.startswith("_") or _name in {"config", "cli", "runtime"}:
        continue
    if _name not in __all__:
        __all__.append(_name)

# Compatibility shims for legacy imports (tests rely on these attributes existing).
_LEGACY_EXPORTS = (
    "AutoTokenizer",
    "HuggingFaceTokenizer",
    "HybridChunker",
    "ProvenanceMetadata",
    "ChunkRow",
    "get_docling_version",
    "manifest_log_failure",
    "manifest_log_success",
    "manifest_log_skip",
    "atomic_write",
    "_LOGGER",
)

for _legacy_name in _LEGACY_EXPORTS:
    if hasattr(_runtime, _legacy_name):
        globals()[_legacy_name] = getattr(_runtime, _legacy_name)
        if _legacy_name not in __all__:
            __all__.append(_legacy_name)
    elif _legacy_name in globals():
        # Provided via DocsToKG.DocParsing.logging imports above.
        if _legacy_name not in __all__:
            __all__.append(_legacy_name)

del _legacy_name, _LEGACY_EXPORTS
del _name
