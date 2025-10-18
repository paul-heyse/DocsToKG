"""Chunking stage package with modular subcomponents."""

from .config import CHUNK_PROFILE_PRESETS, ChunkerCfg, SOFT_BARRIER_MARGIN
from .cli import CHUNK_CLI_OPTIONS, build_parser, parse_args
from .runtime import main
from .runtime import *  # noqa: F401,F403 - re-export legacy runtime symbols
from . import runtime as _runtime
from DocsToKG.DocParsing.logging import (
    manifest_log_failure,
    manifest_log_skip,
    manifest_log_success,
)
from DocsToKG.DocParsing.io import atomic_write

__all__ = [
    "ChunkerCfg",
    "CHUNK_PROFILE_PRESETS",
    "SOFT_BARRIER_MARGIN",
    "CHUNK_CLI_OPTIONS",
    "build_parser",
    "parse_args",
    "main",
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
)

for _legacy_name in _LEGACY_EXPORTS:
    if hasattr(_runtime, _legacy_name):
        globals()[_legacy_name] = getattr(_runtime, _legacy_name)
        if _legacy_name not in __all__:
            __all__.append(_legacy_name)
    else:
        # Provided via DocsToKG.DocParsing.logging imports above.
        globals()[_legacy_name] = globals()[_legacy_name]
        if _legacy_name not in __all__:
            __all__.append(_legacy_name)

del _legacy_name, _LEGACY_EXPORTS
del _name
