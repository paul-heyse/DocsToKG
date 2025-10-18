"""Embedding stage package with modular subcomponents."""

from .config import EmbedCfg, EMBED_PROFILE_PRESETS
from .cli import EMBED_CLI_OPTIONS, build_parser, parse_args
from .runtime import main
from .runtime import *  # noqa: F401,F403 - re-export legacy runtime symbols
from . import runtime as _runtime
from DocsToKG.DocParsing.formats import BM25Vector, DenseVector, SPLADEVector, VectorRow
from DocsToKG.DocParsing.logging import (
    manifest_log_failure,
    manifest_log_skip,
    manifest_log_success,
)

__all__ = [
    "EmbedCfg",
    "EMBED_PROFILE_PRESETS",
    "EMBED_CLI_OPTIONS",
    "build_parser",
    "parse_args",
    "main",
]

# Export the names brought in from runtime to maintain backwards compatibility.
for _name in list(globals()):
    if _name.startswith("_") or _name in {"config", "cli", "runtime"}:
        continue
    if _name not in __all__:
        __all__.append(_name)

for _compat_name in ("_ensure_splade_dependencies", "_ensure_qwen_dependencies"):
    if hasattr(_runtime, _compat_name):
        globals()[_compat_name] = getattr(_runtime, _compat_name)
        if _compat_name not in __all__:
            __all__.append(_compat_name)
del _compat_name

del _name
