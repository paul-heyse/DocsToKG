"""Chunking stage package with modular subcomponents."""

from .config import CHUNK_PROFILE_PRESETS, ChunkerCfg, SOFT_BARRIER_MARGIN
from .cli import CHUNK_CLI_OPTIONS, build_parser, parse_args
from .runtime import main
from .runtime import *  # noqa: F401,F403 - re-export legacy runtime symbols

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

del _name
