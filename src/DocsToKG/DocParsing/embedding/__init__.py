"""Embedding stage package with modular subcomponents."""

from .config import EmbedCfg, EMBED_PROFILE_PRESETS
from .cli import EMBED_CLI_OPTIONS, build_parser, parse_args
from .runtime import main
from .runtime import *  # noqa: F401,F403 - re-export legacy runtime symbols

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

del _name
