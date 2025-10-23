# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.pyalex_shim",
#   "purpose": "PyAlex configuration shims for polite resolver defaults.",
#   "sections": [
#     {
#       "id": "load-pyalex-module",
#       "name": "_load_pyalex_module",
#       "anchor": "function-load-pyalex-module",
#       "kind": "function"
#     },
#     {
#       "id": "get-config",
#       "name": "get_config",
#       "anchor": "function-get-config",
#       "kind": "function"
#     },
#     {
#       "id": "set-config-attr",
#       "name": "_set_config_attr",
#       "anchor": "function-set-config-attr",
#       "kind": "function"
#     },
#     {
#       "id": "apply-mailto",
#       "name": "apply_mailto",
#       "anchor": "function-apply-mailto",
#       "kind": "function"
#     },
#     {
#       "id": "configproxy",
#       "name": "ConfigProxy",
#       "anchor": "class-configproxy",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""PyAlex configuration shims for polite resolver defaults.

Responsibilities
----------------
- Resolve the active ``pyalex`` module/config at call time, even when tests or
  harnesses monkeypatch the import after this module has loaded.
- Provide :func:`apply_mailto` so CLI invocations can set polite contact
  headers without depending on the pyalex import strategy used by callers.
- Expose :class:`ConfigProxy`, a lightweight proxy that forwards attribute
  access to the live config object, keeping resolver setup code agnostic to
  whether ``pyalex.config`` is a module or mapping.

Design Notes
------------
- All helpers tolerate the ``pyalex`` package being absent; optional dependencies
  are resolved lazily to avoid hard runtime requirements during tests.
- Attribute assignment falls back to mapping semantics to support both object
  and dict-style configs shipped by different pyalex versions.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

__all__ = ["apply_mailto", "ConfigProxy", "get_config"]


def _load_pyalex_module() -> Any | None:
    """Return the currently registered ``pyalex`` module if available."""

    module = sys.modules.get("pyalex")
    if module is not None:
        return module
    try:
        return importlib.import_module("pyalex")
    except ImportError:
        return None


def get_config() -> Any | None:
    """Return the live ``pyalex`` config object, handling late module swaps."""

    module = _load_pyalex_module()
    if module is None:
        return None
    return getattr(module, "config", None)


def _set_config_attr(config: Any, name: str, value: Any) -> None:
    """Assign an attribute on the config, falling back to mapping semantics."""

    try:
        setattr(config, name, value)
    except AttributeError:
        if isinstance(config, dict):
            config[name] = value
    except Exception:
        if isinstance(config, dict):
            config[name] = value


def apply_mailto(mailto: str) -> None:
    """Update the pyalex config contact fields when a mailto is provided."""

    if not mailto:
        return
    config = get_config()
    if config is None:
        return
    for field in ("email", "mailto"):
        _set_config_attr(config, field, mailto)


class ConfigProxy:
    """Dynamic proxy that forwards attribute access to the live pyalex config."""

    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        config = get_config()
        if config is None:
            raise AttributeError("pyalex config is unavailable")  # pragma: no cover
        return getattr(config, name)

    def __setattr__(self, name: str, value: Any) -> None:
        config = get_config()
        if config is None:
            raise AttributeError("pyalex config is unavailable")  # pragma: no cover
        _set_config_attr(config, name, value)
