"""Helpers for interacting with the pyalex configuration safely.

Tests swap in fake ``pyalex`` modules (and configs) to avoid hitting the real
network client. Import-time aliases to ``pyalex.config`` become stale when that
swap happens later, so this module provides a small proxy and helper utilities
that always resolve the active configuration object at call time.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any, Optional

__all__ = ["apply_mailto", "ConfigProxy", "get_config"]


def _load_pyalex_module() -> Optional[Any]:
    """Return the currently registered ``pyalex`` module if available."""

    module = sys.modules.get("pyalex")
    if module is not None:
        return module
    try:
        return importlib.import_module("pyalex")
    except ImportError:
        return None


def get_config() -> Optional[Any]:
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
