"""Optional dependency helpers for the ontology downloader.

Several validators and resolvers rely on heavy libraries (pystow, rdflib,
pronto, owlready2).  Importing them unconditionally would slow startup and
break minimal deployments.  This module centralises the guarded import logic,
provides cache-aware accessors, and exposes hooks for tests to override the
loader behaviour.
"""

from __future__ import annotations

import sys
from typing import Any, Callable, Optional

from . import settings as _settings

__all__ = [
    "_import_module",
    "get_pystow",
    "get_rdflib",
    "get_pronto",
    "get_owlready2",
]

_BASE_IMPORT = _settings._import_module

_CACHE_ATTRS = {
    "get_pystow": ("_pystow", "pystow"),
    "get_rdflib": ("_rdflib", "rdflib"),
    "get_pronto": ("_pronto", "pronto"),
    "get_owlready2": ("_owlready2", "owlready2"),
}


def _import_module(name: str) -> Any:
    return _BASE_IMPORT(name)


def _call_with_override(callback: Callable[[], Any], cache_key: Optional[str]) -> Any:
    original_import = _settings._import_module
    current_import = globals()["_import_module"]

    def _wrapped_import(name: str) -> Any:
        try:
            return current_import(name)
        except ImportError as exc:  # pragma: no cover - exercised in tests
            raise ModuleNotFoundError(str(exc)) from exc

    cache_tuple = _CACHE_ATTRS.get(cache_key or "")
    cache_name: Optional[str]
    module_name: Optional[str]
    if cache_tuple:
        cache_name, module_name = cache_tuple
    else:
        cache_name = module_name = None

    previous_cache = None
    previous_module = None
    if cache_name and hasattr(_settings, cache_name):
        previous_cache = getattr(_settings, cache_name)
        setattr(_settings, cache_name, None)
    if module_name:
        previous_module = sys.modules.get(module_name)

    _settings._import_module = _wrapped_import
    try:
        result = callback()
    finally:
        _settings._import_module = original_import
        if cache_name and hasattr(_settings, cache_name):
            setattr(_settings, cache_name, previous_cache)
        if module_name:
            if previous_module is not None:
                sys.modules[module_name] = previous_module
            else:
                sys.modules.pop(module_name, None)
    return result


def get_pystow() -> Any:
    """Return the ``pystow`` module, loading a stub when optional deps are missing.

    Returns:
        Any: Imported ``pystow`` module or fallback stub.
    """

    return _call_with_override(_settings.get_pystow, "get_pystow")


def get_rdflib() -> Any:
    """Return an ``rdflib`` module or stub ``Graph`` implementation for testing.

    Returns:
        Any: Real ``rdflib`` module or a stub with a ``Graph`` attribute.
    """

    module = _call_with_override(_settings.get_rdflib, "get_rdflib")
    if getattr(module, "_ontofetch_stub", False):
        return module.Graph()
    return module


def get_pronto() -> Any:
    """Return the ``pronto`` module, respecting optional dependency overrides.

    Returns:
        Any: Imported ``pronto`` module or stub.
    """

    return _call_with_override(_settings.get_pronto, "get_pronto")


def get_owlready2() -> Any:
    """Return the ``owlready2`` module, applying opt-dependency fallbacks when needed.

    Returns:
        Any: Imported ``owlready2`` module or fallback stub.
    """

    return _call_with_override(_settings.get_owlready2, "get_owlready2")
