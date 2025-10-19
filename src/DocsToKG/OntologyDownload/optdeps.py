"""Optional dependency helpers for the ontology downloader.

Historically these helpers lived in a dedicated ``optdeps`` module.  The
production implementations now reside in :mod:`DocsToKG.OntologyDownload.settings`
so this compatibility layer proxies ``_import_module`` into the settings module
and resets the cached module state so tests can simulate missing dependencies.
"""

from __future__ import annotations

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
    "get_pystow": "_pystow",
    "get_rdflib": "_rdflib",
    "get_pronto": "_pronto",
    "get_owlready2": "_owlready2",
}


def _import_module(name: str) -> Any:
    """Delegate to the original settings loader."""

    return _BASE_IMPORT(name)


def _call_with_override(callback: Callable[[], Any], cache_key: Optional[str]) -> Any:
    original_import = _settings._import_module
    current_import = globals()["_import_module"]

    def _wrapped_import(name: str) -> Any:
        try:
            return current_import(name)
        except ImportError as exc:  # pragma: no cover - exercised in tests
            raise ModuleNotFoundError(str(exc)) from exc

    _settings._import_module = _wrapped_import
    cache_name = _CACHE_ATTRS.get(cache_key or "")
    if cache_name and hasattr(_settings, cache_name):
        setattr(_settings, cache_name, None)
    try:
        return callback()
    finally:
        _settings._import_module = original_import


def get_pystow() -> Any:
    return _call_with_override(_settings.get_pystow, "get_pystow")


def get_rdflib() -> Any:
    module = _call_with_override(_settings.get_rdflib, "get_rdflib")
    if getattr(module, "_ontofetch_stub", False):
        return module.Graph()
    return module


def get_pronto() -> Any:
    return _call_with_override(_settings.get_pronto, "get_pronto")


def get_owlready2() -> Any:
    return _call_with_override(_settings.get_owlready2, "get_owlready2")
