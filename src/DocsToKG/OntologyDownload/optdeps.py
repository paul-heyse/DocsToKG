"""Optional dependency helpers for the ontology downloader.

Historically these helpers lived in a dedicated ``optdeps`` module, and the
tests still import them from this location.  The production implementation has
since consolidated the logic inside :mod:`DocsToKG.OntologyDownload.settings`.
To preserve the public surface area we simply re-export the maintained helpers
from ``settings``.
"""

from __future__ import annotations

from typing import Any

from .settings import (
    _import_module as _settings_import_module,
)
from .settings import (
    get_owlready2 as _settings_get_owlready2,
)
from .settings import (
    get_pronto as _settings_get_pronto,
)
from .settings import (
    get_pystow as _settings_get_pystow,
)
from .settings import (
    get_rdflib as _settings_get_rdflib,
)

__all__ = [
    "_import_module",
    "get_owlready2",
    "get_pronto",
    "get_pystow",
    "get_rdflib",
]


def _import_module(name: str) -> Any:
    """Delegate to the settings helper for importing modules.

    Exposed so tests can patch it and simulate missing dependencies.
    """

    return _settings_import_module(name)


def get_pystow() -> Any:
    """Return the ``pystow`` module, deferring to the settings implementation."""

    return _settings_get_pystow()


def get_rdflib() -> Any:
    """Return the ``rdflib`` module with stub fallback."""

    return _settings_get_rdflib()


def get_pronto() -> Any:
    """Return the ``pronto`` module with stub fallback."""

    return _settings_get_pronto()


def get_owlready2() -> Any:
    """Return the ``owlready2`` module with stub fallback."""

    return _settings_get_owlready2()

