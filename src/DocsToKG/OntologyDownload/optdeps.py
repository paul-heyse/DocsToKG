# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.optdeps",
#   "purpose": "Implements DocsToKG.OntologyDownload.optdeps behaviors and helpers",
#   "sections": [
#     {
#       "id": "_opt_deps_proxy",
#       "name": "_OptDepsProxy",
#       "anchor": "OPTD",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""Optional dependency helpers for the ontology downloader."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any, Iterable

from . import ontology_download as _core

# Names that should return the corresponding attribute from the core module.
_FORWARD_EXPORTS = {
    "get_pystow": _core.get_pystow,
    "get_rdflib": _core.get_rdflib,
    "get_pronto": _core.get_pronto,
    "get_owlready2": _core.get_owlready2,
}

# Attributes whose mutation should be forwarded to the core module so that test
# helpers clearing caches or monkeypatching helper functions affect the shared
# implementation.
_MUTABLE_EXPORTS = {
    "_pystow",
    "_rdflib",
    "_pronto",
    "_owlready2",
    "_import_module",
}


class _OptDepsProxy(ModuleType):
    """Module proxy that forwards access to ``ontology_download`` internals.

    Attributes:
        __slots__: Prevents accidental attribute assignment on the proxy type.

    Examples:
        >>> proxy = _OptDepsProxy('DocsToKG.OntologyDownload.optdeps')
        >>> callable(proxy.get_rdflib)
        True
    """

    __slots__ = ()

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - exercised via tests
        if name in _FORWARD_EXPORTS:
            return _FORWARD_EXPORTS[name]
        if name in _MUTABLE_EXPORTS:
            return getattr(_core, name)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _MUTABLE_EXPORTS:
            setattr(_core, name, value)
            return
        super().__setattr__(name, value)

    def __dir__(self) -> Iterable[str]:
        return sorted(set(super().__dir__()) | set(_FORWARD_EXPORTS) | _MUTABLE_EXPORTS)


_proxy = _OptDepsProxy(__name__)
_proxy.__dict__.update(
    {
        "__doc__": __doc__,
        "__all__": tuple(_FORWARD_EXPORTS),
    }
)
# Register the proxy so future imports receive the forwarding module.
sys.modules[__name__] = _proxy
