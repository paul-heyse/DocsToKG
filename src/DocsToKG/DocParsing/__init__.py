"""High-level facade exposing consolidated DocParsing modules."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:  # pragma: no cover - import-time hints only
    from . import chunking as chunking  # noqa: F401  (re-exported at runtime)
    from . import core as core  # noqa: F401 (re-exported at runtime)
    from . import doctags as doctags  # noqa: F401 (re-exported at runtime)
    from . import embedding as embedding  # noqa: F401 (re-exported at runtime)
    from . import formats as formats  # noqa: F401 (re-exported at runtime)
    from . import token_profiles as token_profiles  # noqa: F401 (re-exported at runtime)


_LAZY_ATTR_MODULES: Dict[str, str] = {
    "chunking": "DocsToKG.DocParsing.chunking",
    "core": "DocsToKG.DocParsing.core",
    "doctags": "DocsToKG.DocParsing.doctags",
    "embedding": "DocsToKG.DocParsing.embedding",
    "formats": "DocsToKG.DocParsing.formats",
    "token_profiles": "DocsToKG.DocParsing.token_profiles",
}

_MODULE_CACHE: Dict[str, ModuleType] = {}


def _import_module(module_name: str):
    """Wrapper around :func:`import_module` for monkeypatch-friendly indirection."""

    return import_module(module_name)


def _load_module(name: str) -> ModuleType:
    """Load a module by name, using cache for performance."""
    if name in _MODULE_CACHE:
        return _MODULE_CACHE[name]

    module_name = _LAZY_ATTR_MODULES[name]
    try:
        module = _import_module(module_name)
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in tests
        missing = getattr(exc, "name", None) or str(exc)
        raise ImportError(
            "DocsToKG.DocParsing.{name} could not be imported because the optional "
            "dependency '{missing}' is not installed. Install the appropriate extras, "
            'for example `pip install "DocsToKG[docling,gpu]"` to enable this module.'.format(
                name=name, missing=missing
            )
        ) from exc

    globals()[name] = module
    _MODULE_CACHE[name] = module
    return module


__all__ = [
    "core",
    "formats",
    "doctags",
    "chunking",
    "embedding",
    "token_profiles",
    "plan",
    "manifest",
    "pdf_build_parser",
    "pdf_parse_args",
    "pdf_main",
    "html_build_parser",
    "html_parse_args",
    "html_main",
]


def __getattr__(name: str) -> Any:
    """Dynamically import submodules only when they are requested."""

    if name in _LAZY_ATTR_MODULES:
        return _load_module(name)
    raise AttributeError(f"module 'DocsToKG.DocParsing' has no attribute '{name}'")


def __dir__() -> list[str]:
    """Ensure lazily exposed attributes appear in :func:`dir` results."""

    return sorted(set(globals()) | set(_LAZY_ATTR_MODULES))


def plan(*args, **kwargs):
    """Proxy to :func:`DocsToKG.DocParsing.core.plan` with lazy loading."""

    function = _load_module("core").plan
    globals()["plan"] = function
    return function(*args, **kwargs)


def manifest(*args, **kwargs):
    """Proxy to :func:`DocsToKG.DocParsing.core.manifest` with lazy loading."""

    function = _load_module("core").manifest
    globals()["manifest"] = function
    return function(*args, **kwargs)


def pdf_build_parser(*args, **kwargs):
    """Proxy to :func:`DocsToKG.DocParsing.doctags.pdf_build_parser`."""

    function = _load_module("doctags").pdf_build_parser
    globals()["pdf_build_parser"] = function
    return function(*args, **kwargs)


def pdf_parse_args(*args, **kwargs):
    """Proxy to :func:`DocsToKG.DocParsing.doctags.pdf_parse_args`."""

    function = _load_module("doctags").pdf_parse_args
    globals()["pdf_parse_args"] = function
    return function(*args, **kwargs)


def pdf_main(*args, **kwargs):
    """Proxy to :func:`DocsToKG.DocParsing.doctags.pdf_main`."""

    function = _load_module("doctags").pdf_main
    globals()["pdf_main"] = function
    return function(*args, **kwargs)


def html_build_parser(*args, **kwargs):
    """Proxy to :func:`DocsToKG.DocParsing.doctags.html_build_parser`."""

    function = _load_module("doctags").html_build_parser
    globals()["html_build_parser"] = function
    return function(*args, **kwargs)


def html_parse_args(*args, **kwargs):
    """Proxy to :func:`DocsToKG.DocParsing.doctags.html_parse_args`."""

    function = _load_module("doctags").html_parse_args
    globals()["html_parse_args"] = function
    return function(*args, **kwargs)


def html_main(*args, **kwargs):
    """Proxy to :func:`DocsToKG.DocParsing.doctags.html_main`."""

    function = _load_module("doctags").html_main
    globals()["html_main"] = function
    return function(*args, **kwargs)
