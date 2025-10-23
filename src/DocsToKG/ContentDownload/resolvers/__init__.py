"""Resolver subsystem public exports."""

from __future__ import annotations

import importlib
import sys
from typing import Any

__all__ = [
    "ApiResolverBase",
    "ArxivResolver",
    "CoreResolver",
    "CrossrefResolver",
    "DEFAULT_RESOLVER_TOGGLES",
    "DoajResolver",
    "EuropePmcResolver",
    "FigshareResolver",
    "HalResolver",
    "LandingPageResolver",
    "OpenAireResolver",
    "OpenAlexResolver",
    "OsfResolver",
    "PmcResolver",
    "Resolver",
    "ResolverEvent",
    "ResolverEventReason",
    "ResolverResult",
    "SemanticScholarResolver",
    "UnpaywallResolver",
    "WaybackResolver",
    "ZenodoResolver",
    "find_pdf_via_anchor",
    "find_pdf_via_link",
    "find_pdf_via_meta",
    "build_resolvers",
    "get_registry",
    "register_v2",
    "ResolverProtocol",
]

_EXPORT_MAP: dict[str, tuple[str, str]] = {
    "ApiResolverBase": (".base", "ApiResolverBase"),
    "ArxivResolver": (".arxiv", "ArxivResolver"),
    "CoreResolver": (".core", "CoreResolver"),
    "CrossrefResolver": (".crossref", "CrossrefResolver"),
    "DEFAULT_RESOLVER_TOGGLES": (".base", "DEFAULT_RESOLVER_TOGGLES"),
    "DoajResolver": (".doaj", "DoajResolver"),
    "EuropePmcResolver": (".europe_pmc", "EuropePmcResolver"),
    "FigshareResolver": (".figshare", "FigshareResolver"),
    "HalResolver": (".hal", "HalResolver"),
    "LandingPageResolver": (".landing_page", "LandingPageResolver"),
    "OpenAireResolver": (".openaire", "OpenAireResolver"),
    "OpenAlexResolver": (".openalex", "OpenAlexResolver"),
    "OsfResolver": (".osf", "OsfResolver"),
    "Resolver": (".base", "Resolver"),
    "ResolverEvent": (".base", "ResolverEvent"),
    "ResolverEventReason": (".base", "ResolverEventReason"),
    "ResolverResult": (".base", "ResolverResult"),
    "ResolverProtocol": (".registry_v2", "ResolverProtocol"),
    "SemanticScholarResolver": (".semantic_scholar", "SemanticScholarResolver"),
    "UnpaywallResolver": (".unpaywall", "UnpaywallResolver"),
    "WaybackResolver": (".wayback", "WaybackResolver"),
    "ZenodoResolver": (".zenodo", "ZenodoResolver"),
    "build_resolvers": (".registry_v2", "build_resolvers"),
    "find_pdf_via_anchor": (".base", "find_pdf_via_anchor"),
    "find_pdf_via_link": (".base", "find_pdf_via_link"),
    "find_pdf_via_meta": (".base", "find_pdf_via_meta"),
    "get_registry": (".registry_v2", "get_registry"),
    "register_v2": (".registry_v2", "register_v2"),
}

_ALIAS_EXPORTS: dict[str, tuple[str, str]] = {
    "PmcResolver": (".europe_pmc", "EuropePmcResolver"),
}


def __getattr__(name: str) -> Any:  # pragma: no cover - exercised in tests
    if name in _EXPORT_MAP:
        module_path, attr_name = _EXPORT_MAP[name]
        module = importlib.import_module(f"{__name__}{module_path}")
        value = getattr(module, attr_name)
        setattr(sys.modules[__name__], name, value)
        return value
    if name in _ALIAS_EXPORTS:
        module_path, target_name = _ALIAS_EXPORTS[name]
        module = importlib.import_module(f"{__name__}{module_path}")
        value = getattr(module, target_name)
        setattr(sys.modules[__name__], name, value)
        return value
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:  # pragma: no cover - tooling helper
    return sorted(set(globals()) | set(__all__))
