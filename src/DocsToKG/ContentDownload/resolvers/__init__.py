"""Resolver subsystem public exports."""

from __future__ import annotations

from .arxiv import ArxivResolver
from .base import (
    ApiResolverBase,
    DEFAULT_RESOLVER_TOGGLES,
    Resolver,
    ResolverEvent,
    ResolverEventReason,
    ResolverResult,
    find_pdf_via_anchor,
    find_pdf_via_link,
    find_pdf_via_meta,
)
from .core import CoreResolver
from .crossref import CrossrefResolver
from .doaj import DoajResolver
from .europe_pmc import EuropePmcResolver
from .figshare import FigshareResolver
from .hal import HalResolver
from .landing_page import LandingPageResolver
from .openaire import OpenAireResolver
from .openalex import OpenAlexResolver
from .osf import OsfResolver
from .semantic_scholar import SemanticScholarResolver
from .unpaywall import UnpaywallResolver
from .wayback import WaybackResolver
from .zenodo import ZenodoResolver

# Legacy alias maintained for compatibility with older tests
PmcResolver = EuropePmcResolver

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
]
