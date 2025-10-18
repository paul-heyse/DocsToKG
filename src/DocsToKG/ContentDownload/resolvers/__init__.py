# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.resolvers",
#   "purpose": "Resolver registry and concrete resolver exports",
#   "sections": [
#     {
#       "id": "default-resolvers",
#       "name": "default_resolvers",
#       "anchor": "function-default-resolvers",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===
"""Resolver package providing registry helpers and concrete implementations."""

from __future__ import annotations

from typing import List

from .base import (
    DEFAULT_RESOLVER_ORDER,
    DEFAULT_RESOLVER_TOGGLES,
    ApiResolverBase,
    BeautifulSoup,
    RegisteredResolver,
    Resolver,
    ResolverEvent,
    ResolverEventReason,
    ResolverRegistry,
    ResolverResult,
    XMLParsedAsHTMLWarning,
    _absolute_url,
    _collect_candidate_urls,
    _fetch_semantic_scholar_data,
    _fetch_unpaywall_data,
    find_pdf_via_anchor,
    find_pdf_via_link,
    find_pdf_via_meta,
)


def default_resolvers() -> List[Resolver]:
    """Instantiate the default resolver stack in priority order."""

    return ResolverRegistry.create_default()


# Import concrete resolvers for registration side effects.
from .arxiv import ArxivResolver  # noqa: E402,F401
from .core import CoreResolver  # noqa: E402,F401
from .crossref import CrossrefResolver  # noqa: E402,F401
from .doaj import DoajResolver  # noqa: E402,F401
from .europe_pmc import EuropePmcResolver  # noqa: E402,F401
from .figshare import FigshareResolver  # noqa: E402,F401
from .hal import HalResolver  # noqa: E402,F401
from .landing_page import LandingPageResolver  # noqa: E402,F401
from .openaire import OpenAireResolver  # noqa: E402,F401
from .openalex import OpenAlexResolver  # noqa: E402,F401
from .osf import OsfResolver  # noqa: E402,F401
from .pmc import PmcResolver  # noqa: E402,F401
from .semantic_scholar import SemanticScholarResolver  # noqa: E402,F401
from .unpaywall import UnpaywallResolver  # noqa: E402,F401
from .wayback import WaybackResolver  # noqa: E402,F401
from .zenodo import ZenodoResolver  # noqa: E402,F401

__all__ = [
    "ApiResolverBase",
    "ArxivResolver",
    "BeautifulSoup",
    "CoreResolver",
    "CrossrefResolver",
    "DEFAULT_RESOLVER_ORDER",
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
    "ResolverRegistry",
    "ResolverResult",
    "RegisteredResolver",
    "SemanticScholarResolver",
    "UnpaywallResolver",
    "WaybackResolver",
    "XMLParsedAsHTMLWarning",
    "ZenodoResolver",
    "_absolute_url",
    "_collect_candidate_urls",
    "_fetch_semantic_scholar_data",
    "_fetch_unpaywall_data",
    "default_resolvers",
    "find_pdf_via_anchor",
    "find_pdf_via_link",
    "find_pdf_via_meta",
]
