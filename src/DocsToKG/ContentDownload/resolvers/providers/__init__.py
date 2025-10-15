"""
Resolver Provider Registry

This package aggregates resolver provider implementations and exposes a helper
for constructing the default resolver stack used by the content download
pipeline.

Key Features:
- Explicit imports of all resolver provider classes for easy discovery.
- ``default_resolvers`` helper that instantiates providers in priority order.

Usage:
    from DocsToKG.ContentDownload.resolvers.providers import default_resolvers

    resolvers = default_resolvers()
"""

from typing import List

from ..types import Resolver
from .arxiv import ArxivResolver
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
from .pmc import PmcResolver
from .semantic_scholar import SemanticScholarResolver
from .unpaywall import UnpaywallResolver
from .wayback import WaybackResolver
from .zenodo import ZenodoResolver


def default_resolvers() -> List[Resolver]:
    """Return default resolver instances in priority order.

    Args:
        None

    Returns:
        List of resolver instances ordered by preferred execution priority.
    """

    return [
        OpenAlexResolver(),
        UnpaywallResolver(),
        CrossrefResolver(),
        LandingPageResolver(),
        ArxivResolver(),
        PmcResolver(),
        EuropePmcResolver(),
        CoreResolver(),
        ZenodoResolver(),
        FigshareResolver(),
        DoajResolver(),
        SemanticScholarResolver(),
        OpenAireResolver(),
        HalResolver(),
        OsfResolver(),
        WaybackResolver(),
    ]


__all__ = ["default_resolvers"]
