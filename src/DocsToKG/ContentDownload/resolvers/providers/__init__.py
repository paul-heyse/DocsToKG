"""Resolver provider implementations and registry."""

from typing import List

from ..types import Resolver
from .arxiv import ArxivResolver
from .core import CoreResolver
from .crossref import CrossrefResolver
from .doaj import DoajResolver
from .europe_pmc import EuropePmcResolver
from .hal import HalResolver
from .landing_page import LandingPageResolver
from .openaire import OpenAireResolver
from .osf import OsfResolver
from .pmc import PmcResolver
from .semantic_scholar import SemanticScholarResolver
from .unpaywall import UnpaywallResolver
from .wayback import WaybackResolver


def default_resolvers() -> List[Resolver]:
    """Return default resolver instances in priority order."""

    return [
        UnpaywallResolver(),
        CrossrefResolver(),
        LandingPageResolver(),
        ArxivResolver(),
        PmcResolver(),
        EuropePmcResolver(),
        CoreResolver(),
        DoajResolver(),
        SemanticScholarResolver(),
        OpenAireResolver(),
        HalResolver(),
        OsfResolver(),
        WaybackResolver(),
    ]


__all__ = ["default_resolvers"]
