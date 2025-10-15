"""Resolver provider implementations and registry used by the content download pipeline."""

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
from .zenodo import ZenodoResolver
from .unpaywall import UnpaywallResolver
from .wayback import WaybackResolver


def default_resolvers() -> List[Resolver]:
    """Return default resolver instances in priority order.

    Args:
        None

    Returns:
        List of resolver instances ordered by preferred execution priority.
    """

    return [
        UnpaywallResolver(),
        CrossrefResolver(),
        LandingPageResolver(),
        ArxivResolver(),
        PmcResolver(),
        EuropePmcResolver(),
        CoreResolver(),
        ZenodoResolver(),
        DoajResolver(),
        SemanticScholarResolver(),
        OpenAireResolver(),
        HalResolver(),
        OsfResolver(),
        WaybackResolver(),
    ]


__all__ = ["default_resolvers"]
