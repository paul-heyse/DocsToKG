# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.resolvers",
#   "purpose": "Resolver registry and resolver exports",
#   "sections": [
#     {
#       "id": "registered-resolvers",
#       "name": "Registered Resolvers",
#       "anchor": "section-registered-resolvers",
#       "kind": "section"
#     }
#   ]
# }
# === /NAVMAP ===
"""Resolver package with registry and concrete implementations.

All resolver classes are automatically registered via import-time side effects.
Access via registry_v2.get_registry() or registry_v2.build_resolvers().
"""

from __future__ import annotations

# ============================================================================
# Import concrete resolvers for registration side effects
# ============================================================================
# These imports trigger @register_v2 decorators and populate the registry
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

# ============================================================================
# Registry API (modern system)
# ============================================================================
from .registry_v2 import (
    ResolverProtocol,
    build_resolvers,
    get_registry,
    get_resolver_class,
    register_v2,
)
from .semantic_scholar import SemanticScholarResolver  # noqa: E402,F401
from .unpaywall import UnpaywallResolver  # noqa: E402,F401
from .wayback import WaybackResolver  # noqa: E402,F401
from .zenodo import ZenodoResolver  # noqa: E402,F401

__all__ = [
    # Concrete resolvers
    "ArxivResolver",
    "CoreResolver",
    "CrossrefResolver",
    "DoajResolver",
    "EuropePmcResolver",
    "FigshareResolver",
    "HalResolver",
    "LandingPageResolver",
    "OpenAireResolver",
    "OpenAlexResolver",
    "OsfResolver",
    "SemanticScholarResolver",
    "UnpaywallResolver",
    "WaybackResolver",
    "ZenodoResolver",
    # Registry API
    "ResolverProtocol",
    "build_resolvers",
    "get_registry",
    "get_resolver_class",
    "register_v2",
]
