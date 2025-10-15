"""Resolver pipeline and provider implementations.

This module maintains backward compatibility by re-exporting all public APIs.
New code should import from submodules (pipeline, types, providers) directly.
"""

from .pipeline import ResolverPipeline
from .providers import (
    ArxivResolver,
    CoreResolver,
    CrossrefResolver,
    DoajResolver,
    EuropePmcResolver,
    HalResolver,
    LandingPageResolver,
    OpenAireResolver,
    OsfResolver,
    PmcResolver,
    SemanticScholarResolver,
    UnpaywallResolver,
    WaybackResolver,
    default_resolvers,
)
from .types import (
    DEFAULT_RESOLVER_ORDER as _DEFAULT_RESOLVER_ORDER,
)
from .types import (
    AttemptLogger,
    AttemptRecord,
    DownloadFunc,
    DownloadOutcome,
    PipelineResult,
    Resolver,
    ResolverConfig,
    ResolverMetrics,
    ResolverResult,
)

DEFAULT_RESOLVER_ORDER = _DEFAULT_RESOLVER_ORDER


def clear_resolver_caches() -> None:
    """Clear resolver-level LRU caches.

    Args:
        None

    Returns:
        None
    """

    from .providers.crossref import _fetch_crossref_data
    from .providers.semantic_scholar import _fetch_semantic_scholar_data
    from .providers.unpaywall import _fetch_unpaywall_data

    _fetch_unpaywall_data.cache_clear()
    _fetch_crossref_data.cache_clear()
    _fetch_semantic_scholar_data.cache_clear()


__all__ = [
    "AttemptRecord",
    "AttemptLogger",
    "DownloadOutcome",
    "PipelineResult",
    "Resolver",
    "ResolverConfig",
    "ResolverPipeline",
    "ResolverResult",
    "ResolverMetrics",
    "DownloadFunc",
    "default_resolvers",
    "DEFAULT_RESOLVER_ORDER",
    "clear_resolver_caches",
    "ArxivResolver",
    "CoreResolver",
    "CrossrefResolver",
    "DoajResolver",
    "EuropePmcResolver",
    "HalResolver",
    "LandingPageResolver",
    "OpenAireResolver",
    "OsfResolver",
    "PmcResolver",
    "SemanticScholarResolver",
    "UnpaywallResolver",
    "WaybackResolver",
]
