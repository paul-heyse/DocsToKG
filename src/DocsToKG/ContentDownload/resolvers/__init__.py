"""Resolver pipeline and provider implementations.

This module maintains backward compatibility by re-exporting legacy entry points
so existing integrations and tests can continue to monkeypatch ``requests`` or
``time`` on the resolver namespace.
"""

import time as _time

import requests as _requests

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
    OpenAlexResolver,
    OsfResolver,
    PmcResolver,
    SemanticScholarResolver,
    UnpaywallResolver,
    WaybackResolver,
    ZenodoResolver,
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
time = _time
requests = _requests


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
    "OpenAlexResolver",
    "OpenAireResolver",
    "OsfResolver",
    "ZenodoResolver",
    "PmcResolver",
    "SemanticScholarResolver",
    "UnpaywallResolver",
    "WaybackResolver",
    "time",
    "requests",
]
