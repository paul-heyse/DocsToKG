"""Resolver pipeline and provider implementations with compatibility shims."""

import time as _time
import warnings

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
from .types import DEFAULT_RESOLVER_ORDER as _DEFAULT_RESOLVER_ORDER
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

_LEGACY_EXPORTS = {
    "time": _time,
    "requests": _requests,
}

_DEPRECATION_MESSAGES = {
    "time": (
        "DocsToKG.ContentDownload.resolvers.time is deprecated; import 'time' "
        "directly. This alias will be removed in a future release."
    ),
    "requests": (
        "DocsToKG.ContentDownload.resolvers.requests is deprecated; import the "
        "'requests' package directly. This alias will be removed in a future release."
    ),
}

warnings.warn(
    (
        "Importing resolver classes from DocsToKG.ContentDownload.resolvers is "
        "deprecated and will be removed in a future release. Import from the "
        "explicit submodules (e.g. DocsToKG.ContentDownload.resolvers.pipeline) "
        "instead."
    ),
    DeprecationWarning,
    stacklevel=2,
)


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


def __getattr__(name: str):
    """Return legacy exports while emitting :class:`DeprecationWarning`."""

    if name in _LEGACY_EXPORTS:
        warnings.warn(
            _DEPRECATION_MESSAGES.get(
                name,
                f"DocsToKG.ContentDownload.resolvers.{name} is deprecated",
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        return _LEGACY_EXPORTS[name]
    raise AttributeError(name)
