"""
Resolver Package Facade

This module exposes the public interface for resolver pipeline utilities and
provider implementations used by the modular content download architecture. It
re-exports pipeline classes, resolver types, and concrete provider factories to
preserve backward compatibility with legacy imports while gently steering
callers towards explicit submodules.

Key Features:
- Facade for the resolver pipeline orchestration classes and metrics types.
- Explicit export of default resolver configuration and provider implementations.
- Backward-compatible aliases for deprecated imports (``time`` and ``requests``).

Usage:
    from DocsToKG.ContentDownload.resolvers import ResolverPipeline, default_resolvers

    pipeline = ResolverPipeline(
        resolvers=default_resolvers(),
        config=ResolverConfig(),
        download_func=lambda *args, **kwargs: None,
        logger=lambda record: None,
        metrics=ResolverMetrics(),
    )
"""

import time as _time
import warnings

import requests as _requests

from .cache import clear_resolver_caches
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

# Legacy convenience exports retained temporarily for backward compatibility.
#
# Downstream consumers historically imported ``time`` and ``requests`` via the
# resolver package. These aliases will be removed in the next minor release
# (after 0.x) to encourage direct imports from the standard library and PyPI.
# The deprecation warnings emitted by :func:`__getattr__` provide advance notice
# so integrations can migrate before the removal occurs.
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
    """Return legacy exports while emitting :class:`DeprecationWarning`.

    Args:
        name: Attribute name requested by the caller.

    Returns:
        Either the legacy export object or raises :class:`AttributeError` when unknown.

    Raises:
        AttributeError: If ``name`` is not a recognised legacy export.
    """

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
