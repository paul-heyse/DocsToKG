# 1. Module: __init__

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.resolvers.__init__``.

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

## 1. Functions

### `__getattr__(name)`

Return legacy exports while emitting :class:`DeprecationWarning`.

Args:
name: Attribute name requested by the caller.

Returns:
Either the legacy export object or raises :class:`AttributeError` when unknown.

Raises:
AttributeError: If ``name`` is not a recognised legacy export.
