# 1. Migration Guide: Resolver Modularisation

The resolver subsystem now lives under ``DocsToKG.ContentDownload.resolvers``.
Each resolver sits in its own module and registers with ``ResolverRegistry`` on
import. ``pipeline.py`` continues to re-export the resolver classes for backwards
compatibility. This guide summarises the changes needed when upgrading.

## 1. Update Import Paths

| Replace | With |
| --- | --- |
| ``from DocsToKG.ContentDownload.pipeline import OpenAlexResolver`` | ``from DocsToKG.ContentDownload.resolvers import OpenAlexResolver`` (or retain the pipeline import for compatibility) |
| ``from DocsToKG.ContentDownload.pipeline import default_resolvers`` | ``from DocsToKG.ContentDownload.resolvers import default_resolvers`` |
| ``from DocsToKG.ContentDownload.pipeline import RegisteredResolver`` | ``from DocsToKG.ContentDownload.resolvers import RegisteredResolver`` |

Existing imports from ``pipeline`` keep working because the module now
re-exports the resolver classes. New code should prefer importing from the
``resolvers`` package to make dependencies explicit.

## 2. Registering Custom Resolvers

Place new resolvers in ``ContentDownload/resolvers/`` and inherit from
``RegisteredResolver`` or ``ApiResolverBase``. Import the module from
``resolvers/__init__.py`` (mirroring the built-in resolvers) so registration runs
at start-up. To participate in the default order, append an instance to the list
returned by ``default_resolvers()`` in your own entry point.

## 3. Behaviour Changes

- ``ResolverRegistry`` owns discovery and ordering. ``default_resolvers()`` now
  simply instantiates the registry.
- ``pipeline.py`` focuses on orchestration and telemetry. Resolver-specific code
  (parsing helpers, HTML scrapers, etc.) moved into ``resolvers/base.py``.
- Strategy helpers in ``download.py`` now handle classification validation,
  resume logic, and corruption heuristics, so custom resolvers no longer need to
  duplicate that logic.

## 4. Testing Checklist

1. Run ``pytest tests/content_download`` to ensure resolver coverage passes.
2. Verify custom resolvers appear in ``ResolverRegistry.all()`` during
   application start-up.
3. Inspect manifests to confirm resolver names and metadata remain unchanged.

## 5. Support

Open a discussion in ``#docs-to-kg`` with resolver names, stack traces, and
configuration snippets if migration issues arise.
