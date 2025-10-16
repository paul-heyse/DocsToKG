# 1. Migration Guide: Unified Content Download Resolvers

The resolver subsystem has been recomposed into a single module,
`DocsToKG.ContentDownload.resolvers`. This guide explains how to migrate code
that previously relied on the intermediate submodules (`.types`, `.pipeline`,
`.providers`, `.headers`, `.cache`).

## 1. Update Import Paths

| Replace | With |
| --- | --- |
| `from DocsToKG.ContentDownload.resolvers.types import …` | `from DocsToKG.ContentDownload.resolvers import …` |
| `from DocsToKG.ContentDownload.resolvers.pipeline import …` | `from DocsToKG.ContentDownload.resolvers import …` |
| `from DocsToKG.ContentDownload.resolvers.providers import …` | `from DocsToKG.ContentDownload.resolvers import …` |
| `from DocsToKG.ContentDownload.resolvers.headers import headers_cache_key` | `from DocsToKG.ContentDownload.resolvers import headers_cache_key` |

All resolver-related dataclasses (`ResolverResult`, `AttemptRecord`, etc.),
pipeline helpers, and provider implementations are now re-exported directly
from the top-level module.

> **Tip:** Legacy import paths will raise `ImportError`. Update your imports to
> ensure compatibility with the unified module.

## 2. Registering Custom Resolvers

Subclass `DocsToKG.ContentDownload.resolvers.RegisteredResolver` and make sure
the subclass is imported during start-up so it registers with the resolver
registry. To take part in the default order, append instances to
`default_resolvers()` inside `resolvers.py` or extend the returned list within
your application.

## 3. Configuration & Behaviour

No configuration keys changed during the consolidation. Existing YAML files
that reference `ResolverConfig` fields remain valid. Behaviour such as
centralised retries, HEAD preflight checks, and structured logging carries
over unchanged.

## 4. Testing Checklist

1. Run `pytest tests/` to confirm your integration still passes the resolver
   suite.
2. If you maintain custom resolvers, verify they are imported so registration
   occurs.
3. Review logs for unexpected warnings after the import changes.

## 5. Support

Reach out in `#docs-to-kg` with stack traces, configuration snippets, and
resolver names if you encounter migration issues.
