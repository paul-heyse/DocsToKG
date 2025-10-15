# Migration Guide: Modular Resolver Architecture

This guide describes how to upgrade custom tooling to the modular resolver
layout introduced by the `modularize-content-download-architecture` change.

## Legacy Import Paths

Existing code commonly imported pipeline classes and helper utilities directly
from the monolithic namespace:

```python
from DocsToKG.ContentDownload.resolvers import ResolverPipeline
from DocsToKG.ContentDownload.resolvers import ResolverConfig
from DocsToKG.ContentDownload.resolvers import default_resolvers
```

These imports continue to work and emit a `DeprecationWarning` when accessing the
legacy `time` or `requests` shims.

## Recommended Import Paths

Prefer importing directly from the new module hierarchy to benefit from static
typing and clearer dependencies:

```python
from DocsToKG.ContentDownload.resolvers.pipeline import ResolverPipeline
from DocsToKG.ContentDownload.resolvers.types import ResolverConfig
from DocsToKG.ContentDownload.resolvers.providers import default_resolvers
```

Individual providers may be imported from
`DocsToKG.ContentDownload.resolvers.providers.<name>`.

## Updated Configuration Options

The refactor introduces the following configuration settings. They are available
on `ResolverConfig` and in YAML/CLI overrides:

| Option | Default | Description |
|--------|---------|-------------|
| `max_concurrent_resolvers` | `1` | Upper bound on concurrent resolver threads per work item. |
| `enable_head_precheck` | `True` | Toggle HEAD preflight checks that skip obvious HTML/empty responses. |
| `resolver_head_precheck` | `{}` | Per-resolver override for HEAD preflight behaviour. |
| `resolver_min_interval_s` | `{}` | Fine-grained rate limiting per resolver (supersedes `resolver_rate_limits`). |
| `resolver_timeouts` | `{}` | Resolver-specific timeout overrides. |

## Behaviour Changes

* Resolver executions are routed through a shared `request_with_retries`
  helper that respects `Retry-After` headers and consistent backoff policies.
* Resolver exceptions are captured and surfaced as structured events instead of
  bubbling through the pipeline.
* Zenodo and Figshare resolvers are now part of the default resolver set.
* Legacy `DocsToKG.ContentDownload.resolvers.time` and
  `DocsToKG.ContentDownload.resolvers.requests` aliases emit deprecation
  warnings and will be removed in a future release.

## Validation Checklist

1. Run the full resolver test suite: `pytest tests/test_resolver_pipeline.py`.
2. Execute your custom resolver integration using the new imports.
3. Confirm that manifests still record wall-clock timing via
   `resolver_wall_time_ms`.
4. Update any documentation or onboarding material to reference the new
   modules.
