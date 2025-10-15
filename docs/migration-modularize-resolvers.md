# Migration Guide: Modular Content Download Resolvers

This guide helps teams upgrade from the monolithic
`DocsToKG.ContentDownload.resolvers` module to the modular resolver
architecture introduced by the `modularize-content-download-architecture`
change proposal.

## 1. Import Path Updates

| Legacy Import | Recommended Import | Notes |
| --- | --- | --- |
| `from DocsToKG.ContentDownload.resolvers import ResolverPipeline` | `from DocsToKG.ContentDownload.resolvers.pipeline import ResolverPipeline` | New path exposes the same API with improved typing. |
| `from DocsToKG.ContentDownload.resolvers import default_resolvers` | `from DocsToKG.ContentDownload.resolvers.providers import default_resolvers` | Provider registry now lives alongside individual resolver modules. |
| `from DocsToKG.ContentDownload.resolvers import ResolverConfig` | `from DocsToKG.ContentDownload.resolvers.types import ResolverConfig` | Configuration dataclasses reside in the dedicated `types` module. |

> **Compatibility:** All legacy imports continue working via re-exports, but
> they now emit `DeprecationWarning`. Update imports to silence the warning
> and prepare for future removals.

## 2. New Configuration Options

| Option | Default | Description |
| --- | --- | --- |
| `max_concurrent_resolvers` | `1` | Bounds intra-work concurrency. Set to `>1` to enable threaded resolver execution. |
| `enable_head_precheck` | `True` | Performs HEAD requests before downloads to filter HTML and zero-byte responses. |
| `resolver_head_precheck` | `{}` | Per-resolver override for HEAD behaviour (set to `False` for resolvers that reject HEAD). |
| `resolver_timeouts` | `{}` | Resolver-specific timeout overrides (fallbacks to `timeout`). |
| `resolver_min_interval_s` | `{}` | Granular rate-limiting expressed as minimum seconds between resolver invocations. |

All new options are optional. Existing configuration files that omit the
fields continue to function without modification.

## 3. Behavioural Changes

- **OpenAlexResolver:** OpenAlex-provided URLs now flow through the standard
  resolver pipeline as the first resolver. Remove any custom
  `attempt_openalex_candidates()` integration code.
- **Centralised HTTP retries:** All resolvers use
  `DocsToKG.ContentDownload.http.request_with_retries()` for consistent
  backoff, timeout, and `Retry-After` handling.
- **Conditional requests:** Download code paths share the
  `ConditionalRequestHelper` utility to honour ETag and Last-Modified headers.
- **Structured logging:** Attempt records include `resolver_wall_time_ms`
  for total resolver wall-clock tracking.

## 4. Testing Checklist

1. Update imports to the recommended module paths.
2. Run `pytest tests/` to confirm backward compatibility.
3. Inspect logs for any unexpected `DeprecationWarning` occurrences.
4. If you provide custom resolvers, ensure they are registered via
   `DocsToKG.ContentDownload.resolvers.providers.default_resolvers()`.

## 5. Support

Reach out in the `#docs-to-kg` channel if you encounter migration issues or
spot regressions. Include stack traces, configuration snippets, and resolver
names to speed up triage.
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
