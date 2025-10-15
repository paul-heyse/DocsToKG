# 1. Migration Guide: Modular Content Download Resolvers

This guide explains how to upgrade existing DocsToKG integrations from the
monolithic `DocsToKG.ContentDownload.resolvers` module to the modular resolver
architecture introduced in the `modularize-content-download-architecture`
proposal.

## 1. Update Import Paths

- `ResolverPipeline`<br>
  Legacy: `from DocsToKG.ContentDownload.resolvers import ResolverPipeline`<br>
  Recommended: `from DocsToKG.ContentDownload.resolvers.pipeline import ResolverPipeline`
- `default_resolvers`<br>
  Legacy: `from DocsToKG.ContentDownload.resolvers import default_resolvers`<br>
  Recommended: `from DocsToKG.ContentDownload.resolvers.providers import default_resolvers`
- `ResolverConfig`<br>
  Legacy: `from DocsToKG.ContentDownload.resolvers import ResolverConfig`<br>
  Recommended: `from DocsToKG.ContentDownload.resolvers.types import ResolverConfig`

> **Compatibility:** Legacy imports continue working via re-exports but emit
> `DeprecationWarning`. Update code to silence the warning and prepare for
> future removals.

## 2. Adopt New Configuration Options

| Option | Default | Description |
| --- | --- | --- |
| `max_concurrent_resolvers` | `1` | Bounds intra-work concurrency. Set to `>1` to enable threaded resolver execution. |
| `enable_head_precheck` | `True` | Enables HEAD requests that filter HTML or zero-byte responses before downloading. |
| `resolver_head_precheck` | `{}` | Per-resolver override that can disable HEAD checks for services that reject them. |
| `resolver_timeouts` | `{}` | Resolver-specific timeout overrides (falls back to the global `timeout`). |
| `resolver_min_interval_s` | `{}` | Fine-grained rate limiting expressed as minimum seconds between resolver attempts. |

All options are optional—existing configuration files remain valid without
modification.

## 3. Review Behavioural Changes

- **OpenAlexResolver:** Now participates as the first resolver in the default
  pipeline. Remove bespoke `attempt_openalex_candidates()` integrations.
- **Centralised HTTP retries:** Every resolver uses
  `DocsToKG.ContentDownload.network.request_with_retries()` for consistent backoff,
  timeout handling, and `Retry-After` support.
- **Conditional requests:** Download code paths share the
  `ConditionalRequestHelper` utility to honour ETag and Last-Modified headers.
- **Structured logging:** Attempt records now include `resolver_wall_time_ms`
  for precise wall-clock tracking.

## 4. Testing Checklist

1. Update imports to the recommended module paths.
2. Run `pytest tests/` to confirm backward compatibility.
3. Inspect logs for unexpected `DeprecationWarning` entries.
4. If you maintain custom resolvers, ensure they register via
   `DocsToKG.ContentDownload.resolvers.providers.default_resolvers()`.

## 5. Support

Reach out in the `#docs-to-kg` channel if you encounter migration issues or
spot regressions. Include stack traces, configuration snippets, and resolver
names to speed up triage.
