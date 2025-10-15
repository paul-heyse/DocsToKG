# 1. Module: __init__

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.resolvers.providers.__init__``.

Resolver provider implementations and registry used by the content download pipeline.

The registry ships with resolvers for Unpaywall, Crossref, OpenAlex, Wayback,
Zenodo, Figshare, and additional open-access aggregators. ``default_resolvers()``
returns provider instances in execution order so callers can build customised
pipelines or extend the registry with project-specific resolvers.

## 1. Functions

### `default_resolvers()`

Return default resolver instances in priority order.

Args:
None

Returns:
List of resolver instances ordered by preferred execution priority.
