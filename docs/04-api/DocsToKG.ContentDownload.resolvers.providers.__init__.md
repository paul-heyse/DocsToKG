# 1. Module: __init__

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.resolvers.providers.__init__``.

Resolver Provider Registry

This package aggregates resolver provider implementations and exposes a helper
for constructing the default resolver stack used by the content download
pipeline.

Key Features:
- Explicit imports of all resolver provider classes for easy discovery.
- ``default_resolvers`` helper that instantiates providers in priority order.

Usage:
    from DocsToKG.ContentDownload.resolvers.providers import default_resolvers

    resolvers = default_resolvers()

## 1. Functions

### `default_resolvers()`

Return default resolver instances in priority order.

Args:
None

Returns:
List of resolver instances ordered by preferred execution priority.
