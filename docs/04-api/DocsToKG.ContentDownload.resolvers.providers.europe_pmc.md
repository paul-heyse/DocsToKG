# 1. Module: europe_pmc

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.resolvers.providers.europe_pmc``.

Europe PMC Resolver Provider

This module integrates with the Europe PMC REST API to locate open-access PDFs
hosted across European repositories. It complements other resolver providers by
covering funder-mandated repositories aggregated by Europe PMC.

Key Features:
- Query construction against Europe PMC's search endpoint using DOI filters.
- Deduplication of PDF URLs within API responses.
- Structured error reporting for network and parsing failures.

Usage:
    from DocsToKG.ContentDownload.resolvers.providers.europe_pmc import EuropePmcResolver

    resolver = EuropePmcResolver()
    urls = list(resolver.iter_urls(session, config, artifact))

## 1. Functions

### `is_enabled(self, config, artifact)`

Return ``True`` when the artifact has a DOI suitable for lookup.

Args:
config: Resolver configuration providing Europe PMC preferences.
artifact: Work artifact containing DOI metadata.

Returns:
Boolean indicating whether the resolver should attempt a lookup.

### `iter_urls(self, session, config, artifact)`

Yield candidate URLs from the Europe PMC API.

Args:
session: HTTP session used to execute Europe PMC API requests.
config: Resolver configuration including polite headers and timeouts.
artifact: Work artifact representing the target scholarly output.

Returns:
Iterable of resolver results containing discovered URLs.

## 2. Classes

### `EuropePmcResolver`

Resolve Open Access links via the Europe PMC REST API.

Attributes:
name: Resolver identifier exposed to the orchestration pipeline.

Examples:
>>> resolver = EuropePmcResolver()
>>> resolver.name
'europe_pmc'
