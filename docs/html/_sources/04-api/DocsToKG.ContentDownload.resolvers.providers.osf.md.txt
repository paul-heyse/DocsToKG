# 1. Module: osf

This reference describes the ``OsfResolver`` class provided by the consolidated module ``DocsToKG.ContentDownload.pipeline.providers``.

OSF Resolver Provider

This module queries the Open Science Framework (OSF) API to retrieve download
links for preprints and other research artefacts hosted on the platform.

Key Features:
- Support for both direct download links and primary file relationships.
- Structured error handling for timeouts, connection failures, and JSON errors.
- Deduplication of URLs to avoid redundant download attempts.

Usage:
    from DocsToKG.ContentDownload.pipeline.providers import OsfResolver

    resolver = OsfResolver()
    results = list(resolver.iter_urls(session, config, artifact))

## 1. Functions

### `is_enabled(self, config, artifact)`

Return ``True`` when the artifact has a DOI for OSF lookup.

Args:
config: Resolver configuration providing OSF request details.
artifact: Work artifact potentially containing a DOI value.

Returns:
Boolean indicating whether the resolver should run.

### `iter_urls(self, session, config, artifact)`

Yield candidate download URLs from the OSF API.

Args:
session: HTTP session available for API requests.
config: Resolver configuration including polite headers and timeouts.
artifact: Work artifact describing the record under consideration.

Returns:
Iterable of resolver results containing candidate download URLs.

## 2. Classes

### `OsfResolver`

Resolve artefacts hosted on the Open Science Framework.

Attributes:
name: Resolver identifier shared with the pipeline.

Examples:
>>> resolver = OsfResolver()
>>> resolver.name
'osf'
