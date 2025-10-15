# 1. Module: pmc

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.resolvers.providers.pmc``.

PubMed Central Resolver Provider

This module integrates with PubMed Central (PMC) utilities to resolve open
access PDFs using DOI, PMID, or PMCID identifiers.

Key Features:
- Conversion of DOI/PMID identifiers into PMCIDs using NCBI idconv service.
- Retrieval of PMC Open Access links with fallback to static PDF endpoints.
- Deduplication of PMCIDs and robust error reporting via resolver events.

Usage:
    from DocsToKG.ContentDownload.resolvers.providers.pmc import PmcResolver

    resolver = PmcResolver()
    results = list(resolver.iter_urls(session, config, artifact))

## 1. Functions

### `_absolute_url(base, href)`

Resolve relative ``href`` values against ``base`` to obtain absolute URLs.

Args:
base: Base URL used as the reference for resolving the link.
href: Relative or absolute URL extracted from PMC HTML content.

Returns:
Absolute URL pointing to the resource referenced by ``href``.

### `is_enabled(self, config, artifact)`

Return ``True`` when the artifact has PMC, PMID, or DOI identifiers.

Args:
config: Resolver configuration providing PMC connectivity hints.
artifact: Work artifact capturing PMC/PMID/DOI metadata.

Returns:
Boolean indicating whether PMC resolution should be attempted.

### `_lookup_pmcids(self, session, identifiers, config)`

Return PMCIDs resolved from DOI/PMID identifiers using NCBI utilities.

Args:
session: Requests session reused across resolver calls.
identifiers: DOI or PMID identifiers to convert to PMCIDs.
config: Resolver configuration providing timeout and polite headers.

Returns:
List of PMCIDs corresponding to the supplied identifiers.

### `iter_urls(self, session, config, artifact)`

Yield candidate PMC download URLs derived from identifiers.

Args:
session: HTTP session used to issue PMC and utility API requests.
config: Resolver configuration, including timeouts and headers.
artifact: Work artifact describing the scholarly item under resolution.

Returns:
Iterable of resolver results pointing to PMC hosted PDFs.

## 2. Classes

### `PmcResolver`

Resolve PubMed Central articles via identifiers and lookups.

Attributes:
name: Resolver identifier published to the pipeline scheduler.

Examples:
>>> resolver = PmcResolver()
>>> resolver.name
'pmc'
