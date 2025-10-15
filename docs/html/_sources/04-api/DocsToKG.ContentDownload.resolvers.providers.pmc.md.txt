# 1. Module: pmc

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.resolvers.providers.pmc``.

PubMed Central resolver leveraging NCBI utilities and OA endpoints.

## 1. Functions

### `_absolute_url(base, href)`

Resolve relative ``href`` values against ``base`` to obtain absolute URLs.

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
