# Module: pmc

PubMed Central resolver leveraging NCBI utilities and OA endpoints.

## Functions

### `_absolute_url(base, href)`

*No documentation available.*

### `is_enabled(self, config, artifact)`

Return ``True`` when the artifact has PMC, PMID, or DOI identifiers.

Args:
config: Resolver configuration providing PMC connectivity hints.
artifact: Work artifact capturing PMC/PMID/DOI metadata.

Returns:
Boolean indicating whether PMC resolution should be attempted.

### `_lookup_pmcids(self, session, identifiers, config)`

*No documentation available.*

### `iter_urls(self, session, config, artifact)`

Yield candidate PMC download URLs derived from identifiers.

Args:
session: HTTP session used to issue PMC and utility API requests.
config: Resolver configuration, including timeouts and headers.
artifact: Work artifact describing the scholarly item under resolution.

Returns:
Iterable of resolver results pointing to PMC hosted PDFs.

## Classes

### `PmcResolver`

Resolve PubMed Central articles via identifiers and lookups.

Attributes:
name: Resolver identifier published to the pipeline scheduler.

Examples:
>>> resolver = PmcResolver()
>>> resolver.name
'pmc'
