# 1. Module: pmc

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.resolvers.pmc``.

## 1. Overview

Resolver implementation for PubMed Central content.

## 2. Functions

### `is_enabled(self, config, artifact)`

Return ``True`` when PMC identifiers or DOI metadata are available.

Args:
config: Resolver configuration (unused for enablement checks).
artifact: Work record containing identifiers such as DOI, PMID, or PMCID.

Returns:
bool: Whether the resolver should attempt the work.

### `_lookup_pmcids(self, session, identifiers, config)`

*No documentation available.*

### `iter_urls(self, session, config, artifact)`

Yield PubMed Central PDF URLs matched to ``artifact``.

Args:
session: Requests session for HTTP requests.
config: Resolver configuration providing timeouts and headers.
artifact: Work metadata supplying PMC identifiers.

Yields:
ResolverResult: Candidate PDF URLs or diagnostic events.

## 3. Classes

### `PmcResolver`

Resolve PubMed Central articles via identifiers and lookups.
