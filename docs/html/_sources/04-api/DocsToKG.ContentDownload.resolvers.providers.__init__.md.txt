# 1. Module: __init__

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.resolvers.providers.__init__``.

Resolver Providers

This module consolidates all resolver provider implementations into a single
registry-backed module. Providers register themselves upon subclassing the
``RegisteredResolver`` base, allowing ``default_resolvers`` to materialise the
prioritised resolver stack without manual bookkeeping.

The consolidation centralises shared imports, caching helpers, and optional
third-party dependencies while preserving the public API expected by the
resolver pipeline and caching utilities.

## 1. Functions

### `_absolute_url(base, href)`

Resolve relative ``href`` values against ``base`` to obtain absolute URLs.

### `_collect_candidate_urls(node, results)`

Recursively collect HTTP(S) URLs from nested response payloads.

### `_fetch_crossref_data(doi, mailto, timeout, headers_key)`

Retrieve Crossref metadata for ``doi`` with polite header caching.

### `_fetch_unpaywall_data(doi, email, timeout, headers_key)`

Fetch Unpaywall metadata for ``doi`` using polite caching.

### `_fetch_semantic_scholar_data(doi, api_key, timeout, headers_key)`

Fetch Semantic Scholar Graph API metadata for ``doi`` with caching.

### `default_resolvers()`

Return default resolver instances in priority order.

### `register(cls, resolver_cls)`

*No documentation available.*

### `create_default(cls)`

*No documentation available.*

### `__init_subclass__(cls, register)`

*No documentation available.*

### `is_enabled(self, config, artifact)`

*No documentation available.*

### `iter_urls(self, session, config, artifact)`

*No documentation available.*

### `is_enabled(self, config, artifact)`

*No documentation available.*

### `iter_urls(self, session, config, artifact)`

*No documentation available.*

### `is_enabled(self, config, artifact)`

*No documentation available.*

### `iter_urls(self, session, config, artifact)`

*No documentation available.*

### `is_enabled(self, config, artifact)`

*No documentation available.*

### `iter_urls(self, session, config, artifact)`

*No documentation available.*

### `is_enabled(self, config, artifact)`

*No documentation available.*

### `iter_urls(self, session, config, artifact)`

*No documentation available.*

### `is_enabled(self, config, artifact)`

*No documentation available.*

### `iter_urls(self, session, config, artifact)`

*No documentation available.*

### `is_enabled(self, config, artifact)`

*No documentation available.*

### `iter_urls(self, session, config, artifact)`

*No documentation available.*

### `is_enabled(self, config, artifact)`

*No documentation available.*

### `iter_urls(self, session, config, artifact)`

*No documentation available.*

### `is_enabled(self, config, artifact)`

*No documentation available.*

### `iter_urls(self, session, config, artifact)`

*No documentation available.*

### `is_enabled(self, config, artifact)`

*No documentation available.*

### `iter_urls(self, session, config, artifact)`

*No documentation available.*

### `is_enabled(self, config, artifact)`

*No documentation available.*

### `iter_urls(self, session, config, artifact)`

*No documentation available.*

### `is_enabled(self, config, artifact)`

*No documentation available.*

### `_lookup_pmcids(self, session, identifiers, config)`

*No documentation available.*

### `iter_urls(self, session, config, artifact)`

*No documentation available.*

### `is_enabled(self, config, artifact)`

*No documentation available.*

### `iter_urls(self, session, config, artifact)`

*No documentation available.*

### `is_enabled(self, config, artifact)`

*No documentation available.*

### `iter_urls(self, session, config, artifact)`

*No documentation available.*

### `is_enabled(self, config, artifact)`

*No documentation available.*

### `iter_urls(self, session, config, artifact)`

*No documentation available.*

### `is_enabled(self, config, artifact)`

*No documentation available.*

### `iter_urls(self, session, config, artifact)`

*No documentation available.*

## 2. Classes

### `ResolverRegistry`

Registry tracking resolver classes by their ``name`` attribute.

### `RegisteredResolver`

Mixin ensuring subclasses register with :class:`ResolverRegistry`.

### `ArxivResolver`

Resolve arXiv preprints using arXiv identifier lookups.

### `CoreResolver`

Resolve PDFs using the CORE API.

### `CrossrefResolver`

Resolve candidate URLs from the Crossref metadata API.

### `DoajResolver`

Resolve Open Access links using the DOAJ API.

### `EuropePmcResolver`

Resolve Open Access links via the Europe PMC REST API.

### `FigshareResolver`

Resolve Figshare repository metadata into download URLs.

### `HalResolver`

Resolve publications from the HAL open archive.

### `LandingPageResolver`

Attempt to scrape landing pages when explicit PDFs are unavailable.

### `OpenAireResolver`

Resolve URLs using the OpenAIRE API.

### `OpenAlexResolver`

Resolve OpenAlex work metadata into candidate download URLs.

### `OsfResolver`

Resolve artefacts hosted on the Open Science Framework.

### `PmcResolver`

Resolve PubMed Central articles via identifiers and lookups.

### `SemanticScholarResolver`

Resolve PDFs via the Semantic Scholar Graph API.

### `UnpaywallResolver`

Resolve PDFs via the Unpaywall API.

### `WaybackResolver`

Fallback resolver that queries the Internet Archive Wayback Machine.

### `ZenodoResolver`

Resolve Zenodo records into downloadable open-access PDF URLs.
