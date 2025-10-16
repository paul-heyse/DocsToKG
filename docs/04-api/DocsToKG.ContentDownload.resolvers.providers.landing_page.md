# 1. Module: landing_page

This reference describes the ``LandingPageResolver`` class provided by the consolidated module ``DocsToKG.ContentDownload.pipeline.providers``.

Landing Page Scraper Resolver

This module uses lightweight HTML parsing to discover PDF links embedded within
landing pages when resolvers fail to return direct download URLs.

Key Features:
- Detection of ``citation_pdf_url`` meta tags and alternate link elements.
- Heuristic scanning of anchor text and href targets for PDF references.
- Graceful degradation when BeautifulSoup or lxml is unavailable.

Usage:
    from DocsToKG.ContentDownload.pipeline.providers import LandingPageResolver

    resolver = LandingPageResolver()
    results = list(resolver.iter_urls(session, config, artifact))

## 1. Functions

### `_absolute_url(base, href)`

Resolve relative links on a landing page against the page URL.

Args:
base: Landing page URL used as the base for resolution.
href: Relative or absolute link discovered in the page.

Returns:
Absolute URL string that can be used for downstream requests.

### `is_enabled(self, config, artifact)`

Return ``True`` when the artifact exposes landing page URLs.

Args:
config: Resolver configuration controlling scraping behaviour.
artifact: Work artifact containing landing page URLs.

Returns:
Boolean indicating whether landing pages should be scraped.

### `iter_urls(self, session, config, artifact)`

Yield candidate URLs discovered by scraping landing pages.

Args:
session: HTTP session used to download landing pages.
config: Resolver configuration providing headers and timeouts.
artifact: Work artifact listing landing page URLs to inspect.

Returns:
Iterable of resolver results representing discovered download URLs.

## 2. Classes

### `LandingPageResolver`

Attempt to scrape landing pages when explicit PDFs are unavailable.

Attributes:
name: Resolver identifier surfaced to the pipeline dispatcher.

Examples:
>>> resolver = LandingPageResolver()
>>> resolver.name
'landing_page'
