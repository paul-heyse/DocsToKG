# 1. Module: landing_page

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.resolvers.providers.landing_page``.

Landing page scraper resolver using BeautifulSoup.

## 1. Functions

### `_absolute_url(base, href)`

Resolve relative links on a landing page against the page URL.

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
