# 1. Module: landing_page

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.resolvers.landing_page``.

## 1. Overview

Resolver that scrapes landing pages for PDF links when metadata fails.

## 2. Functions

### `is_enabled(self, config, artifact)`

Return ``True`` when landing page URLs are available to scrape.

Args:
config: Resolver configuration (unused for enablement).
artifact: Work record containing landing URLs.

Returns:
bool: Whether the resolver should attempt scraping.

### `iter_urls(self, session, config, artifact)`

Scrape landing pages for PDF links and yield matching results.

Args:
session: Requests session for HTTP interactions.
config: Resolver configuration providing timeouts and headers.
artifact: Work metadata containing landing URLs.

Yields:
ResolverResult: Candidate download URLs or diagnostic events.

## 3. Classes

### `LandingPageResolver`

Attempt to scrape landing pages when explicit PDFs are unavailable.
