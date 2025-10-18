# 1. Module: figshare

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.resolvers.figshare``.

## 1. Overview

Resolver implementation for the Figshare repository API.

## 2. Functions

### `is_enabled(self, config, artifact)`

Return ``True`` when a DOI is available for Figshare searches.

Args:
config: Resolver configuration (unused for enablement checks).
artifact: Work record describing the target document.

Returns:
bool: Whether this resolver should be activated.

### `iter_urls(self, session, config, artifact)`

Yield Figshare file download URLs associated with ``artifact``.

Args:
session: Requests session for issuing HTTP requests.
config: Resolver configuration controlling Figshare access.
artifact: Work metadata used to seed the query.

Yields:
ResolverResult: Candidate download URLs or skip events.

## 3. Classes

### `FigshareResolver`

Resolve Figshare repository metadata into download URLs.
