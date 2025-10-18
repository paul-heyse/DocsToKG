# 1. Module: europe_pmc

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.resolvers.europe_pmc``.

## 1. Overview

Resolver implementation for the Europe PMC REST API.

## 2. Functions

### `is_enabled(self, config, artifact)`

Return ``True`` when a DOI is available for Europe PMC lookups.

Args:
config: Resolver configuration (unused but part of the interface).
artifact: Work record we may attempt to resolve.

Returns:
bool: Whether this resolver should run for the work.

### `iter_urls(self, session, config, artifact)`

Yield PDF URLs announced by the Europe PMC REST API.

Args:
session: Requests session for HTTP calls.
config: Resolver configuration controlling limits.
artifact: Work record that triggered this resolver.

Yields:
ResolverResult: Candidate download URLs or skip events.

## 3. Classes

### `EuropePmcResolver`

Resolve Open Access links via the Europe PMC REST API.
