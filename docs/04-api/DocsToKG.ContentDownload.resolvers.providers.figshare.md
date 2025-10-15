# 1. Module: figshare

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.resolvers.providers.figshare``.

Figshare repository resolver for DOI-indexed research outputs.

## 1. Functions

### `is_enabled(self, config, artifact)`

Enable when artifact has a DOI.

### `iter_urls(self, session, config, artifact)`

Search Figshare API by DOI and yield PDF file URLs.

## 2. Classes

### `FigshareResolver`

Resolver for Figshare repository.
