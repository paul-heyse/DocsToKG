# 1. Module: zenodo

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.resolvers.providers.zenodo``.

Zenodo repository resolver for DOI-indexed research outputs.

## 1. Functions

### `is_enabled(self, config, artifact)`

Enable when artifact has a DOI.

### `iter_urls(self, session, config, artifact)`

Query Zenodo API by DOI and yield PDF file URLs.

## 2. Classes

### `ZenodoResolver`

Resolver for Zenodo open access repository.
