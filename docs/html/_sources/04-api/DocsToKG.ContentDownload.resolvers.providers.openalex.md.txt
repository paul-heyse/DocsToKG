# 1. Module: openalex

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.resolvers.providers.openalex``.

OpenAlex direct URL resolver (position 0 in pipeline).

## 1. Functions

### `is_enabled(self, config, artifact)`

Enable when artifact has pdf_urls or open_access_url.

### `iter_urls(self, session, config, artifact)`

Yield all PDF URLs from OpenAlex work metadata.

## 2. Classes

### `OpenAlexResolver`

Resolver for PDF URLs directly provided by OpenAlex metadata.
