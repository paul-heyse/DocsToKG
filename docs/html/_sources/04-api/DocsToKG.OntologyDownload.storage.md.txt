# 1. Module: storage

This reference documents the DocsToKG module ``DocsToKG.OntologyDownload.storage``.

## 1. Overview

Storage backend facade for the ontology downloader.

## 2. Functions

### `get_storage_backend()`

Return the active storage backend and sync the core facade.

Args:
None

Returns:
StorageBackend: Instance implementing storage operations for ontology artifacts.
