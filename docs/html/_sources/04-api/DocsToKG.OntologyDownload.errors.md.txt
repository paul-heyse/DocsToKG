# 1. Module: errors

This reference documents the DocsToKG module ``DocsToKG.OntologyDownload.errors``.

## 1. Overview

Shared exception hierarchy for the ontology downloader.

## 2. Classes

### `UserConfigError`

Raised when CLI arguments or YAML configuration inputs are invalid.

### `OntologyDownloadError`

Base class for runtime failures during ontology planning or download.

### `ResolverError`

Raised when resolver planning cannot produce a usable fetch plan.

### `ValidationError`

Raised when ontology validation encounters unrecoverable problems.

### `PolicyError`

Raised when security, licensing, or rate limit policies are violated.

### `DownloadFailure`

Raised when an HTTP download attempt fails.
