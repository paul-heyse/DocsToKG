# 1. Module: errors

This reference documents the DocsToKG module ``DocsToKG.OntologyDownload.errors``.

## 1. Overview

Shared exception hierarchy for DocsToKG ontology downloads.

## 2. Classes

### `OntologyDownloadError`

Base exception for ontology planning, download, or validation failures.

### `ConfigurationError`

Raised when configuration inputs or manifests are invalid.

### `ResolverError`

Raised when resolver planning cannot produce a usable fetch plan.

### `ValidationError`

Raised when ontology validation encounters unrecoverable issues.

### `PolicyError`

Raised when security, licensing, or rate limit policies are violated.

### `DownloadFailure`

Raised when an HTTP download attempt fails.

### `UserConfigError`

Raised when CLI arguments or YAML configuration inputs are invalid.
