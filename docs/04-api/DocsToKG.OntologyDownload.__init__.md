# 1. Module: __init__

This reference documents the DocsToKG module ``DocsToKG.OntologyDownload.__init__``.

## 1. Overview

Public API for the DocsToKG ontology downloader and resolver pipeline.

This facade exposes the primary fetch utilities used by external callers to
plan resolver fallback chains, download ontologies with hardened validation,
perform stream normalization, and emit schema-compliant manifests with
deterministic fingerprints.

## 2. Functions

### `__getattr__(name)`

Lazily import API exports to avoid resolver dependencies at import time.

### `__dir__()`

Expose lazily-populated attributes in ``dir()`` results.
