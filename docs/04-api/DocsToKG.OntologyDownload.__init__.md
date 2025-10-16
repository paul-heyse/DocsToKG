# 1. Module: __init__

This reference documents the DocsToKG module ``DocsToKG.OntologyDownload.__init__``.

## 1. Overview

Public API for the DocsToKG ontology downloader and resolver pipeline.

This facade exposes the primary fetch utilities used by external callers to
plan resolver fallback chains, download ontologies with hardened validation,
perform stream normalization, and emit schema-compliant manifests with
deterministic fingerprints.
