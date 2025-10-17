# 1. Module: providers

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.providers``.

## 1. Overview

Source adapter interfaces for the DocsToKG content download pipeline.

This module formalises the boundary between the CLI and upstream catalogues by
exposing a :class:`WorkProvider` protocol. Providers are responsible for
yielding :class:`~DocsToKG.ContentDownload.core.WorkArtifact` instances that
downstream resolver components can process without needing to know which
catalogue supplied the metadata. Today we ship an OpenAlex implementation, but
the protocol enables future adapters (e.g., Crossref, arXiv) to plug in without
modifying the pipeline.

## 2. Functions

### `iter_artifacts(self)`

Yield normalized work artifacts ready for download processing.

### `__iter__(self)`

*No documentation available.*

### `iter_artifacts(self)`

Iterate over OpenAlex works and yield normalized :class:`WorkArtifact` objects.

### `__iter__(self)`

*No documentation available.*

### `_iterate_openalex(self)`

*No documentation available.*

### `_iter_source(self)`

*No documentation available.*

## 3. Classes

### `WorkProvider`

Protocol describing a source of :class:`WorkArtifact` instances.

### `OpenAlexWorkProvider`

Produce :class:`WorkArtifact` instances from an OpenAlex Works query.
