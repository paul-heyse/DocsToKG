# 1. Module: optdeps

This reference documents the DocsToKG module ``DocsToKG.OntologyDownload.optdeps``.

## 1. Overview

Optional dependency helpers for the ontology downloader.

## 2. Functions

### `__getattr__(self, name)`

*No documentation available.*

### `__setattr__(self, name, value)`

*No documentation available.*

### `__dir__(self)`

*No documentation available.*

## 3. Classes

### `_OptDepsProxy`

Module proxy that forwards access to ``ontology_download`` internals.

Attributes:
__slots__: Prevents accidental attribute assignment on the proxy type.

Examples:
>>> proxy = _OptDepsProxy('DocsToKG.OntologyDownload.optdeps')
>>> callable(proxy.get_rdflib)
True
