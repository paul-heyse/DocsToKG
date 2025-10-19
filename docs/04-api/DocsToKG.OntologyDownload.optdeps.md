# 1. Module: optdeps

This reference documents the DocsToKG module ``DocsToKG.OntologyDownload.optdeps``.

## 1. Overview

Optional dependency helpers for the ontology downloader.

## 2. Functions

### `_import_module(name)`

*No documentation available.*

### `_call_with_override(callback, cache_key)`

*No documentation available.*

### `get_pystow()`

Return the ``pystow`` module, loading a stub when optional deps are missing.

Returns:
Any: Imported ``pystow`` module or fallback stub.

### `get_rdflib()`

Return an ``rdflib`` module or stub ``Graph`` implementation for testing.

Returns:
Any: Real ``rdflib`` module or a stub with a ``Graph`` attribute.

### `get_pronto()`

Return the ``pronto`` module, respecting optional dependency overrides.

Returns:
Any: Imported ``pronto`` module or stub.

### `get_owlready2()`

Return the ``owlready2`` module, applying opt-dependency fallbacks when needed.

Returns:
Any: Imported ``owlready2`` module or fallback stub.

### `_wrapped_import(name)`

*No documentation available.*
