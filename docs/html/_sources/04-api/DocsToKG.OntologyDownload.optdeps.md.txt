# 1. Module: optdeps

This reference documents the DocsToKG module ``DocsToKG.OntologyDownload.optdeps``.

## 1. Overview

Optional dependency helpers retained as a testing shim.

Production code should import optional dependency helpers from
``DocsToKG.OntologyDownload.ontology_download`` directly. This proxy exists so
tests can monkeypatch the internal caches without touching private globals.

## 2. Functions

### `patch_getter(name, replacement)`

Temporarily replace an optional dependency getter for tests.

### `__getattr__(self, name)`

*No documentation available.*

### `__setattr__(self, name, value)`

*No documentation available.*

### `__delattr__(self, name)`

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
