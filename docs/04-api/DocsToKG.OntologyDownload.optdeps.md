# 1. Module: optdeps

This reference documents the DocsToKG module ``DocsToKG.OntologyDownload.optdeps``.

## 1. Overview

Optional dependency helpers for the ontology downloader.

## 2. Functions

### `_create_stub_module(name, attrs)`

*No documentation available.*

### `_create_stub_bnode(value)`

*No documentation available.*

### `_create_stub_literal(value)`

*No documentation available.*

### `_create_stub_uri(value)`

*No documentation available.*

### `_import_module(name)`

*No documentation available.*

### `_create_pystow_stub(root)`

*No documentation available.*

### `_create_rdflib_stub()`

*No documentation available.*

### `_create_pronto_stub()`

*No documentation available.*

### `_create_owlready_stub()`

*No documentation available.*

### `get_pystow()`

Return the ``pystow`` module, supplying a stub when unavailable.

### `get_rdflib()`

Return the ``rdflib`` module, supplying a stub when unavailable.

### `get_pronto()`

Return the ``pronto`` module, supplying a stub when unavailable.

### `get_owlready2()`

Return the ``owlready2`` module, supplying a stub when unavailable.

### `__getitem__(self, key)`

*No documentation available.*

### `bind(self, prefix, namespace)`

Register a namespace binding in the lightweight stub manager.

### `namespaces(self)`

Yield currently registered ``(prefix, namespace)`` pairs.

### `parse(self, source, format)`

Parse a Turtle-like text file into an in-memory triple list.

### `serialize(self, destination, format)`

Serialise parsed triples to the supplied destination.

### `add(self, triple)`

Append a triple to the stub graph, mirroring rdflib behaviour.

### `bind(self, prefix, namespace)`

Register a namespace binding within the stub graph.

### `namespaces(self)`

Yield namespace bindings previously registered via :meth:`bind`.

### `__len__(self)`

*No documentation available.*

### `__iter__(self)`

*No documentation available.*

### `join()`

Mimic :func:`pystow.join` by joining segments onto the stub root.

### `get_ontology(iri)`

Return a stub ontology instance for the provided IRI.

### `terms(self)`

Return a deterministic collection of ontology term identifiers.

### `dump(self, destination, format)`

Write minimal ontology contents to ``destination`` for tests.

### `load(self)`

Provide fluent API parity with owlready2 ontologies.

### `classes(self)`

Return example ontology classes for tests and fallbacks.

## 3. Classes

### `_StubNamespace`

*No documentation available.*

### `_StubNamespaceManager`

*No documentation available.*

### `_StubGraph`

*No documentation available.*

### `_StubOntology`

*No documentation available.*

### `_StubOntology`

*No documentation available.*
