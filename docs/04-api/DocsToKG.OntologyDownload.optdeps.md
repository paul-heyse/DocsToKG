# 1. Module: optdeps

This reference documents the DocsToKG module ``DocsToKG.OntologyDownload.optdeps``.

Centralized optional dependency management with fallback stubs.

The refactored ontology downloader relies on optional tooling for streaming
normalization, diagnostics, and validator subprocesses. These helpers expose
consistent access points that return either the real modules or lightweight
stubs, ensuring features like ROBOT health checks, Pronto/Owlready2 validation,
and filesystem abstractions work even in minimal test environments.

Examples:
    >>> from DocsToKG.OntologyDownload.optdeps import get_pystow
    >>> pystow = get_pystow()
    >>> cache_dir = pystow.join("ontology-fetcher", "cache")
    >>> cache_dir.as_posix()
    '.../ontology-fetcher/cache'

## 1. Functions

### `_create_stub_module(name, attrs)`

Return a :class:`ModuleType` populated with *attrs*.

### `_import_module(name)`

Import a module by name using :mod:`importlib`.

The indirection makes it trivial to monkeypatch the import logic in unit
tests without modifying global interpreter state.

### `get_pystow()`

Return the real :mod:`pystow` module or a fallback stub.

The helper caches the imported module so repeated calls do not incur
additional import overhead.

Args:
None

Returns:
Either the real :mod:`pystow` module or an instance of
:class:`_PystowFallback` when :mod:`pystow` is not installed.

Examples:
>>> pystow = get_pystow()
>>> cache_dir = pystow.join("ontology-fetcher", "cache")
>>> isinstance(cache_dir, Path)
True

### `get_rdflib()`

Return the real :mod:`rdflib` module or a stub replacement.

Args:
None

Returns:
Either the actual :mod:`rdflib` module or an instance of
:class:`_StubRDFLib` when :mod:`rdflib` is missing.

### `get_pronto()`

Return the real :mod:`pronto` module or a stub replacement.

Args:
None

Returns:
Either the actual :mod:`pronto` module or a :class:`_StubPronto`
instance when the dependency is unavailable.

### `get_owlready2()`

Return the real :mod:`owlready2` module or a stub replacement.

Args:
None

Returns:
Either the actual :mod:`owlready2` module or a :class:`_StubOwlready2`
instance when the dependency import fails.

### `join(self)`

Build a path relative to the fallback root directory.

Args:
*segments: Path segments appended to the root directory.

Returns:
Path object pointing to the requested cache location.

### `parse(self, source, format)`

Record the source path for later inspection.

Args:
source: Path to the RDF document to parse.
format: RDF serialization format. Ignored by the stub.

Returns:
None

Raises:
None.

### `__len__(self)`

Return the number of recorded triples.

Args:
None

Returns:
Integer representing the number of stored triples.

### `serialize(self, destination, format)`

Write a stub serialization result or return inline output.

Args:
destination: Output path, file-like object, or ``None`` for inline output.
format: Serialization format. Ignored by the stub.

Returns:
``None`` for file destinations, or a string when ``destination`` is ``None``.

Raises:
None.

### `terms(self)`

Yield placeholder ontology terms.

Args:
None

Returns:
Iterable of deterministic term identifiers.

Raises:
None.

### `dump(self, file, format)`

Write a stub ontology document to ``file``.

Args:
file: Destination file path.
format: Output format, either ``obo`` or ``obojson``.

Returns:
None

Raises:
None.

### `classes(self)`

Return a deterministic set of class identifiers.

Args:
None

Returns:
Copy of the stubbed class list to mimic Owlready2 behaviour.

Raises:
None.

### `load(self)`

Return a stub loaded ontology instance.

Args:
None

Returns:
Fresh :class:`_StubLoadedOntology` mirroring Owlready2 semantics.

Raises:
None.

### `get_ontology(iri)`

Return a stub ontology wrapper for the provided ``iri``.

Args:
iri: Ontology IRI to associate with the returned wrapper.

Returns:
New :class:`_StubOntologyWrapper` bound to the supplied IRI.

## 2. Classes

### `_PystowFallback`

Minimal ``pystow`` replacement used when the dependency is absent.

The real :mod:`pystow` module exposes a :func:`join` helper used throughout
the downloader. The fallback mirrors the behaviour closely enough for
testing by honouring the ``PYSTOW_HOME`` environment variable and defaulting
to ``~/.data`` when it is not defined.

Attributes:
_root: Root directory used to construct cache paths.

Examples:
>>> fallback = _PystowFallback()
>>> isinstance(fallback.join("ontology"), Path)
True

### `_StubGraph`

Minimal :class:`rdflib.Graph` replacement used for testing.

The stub captures the most common operations used in the downloader:
``parse`` to read a source file, ``serialize`` to write output, and
``__len__`` to introspect triple counts. Behaviour is intentionally simple
but deterministic so tests can assert on the resulting files.

Attributes:
_source: Path of the last parsed RDF document.
_triples: List of stub triples captured for serialization tests.

Examples:
>>> graph = _StubGraph()
>>> graph.parse("ontology.ttl")
>>> len(graph)
1

### `_StubRDFLib`

Stub module mirroring :mod:`rdflib` surface used in tests.

Attributes:
Graph: Replacement graph class mimicking ``rdflib.Graph``.

Examples:
>>> module = _StubRDFLib()
>>> isinstance(module.Graph(), _StubGraph)
True

### `_StubOntology`

Minimal :class:`pronto.Ontology` replacement.

Attributes:
_path: Filesystem path representing the ontology handle.

Examples:
>>> ontology = _StubOntology("example.obo")
>>> list(ontology.terms())
['TERM:0000001', 'TERM:0000002']

### `_StubPronto`

Stub module mirroring :mod:`pronto` surface used in tests.

Attributes:
Ontology: Stub ontology class mirroring :mod:`pronto`.

Examples:
>>> module = _StubPronto()
>>> isinstance(module.Ontology("example.obo"), _StubOntology)
True

### `_StubLoadedOntology`

Stub object returned by :func:`owlready2.get_ontology().load()`.

Attributes:
_classes: Deterministic class list for unit tests.

Examples:
>>> _StubLoadedOntology().classes()
['Class1', 'Class2', 'Class3']

### `_StubOntologyWrapper`

Stub ontology wrapper returned from ``get_ontology``.

Attributes:
_iri: IRI associated with the stub ontology.

Examples:
>>> wrapper = _StubOntologyWrapper("https://example.org/onto")
>>> isinstance(wrapper.load(), _StubLoadedOntology)
True

### `_StubOwlready2`

Stub module mirroring :mod:`owlready2` interface used in tests.

Attributes:
get_ontology: Callable returning stub ontology wrappers.

Examples:
>>> module = _StubOwlready2()
>>> module.get_ontology("https://example.org/onto").load().classes()
['Class1', 'Class2', 'Class3']
