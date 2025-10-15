"""Centralized optional dependency management with fallback stubs.

This module exposes helper functions that return optional third-party
dependencies used across the ontology downloader. When a dependency is not
installed in the current environment the helpers return lightweight stub
implementations that provide just enough behaviour for tests and basic
execution paths. The approach keeps optional dependencies truly optional
while avoiding scattered try/except blocks throughout the codebase.

Examples:
    >>> from DocsToKG.OntologyDownload.optdeps import get_pystow
    >>> pystow = get_pystow()
    >>> cache_dir = pystow.join("ontology-fetcher", "cache")
    >>> cache_dir.as_posix()
    '.../ontology-fetcher/cache'
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any, Iterable, List, Optional

__all__ = [
    "get_pystow",
    "get_rdflib",
    "get_pronto",
    "get_owlready2",
]

# Module level caches ensure we only attempt imports once per interpreter
# session. Each cache stores either the real module or the corresponding stub
# implementation after the first lookup.
_pystow: Optional[Any] = None
_rdflib: Optional[Any] = None
_pronto: Optional[Any] = None
_owlready2: Optional[Any] = None


def _import_module(name: str) -> Any:
    """Import a module by name using :mod:`importlib`.

    The indirection makes it trivial to monkeypatch the import logic in unit
    tests without modifying global interpreter state.

    Args:
        name: Fully qualified module name to import.

    Returns:
        Imported module object.
    """

    return importlib.import_module(name)


class _PystowFallback:
    """Minimal ``pystow`` replacement used when the dependency is absent.

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
    """

    def __init__(self) -> None:
        root = os.environ.get("PYSTOW_HOME")
        self._root = Path(root) if root is not None else Path.home() / ".data"

    def join(self, *segments: str) -> Path:
        """Build a path relative to the fallback root directory.

        Args:
            *segments: Path segments appended to the root directory.

        Returns:
            Path object pointing to the requested cache location.
        """

        return self._root.joinpath(*segments)


def get_pystow() -> Any:
    """Return the real :mod:`pystow` module or a fallback stub.

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
    """

    global _pystow
    if _pystow is None:
        try:  # pragma: no cover - exercised in environments with dependency
            _pystow = _import_module("pystow")  # type: ignore
        except ModuleNotFoundError:  # pragma: no cover - stub path tested
            _pystow = _PystowFallback()
            sys.modules.setdefault("pystow", _pystow)  # type: ignore[arg-type]
    return _pystow


class _StubGraph:
    """Minimal :class:`rdflib.Graph` replacement used for testing.

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
    """

    def __init__(self) -> None:
        self._source: Optional[Path] = None
        self._triples: List[tuple[Any, ...]] = []

    def parse(self, source: str, format: Optional[str] = None) -> None:
        """Record the source path for later inspection.

        Args:
            source: Path to the RDF document to parse.
            format: RDF serialization format. Ignored by the stub.

        Returns:
            None

        Raises:
            None.
        """

        self._source = Path(source)
        self._triples = [("s", "p", "o")]

    def __len__(self) -> int:
        """Return the number of recorded triples.

        Args:
            None

        Returns:
            Integer representing the number of stored triples.
        """

        return len(self._triples) or 1

    def serialize(self, destination: Any, format: str = "turtle") -> None:
        """Write a stub serialization result.

        Args:
            destination: Output path or file-like object.
            format: Serialization format. Ignored by the stub.

        Returns:
            None

        Raises:
            None.
        """

        if isinstance(destination, (str, Path)):
            dest_path = Path(destination)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            if self._source and self._source.exists():
                dest_path.write_bytes(self._source.read_bytes())
            else:
                dest_path.write_text("# Stub TTL output\n")
        else:
            destination.write(b"# Stub TTL output\n")


class _StubRDFLib:
    """Stub module mirroring :mod:`rdflib` surface used in tests.

    Attributes:
        Graph: Replacement graph class mimicking ``rdflib.Graph``.

    Examples:
        >>> module = _StubRDFLib()
        >>> isinstance(module.Graph(), _StubGraph)
        True
    """

    Graph = _StubGraph


def get_rdflib() -> Any:
    """Return the real :mod:`rdflib` module or a stub replacement.

    Args:
        None

    Returns:
        Either the actual :mod:`rdflib` module or an instance of
        :class:`_StubRDFLib` when :mod:`rdflib` is missing.
    """

    global _rdflib
    if _rdflib is None:
        try:  # pragma: no cover - requires dependency
            _rdflib = _import_module("rdflib")  # type: ignore
        except ModuleNotFoundError:  # pragma: no cover - stub behaviour tested
            _rdflib = _StubRDFLib()
            sys.modules.setdefault("rdflib", _rdflib)  # type: ignore[arg-type]
    return _rdflib


class _StubOntology:
    """Minimal :class:`pronto.Ontology` replacement.

    Attributes:
        _path: Filesystem path representing the ontology handle.

    Examples:
        >>> ontology = _StubOntology("example.obo")
        >>> list(ontology.terms())
        ['TERM:0000001', 'TERM:0000002']
    """

    def __init__(self, handle: str) -> None:
        self._path = Path(handle)

    def terms(self) -> Iterable[str]:
        """Yield placeholder ontology terms.

        Args:
            None

        Returns:
            Iterable of deterministic term identifiers.

        Raises:
            None.
        """

        return ["TERM:0000001", "TERM:0000002"]

    def dump(self, file: str, format: str = "obo") -> None:
        """Write a stub ontology document to ``file``.

        Args:
            file: Destination file path.
            format: Output format, either ``obo`` or ``obojson``.

        Returns:
            None

        Raises:
            None.
        """

        output_path = Path(file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if format == "obojson":
            output_path.write_text('{"graphs": []}')
        else:
            output_path.write_text("format-version: 1.2\n")


class _StubPronto:
    """Stub module mirroring :mod:`pronto` surface used in tests.

    Attributes:
        Ontology: Stub ontology class mirroring :mod:`pronto`.

    Examples:
        >>> module = _StubPronto()
        >>> isinstance(module.Ontology("example.obo"), _StubOntology)
        True
    """

    Ontology = _StubOntology


def get_pronto() -> Any:
    """Return the real :mod:`pronto` module or a stub replacement.

    Args:
        None

    Returns:
        Either the actual :mod:`pronto` module or a :class:`_StubPronto`
        instance when the dependency is unavailable.
    """
    global _pronto
    if _pronto is None:
        try:  # pragma: no cover - requires dependency
            _pronto = _import_module("pronto")  # type: ignore
        except ModuleNotFoundError:  # pragma: no cover - stub path tested
            _pronto = _StubPronto()
            sys.modules.setdefault("pronto", _pronto)  # type: ignore[arg-type]
    return _pronto


class _StubLoadedOntology:
    """Stub object returned by :func:`owlready2.get_ontology().load()`.

    Attributes:
        _classes: Deterministic class list for unit tests.

    Examples:
        >>> _StubLoadedOntology().classes()
        ['Class1', 'Class2', 'Class3']
    """

    def __init__(self) -> None:
        self._classes: List[str] = ["Class1", "Class2", "Class3"]

    def classes(self) -> List[str]:
        """Return a deterministic set of class identifiers.

        Args:
            None

        Returns:
            Copy of the stubbed class list to mimic Owlready2 behaviour.

        Raises:
            None.
        """

        return self._classes.copy()


class _StubOntologyWrapper:
    """Stub ontology wrapper returned from ``get_ontology``.

    Attributes:
        _iri: IRI associated with the stub ontology.

    Examples:
        >>> wrapper = _StubOntologyWrapper("https://example.org/onto")
        >>> isinstance(wrapper.load(), _StubLoadedOntology)
        True
    """

    def __init__(self, iri: str) -> None:
        self._iri = iri

    def load(self) -> _StubLoadedOntology:
        """Return a stub loaded ontology instance.

        Args:
            None

        Returns:
            Fresh :class:`_StubLoadedOntology` mirroring Owlready2 semantics.

        Raises:
            None.
        """

        return _StubLoadedOntology()


class _StubOwlready2:
    """Stub module mirroring :mod:`owlready2` interface used in tests.

    Attributes:
        get_ontology: Callable returning stub ontology wrappers.

    Examples:
        >>> module = _StubOwlready2()
        >>> module.get_ontology("https://example.org/onto").load().classes()
        ['Class1', 'Class2', 'Class3']
    """

    @staticmethod
    def get_ontology(iri: str) -> _StubOntologyWrapper:
        """Return a stub ontology wrapper for the provided ``iri``.

        Args:
            iri: Ontology IRI to associate with the returned wrapper.

        Returns:
            New :class:`_StubOntologyWrapper` bound to the supplied IRI.
        """

        return _StubOntologyWrapper(iri)


def get_owlready2() -> Any:
    """Return the real :mod:`owlready2` module or a stub replacement.

    Args:
        None

    Returns:
        Either the actual :mod:`owlready2` module or a :class:`_StubOwlready2`
        instance when the dependency import fails.
    """
    global _owlready2
    if _owlready2 is None:
        try:  # pragma: no cover - requires dependency
            _owlready2 = _import_module("owlready2")  # type: ignore
        except ModuleNotFoundError:  # pragma: no cover - stub behaviour tested
            _owlready2 = _StubOwlready2()
            sys.modules.setdefault("owlready2", _owlready2)  # type: ignore[arg-type]
    return _owlready2
