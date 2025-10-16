"""Optional dependency helpers for the ontology downloader."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, List, Optional

_STUB_ATTR = "_ontofetch_stub"
_BNODE_COUNTER = 0

_pystow: Optional[Any] = None
_rdflib: Optional[Any] = None
_pronto: Optional[Any] = None
_owlready2: Optional[Any] = None


def _create_stub_module(name: str, attrs: Dict[str, Any]) -> ModuleType:
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    setattr(module, _STUB_ATTR, True)
    return module


def _create_stub_bnode(value: Optional[str] = None) -> str:
    global _BNODE_COUNTER
    if value is not None:
        return value
    _BNODE_COUNTER += 1
    return f"_:b{_BNODE_COUNTER}"


def _create_stub_literal(value: Any = None) -> str:
    if value is None:
        return '""'
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)


def _create_stub_uri(value: Optional[str] = None) -> str:
    if value is None:
        return "<>"
    if value.startswith("<") and value.endswith(">"):
        return value
    return f"<{value}>"


class _StubNamespace:
    def __init__(self, base: str) -> None:
        self._base = base

    def __getitem__(self, key: str) -> str:
        return f"{self._base}{key}"


class _StubNamespaceManager:
    def __init__(self) -> None:
        self._bindings: Dict[str, str] = {}

    def bind(self, prefix: str, namespace: str) -> None:
        self._bindings[prefix] = namespace

    def namespaces(self) -> Iterable[tuple[str, str]]:
        return self._bindings.items()


class _StubGraph:
    _ontofetch_stub = True

    def __init__(self) -> None:
        self._triples: List[tuple[str, str, str]] = []
        self._last_text = "# Stub TTL output\n"
        self.namespace_manager = _StubNamespaceManager()

    def parse(self, source: str, format: Optional[str] = None, **_kwargs: object) -> "_StubGraph":
        text = Path(source).read_text()
        self._last_text = text
        triples: List[tuple[str, str, str]] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("@prefix"):
                try:
                    _, remainder = line.split(None, 1)
                except ValueError:
                    continue
                parts = [segment.strip() for segment in remainder.split(None, 2)]
                if len(parts) >= 2:
                    prefix = parts[0].rstrip(":")
                    namespace = parts[1].strip("<>").rstrip(".")
                    self.namespace_manager.bind(prefix, namespace)
                line = parts[2].strip() if len(parts) == 3 else ""
                if not line:
                    continue
            if line.endswith("."):
                line = line[:-1].strip()
            if not line:
                continue
            pieces = line.split(None, 2)
            if len(pieces) < 3:
                continue
            triples.append(tuple(pieces))
        self._triples = triples
        return self

    def serialize(self, destination: Optional[Any] = None, format: Optional[str] = None, **_kwargs: object):
        if destination is None:
            return self._last_text
        if isinstance(destination, (str, Path)):
            Path(destination).write_text(self._last_text)
            return destination
        destination.write(b"# Stub TTL output\n")
        return destination

    def bind(self, prefix: str, namespace: str) -> None:
        self.namespace_manager.bind(prefix, namespace)

    def namespaces(self) -> Iterable[tuple[str, str]]:
        return self.namespace_manager.namespaces()

    def __len__(self) -> int:
        return len(self._triples)

    def __iter__(self):
        return iter(self._triples)


def _import_module(name: str) -> ModuleType:
    existing = sys.modules.get(name)
    if existing is not None and getattr(existing, _STUB_ATTR, False):
        sys.modules.pop(name, None)
    return importlib.import_module(name)


def _create_pystow_stub(root: Path) -> ModuleType:
    root.mkdir(parents=True, exist_ok=True)

    def join(*segments: str) -> Path:
        return root.joinpath(*segments)

    module = _create_stub_module("pystow", {"join": join})
    module._root = root  # type: ignore[attr-defined]
    return module


def _create_rdflib_stub() -> ModuleType:
    namespace_module = _create_stub_module(
        "rdflib.namespace",
        {
            "Namespace": _StubNamespace,
            "NamespaceManager": _StubNamespaceManager,
        },
    )
    graph_module = _create_stub_module(
        "rdflib.graph",
        {
            "Graph": _StubGraph,
        },
    )
    stub = _create_stub_module(
        "rdflib",
        {
            "Graph": _StubGraph,
            "Namespace": _StubNamespace,
            "NamespaceManager": _StubNamespaceManager,
            "BNode": _create_stub_bnode,
            "Literal": _create_stub_literal,
            "URIRef": _create_stub_uri,
        },
    )
    sys.modules.setdefault("rdflib.namespace", namespace_module)
    sys.modules.setdefault("rdflib.graph", graph_module)
    return stub


def _create_pronto_stub() -> ModuleType:
    class _StubOntology:
        _ontofetch_stub = True

        def __init__(self, _path: Optional[str] = None) -> None:
            self.path = _path

        def terms(self) -> Iterable[str]:
            return ["TERM:0000001", "TERM:0000002"]

        def dump(self, destination: str, format: str = "obojson") -> None:
            Path(destination).write_text('{"graphs": []}')

    return _create_stub_module("pronto", {"Ontology": _StubOntology})


def _create_owlready_stub() -> ModuleType:
    class _StubOntology:
        _ontofetch_stub = True

        def __init__(self, iri: str) -> None:
            self.iri = iri

        def load(self) -> "_StubOntology":
            return self

        def classes(self) -> List[str]:
            return ["Class1", "Class2", "Class3"]

    def get_ontology(iri: str) -> _StubOntology:
        return _StubOntology(iri)

    return _create_stub_module("owlready2", {"get_ontology": get_ontology})


def get_pystow() -> Any:
    global _pystow
    if _pystow is not None:
        return _pystow
    try:
        _pystow = _import_module("pystow")
    except ModuleNotFoundError:
        root = Path(os.environ.get("PYSTOW_HOME") or (Path.home() / ".data"))
        _pystow = _create_pystow_stub(root)
        sys.modules.setdefault("pystow", _pystow)
    return _pystow


def get_rdflib() -> Any:
    global _rdflib
    if _rdflib is not None:
        return _rdflib
    try:
        _rdflib = _import_module("rdflib")
    except ModuleNotFoundError:
        _rdflib = _create_rdflib_stub()
        sys.modules.setdefault("rdflib", _rdflib)
    return _rdflib


def get_pronto() -> Any:
    global _pronto
    if _pronto is not None:
        return _pronto
    try:
        _pronto = _import_module("pronto")
    except ModuleNotFoundError:
        _pronto = _create_pronto_stub()
        sys.modules.setdefault("pronto", _pronto)
    return _pronto


def get_owlready2() -> Any:
    global _owlready2
    if _owlready2 is not None:
        return _owlready2
    try:
        _owlready2 = _import_module("owlready2")
    except ModuleNotFoundError:
        _owlready2 = _create_owlready_stub()
        sys.modules.setdefault("owlready2", _owlready2)
    return _owlready2
