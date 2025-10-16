# === NAVMAP v1 ===
# {
#   "module": "tests.ontology_download.test_optdeps",
#   "purpose": "Pytest coverage for ontology download optdeps scenarios",
#   "sections": [
#     {
#       "id": "reset-caches",
#       "name": "_reset_caches",
#       "anchor": "function-reset-caches",
#       "kind": "function"
#     },
#     {
#       "id": "fake-import",
#       "name": "_fake_import",
#       "anchor": "function-fake-import",
#       "kind": "function"
#     },
#     {
#       "id": "failing-import",
#       "name": "_failing_import",
#       "anchor": "function-failing-import",
#       "kind": "function"
#     },
#     {
#       "id": "test-get-pystow-with-real-module",
#       "name": "test_get_pystow_with_real_module",
#       "anchor": "function-test-get-pystow-with-real-module",
#       "kind": "function"
#     },
#     {
#       "id": "test-get-pystow-fallback-when-missing",
#       "name": "test_get_pystow_fallback_when_missing",
#       "anchor": "function-test-get-pystow-fallback-when-missing",
#       "kind": "function"
#     },
#     {
#       "id": "test-pystow-fallback-respects-env-var",
#       "name": "test_pystow_fallback_respects_env_var",
#       "anchor": "function-test-pystow-fallback-respects-env-var",
#       "kind": "function"
#     },
#     {
#       "id": "test-get-rdflib-with-real-module",
#       "name": "test_get_rdflib_with_real_module",
#       "anchor": "function-test-get-rdflib-with-real-module",
#       "kind": "function"
#     },
#     {
#       "id": "test-rdflib-stub-parse-and-serialize",
#       "name": "test_rdflib_stub_parse_and_serialize",
#       "anchor": "function-test-rdflib-stub-parse-and-serialize",
#       "kind": "function"
#     },
#     {
#       "id": "test-pronto-stub-behaviour",
#       "name": "test_pronto_stub_behaviour",
#       "anchor": "function-test-pronto-stub-behaviour",
#       "kind": "function"
#     },
#     {
#       "id": "test-owlready2-stub-load",
#       "name": "test_owlready2_stub_load",
#       "anchor": "function-test-owlready2-stub-load",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Tests for optional dependency helpers."""

from __future__ import annotations

import importlib
import io
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("pydantic_settings")

from DocsToKG.OntologyDownload import optdeps


@pytest.fixture(autouse=True)
# --- Helper Functions ---


def _reset_caches() -> None:
    """Ensure module caches are reset before each test."""

    optdeps._pystow = None  # type: ignore[attr-defined]
    optdeps._rdflib = None  # type: ignore[attr-defined]
    optdeps._pronto = None  # type: ignore[attr-defined]
    optdeps._owlready2 = None  # type: ignore[attr-defined]
    for module_name in ("pystow", "rdflib", "pronto", "owlready2"):
        sys.modules.pop(module_name, None)


def _fake_import(module_name: str, module: Any):
    """Return a patch function that mimics :func:`importlib.import_module`."""

    def _import(name: str) -> Any:
        if name == module_name:
            return module
        return importlib.import_module(name)

    return _import


def _failing_import(module_name: str):
    """Return a patch function that raises :class:`ModuleNotFoundError`."""

    def _import(name: str) -> Any:
        if name == module_name:
            raise ModuleNotFoundError(name)
        return importlib.import_module(name)

    return _import


# --- Test Cases ---


def test_get_pystow_with_real_module(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``get_pystow`` returns the imported module when available."""

    fake_pystow = SimpleNamespace(join=lambda *parts: tmp_path.joinpath(*parts))
    monkeypatch.setattr(optdeps, "_import_module", _fake_import("pystow", fake_pystow))

    module = optdeps.get_pystow()
    again = optdeps.get_pystow()

    assert module is fake_pystow
    assert again is module
    assert module.join("a", "b") == tmp_path / "a" / "b"


def test_get_pystow_fallback_when_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``get_pystow`` returns fallback when import fails."""

    monkeypatch.setenv("PYSTOW_HOME", str(tmp_path))
    monkeypatch.setattr(optdeps, "_import_module", _failing_import("pystow"))

    module = optdeps.get_pystow()

    assert isinstance(module, ModuleType)
    assert hasattr(module, "join")
    assert module.join("cache").is_relative_to(tmp_path)


def test_pystow_fallback_respects_env_var(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Fallback root honours the ``PYSTOW_HOME`` environment variable."""

    monkeypatch.setenv("PYSTOW_HOME", str(tmp_path))
    monkeypatch.setattr(optdeps, "_import_module", _failing_import("pystow"))

    module = optdeps.get_pystow()

    cache_path = module.join("ontology")  # type: ignore[attr-defined]
    assert cache_path == tmp_path / "ontology"


def test_get_rdflib_with_real_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """``get_rdflib`` returns imported module when available."""

    fake_graph = SimpleNamespace()
    fake_rdflib = SimpleNamespace(Graph=fake_graph)
    monkeypatch.setattr(optdeps, "_import_module", _fake_import("rdflib", fake_rdflib))

    module = optdeps.get_rdflib()

    assert module is fake_rdflib
    assert module.Graph is fake_graph


def test_rdflib_stub_parse_and_serialize(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub ``Graph`` records source and writes serialized output."""

    monkeypatch.setattr(optdeps, "_import_module", _failing_import("rdflib"))
    module = optdeps.get_rdflib()
    graph = module.Graph()

    source = tmp_path / "test.ttl"
    source.write_text("@prefix : <#>. :a :b :c .\n")
    graph.parse(str(source))
    assert len(graph) == 1

    destination = tmp_path / "out.ttl"
    graph.serialize(destination)
    assert destination.read_text() == source.read_text()

    buffer = io.BytesIO()
    graph.serialize(buffer)
    assert buffer.getvalue() == b"# Stub TTL output\n"


def test_pronto_stub_behaviour(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Pronto stub provides ``terms`` and ``dump`` methods."""

    monkeypatch.setattr(optdeps, "_import_module", _failing_import("pronto"))
    module = optdeps.get_pronto()
    ontology = module.Ontology(str(tmp_path / "ont.obo"))

    terms = list(ontology.terms())
    assert terms == ["TERM:0000001", "TERM:0000002"]

    output = tmp_path / "out.obojson"
    ontology.dump(str(output), format="obojson")
    assert output.read_text() == '{"graphs": []}'


def test_owlready2_stub_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """Owlready2 stub returns wrapper capable of loading classes."""

    monkeypatch.setattr(optdeps, "_import_module", _failing_import("owlready2"))
    module = optdeps.get_owlready2()
    wrapper = module.get_ontology("file:///tmp/ont.owl")
    loaded = wrapper.load()
    assert loaded.classes() == ["Class1", "Class2", "Class3"]
