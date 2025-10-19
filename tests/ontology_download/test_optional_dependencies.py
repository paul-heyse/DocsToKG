"""Tests covering optional dependency fallbacks used by resolvers."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from DocsToKG.OntologyDownload import optdeps


def test_get_pystow_falls_back_to_stub(tmp_path: Path) -> None:
    """When pystow is unavailable the shim should provide a writable home."""

    with patch.object(optdeps, "_import_module", side_effect=ImportError):
        with patch.dict(os.environ, {"PYSTOW_HOME": str(tmp_path)}):
            module = optdeps.get_pystow()

    assert module.join("ontologies").parent == tmp_path


def test_get_rdflib_stub_handles_parse_and_serialize(tmp_path: Path) -> None:
    """Stub rdflib should expose parse/serialize helpers for tests."""

    with patch.object(optdeps, "_import_module", side_effect=ImportError):
        graph = optdeps.get_rdflib()

    source = tmp_path / "example.ttl"
    source.write_text("@prefix ex: <http://example.org/> . ex:a ex:b ex:c .\n")
    parsed = graph.parse(source)
    assert parsed is graph

    destination = tmp_path / "out.ttl"
    graph.serialize(destination)
    assert destination.exists()
