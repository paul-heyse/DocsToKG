"""Shared fixtures for ontology_download test suite."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

from DocsToKG.OntologyDownload.testing import TestingEnvironment


@pytest.fixture
def monkeypatch() -> None:  # type: ignore[override]
    """Disallow the built-in monkeypatch fixture inside ontology_download tests."""

    pytest.fail(
        "The pytest.monkeypatch fixture is not permitted in ontology_download tests.\n"
        "Use DocsToKG.OntologyDownload.testing.TestingEnvironment and the provided "
        "temporary_resolver/temporary_validator helpers instead of patching internals.",
        pytrace=False,
    )


def _ensure_optional_dependency_stubs() -> None:
    """Provide lightweight stubs for optional packages used by resolvers."""

    if "bioregistry" not in sys.modules:
        bioregistry = types.ModuleType("bioregistry")
        bioregistry.get_obo_download = lambda prefix: None  # type: ignore[attr-defined]
        bioregistry.get_owl_download = lambda prefix: None  # type: ignore[attr-defined]
        bioregistry.get_rdf_download = lambda prefix: None  # type: ignore[attr-defined]
        sys.modules["bioregistry"] = bioregistry

    if "ols_client" not in sys.modules:
        ols_client = types.ModuleType("ols_client")
        ols_client.OlsClient = lambda *args, **kwargs: types.SimpleNamespace()  # type: ignore[attr-defined]
        sys.modules["ols_client"] = ols_client

    if "ontoportal_client" not in sys.modules:
        onto_client = types.ModuleType("ontoportal_client")
        onto_client.BioPortalClient = lambda *args, **kwargs: types.SimpleNamespace()  # type: ignore[attr-defined]
        sys.modules["ontoportal_client"] = onto_client

    if "pystow" not in sys.modules:
        def _pystow_join(*segments):
            root = Path(os.environ.get("PYSTOW_HOME", Path.home() / ".data"))
            return root.joinpath(*segments)

        pystow_mod = types.ModuleType("pystow")
        pystow_mod.join = _pystow_join  # type: ignore[attr-defined]
        sys.modules["pystow"] = pystow_mod


_ensure_optional_dependency_stubs()


@pytest.fixture(scope="function")
def ontology_env():
    """Provision an isolated ontology download environment for tests."""

    with TestingEnvironment() as env:
        yield env


@pytest.fixture(scope="function")
def download_config(ontology_env):
    return ontology_env.build_download_config()


@pytest.fixture(scope="function")
def resolved_config(ontology_env):
    return ontology_env.build_resolved_config()
