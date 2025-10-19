"""Shared fixtures for ontology_download test suite."""

from __future__ import annotations

import importlib
import importlib.util
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

    if "pystow" not in sys.modules and importlib.util.find_spec("pystow") is None:
        data_root = Path(os.environ.get("PYSTOW_HOME", Path.home() / ".data"))

        def _ensure_parent(path: Path) -> Path:
            path.parent.mkdir(parents=True, exist_ok=True)
            return path

        def _pystow_join(*segments: str) -> Path:
            return _ensure_parent(data_root.joinpath(*segments))

        class _PystowModule:
            def __init__(self, base: Path) -> None:
                self.base = base

            def join(self, *segments: str) -> Path:
                return _ensure_parent(self.base.joinpath(*segments))

            joinpath = join

            def ensure(self, *segments: str, directory: bool = True) -> Path:
                target = self.base.joinpath(*segments)
                if directory:
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    _ensure_parent(target)
                    target.touch(exist_ok=True)
                return target

            def module(self, *segments: str, ensure_exists: bool = True) -> "_PystowModule":
                return _pystow_module(*segments, ensure_exists=ensure_exists)

        def _pystow_module(*segments: str, ensure_exists: bool = True) -> _PystowModule:
            module_root = data_root.joinpath(*segments)
            if ensure_exists:
                module_root.mkdir(parents=True, exist_ok=True)
            return _PystowModule(module_root)

        def _pystow_get_config(_module: str, _key: str, *, default=None, **_kwargs):
            return default

        pystow_mod = types.ModuleType("pystow")
        pystow_mod.join = _pystow_join  # type: ignore[attr-defined]
        pystow_mod.joinpath = _pystow_join  # type: ignore[attr-defined]
        pystow_mod.module = _pystow_module  # type: ignore[attr-defined]
        pystow_mod.ensure = lambda *segments, **kwargs: _pystow_module(*segments, **kwargs).base  # type: ignore[attr-defined]
        pystow_mod.get_config = _pystow_get_config  # type: ignore[attr-defined]
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
