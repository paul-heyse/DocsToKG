"""Shared fixtures for ontology_download test suite."""

from __future__ import annotations

import os
import sys
import types
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pytest

from DocsToKG.OntologyDownload.testing import TestingEnvironment


class PatchManager:
    """Temporary patch helper to support legacy tests during migration."""

    def __init__(self) -> None:
        self._stack = ExitStack()

    def close(self) -> None:
        self._stack.close()

    def setattr(self, target, *args, raising: bool = True):
        if isinstance(target, str):
            if len(args) != 1:
                raise TypeError("setattr string target expects exactly one value")
            (value,) = args
            patcher = patch(target, value, create=not raising)
        else:
            if len(args) != 2:
                raise TypeError("setattr object target expects name and value")
            name, value = args
            patcher = patch.object(target, name, value, create=not raising)
        self._stack.enter_context(patcher)
        return value

    def setitem(self, mapping, key, value):
        sentinel = object()
        original = mapping.get(key, sentinel)
        mapping[key] = value

        def restore():
            if original is sentinel:
                mapping.pop(key, None)
            else:
                mapping[key] = original

        self._stack.callback(restore)
        return value

    def setenv(self, key: str, value: str) -> str:
        sentinel = object()
        original = os.environ.get(key, sentinel)
        os.environ[key] = value

        def restore():
            if original is sentinel:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original

        self._stack.callback(restore)
        return value

    def delenv(self, key: str, *, raising: bool = True) -> None:
        sentinel = object()
        if key in os.environ:
            original = os.environ.pop(key)
        else:
            if raising:
                raise KeyError(key)
            original = sentinel

        def restore():
            if original is not sentinel:
                os.environ[key] = original

        self._stack.callback(restore)

    def enter_context(self, ctx):
        return self._stack.enter_context(ctx)

    def callback(self, func):
        self._stack.callback(func)


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


@pytest.fixture
def patch_stack():
    manager = PatchManager()
    try:
        yield manager
    finally:
        manager.close()


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
