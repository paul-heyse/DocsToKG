"""
Ontology Test Environment Stubs

This module wires lightweight stubs for optional dependencies used in the
ontology download suite so tests can run without network access or heavy
third-party packages.

Key Scenarios:
- Provides fallback implementations for requests, BioPortal, OLS, and Pystow
- Supplies stubbed telemetry/logging clients for validator invocation

Usage:
    pytest tests/ontology_download -k ontology_download
"""

import importlib.machinery
import logging
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


class _StubRequestsModule(SimpleNamespace):
    """Lightweight stand-in for the requests module used in tests."""


class _RequestException(Exception):
    pass


class _HTTPError(_RequestException):
    def __init__(self, message: str = "", response=None) -> None:
        super().__init__(message)
        self.response = response


class _Timeout(_RequestException):
    pass


class _ConnectionError(_RequestException):
    pass


class _SSLError(_RequestException):
    pass


class _StubSession:
    def get(self, *args, **kwargs):  # pragma: no cover - patched in tests
        raise RuntimeError("requests.Session stub used without monkeypatching")


class _StubResponse:
    def __init__(self) -> None:
        self.status_code = None
        self.headers = {}
        self.url = ""


class _PlaceholderClient:
    def __init__(self, *args, **kwargs):  # pragma: no cover - should be patched in tests
        pass

    def __getattr__(self, item):  # pragma: no cover - should be replaced in tests
        raise RuntimeError(f"Accessed placeholder client attribute '{item}' without patching")


if "bioregistry" not in sys.modules:
    _stub_bioregistry = types.ModuleType("bioregistry")
    _stub_bioregistry.get_obo_download = lambda prefix: None  # type: ignore[attr-defined]
    _stub_bioregistry.get_owl_download = lambda prefix: None  # type: ignore[attr-defined]
    _stub_bioregistry.get_rdf_download = lambda prefix: None  # type: ignore[attr-defined]
    sys.modules["bioregistry"] = _stub_bioregistry

if "ols_client" not in sys.modules:
    _stub_ols = types.ModuleType("ols_client")
    _stub_ols.OlsClient = _PlaceholderClient  # type: ignore[attr-defined]
    sys.modules["ols_client"] = _stub_ols

if "ontoportal_client" not in sys.modules:
    _stub_ontoportal = types.ModuleType("ontoportal_client")
    _stub_ontoportal.BioPortalClient = _PlaceholderClient  # type: ignore[attr-defined]
    sys.modules["ontoportal_client"] = _stub_ontoportal

if "requests" not in sys.modules:
    try:  # Prefer the real requests library if available
        import requests as _real_requests  # type: ignore
    except ModuleNotFoundError:
        sys.modules["requests"] = _StubRequestsModule(
            Session=_StubSession,
            HTTPError=_HTTPError,
            Timeout=_Timeout,
            ConnectionError=_ConnectionError,
            RequestException=_RequestException,
            Response=_StubResponse,
            exceptions=SimpleNamespace(SSLError=_SSLError),
        )
    else:
        sys.modules["requests"] = _real_requests

if "pystow" not in sys.modules:

    def _pystow_join(*segments):
        root = Path(os.environ.get("PYSTOW_HOME", Path.home() / ".data"))
        return root.joinpath(*segments)

    _stub_pystow = types.ModuleType("pystow")
    _stub_pystow.join = _pystow_join  # type: ignore[attr-defined]
    sys.modules["pystow"] = _stub_pystow

if "psutil" not in sys.modules:
    try:
        import psutil as _real_psutil  # type: ignore
    except ModuleNotFoundError:

        class _StubProcess:
            def memory_info(self):
                return SimpleNamespace(rss=0)

        _stub_psutil = types.ModuleType("psutil")
        _stub_psutil.Process = lambda: _StubProcess()  # type: ignore[attr-defined]
        _stub_psutil.__spec__ = importlib.machinery.ModuleSpec("psutil", loader=None)
        sys.modules["psutil"] = _stub_psutil
    else:
        sys.modules["psutil"] = _real_psutil

if "pooch" not in sys.modules:

    class _HTTPDownloader:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(
            self, url, output_file, pooch_logger
        ):  # pragma: no cover - overridden in tests
            raise NotImplementedError("pooch downloader stub should be overridden")

    def _retrieve(url, *, path, fname, downloader, **kwargs):
        cache_dir = Path(path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        output = cache_dir / fname
        downloader(url, str(output), logging.getLogger("pooch"))
        return str(output)

    _stub_pooch = types.ModuleType("pooch")
    _stub_pooch.HTTPDownloader = _HTTPDownloader  # type: ignore[attr-defined]
    _stub_pooch.retrieve = _retrieve  # type: ignore[attr-defined]
    sys.modules["pooch"] = _stub_pooch

from DocsToKG.OntologyDownload import core  # noqa: E402  (after stubs)

_ORIGINAL_BUILD_DESTINATION = core._build_destination


@pytest.fixture(autouse=True)
def _reset_build_destination(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure tests see the canonical _build_destination implementation."""

    monkeypatch.setattr(core, "_build_destination", _ORIGINAL_BUILD_DESTINATION, raising=False)


@pytest.fixture(autouse=True)
def _ontology_env(tmp_path_factory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide deterministic environment variables expected by validators."""

    pystow_home = tmp_path_factory.mktemp("pystow-home")
    monkeypatch.setenv("PYSTOW_HOME", str(pystow_home))
    monkeypatch.setenv("BIOPORTAL_API_KEY", "test-bioportal-key")
