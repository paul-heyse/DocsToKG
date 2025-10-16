# === NAVMAP v1 ===
# {
#   "module": "tests.conftest",
#   "purpose": "Shared pytest fixtures for suite",
#   "sections": [
#     {
#       "id": "install-pyalex-stub",
#       "name": "_install_pyalex_stub",
#       "anchor": "function-install-pyalex-stub",
#       "kind": "function"
#     },
#     {
#       "id": "pytest-addoption",
#       "name": "pytest_addoption",
#       "anchor": "function-pytest-addoption",
#       "kind": "function"
#     },
#     {
#       "id": "pytest-configure",
#       "name": "pytest_configure",
#       "anchor": "function-pytest-configure",
#       "kind": "function"
#     },
#     {
#       "id": "pytest-collection-modifyitems",
#       "name": "pytest_collection_modifyitems",
#       "anchor": "function-pytest-collection-modifyitems",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""
Pytest Configuration

This module configures shared pytest behaviour, including dynamic sys.path
management for `src`, CLI options for enabling real-vector suites, and
collection-time markers that gate hardware-intensive tests.

Key Scenarios:
- Adds command-line switches to opt into real-vector and scale suites
- Applies skip markers when vector fixtures are unavailable

Usage:
    pytest --help  # to inspect custom options
"""

from __future__ import annotations

import importlib.util
import sys
import types
import warnings
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import SimpleNamespace

import pytest

# --- Globals ---

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Always prefer the project virtual environment site-packages so tests see the
# same dependency set as the CLI.
VENV_ROOT = ROOT / ".venv"
if VENV_ROOT.exists():
    venv_prefix = VENV_ROOT.resolve()
    current_prefix = Path(sys.prefix).resolve()
    if current_prefix != venv_prefix:
        raise pytest.UsageError(
            f"Tests must run with the project virtual environment at {venv_prefix}. "
            "Activate it (e.g. `. .venv/bin/activate`) or invoke "
            f"`{venv_prefix}/bin/python -m pytest`."
        )

    site_packages_candidates = sorted((VENV_ROOT / "lib").glob("python*/site-packages"))
    for site_packages in site_packages_candidates:
        site_path = str(site_packages)
        if site_packages.exists() and site_path not in sys.path:
            sys.path.insert(0, site_path)
# --- Helper Functions ---


def _install_pyalex_stub() -> None:
    """Install a lightweight ``pyalex`` stub when the real dependency is absent."""

    if importlib.util.find_spec("pyalex") is not None:
        return

    stub = types.ModuleType("pyalex")
    stub.__file__ = "<pyalex-stub>"
    stub.__package__ = "pyalex"
    stub.__spec__ = ModuleSpec("pyalex", loader=None)  # type: ignore[arg-type]

    stub._works_results = []  # type: ignore[attr-defined]
    stub._topics_results = []  # type: ignore[attr-defined]

    class _QueryBase:
        """Minimal query object supporting chaining semantics used by the CLI."""

        def __init__(self, results=None):
            self._results = list(results) if results is not None else []

        def filter(self, **_filters):  # noqa: D401
            return self

        def search(self, _query):  # noqa: D401
            return self

        def select(self, _fields):  # noqa: D401
            return self

        def sort(self, **_fields):  # noqa: D401
            return self

        def paginate(self, per_page=25, n_max=None):  # noqa: D401
            data = list(self._results)

            if not data:
                return iter(())

            def _iterator():
                step = per_page or len(data)
                for start in range(0, len(data), step):
                    yield data[start : start + step]

            return _iterator()

    class Works(_QueryBase):
        def __init__(self, results=None):
            base = stub._works_results if results is None else results  # type: ignore[attr-defined]
            super().__init__(base)

    class Topics:
        def __init__(self, results=None):
            self._results = list(
                stub._topics_results if results is None else results  # type: ignore[attr-defined]
            )

        def search(self, _query):  # noqa: D401
            return self

        def get(self):  # noqa: D401
            return list(self._results)

    stub.Works = Works  # type: ignore[attr-defined]
    stub.Topics = Topics  # type: ignore[attr-defined]
    stub.config = SimpleNamespace(email=None)  # type: ignore[attr-defined]

    sys.modules.setdefault("pyalex", stub)


_install_pyalex_stub()


warnings.filterwarnings(
    "ignore",
    message=".*SwigPy.*__module__ attribute",
    category=DeprecationWarning,
)
# --- Test Cases ---


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--real-vectors",
        action="store_true",
        default=False,
        help="Run tests that require the real-vector fixture",
    )
    parser.addoption(
        "--scale-vectors",
        action="store_true",
        default=False,
        help="Run tests that require the large-scale real-vector fixture",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "real_vectors: mark test as requiring the real-vector fixture"
    )
    config.addinivalue_line(
        "markers", "scale_vectors: mark test as requiring the large-scale real-vector fixture"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    have_real = config.getoption("--real-vectors")
    have_scale = config.getoption("--scale-vectors")
    skip_real = pytest.mark.skip(reason="requires --real-vectors")
    skip_scale = pytest.mark.skip(reason="requires --scale-vectors")
    for item in items:
        if "real_vectors" in item.keywords and not have_real:
            item.add_marker(skip_real)
        if "scale_vectors" in item.keywords and not have_scale:
            item.add_marker(skip_scale)
