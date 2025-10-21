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

import importlib
import importlib.util
import os
import sys
import types
import warnings
from collections.abc import MutableMapping
from contextlib import ExitStack
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Generator

import pytest

# Import determinism fixtures to make them globally available
from tests.fixtures.determinism import (  # noqa: F401
    deterministic_env,
    env_snapshot,
    hypothesis_settings,
    seed_state,
    temporary_env_patch,
    tmp_isolated_dir,
)

# Import Phase 2 fixtures (Optimization 9 Phase 2)
from tests.fixtures.http_mocking import (  # noqa: F401
    http_mock,
    mocked_http_client,
)
from tests.fixtures.duckdb_fixtures import (  # noqa: F401
    duckdb_migrations,
    duckdb_with_test_data,
    ephemeral_duckdb,
)
from tests.fixtures.telemetry_fixtures import (  # noqa: F401
    event_sink,
    mock_event_emitter,
    ratelimit_registry_reset,
)

# Import Phase 4 fixtures (Optimization 9 Phase 4 - Snapshots)
from tests.fixtures.snapshot_fixtures import (  # noqa: F401
    snapshot_manager,
)
from tests.fixtures.snapshot_assertions import (  # noqa: F401
    SnapshotAssertions,
    canonicalize_json,
)

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


_UNSET = object()


class PatchManager:
    """Lightweight patch helper that mirrors pytest's patcher API."""

    def __init__(self) -> None:
        self._stack = ExitStack()

    @staticmethod
    def _resolve_attr_target(target: Any, name: str | None) -> tuple[Any, str]:
        if isinstance(target, str):
            if name is not None:
                raise TypeError("name must be None when patching by dotted path")
            module_path = target
            attribute_chain: list[str] = []
            while True:
                try:
                    module = importlib.import_module(module_path)
                    break
                except ModuleNotFoundError as exc:
                    parent, sep, remainder = module_path.rpartition(".")
                    if not sep:
                        raise exc
                    attribute_chain.insert(0, remainder)
                    module_path = parent

            if not attribute_chain:
                raise TypeError("target must include an attribute name")

            obj: Any = module
            for attr_name in attribute_chain[:-1]:
                obj = getattr(obj, attr_name)
            return obj, attribute_chain[-1]
        if name is None:
            raise TypeError("name must be provided when patching objects")
        return target, name

    def setattr(
        self,
        target: Any,
        name: str | None = None,
        value: Any = _UNSET,
        *,
        raising: bool = True,
    ) -> Any:
        if isinstance(target, str):
            new_value = name if value is _UNSET else value
            if new_value is _UNSET:
                raise TypeError("value must be provided when patching by dotted path")
            obj, attr = self._resolve_attr_target(target, None)
        else:
            if name is None:
                raise TypeError("name must be provided when patching objects")
            if value is _UNSET:
                raise TypeError("value must be provided when patching objects")
            obj, attr = self._resolve_attr_target(target, name)
            new_value = value

        original = getattr(obj, attr, _UNSET)
        if original is _UNSET and raising:
            raise AttributeError(attr)

        setattr(obj, attr, new_value)

        def restore() -> None:
            if original is _UNSET:
                delattr(obj, attr)
            else:
                setattr(obj, attr, original)

        self._stack.callback(restore)
        return new_value

    def setitem(
        self,
        mapping: MutableMapping[Any, Any],
        key: Any,
        value: Any,
    ) -> Any:
        original = mapping.get(key, _UNSET)
        mapping[key] = value

        def restore() -> None:
            if original is _UNSET:
                mapping.pop(key, None)
            else:
                mapping[key] = original

        self._stack.callback(restore)
        return value

    def setenv(
        self,
        name: str,
        value: str,
        *,
        env: MutableMapping[str, str] | None = None,
    ) -> str:
        target_env = env or os.environ
        original = target_env.get(name, _UNSET)
        target_env[name] = value

        def restore() -> None:
            if original is _UNSET:
                target_env.pop(name, None)
            else:
                target_env[name] = original

        self._stack.callback(restore)
        return value

    def delitem(
        self,
        mapping: MutableMapping[Any, Any],
        key: Any,
        *,
        raising: bool = True,
    ) -> None:
        original = mapping.pop(key, _UNSET)
        if original is _UNSET and raising:
            raise KeyError(key)

        def restore() -> None:
            if original is _UNSET:
                return
            mapping[key] = original

        self._stack.callback(restore)

    def delattr(
        self,
        target: Any,
        name: str | None = None,
        *,
        raising: bool = True,
    ) -> None:
        obj, attr = self._resolve_attr_target(target, name)
        original = getattr(obj, attr, _UNSET)
        if original is _UNSET:
            if raising:
                raise AttributeError(attr)

            def restore_missing() -> None:
                if hasattr(obj, attr):
                    delattr(obj, attr)

            self._stack.callback(restore_missing)
            return

        delattr(obj, attr)

        def restore() -> None:
            setattr(obj, attr, original)

        self._stack.callback(restore)

    def delenv(
        self,
        name: str,
        *,
        env: MutableMapping[str, str] | None = None,
        raising: bool = True,
    ) -> None:
        target_env = env or os.environ
        original = target_env.pop(name, _UNSET)
        if original is _UNSET and raising:
            raise KeyError(name)

        def restore() -> None:
            if original is _UNSET:
                return
            target_env[name] = original

        self._stack.callback(restore)

    def syspath_prepend(self, path: str | os.PathLike[str]) -> None:
        str_path = os.fspath(path)
        sys_path = sys.path
        sys_path.insert(0, str_path)

        def restore() -> None:
            try:
                sys_path.remove(str_path)
            except ValueError:
                pass

        self._stack.callback(restore)

    def chdir(self, path: str | os.PathLike[str]) -> None:
        original = Path.cwd()
        os.chdir(path)

        def restore() -> None:
            os.chdir(original)

        self._stack.callback(restore)

    def close(self) -> None:
        self._stack.close()


@pytest.fixture
def patcher() -> Generator[PatchManager, None, None]:
    """Auto-reverting patch helper based on unittest.mock."""

    manager = PatchManager()
    try:
        yield manager
    finally:
        manager.close()


warnings.filterwarnings(
    "ignore",
    message=".*SwigPy.*__module__ attribute",
    category=DeprecationWarning,
)
# --- Determinism Controls (Optimization 9) ---


def _configure_determinism() -> None:
    """
    Initialize global determinism controls for reproducible test runs.

    Controls:
    - PYTHONHASHSEED: Disable hash randomization
    - random.seed: Python's random module seed
    - numpy.random.seed: NumPy random seed (if available)
    - TZ: Set timezone to UTC
    - LOCALE: Fix locale to C.UTF-8
    - Environment: Clear proxy variables, disable telemetry
    """
    # Hash seed (already set by PYTHONHASHSEED env var, but enforce at init)
    os.environ.setdefault("PYTHONHASHSEED", "42")

    # Timezone to UTC for reproducible time-dependent tests
    os.environ["TZ"] = "UTC"

    # Locale to C.UTF-8 for consistent path/string handling across platforms
    os.environ.setdefault("LC_ALL", "C.UTF-8")
    os.environ.setdefault("LANG", "C.UTF-8")

    # Disable proxy/network leakage
    for var in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "NO_PROXY", "no_proxy"]:
        os.environ.pop(var, None)

    # Disable any telemetry "phone home" behavior
    os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

    # Seed Python's random module (seed value is arbitrary but fixed)
    import random

    random.seed(42)

    # Seed NumPy if available
    try:
        import numpy as np

        np.random.seed(42)
    except ImportError:
        pass

    # Configure Hypothesis if available
    try:
        from hypothesis import HealthCheck, settings

        settings.register_profile(
            "test",
            max_examples=100,
            deadline=None,  # No deadline for I/O-bound tests
            suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
        )
        settings.load_profile("test")
    except ImportError:
        pass


# Initialize determinism at module load time
_configure_determinism()

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
    # --- Optimization 9 test strata markers ---
    config.addinivalue_line(
        "markers",
        "unit: mark test as pure unit test (no I/O, <50ms, <50 LOC). "
        "Use for isolated function/method testing without fixtures beyond mocks.",
    )
    config.addinivalue_line(
        "markers",
        "component: mark test as component-level (touches one subsystem, <500ms). "
        "Use for HTTP client, extractor, catalog, or ratelimiter integration tests.",
    )
    config.addinivalue_line(
        "markers",
        "e2e: mark test as end-to-end (small dataset, plan→pull→extract→validate, <5s). "
        "Use for full pipeline verification with tiny archives.",
    )
    config.addinivalue_line(
        "markers",
        "property: mark test as property-based (Hypothesis). "
        "Use for generative testing of gates, URLs, paths, extraction ratios.",
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow/heavy (opt-in for nightly/local runs). "
        "Use for large-dataset, long-running, or resource-intensive tests.",
    )
    config.addinivalue_line(
        "markers",
        "windows_only: mark test as Windows-specific. "
        "Use for tests that require Windows-only path handling or APIs.",
    )
    config.addinivalue_line(
        "markers",
        "posix_only: mark test as POSIX-specific (Linux/macOS). "
        "Use for tests that require POSIX path handling or APIs.",
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
