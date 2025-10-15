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

import sys
import warnings
from pathlib import Path

import pytest

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


warnings.filterwarnings(
    "ignore",
    message=".*SwigPy.*__module__ attribute",
    category=DeprecationWarning,
)


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
