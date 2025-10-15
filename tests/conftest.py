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
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


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
