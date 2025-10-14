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


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "real_vectors: mark test as requiring the real-vector fixture"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--real-vectors"):
        return
    skip_marker = pytest.mark.skip(reason="requires --real-vectors")
    for item in items:
        if "real_vectors" in item.keywords:
            item.add_marker(skip_marker)
