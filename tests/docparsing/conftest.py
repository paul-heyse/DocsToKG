"""Shared pytest fixtures for DocParsing test suite."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _restore_environ() -> None:
    """Snapshot environment variables and restore them after each test."""

    original = os.environ.copy()
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(original)


@pytest.fixture(autouse=True)
def _restore_cwd() -> None:
    """Ensure tests leave the current working directory unchanged."""

    original_cwd = Path.cwd()
    try:
        yield
    finally:
        os.chdir(original_cwd)
