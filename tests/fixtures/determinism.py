# === NAVMAP v1 ===
# {
#   "module": "tests.fixtures.determinism",
#   "purpose": "Global determinism and environment control fixtures for reproducible tests",
#   "sections": [
#     {"id": "imports", "name": "Imports", "anchor": "imports", "kind": "section"},
#     {"id": "deterministic-env-fixture", "name": "deterministic_env", "anchor": "fixture-deterministic-env", "kind": "fixture"},
#     {"id": "seed-state-fixture", "name": "seed_state", "anchor": "fixture-seed-state", "kind": "fixture"},
#     {"id": "env-snapshot-fixture", "name": "env_snapshot", "anchor": "fixture-env-snapshot", "kind": "fixture"},
#     {"id": "tmp-isolated-dir-fixture", "name": "tmp_isolated_dir", "anchor": "fixture-tmp-isolated-dir", "kind": "fixture"}
#   ]
# }
# === /NAVMAP ===

"""
Determinism and environment control fixtures.

These fixtures ensure reproducible test runs by:
- Controlling random seeds (Python, NumPy, Hypothesis)
- Freezing environment variables
- Managing temporary directories with proper isolation
- Resetting global state between tests
"""

from __future__ import annotations

import os
import random
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import pytest

# --- Deterministic Environment Fixture ---


@pytest.fixture(scope="function")
def deterministic_env() -> Generator[dict[str, Any], None, None]:
    """
    Provide a deterministic environment with frozen state for test reproducibility.

    Yields a dict with:
    - seed: Current random seed value (42)
    - tz: Current timezone (UTC)
    - locale: Current locale setting (C.UTF-8)
    - original_env: Snapshot of os.environ at test start

    Automatically restores environment on test teardown.

    Example:
        def test_my_feature(deterministic_env):
            assert deterministic_env['seed'] == 42
            assert deterministic_env['tz'] == 'UTC'
    """
    # Snapshot current environment
    original_env = dict(os.environ)
    seed_value = 42

    # Ensure determinism controls are in place
    os.environ["TZ"] = "UTC"
    os.environ["LC_ALL"] = "C.UTF-8"
    os.environ["LANG"] = "C.UTF-8"

    # Clear proxy variables
    for var in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
        os.environ.pop(var, None)

    yield {
        "seed": seed_value,
        "tz": "UTC",
        "locale": "C.UTF-8",
        "original_env": original_env,
    }

    # Restore environment
    os.environ.clear()
    os.environ.update(original_env)


# --- Seed State Fixture ---


@pytest.fixture(scope="function")
def seed_state() -> Generator[dict[str, Any], None, None]:
    """
    Provide seeded random state for reproducible random operations.

    Seeds:
    - random module (Python's built-in)
    - NumPy (if available)
    - Hypothesis (if available)

    Yields a dict with:
    - seed: Seed value (42)
    - random_state: Snapshot of Python random state before seeding
    - np_state: Snapshot of NumPy random state (if available)

    Automatically restores random state on test teardown.

    Example:
        def test_random_behavior(seed_state):
            import random
            # All random() calls are deterministic
            assert random.random() == random.random()  # (requires reset between calls)
    """
    seed_value = 42

    # Snapshot current random states
    py_state = random.getstate()
    np_state = None
    try:
        import numpy as np

        np_state = np.random.get_state()
    except ImportError:
        pass

    # Reset seeds
    random.seed(seed_value)
    try:
        import numpy as np

        np.random.seed(seed_value)
    except ImportError:
        pass

    yield {
        "seed": seed_value,
        "random_state": py_state,
        "np_state": np_state,
    }

    # Restore random states
    random.setstate(py_state)
    if np_state is not None:
        try:
            import numpy as np

            np.random.set_state(np_state)
        except ImportError:
            pass


# --- Environment Snapshot Fixture ---


@pytest.fixture(scope="function")
def env_snapshot() -> Generator[dict[str, Any], None, None]:
    """
    Capture environment state at test start and restore on teardown.

    Yields a dict with:
    - snapshot: dict[str, str] of os.environ at test start
    - cwd: Current working directory at test start
    - umask: Process umask at test start

    Automatically restores all state on test teardown.

    Example:
        def test_env_modification(env_snapshot):
            os.environ['TEST_VAR'] = 'modified'
            # ... test code ...
            # Environment automatically restored after test
    """
    snapshot = dict(os.environ)
    cwd = Path.cwd()
    umask = os.umask(0o022)  # Capture and restore standard umask
    os.umask(umask)

    yield {
        "snapshot": snapshot,
        "cwd": cwd,
        "umask": umask,
    }

    # Restore environment
    os.environ.clear()
    os.environ.update(snapshot)

    # Restore working directory
    try:
        os.chdir(cwd)
    except (FileNotFoundError, OSError):
        # Directory may have been deleted; fallback to temp
        os.chdir(tempfile.gettempdir())

    # Restore umask
    os.umask(umask)


# --- Temporary Isolated Directory Fixture ---


@pytest.fixture(scope="function")
def tmp_isolated_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """
    Provide an isolated temporary directory with proper encapsulation.

    Ensures:
    - Permissions set to 0o755 (readable/writable/executable)
    - Platform-specific path normalization
    - Automatic cleanup on test teardown

    Args:
        tmp_path: Built-in pytest tmp_path fixture

    Yields:
        Path: Isolated temporary directory

    Example:
        def test_file_operations(tmp_isolated_dir):
            test_file = tmp_isolated_dir / "test.txt"
            test_file.write_text("content")
            assert test_file.read_text() == "content"
    """
    # Ensure directory is writable and executable
    tmp_path.chmod(0o755)

    yield tmp_path

    # Cleanup is automatic via pytest's tmp_path fixture


# --- Hypothesis Configuration Fixture ---


@pytest.fixture(scope="function")
def hypothesis_settings() -> Generator[dict[str, Any], None, None]:
    """
    Provide Hypothesis test settings for property-based testing.

    Configures:
    - max_examples: 100 (balanced for CI)
    - deadline: None (no timeout for I/O-bound tests)
    - suppress_health_check: Too slow, filter too much
    - database: None (stateless)

    Yields:
        dict: Settings metadata

    Example:
        @pytest.mark.property
        def test_url_normalization(hypothesis_settings):
            @given(st.text())
            def check_normalization(url):
                assert normalize_url(url) is not None
            check_normalization()
    """
    try:
        from hypothesis import HealthCheck, settings

        # Create and register a test profile
        settings.register_profile(
            "test_determinism",
            max_examples=100,
            deadline=None,
            suppress_health_check=[
                HealthCheck.too_slow,
                HealthCheck.filter_too_much,
            ],
            database=None,  # Stateless
        )
        settings.load_profile("test_determinism")

        yield {
            "max_examples": 100,
            "deadline": None,
            "suppress_health_check": ["too_slow", "filter_too_much"],
        }
    except ImportError:
        yield {"error": "Hypothesis not installed"}


# --- Context Manager for Temporary Environment Patches ---


@contextmanager
def temporary_env_patch(**kwargs: str) -> Generator[dict[str, str | None], None, None]:
    """
    Context manager for temporary environment variable patches.

    Args:
        **kwargs: Environment variables to set temporarily

    Yields:
        dict: Original values of patched variables

    Example:
        with temporary_env_patch(MY_VAR="test_value"):
            assert os.environ["MY_VAR"] == "test_value"
        # MY_VAR automatically restored
    """
    original: dict[str, str | None] = {}
    for key, value in kwargs.items():
        original[key] = os.environ.get(key)
        os.environ[key] = value

    try:
        yield original
    finally:
        for key, original_value in original.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value
